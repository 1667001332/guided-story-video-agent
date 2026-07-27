from __future__ import annotations

import json
import re
import warnings
from copy import deepcopy
from functools import wraps
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from .agent import IDEA_COUNT, RuleBasedStoryAgent, StoryAgent
from .continuity import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VISUAL_REFERENCE_USAGES,
    freeze_confirmed_visual_inputs,
    verify_confirmed_visual_inputs,
)
from .models import (
    ArtifactReview,
    ArtifactRevision,
    ContinuityState,
    CreativeBrief,
    CreativeSuggestion,
    DraftBundle,
    ElementOption,
    ElementPalette,
    GuideTurnResult,
    IdeaBatch,
    IdeaCard,
    IdeationTurnResult,
    RenderManifest,
    SelectionState,
    SourceAttribution,
    Stage,
    StoryBeat,
    StoryCharacter,
    StoryDraft,
    StoryFacts,
    StoryLocation,
    StoryOutline,
    StoryScene,
    StoryScript,
    StoryboardPlan,
    StoryboardShot,
    VideoArtifact,
    VisualAsset,
    VisualBible,
    VisualReference,
    to_plain_data,
)
from .narration import normalize_narration_timeline
from .storyboard import (
    build_storyboard,
    derive_retake_seed,
    fit_scenes_to_duration,
)
from .timing import allocate_durations, estimate_story_duration
from .video_provider import sanitize_remote_url


MAX_USER_INPUT_CHARS = 4_000
CURRENT_STAGES = {
    Stage.IDEATING,
    Stage.STORY_REVIEW,
    Stage.SCRIPT_REVIEW,
    Stage.STORYBOARD_REVIEW,
    Stage.RENDER_READY,
    Stage.COMPLETED,
}


def _state_mutation(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._state_lock:
            self._require_not_rendering()
            self._require_no_pending_render_tasks()
            return method(self, *args, **kwargs)

    return guarded


class GuidedStorySession:
    """Single source of truth for the low-pressure idea garden."""

    schema_version = 4

    def __init__(
        self,
        brief: CreativeBrief | None = None,
        agent: StoryAgent | None = None,
    ) -> None:
        self._state_lock = RLock()
        self._render_in_progress = False
        self.brief = brief or CreativeBrief()
        self.brief.validate()
        self.agent = agent or RuleBasedStoryAgent()
        self.stage = Stage.IDEATING
        self.direction = ""
        self.idea_batches: list[IdeaBatch] = []
        self.selected_idea_ids: list[str] = []
        self.rejected_idea_ids: list[str] = []
        self.element_palette: ElementPalette | None = None
        self.selected_elements: dict[str, str] = {}
        self.chat_history: list[dict[str, str]] = []
        self.story: StoryDraft | None = None
        self.story_history: list[StoryDraft] = []
        self.draft: DraftBundle | None = None
        self.draft_history: list[DraftBundle] = []
        self.outline: StoryOutline | None = None
        self.script: StoryScript | None = None
        self.storyboard: StoryboardPlan | None = None
        self.render_manifest: RenderManifest | None = None
        self._confirmed_storyboard_signature = ""
        self.legacy_facts = StoryFacts(genre=self.brief.genre)
        self.revisions: dict[str, list[ArtifactRevision]] = {}
        self.revision_cursor: dict[str, int] = {}
        self.user_action_count = 0
        self.free_text_count = 0

    @property
    def current_batch(self) -> IdeaBatch | None:
        return self.idea_batches[-1] if self.idea_batches else None

    @property
    def render_in_progress(self) -> bool:
        with self._state_lock:
            return self._render_in_progress

    @property
    def all_ideas(self) -> list[IdeaCard]:
        return [card for batch in self.idea_batches for card in batch.cards]

    @property
    def selected_cards(self) -> list[IdeaCard]:
        lookup = {card.idea_id: card for card in self.all_ideas}
        return [lookup[item] for item in self.selected_idea_ids if item in lookup]

    @property
    def selection_state(self) -> SelectionState:
        return SelectionState(
            selected_idea_ids=list(self.selected_idea_ids),
            selected_elements=dict(self.selected_elements),
            can_generate_story=bool(self.direction),
            can_generate_draft=bool(self.direction),
        )

    @property
    def facts(self) -> StoryFacts:
        return self._story_facts()

    @property
    def story_bible(self) -> StoryFacts:
        return self.facts

    @property
    def valid_turns(self) -> int:
        return self.free_text_count

    @property
    def can_build_outline(self) -> bool:
        return bool(self.direction)

    @property
    def can_build_script(self) -> bool:
        return bool(self.story and self.story.confirmed)

    @property
    def effective_target_seconds(self) -> int:
        if self.script is not None:
            return int(self.script.target_seconds)
        if self.storyboard is not None:
            return int(self.storyboard.target_seconds)
        if self.brief.resolved_target_seconds is not None:
            return int(self.brief.resolved_target_seconds)
        if self.brief.target_seconds is not None:
            return int(self.brief.target_seconds)
        raise RuntimeError("自动时长将在完整故事确认后计算。")

    @property
    def current_question(self) -> str:
        if not self.direction:
            return "随便说一个方向就行，例如：校园里发生一件带点悬疑的事。"
        return "你可以选卡、换一批、混合，或者现在就生成完整故事。"

    @_state_mutation
    def start_ideation(self, direction: str) -> IdeaBatch:
        cleaned = self._clean_user_input(
            direction,
            empty_message="请先说一个大概方向，几个词也可以。",
        )
        batch = deepcopy(self.agent.generate_ideas(cleaned, round_number=1))
        self._validate_batch(batch, previous_cards=[])

        self.direction = cleaned
        self.stage = Stage.IDEATING
        self.idea_batches = []
        self.selected_idea_ids = []
        self.rejected_idea_ids = []
        self.element_palette = None
        self.selected_elements = {}
        self.draft = None
        self.outline = None
        self.script = None
        self.storyboard = None
        self.render_manifest = None
        self.legacy_facts = StoryFacts(genre=self.brief.genre)
        self.chat_history = [{"role": "user", "content": cleaned}]
        self.story = None
        self.story_history = []
        self.draft_history = []
        self.revisions = {}
        self.revision_cursor = {}
        self.user_action_count = 0
        self.free_text_count = 1
        if self.brief.duration_mode == "auto":
            self.brief.resolved_target_seconds = None
        self.idea_batches.append(batch)
        self.chat_history.append(
            {
                "role": "assistant",
                "content": "我先铺开8个不同方向。你不需要回答问题，只要挑喜欢的。",
            }
        )
        return batch

    @_state_mutation
    def refresh_ideas(self, feedback: str = "") -> IdeaBatch:
        self._require_ideating()
        self._require_direction()
        cleaned_feedback = self._clean_user_input(
            feedback,
            allow_empty=True,
            empty_message="",
        )
        previous_cards = deepcopy(self.all_ideas)
        batch = deepcopy(
            self.agent.generate_ideas(
                self.direction,
                round_number=len(self.idea_batches) + 1,
                feedback=cleaned_feedback,
                previous_cards=deepcopy(previous_cards),
            )
        )
        self._validate_batch(batch, previous_cards=previous_cards)
        rejected = list(self.rejected_idea_ids)
        if self.current_batch:
            rejected.extend(
                card.idea_id
                for card in self.current_batch.cards
                if card.idea_id not in self.selected_idea_ids and card.idea_id not in rejected
            )
        self.rejected_idea_ids = rejected
        self.idea_batches.append(batch)
        self.user_action_count += 1
        return batch

    @_state_mutation
    def select_ideas(self, idea_ids: list[str]) -> SelectionState:
        self._require_ideating()
        unique = list(dict.fromkeys(str(item) for item in idea_ids if str(item)))
        if len(unique) > 3:
            raise ValueError("最多同时保留3张创意卡。")
        known = {card.idea_id for card in self.all_ideas}
        unknown = [item for item in unique if item not in known]
        if unknown:
            raise ValueError("选择中包含已经过期的创意卡。")
        if unique != self.selected_idea_ids:
            self._invalidate_after_upstream_change()
        self.selected_idea_ids = unique
        self.user_action_count += 1
        return self.selection_state

    @_state_mutation
    def more_like(self, idea_id: str) -> IdeaBatch:
        self._require_ideating()
        card = self._idea_by_id(idea_id)
        previous_cards = deepcopy(self.all_ideas)
        batch = deepcopy(
            self.agent.generate_ideas(
                self.direction,
                round_number=len(self.idea_batches) + 1,
                previous_cards=deepcopy(previous_cards),
                mode="similar",
                anchors=[deepcopy(card)],
            )
        )
        self._validate_batch(batch, previous_cards=previous_cards)
        for generated in batch.cards:
            generated.source_idea_ids = [card.idea_id]
        self.idea_batches.append(batch)
        self.user_action_count += 1
        return batch

    @_state_mutation
    def mix_selected(self) -> IdeaBatch:
        self._require_ideating()
        if not 1 <= len(self.selected_idea_ids) <= 3:
            raise RuntimeError("请先选择1到3张想混合的卡。")
        previous_cards = deepcopy(self.all_ideas)
        batch = deepcopy(
            self.agent.generate_ideas(
                self.direction,
                round_number=len(self.idea_batches) + 1,
                previous_cards=deepcopy(previous_cards),
                mode="mix",
                anchors=deepcopy(self.selected_cards),
            )
        )
        self._validate_batch(batch, previous_cards=previous_cards)
        required_sources = set(self.selected_idea_ids)
        for card in batch.cards:
            card.source_idea_ids = list(required_sources)
        self.idea_batches.append(batch)
        self.user_action_count += 1
        return batch

    @_state_mutation
    def expand_selected(self) -> ElementPalette:
        self._require_ideating()
        self._require_direction()
        palette = deepcopy(
            self.agent.expand_elements(self.direction, deepcopy(self.selected_cards))
        )
        for kind in ("character", "conflict", "turning_point", "ending"):
            options = palette.options.get(kind, [])
            if len(options) != 4:
                raise ValueError(f"{kind} 必须提供4个选项。")
            if any(
                item.kind != kind or not item.title.strip() or not item.content.strip()
                for item in options
            ):
                raise ValueError(f"{kind} 包含无效或空白选项。")
        if self.selected_elements:
            self._invalidate_after_upstream_change()
        self.element_palette = palette
        self.selected_elements = {}
        self.user_action_count += 1
        return self.element_palette

    @_state_mutation
    def choose_element(self, kind: str, option_id: str) -> SelectionState:
        self._require_ideating()
        option = self._element_by_id(kind, option_id)
        if self.selected_elements.get(kind) != option.option_id:
            self._invalidate_after_upstream_change(clear_palette=False)
        self.selected_elements[kind] = option.option_id
        self.user_action_count += 1
        return self.selection_state

    @_state_mutation
    def auto_choose(self) -> SelectionState:
        self._require_ideating()
        self._require_direction()
        if self.current_batch is None:
            raise RuntimeError("还没有创意卡。")
        selected = self.current_batch.recommended_id or self.current_batch.cards[0].idea_id
        if selected not in {card.idea_id for card in self.current_batch.cards}:
            raise ValueError("推荐创意卡不存在。")
        if self.selected_idea_ids != [selected]:
            self._invalidate_after_upstream_change()
        self.selected_idea_ids = [selected]
        self.user_action_count += 1
        return self.selection_state

    @_state_mutation
    def chat_ideation(self, text: str) -> IdeationTurnResult:
        cleaned = self._clean_user_input(
            text,
            empty_message="可以随便说一句想调整的方向。",
        )
        if not self.direction:
            batch = self.start_ideation(cleaned)
        else:
            self._require_ideating()
            batch = self.refresh_ideas(cleaned)
            self.chat_history.append({"role": "user", "content": cleaned})
            self.free_text_count += 1
        message = f"已按“{cleaned}”换出8个新方向；你仍然可以直接生成完整故事。"
        self.chat_history.append({"role": "assistant", "content": message})
        return IdeationTurnResult(
            message=message,
            batch=batch,
            selection=self.selection_state,
            available_actions=["select", "more_like", "mix", "expand", "story"],
            used_fallback=bool(getattr(self.agent, "last_used_fallback", False)),
        )

    @_state_mutation
    def generate_story(self) -> StoryDraft:
        if self.stage not in {Stage.IDEATING, Stage.STORY_REVIEW}:
            raise RuntimeError("请先返回灵感区，再生成新的完整故事。")
        self._require_direction()
        selected_options = {
            kind: self._element_by_id(kind, option_id)
            for kind, option_id in self.selected_elements.items()
        }
        story = deepcopy(
            self.agent.generate_story(
                self.direction,
                deepcopy(self.selected_cards),
                deepcopy(selected_options),
            )
        )
        self._preserve_user_choices(story, selected_options)
        self._validate_story(story)
        self._validate_selected_constraints(story, selected_options)
        story.version = len(self.story_history) + 1
        story.confirmed = False
        self.story = story
        self.story_history.append(deepcopy(story))
        if self.brief.duration_mode == "auto":
            self.brief.resolved_target_seconds = None
        self.script = None
        self.draft = None
        self.outline = None
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.STORY_REVIEW
        self.user_action_count += 1
        self._snapshot("story", to_plain_data(story))
        return story

    @_state_mutation
    def revise_story(self, feedback: str) -> StoryDraft:
        if self.story is None or self.stage != Stage.STORY_REVIEW:
            raise RuntimeError("请先生成一版完整故事。")
        cleaned_feedback = self._clean_user_input(
            feedback,
            empty_message="请用一句话说明想怎样修改故事。",
        )
        revised = deepcopy(self.agent.revise_story(deepcopy(self.story), cleaned_feedback))
        selected_options = {
            kind: self._element_by_id(kind, option_id)
            for kind, option_id in self.selected_elements.items()
        }
        self._preserve_user_choices(revised, selected_options)
        self._validate_story(revised)
        self._validate_selected_constraints(revised, selected_options)
        revised.version = len(self.story_history) + 1
        revised.confirmed = False
        self.story = revised
        self.story_history.append(deepcopy(revised))
        if self.brief.duration_mode == "auto":
            self.brief.resolved_target_seconds = None
        self.script = None
        self.draft = None
        self.outline = None
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.STORY_REVIEW
        self.free_text_count += 1
        self._snapshot(
            "story",
            to_plain_data(revised),
            user_feedback=cleaned_feedback,
        )
        return revised

    @_state_mutation
    def back_to_ideation(self) -> None:
        self._require_direction()
        self.stage = Stage.IDEATING

    @_state_mutation
    def confirm_story(self) -> None:
        if self.story is None:
            raise RuntimeError("当前没有等待确认的完整故事。")
        if self.story.confirmed:
            return
        if self.stage != Stage.STORY_REVIEW:
            raise RuntimeError("当前没有等待确认的完整故事。")
        review = self.review_current_artifact("story")
        if not review.can_confirm:
            raise RuntimeError("故事仍有必须修复的问题：" + "；".join(review.hard_errors))
        self.story.confirmed = True
        self._snapshot("story", to_plain_data(self.story), confirmed=True)

    @_state_mutation
    def generate_script(self) -> StoryScript:
        if self.story is None or not self.story.confirmed or self.stage != Stage.STORY_REVIEW:
            raise RuntimeError("请先确认完整故事。")
        target_seconds = self._compute_target_seconds()
        script = deepcopy(self.agent.generate_script(deepcopy(self.story), target_seconds))
        self._validate_or_repair_script(script, target_seconds=target_seconds)
        script.confirmed = False
        self.brief.resolved_target_seconds = target_seconds
        self.brief.validate()
        self.script = script
        self.draft = None
        self.outline = None
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.SCRIPT_REVIEW
        self.user_action_count += 1
        self._snapshot("script", to_plain_data(self.script))
        return self.script

    @_state_mutation
    def revise_script(self, feedback: str) -> StoryScript:
        if (
            self.story is None
            or not self.story.confirmed
            or self.script is None
            or self.stage != Stage.SCRIPT_REVIEW
        ):
            raise RuntimeError("请先确认故事并生成剧本。")
        cleaned_feedback = self._clean_user_input(
            feedback,
            empty_message="请用一句话说明想怎样修改剧本。",
        )
        script = deepcopy(
            self.agent.revise_script(
                deepcopy(self.story),
                deepcopy(self.script),
                cleaned_feedback,
            )
        )
        self._validate_or_repair_script(
            script,
            target_seconds=self.script.target_seconds,
        )
        script.confirmed = False
        self.script = script
        self.draft = None
        self.outline = None
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.SCRIPT_REVIEW
        self.free_text_count += 1
        self._snapshot(
            "script",
            to_plain_data(self.script),
            user_feedback=cleaned_feedback,
        )
        return self.script

    @_state_mutation
    def confirm_script(self) -> None:
        if self.script is None or self.stage != Stage.SCRIPT_REVIEW:
            raise RuntimeError("当前没有等待确认的剧本。")
        review = self.review_current_artifact("script")
        if not review.can_confirm:
            raise RuntimeError("剧本仍有必须修复的问题：" + "；".join(review.hard_errors))
        self.script.confirmed = True
        self._snapshot("script", to_plain_data(self.script), confirmed=True)

    @_state_mutation
    def build_storyboard(self) -> StoryboardPlan:
        if self.script is None or not self.script.confirmed or self.stage != Stage.SCRIPT_REVIEW:
            raise RuntimeError("请先确认剧本。")
        storyboard = build_storyboard(self.script, self._story_facts())
        self._validate_storyboard_plan(storyboard)
        self.storyboard = storyboard
        self.stage = Stage.STORYBOARD_REVIEW
        self._snapshot("storyboard", to_plain_data(self.storyboard))
        return self.storyboard

    @_state_mutation
    def add_visual_reference(
        self,
        *,
        path: str | Path,
        usage: str,
        binding_kind: str,
        binding_id: str,
        content_summary: str = "",
    ) -> VisualReference:
        """Bind one local image to an asset or shot without silently confirming it."""
        if self.storyboard is None:
            raise RuntimeError("请先生成分镜。")
        normalized_usage = str(usage).strip()
        if normalized_usage not in VISUAL_REFERENCE_USAGES:
            raise ValueError(f"无法识别的参考图用途：{normalized_usage or '空'}")
        normalized_kind = str(binding_kind).strip().lower()
        normalized_id = str(binding_id).strip()
        if normalized_kind not in {"asset", "shot"} or not normalized_id:
            raise ValueError("参考图必须明确绑定到一个镜头或视觉资产。")
        if normalized_kind == "asset" and normalized_usage == "start_frame":
            raise ValueError("start_frame 只能绑定到具体镜头，不能绑定到通用资产。")

        candidate = Path(path).expanduser().resolve()
        if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的图片类型：{candidate.suffix or '无扩展名'}")
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise ValueError("上传的参考图不存在或为空。")
        digest = self._file_digest(candidate)
        reference = VisualReference(
            reference_id=f"visual-{uuid4().hex[:16]}",
            path=str(candidate),
            usage=normalized_usage,
            content_digest=digest,
            content_summary=(
                str(content_summary).strip()
                or f"{candidate.name}，sha256:{digest[:12]}，{candidate.stat().st_size} bytes"
            ),
            confirmed=False,
            binding_kind=normalized_kind,
            binding_id=normalized_id,
        )

        if normalized_kind == "asset":
            asset = next(
                (
                    item
                    for item in self.storyboard.visual_bible.assets
                    if item.asset_id == normalized_id
                ),
                None,
            )
            if asset is None:
                raise ValueError("绑定的视觉资产不存在。")
            target = asset.references
        else:
            try:
                shot_id = int(normalized_id)
            except ValueError as exc:
                raise ValueError("镜头绑定 ID 必须是整数。") from exc
            shot = next(
                (item for item in self.storyboard.shots if item.shot_id == shot_id),
                None,
            )
            if shot is None:
                raise ValueError("绑定的镜头不存在。")
            reference.binding_id = str(shot_id)
            target = shot.confirmed_visual_inputs

        if any(
            item.path == reference.path and item.usage == reference.usage
            for item in target
        ):
            raise ValueError("该绑定中已经存在相同用途的同一张参考图。")
        target.append(reference)
        self._invalidate_storyboard_confirmation()
        self._validate_storyboard_plan(self.storyboard)
        self._snapshot(
            "storyboard",
            to_plain_data(self.storyboard),
            user_feedback=f"add visual {reference.reference_id}",
        )
        return reference

    @_state_mutation
    def remove_visual_reference(self, reference_id: str) -> StoryboardPlan:
        """Remove a visual binding and any resolved copies from the current plan."""
        if self.storyboard is None:
            raise RuntimeError("请先生成分镜。")
        normalized_id = str(reference_id).strip()
        if not normalized_id:
            raise ValueError("请先选择要删除的参考图。")
        removed = False
        for asset in self.storyboard.visual_bible.assets:
            kept = [
                item for item in asset.references if item.reference_id != normalized_id
            ]
            removed = removed or len(kept) != len(asset.references)
            asset.references = kept
        for shot in self.storyboard.shots:
            kept = [
                item
                for item in shot.confirmed_visual_inputs
                if item.reference_id != normalized_id
            ]
            removed = removed or len(kept) != len(shot.confirmed_visual_inputs)
            shot.confirmed_visual_inputs = kept
            shot.reference_image_paths = [
                item.path
                for item in kept
                if item.confirmed and item.usage != "start_frame"
            ]
        if not removed:
            raise ValueError("参考图绑定不存在或已经删除。")
        self._invalidate_storyboard_confirmation()
        self._validate_storyboard_plan(self.storyboard)
        self._snapshot(
            "storyboard",
            to_plain_data(self.storyboard),
            user_feedback=f"remove visual {normalized_id}",
        )
        return self.storyboard

    @_state_mutation
    def confirm_visual_inputs(self) -> list[str]:
        """Freeze selected image contents while keeping the storyboard review gate closed."""
        if self.storyboard is None:
            raise RuntimeError("请先生成分镜。")
        for asset in self.storyboard.visual_bible.assets:
            for reference in asset.references:
                self._freeze_visual_reference(reference)
        for shot in self.storyboard.shots:
            for reference in shot.confirmed_visual_inputs:
                if reference.binding_kind != "asset":
                    self._freeze_visual_reference(reference)
        self._invalidate_storyboard_confirmation()
        diagnostics = freeze_confirmed_visual_inputs(self.storyboard)
        self._validate_storyboard_plan(self.storyboard)
        self._snapshot(
            "storyboard",
            to_plain_data(self.storyboard),
            user_feedback="confirm visual inputs",
        )
        return diagnostics

    # v0.3 compatibility wrappers -------------------------------------------------
    @_state_mutation
    def generate_draft(self) -> DraftBundle:
        warnings.warn(
            "generate_draft 已被 generate_story/confirm_story/generate_script 取代",
            DeprecationWarning,
        )
        story = self.generate_story()
        self.confirm_story()
        script = self.generate_script()
        beats = [
            StoryBeat(
                scene.scene_id,
                scene.title,
                scene.visible_action or scene.action,
                scene.start_state,
                scene.emotional_change,
                scene.duration,
            )
            for scene in script.scenes
        ]
        outline = StoryOutline(
            title=story.title,
            logline=story.logline,
            opening=script.scenes[0].visible_action or script.scenes[0].action,
            protagonist_goal=story.characters[0].description if story.characters else story.logline,
            conflict=story.core_conflict,
            development=story.story_text,
            turning_point="",
            ending=story.ending,
            source_turn_ids=[],
            beats=beats,
        )
        self.draft = DraftBundle(
            outline=outline,
            script=script,
            field_sources=deepcopy(story.field_sources),
            ai_filled_fields=list(story.ai_filled_fields),
            version=len(self.draft_history) + 1,
        )
        self.draft_history.append(deepcopy(self.draft))
        self.outline = outline
        return self.draft

    @_state_mutation
    def revise_draft(self, feedback: str) -> DraftBundle:
        warnings.warn("revise_draft 已被 revise_story/revise_script 取代", DeprecationWarning)
        if self.draft is None:
            raise RuntimeError("请先生成剧本。")
        candidate = deepcopy(self.draft)
        candidate.script = self.revise_script(feedback)
        candidate.version += 1
        self.draft = candidate
        self.outline = candidate.outline
        self.draft_history.append(deepcopy(candidate))
        return self.draft

    @_state_mutation
    def confirm_draft(self) -> None:
        warnings.warn("confirm_draft 已被 confirm_script 取代", DeprecationWarning)
        self.confirm_script()

    @_state_mutation
    def update_storyboard_shot(self, shot_id: int, patch: dict[str, Any]) -> StoryboardPlan:
        if self.storyboard is None:
            raise RuntimeError("尚未生成分镜。")
        shot_index = next(
            (
                index
                for index, item in enumerate(self.storyboard.shots)
                if item.shot_id == int(shot_id)
            ),
            None,
        )
        if shot_index is None:
            raise ValueError("镜头不存在。")
        if not isinstance(patch, dict) or not patch:
            raise ValueError("镜头修改必须包含至少一个字段。")
        allowed = set(StoryboardShot.__dataclass_fields__) - {"shot_id", "scene_id"}
        unsupported = [field for field in patch if field not in allowed]
        if unsupported:
            raise ValueError(f"不支持的镜头字段：{unsupported[0]}")

        candidate_plan = deepcopy(self.storyboard)
        candidate = candidate_plan.shots[shot_index]
        for field, value in patch.items():
            setattr(candidate, field, deepcopy(value))
        if "seed" not in patch:
            candidate.seed = derive_retake_seed(
                candidate.seed,
                candidate.shot_id,
                patch,
            )
        if "duration" in patch:
            candidate.motion_prompt = self._motion_with_duration(
                candidate.motion_prompt,
                candidate.duration,
            )
            candidate.video_prompt = self._motion_with_duration(
                candidate.video_prompt,
                candidate.duration,
            )
        candidate.initial_frame_source_path = ""
        candidate.initial_frame_path = ""
        candidate.initial_frame_url = ""
        candidate.generated_first_frame_path = ""
        candidate.generated_last_frame_path = ""
        candidate_plan.confirmed = False
        self._validate_storyboard_plan(candidate_plan)

        self.storyboard = candidate_plan
        self._confirmed_storyboard_signature = ""
        self.render_manifest = None
        self.stage = Stage.STORYBOARD_REVIEW
        self._snapshot(
            "storyboard", to_plain_data(self.storyboard), user_feedback=f"retake {shot_id}"
        )
        return self.storyboard

    @_state_mutation
    def confirm_storyboard(self) -> None:
        if self.storyboard is None or self.stage != Stage.STORYBOARD_REVIEW:
            raise RuntimeError("当前没有等待确认的分镜。")
        freeze_confirmed_visual_inputs(self.storyboard)
        review = self.review_current_artifact("storyboard")
        if not review.can_confirm:
            raise RuntimeError("分镜仍有必须修复的问题：" + "；".join(review.hard_errors))
        self.storyboard.confirmed = True
        self._confirmed_storyboard_signature = self._storyboard_confirmation_signature(
            self.storyboard
        )
        self.stage = Stage.RENDER_READY
        self._snapshot("storyboard", to_plain_data(self.storyboard), confirmed=True)

    def render_confirmed_plan(self, renderer, output_dir: str | Path) -> RenderManifest:
        with self._state_lock:
            if self._render_in_progress:
                raise RuntimeError("视频生成正在进行，请等待当前任务结束。")
            if (
                self.stage != Stage.RENDER_READY
                or self.storyboard is None
                or not self.storyboard.confirmed
            ):
                raise RuntimeError("必须先确认完整分镜，才能调用视频生成。")
            if (
                self.story is None
                or not self.story.confirmed
                or self.script is None
                or not self.script.confirmed
            ):
                raise RuntimeError("故事、剧本与分镜确认状态不一致，禁止视频生成。")
            review = self.review_current_artifact("storyboard")
            if not review.can_confirm:
                raise RuntimeError("分镜复核失败：" + "；".join(review.hard_errors))
            confirmed_plan = self.storyboard
            current_signature = self._storyboard_confirmation_signature(confirmed_plan)
            if (
                self._confirmed_storyboard_signature
                and current_signature != self._confirmed_storyboard_signature
            ):
                confirmed_plan.confirmed = False
                self._confirmed_storyboard_signature = ""
                self.stage = Stage.STORYBOARD_REVIEW
                raise RuntimeError("分镜内容在确认后发生变化，原确认已失效。")
            visual_errors = verify_confirmed_visual_inputs(confirmed_plan)
            if visual_errors:
                confirmed_plan.confirmed = False
                self._confirmed_storyboard_signature = ""
                self.stage = Stage.STORYBOARD_REVIEW
                raise RuntimeError(
                    "已确认视觉输入发生变化，分镜确认已失效："
                    + "；".join(visual_errors)
                )
            confirmed_snapshot = deepcopy(confirmed_plan)
            signature = self._storyboard_confirmation_signature(confirmed_plan)
            working_plan = deepcopy(confirmed_plan)
            previous_stage = self.stage
            previous_manifest = deepcopy(self.render_manifest)
            self._render_in_progress = True

        invalidated = False
        try:
            manifest = renderer.render(working_plan, output_dir)
            with self._state_lock:
                if (
                    self.storyboard is not confirmed_plan
                    or self.stage != Stage.RENDER_READY
                    or not confirmed_plan.confirmed
                    or self._storyboard_confirmation_signature(confirmed_plan) != signature
                    or self._storyboard_confirmation_signature(working_plan) != signature
                ):
                    restored = deepcopy(confirmed_snapshot)
                    restored.confirmed = False
                    self.storyboard = restored
                    self.stage = Stage.STORYBOARD_REVIEW
                    self.render_manifest = None
                    invalidated = True
                    raise RuntimeError("渲染期间确认计划发生变化，结果未写入当前会话。")
                if not isinstance(manifest, RenderManifest):
                    raise TypeError("渲染器必须返回 RenderManifest。")
                self._validate_render_outputs(working_plan, manifest)
                confirmed_plan.artifacts = deepcopy(working_plan.artifacts)
                confirmed_plan.audio_path = working_plan.audio_path
                confirmed_plan.subtitle_path = working_plan.subtitle_path
                self.render_manifest = deepcopy(manifest)
                if manifest.status in {"succeeded", "succeeded_with_warnings"}:
                    self.stage = Stage.COMPLETED
                return manifest
        except Exception:
            with self._state_lock:
                if not invalidated:
                    can_preserve_runtime_artifacts = (
                        self.storyboard is confirmed_plan
                        and self.stage == Stage.RENDER_READY
                        and confirmed_plan.confirmed
                        and self._storyboard_confirmation_signature(confirmed_plan) == signature
                        and self._storyboard_confirmation_signature(working_plan) == signature
                    )
                    if can_preserve_runtime_artifacts:
                        try:
                            self._validate_storyboard_plan(working_plan)
                            confirmed_plan.artifacts = deepcopy(working_plan.artifacts)
                            confirmed_plan.audio_path = working_plan.audio_path
                            confirmed_plan.subtitle_path = working_plan.subtitle_path
                            self.storyboard = confirmed_plan
                        except (TypeError, ValueError):
                            self.storyboard = deepcopy(confirmed_snapshot)
                    else:
                        self.storyboard = deepcopy(confirmed_snapshot)
                    self.stage = previous_stage
                    self.render_manifest = previous_manifest
            raise
        finally:
            with self._state_lock:
                self._render_in_progress = False

    def review_current_artifact(self, artifact_type: str | None = None) -> ArtifactReview:
        kind = artifact_type or (
            "storyboard" if self.storyboard else "script" if self.script else "story"
        )
        review = ArtifactReview(artifact_type=kind)
        if kind == "story" and self.story:
            if len(self.story.story_text.strip()) < 120:
                review.hard_errors.append("故事正文过短，尚不足以改编成剧本")
            if not self.story.characters:
                review.hard_errors.append("故事缺少明确人物")
            if not self.story.core_conflict.strip() or not self.story.ending.strip():
                review.hard_errors.append("故事缺少核心冲突或结局")
            review.scores["story_completeness"] = 1.0 if not review.hard_errors else 0.5
            review.scores["ai_fill_disclosure"] = 1.0
        elif kind in ("draft", "script") and self.script:
            target_seconds = self.effective_target_seconds
            if abs(self.script.total_duration - target_seconds) > 1:
                review.hard_errors.append("剧本总时长不符合目标")
            if not self.script.scenes:
                review.hard_errors.append("剧本没有可转换的场景")
            if any(scene.duration < 3 for scene in self.script.scenes):
                review.hard_errors.append("存在不足3秒、无法稳定转换为镜头的场景")
            if len(self.script.scenes) > max(1, target_seconds // 3):
                review.hard_errors.append("场景数量超过当前时长可容纳的分镜数量")
            if any(
                not (scene.visible_action or scene.action).strip() for scene in self.script.scenes
            ):
                review.hard_errors.append("存在无法拍摄的空动作场景")
            review.scores["filmability"] = 1.0 if not review.hard_errors else 0.5
        elif kind == "storyboard" and self.storyboard:
            target_seconds = self.effective_target_seconds
            if not self.storyboard.shots:
                review.hard_errors.append("分镜没有镜头")
                review.scores["shot_diversity"] = 0.0
                review.scores["visual_identity_coverage"] = 0.0
                return review
            if abs(self.storyboard.total_duration - target_seconds) > 1:
                review.hard_errors.append("分镜总时长不符合目标")
            minimum = max(1, (target_seconds + 14) // 15)
            maximum = max(1, target_seconds // 3)
            if not minimum <= len(self.storyboard.shots) <= maximum:
                review.hard_errors.append(f"当前时长下分镜数量必须在{minimum}到{maximum}之间")
            if any(not 3 <= shot.duration <= 15 for shot in self.storyboard.shots):
                review.hard_errors.append("存在超出3到15秒限制的镜头")
            if any(
                not shot.first_frame_prompt.strip()
                or not shot.motion_prompt.strip()
                or not shot.end_frame_prompt.strip()
                for shot in self.storyboard.shots
            ):
                review.hard_errors.append("存在缺少首帧、动作或结束帧描述的镜头")
            cameras = {shot.camera for shot in self.storyboard.shots}
            review.scores["shot_diversity"] = min(1.0, len(cameras) / 4)
            referenced = sum(bool(shot.reference_asset_ids) for shot in self.storyboard.shots)
            review.scores["visual_identity_coverage"] = referenced / len(self.storyboard.shots)
        else:
            raise RuntimeError("当前没有可审查的产物。")
        return review

    # v0.2 compatibility wrappers -------------------------------------------------
    @_state_mutation
    def submit_user_turn(self, text: str, *, source: str = "human") -> GuideTurnResult:
        warnings.warn("submit_user_turn 已被 start_ideation/chat_ideation 取代", DeprecationWarning)
        result = self.chat_ideation(text)
        return GuideTurnResult(
            accepted=True,
            assistant_message=result.message,
            next_question=self.current_question,
            suggestions=[card.logline for card in result.batch.cards[:3]] if result.batch else [],
            valid_turns=self.valid_turns,
            missing_fields=[],
            can_build_outline=True,
            readiness_score=1.0,
            recommended_action="generate_story",
            used_fallback=result.used_fallback,
        )

    @_state_mutation
    def answer_detail_question(self, text: str, *, source: str = "human") -> GuideTurnResult:
        return self.submit_user_turn(text, source=source)

    def request_suggestions(self) -> list[CreativeSuggestion]:
        if self.current_batch is None:
            return []
        return [
            CreativeSuggestion(card.idea_id, card.title, card.logline, "idea")
            for card in self.current_batch.cards[:3]
        ]

    @_state_mutation
    def apply_suggestion(self, suggestion_id: str) -> GuideTurnResult:
        self.select_ideas([suggestion_id])
        card = self._idea_by_id(suggestion_id)
        return GuideTurnResult(
            accepted=True,
            assistant_message=f"已保留《{card.title}》。",
            next_question=self.current_question,
            suggestions=[],
            valid_turns=self.valid_turns,
            missing_fields=[],
            can_build_outline=True,
            readiness_score=1.0,
            recommended_action="generate_story",
        )

    @_state_mutation
    def build_outline(self) -> StoryOutline:
        warnings.warn("build_outline 已被 generate_draft 取代", DeprecationWarning)
        return self.generate_draft().outline

    @_state_mutation
    def confirm_outline(self) -> None:
        if self.story is None:
            raise RuntimeError("尚未生成故事。")
        self.confirm_story()

    @_state_mutation
    def build_script(self) -> StoryScript:
        warnings.warn("build_script 已被 generate_script 取代", DeprecationWarning)
        if self.story is None:
            self.generate_story()
        if not self.story.confirmed:
            self.confirm_story()
        return self.generate_script()

    @_state_mutation
    def update_script_scene(self, scene_id: int, patch: dict[str, Any]) -> StoryScript:
        if self.script is None:
            raise RuntimeError("尚未生成剧本。")
        scene_index = next(
            (
                index
                for index, item in enumerate(self.script.scenes)
                if item.scene_id == int(scene_id)
            ),
            None,
        )
        if scene_index is None:
            raise ValueError("场景不存在。")
        if not isinstance(patch, dict) or not patch:
            raise ValueError("场景修改必须包含至少一个字段。")
        allowed = set(StoryScene.__dataclass_fields__) - {"scene_id"}
        unsupported = [field for field in patch if field not in allowed]
        if unsupported:
            raise ValueError(f"不支持的场景字段：{unsupported[0]}")

        candidate_script = deepcopy(self.script)
        candidate = candidate_script.scenes[scene_index]
        for field, value in patch.items():
            setattr(candidate, field, deepcopy(value))
        self._validate_or_repair_script(
            candidate_script,
            target_seconds=self.script.target_seconds,
        )
        candidate_script.confirmed = False

        self.script = candidate_script
        self.draft = None
        self.outline = None
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.SCRIPT_REVIEW
        self._snapshot("script", to_plain_data(self.script), user_feedback=f"edit scene {scene_id}")
        return self.script

    # persistence ----------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def to_dict(self) -> dict[str, Any]:
        with self._state_lock:
            payload = {
                "schema_version": self.schema_version,
                "stage": self.stage.value,
                "brief": to_plain_data(self.brief),
                "direction": self.direction,
                "idea_batches": to_plain_data(self.idea_batches),
                "selected_idea_ids": list(self.selected_idea_ids),
                "rejected_idea_ids": list(self.rejected_idea_ids),
                "element_palette": to_plain_data(self.element_palette)
                if self.element_palette
                else None,
                "selected_elements": dict(self.selected_elements),
                "chat_history": deepcopy(self.chat_history),
                "story": to_plain_data(self.story) if self.story else None,
                "story_history": to_plain_data(self.story_history),
                "script": to_plain_data(self.script) if self.script else None,
                "draft": to_plain_data(self.draft) if self.draft else None,
                "draft_history": to_plain_data(self.draft_history),
                "storyboard": to_plain_data(self.storyboard) if self.storyboard else None,
                "render_manifest": to_plain_data(self.render_manifest)
                if self.render_manifest
                else None,
                "confirmed_storyboard_signature": self._confirmed_storyboard_signature,
                "legacy_facts": to_plain_data(self.legacy_facts),
                "revisions": to_plain_data(self.revisions),
                "revision_cursor": dict(self.revision_cursor),
                "metrics": {
                    "user_action_count": self.user_action_count,
                    "free_text_count": self.free_text_count,
                },
            }
            self._sanitize_persisted_urls(payload)
            return deepcopy(payload)

    @classmethod
    def load(cls, path: str | Path, *, agent: StoryAgent | None = None) -> GuidedStorySession:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("会话文件顶层必须是 JSON 对象。")
        raw_schema = data.get("schema_version", 1)
        if isinstance(raw_schema, bool) or not isinstance(raw_schema, int):
            raise ValueError("schema_version 必须是整数。")
        schema_version = raw_schema
        if not 1 <= schema_version <= cls.schema_version:
            raise ValueError(
                f"不支持的 schema_version={schema_version}；当前最高支持 {cls.schema_version}。"
            )
        brief_data = data.get("brief", {})
        if not isinstance(brief_data, dict):
            raise ValueError("brief 必须是 JSON 对象。")
        session = cls(CreativeBrief(**brief_data), agent=agent)
        if schema_version < 3:
            session._load_v2(data)
            session._validate_loaded_state(schema_version=schema_version)
            return session
        raw_stage = str(data.get("stage", Stage.IDEATING.value))
        if schema_version == 3 and raw_stage == Stage.DRAFT_REVIEW.value:
            raw_stage = Stage.SCRIPT_REVIEW.value
        session.stage = Stage(raw_stage)
        session.direction = str(data.get("direction", ""))
        session.idea_batches = [session._batch_from(item) for item in data.get("idea_batches", [])]
        session.selected_idea_ids = [str(item) for item in data.get("selected_idea_ids", [])]
        session.rejected_idea_ids = [str(item) for item in data.get("rejected_idea_ids", [])]
        if data.get("element_palette"):
            session.element_palette = session._palette_from(data["element_palette"])
        session.selected_elements = {
            str(k): str(v) for k, v in data.get("selected_elements", {}).items()
        }
        session.chat_history = list(data.get("chat_history", []))
        if data.get("story"):
            session.story = session._story_from_data(data["story"])
        session.story_history = [
            session._story_from_data(item) for item in data.get("story_history", [])
        ]
        if data.get("script"):
            session.script = session._script_from(data["script"])
        if data.get("draft"):
            session.draft = session._draft_from(data["draft"])
            session.outline = session.draft.outline
            if session.script is None:
                session.script = session.draft.script
            if session.story is None:
                session.story = session._story_from_legacy_draft(session.draft)
                session.story_history = [deepcopy(session.story)]
        session.draft_history = [
            session._draft_from(item) for item in data.get("draft_history", [])
        ]
        if data.get("storyboard"):
            session.storyboard = session._storyboard_from(data["storyboard"])
            stored_signature = str(data.get("confirmed_storyboard_signature", ""))
            session._confirmed_storyboard_signature = (
                stored_signature
                or (
                    session._storyboard_confirmation_signature(session.storyboard)
                    if session.storyboard.confirmed
                    else ""
                )
            )
        if data.get("render_manifest"):
            session.render_manifest = session._manifest_from(data["render_manifest"])
        session.legacy_facts = session._facts_from(data.get("legacy_facts", {}))
        session._load_revisions(data)
        metrics = data.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("metrics 必须是 JSON 对象。")
        session.user_action_count = int(metrics.get("user_action_count", 0))
        session.free_text_count = int(metrics.get("free_text_count", 0))
        if session.user_action_count < 0 or session.free_text_count < 0:
            raise ValueError("会话计数不能是负数。")
        session._validate_loaded_state(schema_version=schema_version)
        return session

    def _load_v2(self, data: dict[str, Any]) -> GuidedStorySession:
        self.legacy_facts = self._facts_from(data.get("facts", {}))
        outline = self._outline_from(data["outline"]) if data.get("outline") else None
        script = self._script_from(data["script"]) if data.get("script") else None
        self.direction = (
            self.legacy_facts.premise
            or self.legacy_facts.opening
            or (outline.logline if outline else "")
            or "从v0.2迁移的故事"
        )
        if outline and script:
            self.draft = DraftBundle(
                outline=outline,
                script=script,
                ai_filled_fields=[],
                field_sources={
                    "legacy": SourceAttribution(
                        field="legacy",
                        source_type="v0.2_migration",
                        value=self.direction,
                    )
                },
            )
            self.draft_history = [deepcopy(self.draft)]
            self.outline, self.script = outline, script
            self.story = self._story_from_legacy_draft(self.draft)
            self.story_history = [deepcopy(self.story)]
            self.stage = Stage.SCRIPT_REVIEW
        else:
            self.stage = Stage.IDEATING
        if data.get("storyboard"):
            self.storyboard = self._storyboard_from(data["storyboard"])
            old_stage = str(data.get("stage", ""))
            if self.script is None or self.story is None:
                raise ValueError("旧版分镜缺少可迁移的故事或剧本。")
            self.story.confirmed = True
            self.script.confirmed = True
            if old_stage == "render_ready":
                self.storyboard.confirmed = True
            self.stage = (
                Stage.RENDER_READY if old_stage == "render_ready" else Stage.STORYBOARD_REVIEW
            )
        if data.get("render_manifest"):
            self.render_manifest = self._manifest_from(data["render_manifest"])
        self._load_revisions(data)
        return self

    def _validate_loaded_state(self, *, schema_version: int) -> None:
        self.brief.validate()
        if len(self.direction) > MAX_USER_INPUT_CHARS:
            raise ValueError("会话方向超过允许长度。")
        if schema_version >= 3 and self.stage not in CURRENT_STAGES:
            raise ValueError(f"当前版本不再支持会话阶段：{self.stage.value}")

        previous_cards: list[IdeaCard] = []
        for batch in self.idea_batches:
            self._validate_batch(batch, previous_cards=previous_cards)
            previous_cards.extend(batch.cards)
        known_ids = {card.idea_id for card in previous_cards}
        if len(self.selected_idea_ids) > 3 or any(
            idea_id not in known_ids for idea_id in self.selected_idea_ids
        ):
            raise ValueError("保存的创意卡选择无效。")
        if any(idea_id not in known_ids for idea_id in self.rejected_idea_ids):
            raise ValueError("保存的淘汰创意卡不存在。")

        if self.element_palette is not None:
            for kind in ("character", "conflict", "turning_point", "ending"):
                options = self.element_palette.options.get(kind, [])
                if len(options) != 4 or any(item.kind != kind for item in options):
                    raise ValueError(f"保存的 {kind} 故事零件无效。")
        if self.selected_elements:
            if self.element_palette is None:
                raise ValueError("存在故事零件选择，但零件面板缺失。")
            for kind, option_id in self.selected_elements.items():
                self._element_by_id(kind, option_id)

        if not isinstance(self.chat_history, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("role"), str)
            or not isinstance(item.get("content"), str)
            for item in self.chat_history
        ):
            raise ValueError("chat_history 结构无效。")

        if self.story is not None:
            self._validate_story(self.story)
        for story in self.story_history:
            self._validate_story(story)
        if self.script is not None:
            self._validate_loaded_script(self.script)
        if self.storyboard is not None:
            self._validate_storyboard_plan(self.storyboard)
        if self.render_manifest is not None and self.storyboard is None:
            raise ValueError("存在渲染记录，但分镜缺失。")
        if self.render_manifest is not None and self.storyboard is not None:
            self._validate_render_outputs(self.storyboard, self.render_manifest)

        if self.stage == Stage.STORY_REVIEW and self.story is None:
            raise ValueError("故事审查阶段缺少故事。")
        if self.stage == Stage.SCRIPT_REVIEW and (
            self.story is None or not self.story.confirmed or self.script is None
        ):
            raise ValueError("剧本审查阶段的故事或剧本确认链不完整。")
        if self.stage == Stage.STORYBOARD_REVIEW and (
            self.story is None
            or not self.story.confirmed
            or self.script is None
            or not self.script.confirmed
            or self.storyboard is None
        ):
            raise ValueError("分镜审查阶段的确认链不完整。")
        if self.stage in {Stage.RENDER_READY, Stage.COMPLETED} and (
            self.story is None
            or not self.story.confirmed
            or self.script is None
            or not self.script.confirmed
            or self.storyboard is None
            or not self.storyboard.confirmed
        ):
            raise ValueError("可渲染阶段的故事、剧本或分镜确认链不完整。")
        if self.stage == Stage.COMPLETED and (
            self.render_manifest is None
            or self.render_manifest.status not in {"succeeded", "succeeded_with_warnings"}
        ):
            raise ValueError("已完成阶段缺少成功的渲染记录。")

        for kind, cursor in self.revision_cursor.items():
            history = self.revisions.get(kind)
            if history is None or not 0 <= cursor < len(history):
                raise ValueError(f"{kind} 的版本游标无效。")

    def _validate_loaded_script(self, script: StoryScript) -> None:
        if (
            isinstance(script.target_seconds, bool)
            or not isinstance(script.target_seconds, int)
            or not 15 <= script.target_seconds <= 300
        ):
            raise ValueError("保存的剧本目标时长无效。")
        if not script.scenes:
            raise ValueError("保存的剧本没有场景。")
        if len(script.scenes) > script.target_seconds // 3:
            raise ValueError("保存的剧本场景过密，无法转换为三秒以上镜头。")
        seen_ids: set[int] = set()
        for scene in script.scenes:
            self._validate_scene_fields(scene)
            if scene.scene_id in seen_ids:
                raise ValueError("保存的剧本场景 ID 不能重复。")
            seen_ids.add(scene.scene_id)
            if scene.duration < 3:
                raise ValueError("保存的剧本包含不足三秒的场景。")
            if not (scene.visible_action or scene.action).strip():
                raise ValueError("保存的剧本包含空动作场景。")
        if abs(script.total_duration - script.target_seconds) > 1:
            raise ValueError("保存的剧本总时长与目标不一致。")
        self._validate_script_story_boundary(script)

    def _validate_batch(
        self, batch: IdeaBatch, *, previous_cards: list[IdeaCard] | None = None
    ) -> None:
        if len(batch.cards) != IDEA_COUNT:
            raise ValueError("每一轮必须恰好生成8张创意卡。")
        fingerprints = {card.fingerprint for card in batch.cards}
        if len(fingerprints) != IDEA_COUNT:
            raise ValueError("同一批创意卡不能重复。")
        previous = {card.fingerprint for card in (previous_cards or [])}
        if fingerprints & previous:
            raise ValueError("新一批创意卡不能重复已经看过或淘汰的创意。")

    @staticmethod
    def _validate_story(story: StoryDraft) -> None:
        if not story.title.strip() or not story.logline.strip():
            raise ValueError("故事缺少标题或一句话概述。")
        if len(story.story_text.strip()) < 120:
            raise ValueError("故事正文过短，尚不足以进入剧本改编。")
        if not story.characters:
            raise ValueError("故事至少需要一个明确人物。")
        if not story.core_conflict.strip() or not story.ending.strip():
            raise ValueError("故事必须明确核心冲突和结局。")

    def _validate_or_repair_script(
        self,
        script: StoryScript,
        *,
        target_seconds: int | None = None,
    ) -> None:
        if not script.scenes:
            raise ValueError("剧本至少需要一个可拍摄场景。")
        target = int(target_seconds if target_seconds is not None else script.target_seconds)
        if not 15 <= target <= 300:
            raise ValueError("剧本目标时长必须在 15 到 300 秒之间。")
        for scene in script.scenes:
            self._validate_scene_fields(scene)
            if not (scene.visible_action or scene.action).strip():
                scene.action = scene.narration or "主角完成一个清晰可见的动作"
                scene.visible_action = scene.action
        script.scenes = fit_scenes_to_duration(script.scenes, target, minimum=3)
        durations = allocate_durations(
            target,
            len(script.scenes),
            minimum=3,
            maximum=target,
        )
        for scene, duration in zip(script.scenes, durations):
            scene.duration = duration
        script.target_seconds = target
        self._validate_script_story_boundary(script)

    def _compute_target_seconds(self) -> int:
        if self.story is None:
            raise RuntimeError("请先生成完整故事。")
        if self.brief.duration_mode == "custom":
            if self.brief.target_seconds is None:
                raise RuntimeError("自定义时长尚未填写。")
            target = int(self.brief.target_seconds)
        else:
            target = estimate_story_duration(
                self.story.story_text,
                character_count=len(self.story.characters),
                location_count=len(self.story.locations),
            )
        if not 15 <= target <= 300:
            raise ValueError("目标视频时长必须在 15 到 300 秒之间。")
        return target

    def _resolve_target_seconds(self) -> int:
        target = self._compute_target_seconds()
        self.brief.resolved_target_seconds = target
        self.brief.validate()
        return target

    def _preserve_user_choices(
        self,
        story: StoryDraft,
        selected_options: dict[str, ElementOption],
    ) -> None:
        """Apply choices after model output so an LLM cannot silently rewrite them."""
        cards = self.selected_cards
        source_ids = [card.idea_id for card in cards]
        if cards:
            conflicts = "；".join(card.central_conflict for card in cards)
            endings = "；".join(card.ending_direction for card in cards)
            tones = " × ".join(card.tone for card in cards)
            titles = " × ".join(card.title for card in cards)
            story.title = titles
            story.tone = tones
            story.core_conflict = conflicts
            story.ending = endings
            known_names = {item.name for item in story.characters}
            for card in cards:
                if card.protagonist not in known_names:
                    story.characters.append(
                        StoryCharacter(
                            name=card.protagonist,
                            description=card.logline,
                            visual_identity="保持统一外观与服装",
                        )
                    )
                if card.protagonist not in story.story_text:
                    story.story_text = (
                        f"{story.story_text}\n\n{card.protagonist}始终是推动核心行动的角色。"
                    )
            story.field_sources["selected_ideas"] = SourceAttribution(
                field="selected_ideas",
                source_type="selected_card",
                value=json.dumps(to_plain_data(cards), ensure_ascii=False),
                source_ids=source_ids,
            )
        mapping = {
            "character": "protagonist",
            "conflict": "core_conflict",
            "turning_point": "turning_point",
            "ending": "ending",
        }
        for kind, option in selected_options.items():
            field = mapping[kind]
            if kind == "character":
                if story.characters:
                    story.characters[0].description = option.content
                else:
                    story.characters = [StoryCharacter("主角", option.content)]
                if option.content not in story.story_text:
                    story.story_text = f"{story.story_text}\n\n主角设定保持为：{option.content}。"
            elif kind == "turning_point":
                marker = f"确定发生的关键变化：{option.content}"
                if option.content not in story.story_text:
                    story.story_text = f"{story.story_text}\n\n{marker}。"
            else:
                setattr(story, field, option.content)
            story.field_sources[field] = SourceAttribution(
                field=field,
                source_type="selected_element",
                value=option.content,
                source_ids=[option.option_id],
            )
            if field in story.ai_filled_fields:
                story.ai_filled_fields.remove(field)
        for field, source in story.field_sources.items():
            if source.source_type == "ai_fill" and field not in story.ai_filled_fields:
                story.ai_filled_fields.append(field)
        for field in list(story.ai_filled_fields):
            if field not in story.field_sources:
                story.field_sources[field] = SourceAttribution(
                    field=field,
                    source_type="ai_fill",
                    value="由AI在故事生成阶段补全",
                )

    def _validate_selected_constraints(
        self,
        story: StoryDraft,
        selected_options: dict[str, ElementOption],
    ) -> None:
        character_text = "\n".join(f"{item.name} {item.description}" for item in story.characters)
        for card in self.selected_cards:
            if card.protagonist not in character_text or card.protagonist not in story.story_text:
                raise ValueError("所选创意卡的主角没有落实到故事结构中。")
        for kind, option in selected_options.items():
            if kind == "character":
                material = f"{character_text}\n{story.story_text}"
                if option.content not in material:
                    raise ValueError("所选角色设定没有落实到故事结构中。")
            elif kind == "turning_point" and option.content not in story.story_text:
                raise ValueError("所选转折没有落实到故事正文中。")
            elif kind == "conflict" and story.core_conflict != option.content:
                raise ValueError("所选冲突没有保留。")
            elif kind == "ending" and story.ending != option.content:
                raise ValueError("所选结局没有保留。")

    @staticmethod
    def _clean_user_input(
        text: str,
        *,
        empty_message: str,
        allow_empty: bool = False,
    ) -> str:
        cleaned = " ".join(str(text).split())
        if not cleaned and not allow_empty:
            raise ValueError(empty_message)
        if len(cleaned) > MAX_USER_INPUT_CHARS:
            raise ValueError(f"输入过长，请控制在 {MAX_USER_INPUT_CHARS} 个字符以内。")
        return cleaned

    def _require_not_rendering(self) -> None:
        if self._render_in_progress:
            raise RuntimeError("视频生成正在进行，当前不能修改会话状态。")

    def _require_no_pending_render_tasks(self) -> None:
        if self.storyboard is None:
            return
        pending = [
            artifact
            for artifact in self.storyboard.artifacts
            if artifact.status in {"pending", "submission_uncertain"} and artifact.request_id
        ]
        if not pending:
            return
        request_ids = "、".join(
            str(artifact.request_id) for artifact in pending[:3] if artifact.request_id
        )
        raise RuntimeError(
            "仍有远端视频任务正在处理或提交结果尚未确认"
            f"（任务 ID：{request_ids}）。请先再次生成以继续查询，"
            "若提示提交结果不确定，请先到 Provider 后台核对；"
            "当前不能改稿、撤销或开始新项目，以免重复付费。"
        )

    def _require_ideating(self) -> None:
        if self.stage != Stage.IDEATING:
            raise RuntimeError("请先返回灵感区，再修改创意选择。")

    def _invalidate_after_upstream_change(self, *, clear_palette: bool = True) -> None:
        self.story = None
        self.draft = None
        self.outline = None
        self.script = None
        self.storyboard = None
        self.render_manifest = None
        if clear_palette:
            self.element_palette = None
            self.selected_elements = {}
        if self.brief.duration_mode == "auto":
            self.brief.resolved_target_seconds = None
        for kind in ("story", "draft", "script", "storyboard"):
            self.revisions.pop(kind, None)
            self.revision_cursor.pop(kind, None)

    def _invalidate_storyboard_confirmation(self) -> None:
        if self.storyboard is None:
            return
        self.storyboard.confirmed = False
        self._confirmed_storyboard_signature = ""
        self.render_manifest = None
        self.stage = Stage.STORYBOARD_REVIEW

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _freeze_visual_reference(reference: VisualReference) -> None:
        candidate = Path(reference.path).expanduser().resolve()
        if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"参考图 {reference.reference_id or candidate.name} 类型不受支持。"
            )
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise ValueError(
                f"参考图 {reference.reference_id or candidate.name} 不存在或为空。"
            )
        digest = GuidedStorySession._file_digest(candidate)
        reference.path = str(candidate)
        reference.content_digest = digest
        reference.content_summary = (
            reference.content_summary
            or f"{candidate.name}，sha256:{digest[:12]}，{candidate.stat().st_size} bytes"
        )
        reference.confirmed = True

    @staticmethod
    def _validate_scene_fields(scene: StoryScene) -> None:
        if (
            isinstance(scene.scene_id, bool)
            or not isinstance(scene.scene_id, int)
            or scene.scene_id < 1
        ):
            raise ValueError("剧本场景 ID 必须是正整数。")
        string_fields = (
            "title",
            "location",
            "time_of_day",
            "action",
            "narration",
            "dialogue",
            "visible_action",
            "start_state",
            "end_state",
            "emotional_change",
        )
        if any(not isinstance(getattr(scene, field), str) for field in string_fields):
            raise ValueError("剧本场景的文本字段必须是字符串。")
        if not isinstance(scene.characters, list) or not all(
            isinstance(item, str) for item in scene.characters
        ):
            raise ValueError("剧本场景 characters 必须是字符串数组。")
        if not isinstance(scene.props, list) or not all(
            isinstance(item, str) for item in scene.props
        ):
            raise ValueError("剧本场景 props 必须是字符串数组。")
        if isinstance(scene.duration, bool) or not isinstance(scene.duration, int):
            raise ValueError("剧本场景时长必须是整数。")

    def _validate_script_story_boundary(self, script: StoryScript) -> None:
        if self.story is None:
            return
        known_characters = {
            item.name.strip() for item in self.story.characters if item.name.strip()
        }
        if not known_characters:
            return
        script_text = "\n".join(
            [
                *(" ".join(scene.characters) for scene in script.scenes),
                *(
                    f"{scene.visible_action or scene.action} {scene.dialogue} {scene.narration}"
                    for scene in script.scenes
                ),
            ]
        )
        if not any(name in script_text for name in known_characters):
            raise ValueError("剧本没有保留已确认故事中的任何角色。")

    @staticmethod
    def _validate_storyboard_plan(plan: StoryboardPlan) -> None:
        if not isinstance(plan.title, str) or not isinstance(plan.narration_text, str):
            raise ValueError("分镜标题和旁白必须是字符串。")
        if not isinstance(plan.audio_path, str) or not isinstance(plan.subtitle_path, str):
            raise ValueError("分镜音频和字幕路径必须是字符串。")
        if (
            isinstance(plan.target_seconds, bool)
            or not isinstance(plan.target_seconds, int)
            or not 15 <= plan.target_seconds <= 300
        ):
            raise ValueError("分镜目标时长必须在 15 到 300 秒之间。")
        if not isinstance(plan.confirmed, bool):
            raise ValueError("分镜确认状态必须是布尔值。")
        if not plan.shots:
            raise ValueError("分镜至少需要一个镜头。")
        if plan.total_duration != plan.target_seconds:
            raise ValueError("分镜总时长与目标时长不一致。")
        if isinstance(plan.base_seed, bool) or not isinstance(plan.base_seed, int):
            raise ValueError("分镜基础 seed 必须是整数。")
        if not isinstance(plan.visual_bible, VisualBible):
            raise ValueError("分镜视觉圣经类型无效。")
        asset_ids: set[str] = set()
        for asset in plan.visual_bible.assets:
            if not isinstance(asset, VisualAsset):
                raise ValueError("视觉资产类型无效。")
            if not asset.asset_id or asset.asset_id in asset_ids:
                raise ValueError("视觉资产 ID 不能为空或重复。")
            asset_ids.add(asset.asset_id)
            if not isinstance(asset.references, list) or not all(
                isinstance(item, VisualReference) for item in asset.references
            ):
                raise ValueError("视觉资产 references 类型无效。")
            for reference in asset.references:
                GuidedStorySession._validate_visual_reference(reference)
        seen_ids: set[int] = set()
        list_fields = (
            "continuity_notes",
            "visual_anchors",
            "reference_asset_ids",
            "reference_image_paths",
            "continuity_diagnostics",
        )
        text_fields = (
            "character",
            "location",
            "visual",
            "action",
            "camera",
            "lighting",
            "mood",
            "narration",
            "video_prompt",
            "negative_prompt",
            "aspect_ratio",
            "shot_purpose",
            "composition",
            "camera_movement",
            "start_frame",
            "end_frame",
            "shot_kind",
            "duration_reason",
            "first_frame_prompt",
            "motion_prompt",
            "end_frame_prompt",
            "initial_frame_source_path",
            "initial_frame_path",
            "initial_frame_url",
            "continuity_mode",
            "generated_first_frame_path",
            "generated_last_frame_path",
        )
        required_text = (
            "action",
            "video_prompt",
            "first_frame_prompt",
            "motion_prompt",
            "end_frame_prompt",
        )
        for shot in plan.shots:
            if isinstance(shot.shot_id, bool) or not isinstance(shot.shot_id, int):
                raise ValueError("镜头 ID 必须是整数。")
            if shot.shot_id < 1:
                raise ValueError("镜头 ID 必须是正整数。")
            if shot.shot_id in seen_ids:
                raise ValueError("镜头 ID 不能重复。")
            seen_ids.add(shot.shot_id)
            if (
                isinstance(shot.scene_id, bool)
                or not isinstance(shot.scene_id, int)
                or shot.scene_id < 1
            ):
                raise ValueError("镜头对应的场景 ID 必须是正整数。")
            if isinstance(shot.duration, bool) or not isinstance(shot.duration, int):
                raise ValueError("镜头时长必须是整数。")
            if not 3 <= shot.duration <= 15:
                raise ValueError("镜头时长必须在 3 到 15 秒之间。")
            for field in text_fields:
                if not isinstance(getattr(shot, field), str):
                    raise ValueError(f"镜头字段 {field} 必须是字符串。")
            for field in required_text:
                value = getattr(shot, field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"镜头缺少有效字段：{field}")
            for field in list_fields:
                values = getattr(shot, field)
                if not isinstance(values, list) or not all(
                    isinstance(item, str) for item in values
                ):
                    raise ValueError(f"镜头字段 {field} 必须是字符串数组。")
            if not isinstance(shot.confirmed_visual_inputs, list) or not all(
                isinstance(item, VisualReference)
                for item in shot.confirmed_visual_inputs
            ):
                raise ValueError("镜头 confirmed_visual_inputs 类型无效。")
            for reference in shot.confirmed_visual_inputs:
                GuidedStorySession._validate_visual_reference(reference)
            for field in ("duration_weight", "estimated_duration"):
                value = getattr(shot, field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"镜头字段 {field} 必须是数字。")
                if value < 0:
                    raise ValueError(f"镜头字段 {field} 不能为负数。")
            if shot.continuity_mode not in {
                "independent",
                "same_scene_chain",
                "new_scene_reference",
            }:
                raise ValueError("镜头 continuity_mode 无法识别。")
            if shot.previous_shot_id is not None and (
                isinstance(shot.previous_shot_id, bool)
                or not isinstance(shot.previous_shot_id, int)
            ):
                raise ValueError("镜头 previous_shot_id 必须是整数或 null。")
            if shot.continuity_mode == "same_scene_chain":
                if (
                    shot.previous_shot_id is None
                    or shot.previous_shot_id == shot.shot_id
                    or shot.previous_shot_id not in seen_ids
                ):
                    raise ValueError("同场景连续镜头必须引用前面已存在的镜头。")
            elif shot.previous_shot_id is not None:
                raise ValueError("独立或新场景镜头不能继承上一镜头。")
            if shot.seed is not None and (
                isinstance(shot.seed, bool) or not isinstance(shot.seed, int)
            ):
                raise ValueError("镜头 seed 必须是整数或 null。")
            if not isinstance(shot.continuity_state, dict):
                raise ValueError("镜头 continuity_state 必须是对象。")
            if not isinstance(shot.continuity_start_state, ContinuityState) or not isinstance(
                shot.continuity_end_state,
                ContinuityState,
            ):
                raise ValueError("镜头连续性起止状态类型无效。")
        GuidedStorySession._validate_video_artifacts(plan.artifacts, seen_ids)

    @staticmethod
    def _validate_visual_reference(reference: VisualReference) -> None:
        for field in (
            "reference_id",
            "path",
            "usage",
            "content_digest",
            "content_summary",
            "binding_kind",
            "binding_id",
        ):
            if not isinstance(getattr(reference, field), str):
                raise ValueError(f"视觉参考字段 {field} 必须是字符串。")
        if reference.usage not in VISUAL_REFERENCE_USAGES:
            raise ValueError(f"无法识别的视觉参考用途：{reference.usage}")
        if reference.binding_kind not in {"", "asset", "shot"}:
            raise ValueError("视觉参考 binding_kind 无法识别。")
        if reference.binding_kind and not reference.binding_id:
            raise ValueError("视觉参考绑定缺少 binding_id。")
        if not isinstance(reference.confirmed, bool):
            raise ValueError("视觉参考 confirmed 必须是布尔值。")

    @staticmethod
    def _validate_render_outputs(
        plan: StoryboardPlan,
        manifest: RenderManifest,
    ) -> None:
        shot_ids = {shot.shot_id for shot in plan.shots}
        GuidedStorySession._validate_video_artifacts(plan.artifacts, shot_ids)
        text_fields = (
            "status",
            "output_dir",
            "render_run_id",
            "final_video_path",
            "audio_path",
            "subtitle_path",
            "error",
        )
        if any(not isinstance(getattr(manifest, field), str) for field in text_fields):
            raise TypeError("渲染记录的文本字段必须是字符串。")
        for field in (
            "generated_shots",
            "reused_shots",
            "failed_shots",
            "dependency_failed_shots",
            "unreferenced_fallback_shots",
        ):
            values = getattr(manifest, field)
            if not isinstance(values, list) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in values
            ):
                raise TypeError(f"渲染记录字段 {field} 必须是整数数组。")
        GuidedStorySession._validate_video_artifacts(manifest.artifacts, shot_ids)

    @staticmethod
    def _validate_video_artifacts(
        artifacts: list[VideoArtifact],
        shot_ids: set[int],
    ) -> None:
        if not isinstance(artifacts, list) or not all(
            isinstance(artifact, VideoArtifact) for artifact in artifacts
        ):
            raise TypeError("视频产物必须是 VideoArtifact 数组。")
        text_fields = (
            "artifact_id",
            "provider",
            "model",
            "status",
            "local_path",
            "remote_url",
            "prompt",
            "created_at",
            "error_message",
            "initial_frame_source_path",
            "initial_frame_path",
            "initial_frame_url",
            "continuity_mode",
            "input_fingerprint",
            "generated_first_frame_path",
            "generated_last_frame_path",
            "published_last_frame_path",
            "published_last_frame_url",
        )
        for artifact in artifacts:
            if any(not isinstance(getattr(artifact, field), str) for field in text_fields):
                raise TypeError("视频产物的文本字段必须是字符串。")
            if artifact.status not in {
                "pending",
                "submission_uncertain",
                "succeeded",
                "failed",
            }:
                raise ValueError(f"无法识别的视频产物状态：{artifact.status}")
            if (
                isinstance(artifact.shot_id, bool)
                or not isinstance(artifact.shot_id, int)
                or artifact.shot_id not in shot_ids
            ):
                raise ValueError("视频产物引用了不存在的镜头。")
            if (
                isinstance(artifact.duration, bool)
                or not isinstance(artifact.duration, int)
                or artifact.duration < 0
            ):
                raise ValueError("视频产物时长必须是非负整数。")
            if (
                isinstance(artifact.attempt, bool)
                or not isinstance(artifact.attempt, int)
                or artifact.attempt < 1
            ):
                raise ValueError("视频产物尝试次数必须是正整数。")
            if artifact.request_id is not None and not isinstance(artifact.request_id, str):
                raise TypeError("视频产物任务 ID 必须是字符串或 null。")
            for field in ("reference_image_paths", "continuity_diagnostics"):
                values = getattr(artifact, field)
                if not isinstance(values, list) or not all(
                    isinstance(item, str) for item in values
                ):
                    raise TypeError(f"视频产物字段 {field} 必须是字符串数组。")
            if not isinstance(artifact.confirmed_visual_inputs, list) or not all(
                isinstance(item, VisualReference)
                for item in artifact.confirmed_visual_inputs
            ):
                raise TypeError("视频产物 confirmed_visual_inputs 类型无效。")
            for reference in artifact.confirmed_visual_inputs:
                GuidedStorySession._validate_visual_reference(reference)
            if artifact.previous_shot_id is not None and not isinstance(
                artifact.previous_shot_id,
                int,
            ):
                raise TypeError("视频产物 previous_shot_id 必须是整数或 null。")
            if artifact.seed is not None and not isinstance(artifact.seed, int):
                raise TypeError("视频产物 seed 必须是整数或 null。")
            if not isinstance(artifact.used_unreferenced_fallback, bool):
                raise TypeError("视频产物无参考回退标志必须是布尔值。")

    @staticmethod
    def _sanitize_persisted_urls(payload: dict[str, Any]) -> None:
        for container_name in ("storyboard", "render_manifest"):
            container = payload.get(container_name)
            if not isinstance(container, dict):
                continue
            artifacts = container.get("artifacts", [])
            if not isinstance(artifacts, list):
                continue
            for artifact in artifacts:
                if isinstance(artifact, dict):
                    artifact["remote_url"] = sanitize_remote_url(
                        str(artifact.get("remote_url", ""))
                    )

    @staticmethod
    def _storyboard_confirmation_signature(plan: StoryboardPlan) -> str:
        runtime_fields = {
            "reference_image_paths",
            "initial_frame_source_path",
            "initial_frame_path",
            "initial_frame_url",
            "continuity_diagnostics",
            "generated_first_frame_path",
            "generated_last_frame_path",
        }
        shots = []
        for shot in to_plain_data(plan.shots):
            shots.append(
                {key: value for key, value in shot.items() if key not in runtime_fields}
            )
        payload = {
            "title": plan.title,
            "target_seconds": plan.target_seconds,
            "shots": shots,
            "visual_bible": to_plain_data(plan.visual_bible),
            "narration_text": plan.narration_text,
            "confirmed": plan.confirmed,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @staticmethod
    def _motion_with_duration(prompt: str, duration: int) -> str:
        value = str(prompt)
        if re.search(r"在\d+秒内", value):
            return re.sub(r"在\d+秒内", f"在{int(duration)}秒内", value)
        return f"在{int(duration)}秒内完成。{value}"

    def _require_direction(self) -> None:
        if not self.direction:
            raise RuntimeError("请先给出一句方向。")

    def _idea_by_id(self, idea_id: str) -> IdeaCard:
        card = next((item for item in self.all_ideas if item.idea_id == idea_id), None)
        if card is None:
            raise ValueError("创意卡不存在或已经过期。")
        return card

    def _element_by_id(self, kind: str, option_id: str) -> ElementOption:
        if self.element_palette is None:
            raise RuntimeError("请先展开故事零件。")
        option = next(
            (
                item
                for item in self.element_palette.options.get(kind, [])
                if item.option_id == option_id
            ),
            None,
        )
        if option is None:
            raise ValueError("故事零件不存在。")
        return option

    def _story_facts(self) -> StoryFacts:
        if self.story:
            protagonist = "、".join(item.name for item in self.story.characters)
            character_visuals = "；".join(
                f"{item.name}：{item.visual_identity or item.description}"
                for item in self.story.characters
            )
            scene_details = "；".join(
                f"{item.name}：{item.visual_identity or item.description}"
                for item in self.story.locations
            )
            return StoryFacts(
                premise=self.direction,
                genre=self.brief.genre,
                tone=self.story.tone,
                theme=self.story.theme,
                opening=self.story.story_text[:120],
                protagonist=protagonist,
                protagonist_goal=self.story.logline,
                conflict=self.story.core_conflict,
                development=self.story.story_text,
                turning_point=self.story.field_sources.get(
                    "turning_point", SourceAttribution("", "", "")
                ).value,
                ending=self.story.ending,
                character_visuals=character_visuals or "保持人物外观一致",
                scene_details=scene_details or "保持地点空间、光线和主色一致",
                props="；".join(self.story.visual_anchors),
                narration_style="克制旁白，只补充画面不可见信息",
                visual_anchors="；".join(self.story.visual_anchors),
                transitions="动作方向和关键物件匹配剪辑",
            )
        return self.legacy_facts

    def _snapshot(
        self,
        artifact_type: str,
        payload: dict[str, Any],
        *,
        user_feedback: str = "",
        confirmed: bool = False,
    ) -> ArtifactRevision:
        history = self.revisions.setdefault(artifact_type, [])
        cursor = self.revision_cursor.get(artifact_type, len(history) - 1)
        if cursor < len(history) - 1:
            del history[cursor + 1 :]
        revision = ArtifactRevision(
            artifact_type=artifact_type,
            version=len(history) + 1,
            payload=deepcopy(payload),
            parent_version=history[-1].version if history else None,
            user_feedback=user_feedback,
            confirmed=confirmed,
        )
        history.append(revision)
        self.revision_cursor[artifact_type] = len(history) - 1
        return revision

    @_state_mutation
    def undo_artifact(self, artifact_type: str | None = None) -> Any:
        kind = artifact_type or (
            "storyboard" if self.storyboard else "script" if self.script else "story"
        )
        cursor = self.revision_cursor.get(kind, -1)
        if cursor <= 0:
            raise RuntimeError("没有更早的版本可以撤销。")
        next_cursor = cursor - 1
        revision = self.revisions[kind][next_cursor]
        candidate = self._prepare_revision(revision)
        restored = self._commit_revision_candidate(revision.artifact_type, candidate)
        self.revision_cursor[kind] = next_cursor
        return restored

    @_state_mutation
    def redo_artifact(self, artifact_type: str | None = None) -> Any:
        kind = artifact_type or (
            "storyboard" if self.storyboard else "script" if self.script else "story"
        )
        cursor = self.revision_cursor.get(kind, -1)
        history = self.revisions.get(kind, [])
        if cursor < 0 or cursor >= len(history) - 1:
            raise RuntimeError("没有更新的版本可以重做。")
        next_cursor = cursor + 1
        revision = history[next_cursor]
        candidate = self._prepare_revision(revision)
        restored = self._commit_revision_candidate(revision.artifact_type, candidate)
        self.revision_cursor[kind] = next_cursor
        return restored

    def _restore_revision(self, revision: ArtifactRevision) -> Any:
        candidate = self._prepare_revision(revision)
        return self._commit_revision_candidate(revision.artifact_type, candidate)

    def _prepare_revision(self, revision: ArtifactRevision) -> Any:
        if not isinstance(revision.payload, dict):
            raise ValueError("版本 payload 必须是 JSON 对象。")
        if revision.artifact_type == "story":
            candidate = self._story_from_data(deepcopy(revision.payload))
            self._validate_story(candidate)
            return candidate
        if revision.artifact_type == "script":
            if self.story is None or not self.story.confirmed:
                raise ValueError("恢复剧本版本前必须存在已确认故事。")
            candidate = self._script_from(deepcopy(revision.payload))
            self._validate_loaded_script(candidate)
            return candidate
        if revision.artifact_type == "draft":
            if self.story is None or not self.story.confirmed:
                raise ValueError("恢复旧版草稿前必须存在已确认故事。")
            candidate = self._draft_from(deepcopy(revision.payload))
            self._validate_loaded_script(candidate.script)
            return candidate
        if revision.artifact_type == "storyboard":
            if (
                self.story is None
                or not self.story.confirmed
                or self.script is None
                or not self.script.confirmed
            ):
                raise ValueError("恢复分镜版本前必须存在已确认故事和剧本。")
            candidate = self._storyboard_from(deepcopy(revision.payload))
            self._validate_storyboard_plan(candidate)
            return candidate
        raise ValueError("未知版本类型。")

    def _commit_revision_candidate(self, artifact_type: str, candidate: Any) -> Any:
        if artifact_type == "story":
            self.story = candidate
            self.stage = Stage.STORY_REVIEW
            self.draft = None
            self.outline = None
            self.script = None
            self.storyboard = None
            self.render_manifest = None
            return self.story
        if artifact_type == "script":
            self.script = candidate
            self.stage = Stage.SCRIPT_REVIEW
            self.draft = None
            self.outline = None
            self.storyboard = None
            self.render_manifest = None
            return self.script
        if artifact_type == "draft":
            self.draft = candidate
            self.outline, self.script = self.draft.outline, self.draft.script
            self.stage = Stage.SCRIPT_REVIEW
            self.storyboard = None
            self.render_manifest = None
            return self.draft
        if artifact_type == "storyboard":
            self.storyboard = candidate
            self.stage = Stage.STORYBOARD_REVIEW
            self.render_manifest = None
            return self.storyboard
        raise ValueError("未知版本类型。")

    def _load_revisions(self, data: dict[str, Any]) -> None:
        raw_revisions = data.get("revisions", {})
        raw_cursors = data.get("revision_cursor", {})
        if not isinstance(raw_revisions, dict) or not isinstance(raw_cursors, dict):
            raise ValueError("revisions 和 revision_cursor 必须是 JSON 对象。")
        allowed_kinds = {"story", "script", "draft", "storyboard"}
        for raw_kind, entries in raw_revisions.items():
            kind = str(raw_kind)
            if kind not in allowed_kinds:
                raise ValueError(f"未知版本类型：{kind}")
            if not isinstance(entries, list):
                raise ValueError("revisions 中的每项必须是数组。")
            revisions: list[ArtifactRevision] = []
            for expected_version, item in enumerate(entries, start=1):
                if not isinstance(item, dict):
                    raise ValueError("revision 必须是 JSON 对象。")
                if item.get("artifact_type") != kind:
                    raise ValueError(f"{kind} 版本的 artifact_type 不一致。")
                version = item.get("version")
                if (
                    isinstance(version, bool)
                    or not isinstance(version, int)
                    or version != expected_version
                ):
                    raise ValueError(f"{kind} 版本号必须从 1 连续递增。")
                revision_payload = item.get("payload")
                if not isinstance(revision_payload, dict):
                    raise ValueError(f"{kind} 版本 payload 必须是 JSON 对象。")
                parent_version = item.get("parent_version")
                expected_parent = expected_version - 1 if expected_version > 1 else None
                if parent_version != expected_parent or isinstance(parent_version, bool):
                    raise ValueError(f"{kind} 版本的父版本号无效。")
                user_feedback = item.get("user_feedback", "")
                source_turn_ids = item.get("source_turn_ids", [])
                created_at = item.get("created_at", "")
                if not isinstance(user_feedback, str) or not isinstance(created_at, str):
                    raise ValueError(f"{kind} 版本的文本元数据无效。")
                if not isinstance(source_turn_ids, list) or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in source_turn_ids
                ):
                    raise ValueError(f"{kind} 版本的来源轮次无效。")
                confirmed = self._strict_bool(
                    item.get("confirmed", False),
                    "revision.confirmed",
                )
                revisions.append(
                    ArtifactRevision(
                        artifact_type=kind,
                        version=version,
                        payload=deepcopy(revision_payload),
                        parent_version=parent_version,
                        user_feedback=user_feedback,
                        source_turn_ids=list(source_turn_ids),
                        created_at=created_at,
                        confirmed=confirmed,
                    )
                )
            self.revisions[kind] = revisions
        cursors: dict[str, int] = {}
        for raw_kind, value in raw_cursors.items():
            kind = str(raw_kind)
            if kind not in self.revisions or isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{kind} 的版本游标无效。")
            cursors[kind] = value
        self.revision_cursor = cursors

    @staticmethod
    def _strict_bool(value: Any, field_name: str) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{field_name} 必须是 JSON 布尔值。")
        return value

    @staticmethod
    def _batch_from(data: dict[str, Any]) -> IdeaBatch:
        return IdeaBatch(
            round=int(data.get("round", 1)),
            cards=[IdeaCard(**item) for item in data.get("cards", [])],
            recommended_id=str(data.get("recommended_id", "")),
            feedback=str(data.get("feedback", "")),
            generation_kind=str(data.get("generation_kind", "diverge")),
        )

    @staticmethod
    def _palette_from(data: dict[str, Any]) -> ElementPalette:
        return ElementPalette(
            options={
                str(kind): [ElementOption(**item) for item in values]
                for kind, values in data.get("options", {}).items()
            }
        )

    @staticmethod
    def _story_from_data(data: dict[str, Any]) -> StoryDraft:
        return StoryDraft(
            title=str(data.get("title", "")),
            logline=str(data.get("logline", "")),
            story_text=str(data.get("story_text", "")),
            characters=[StoryCharacter(**item) for item in data.get("characters", [])],
            locations=[StoryLocation(**item) for item in data.get("locations", [])],
            tone=str(data.get("tone", "")),
            theme=str(data.get("theme", "")),
            core_conflict=str(data.get("core_conflict", "")),
            ending=str(data.get("ending", "")),
            visual_anchors=[str(item) for item in data.get("visual_anchors", [])],
            field_sources={
                str(kind): SourceAttribution(**item)
                for kind, item in data.get("field_sources", {}).items()
            },
            ai_filled_fields=[str(item) for item in data.get("ai_filled_fields", [])],
            version=int(data.get("version", 1)),
            confirmed=GuidedStorySession._strict_bool(
                data.get("confirmed", False),
                "story.confirmed",
            ),
        )

    @staticmethod
    def _story_from_legacy_draft(draft: DraftBundle) -> StoryDraft:
        story_text = draft.outline.development.strip() or "\n\n".join(
            beat.event for beat in draft.outline.beats if beat.event.strip()
        )
        if len(story_text) < 120:
            story_text = "\n\n".join(
                [
                    draft.outline.opening,
                    draft.outline.protagonist_goal,
                    draft.outline.conflict,
                    draft.outline.development,
                    draft.outline.turning_point,
                    draft.outline.ending,
                ]
            )
        names = []
        for scene in draft.script.scenes:
            for name in scene.characters:
                if name not in names:
                    names.append(name)
        locations = []
        for scene in draft.script.scenes:
            if scene.location not in locations:
                locations.append(scene.location)
        return StoryDraft(
            title=draft.outline.title,
            logline=draft.outline.logline,
            story_text=story_text,
            characters=[
                StoryCharacter(name, draft.outline.protagonist_goal, "保持旧版人物外观一致")
                for name in (names or ["主角"])
            ],
            locations=[
                StoryLocation(name, "从旧版剧本迁移的地点", "保持旧版空间与光线一致")
                for name in (locations or ["故事核心场景"])
            ],
            core_conflict=draft.outline.conflict,
            ending=draft.outline.ending,
            visual_anchors=["旧版人物外观", "旧版关键道具", "旧版场景主色"],
            field_sources=deepcopy(draft.field_sources),
            ai_filled_fields=list(draft.ai_filled_fields),
            version=draft.version,
            confirmed=True,
        )

    @classmethod
    def _draft_from(cls, data: dict[str, Any]) -> DraftBundle:
        return DraftBundle(
            outline=cls._outline_from(data["outline"]),
            script=cls._script_from(data["script"]),
            field_sources={
                str(kind): SourceAttribution(**item)
                for kind, item in data.get("field_sources", {}).items()
            },
            ai_filled_fields=[str(item) for item in data.get("ai_filled_fields", [])],
            version=int(data.get("version", 1)),
        )

    @staticmethod
    def _facts_from(data: dict[str, Any]) -> StoryFacts:
        allowed = StoryFacts.__dataclass_fields__
        return StoryFacts(**{key: str(value) for key, value in data.items() if key in allowed})

    @staticmethod
    def _outline_from(data: dict[str, Any]) -> StoryOutline:
        beats = [StoryBeat(**item) for item in data.get("beats", [])]
        fields = StoryOutline.__dataclass_fields__
        kwargs = {key: value for key, value in data.items() if key in fields and key != "beats"}
        kwargs.setdefault("source_turn_ids", [])
        kwargs["confirmed"] = GuidedStorySession._strict_bool(
            kwargs.get("confirmed", False),
            "outline.confirmed",
        )
        return StoryOutline(**kwargs, beats=beats)

    @staticmethod
    def _script_from(data: dict[str, Any]) -> StoryScript:
        scenes = [StoryScene(**item) for item in data.get("scenes", [])]
        return StoryScript(
            title=str(data.get("title", "")),
            target_seconds=int(data.get("target_seconds", sum(item.duration for item in scenes))),
            scenes=scenes,
            confirmed=GuidedStorySession._strict_bool(
                data.get("confirmed", False),
                "script.confirmed",
            ),
        )

    @staticmethod
    def _storyboard_from(data: dict[str, Any]) -> StoryboardPlan:
        bible_data = data.get("visual_bible", {})
        visual_bible = VisualBible(
            visual_style=str(bible_data.get("visual_style", "电影感写实")),
            color_palette=str(bible_data.get("color_palette", "统一、克制的电影色彩")),
            lighting_rules=str(
                bible_data.get("lighting_rules", "光源方向和时段连续，避免镜头间突变")
            ),
            camera_language=str(
                bible_data.get("camera_language", "镜头服务于动作和情绪，不为变化而变化")
            ),
            assets=[
                VisualAsset(
                    asset_id=str(item.get("asset_id", "")),
                    kind=str(item.get("kind", "")),
                    name=str(item.get("name", "")),
                    description=str(item.get("description", "")),
                    reference_images=[str(path) for path in item.get("reference_images", [])],
                    references=[
                        GuidedStorySession._visual_reference_from(reference)
                        for reference in (item.get("references") or [])
                    ],
                )
                for item in bible_data.get("assets", [])
            ],
            continuity_rules=[str(item) for item in bible_data.get("continuity_rules", [])],
        )
        plan = StoryboardPlan(
            title=str(data.get("title", "")),
            target_seconds=int(data.get("target_seconds", 45)),
            shots=[
                GuidedStorySession._storyboard_shot_from(item)
                for item in data.get("shots", [])
            ],
            narration_text=str(data.get("narration_text", "")),
            visual_bible=visual_bible,
            base_seed=int(data.get("base_seed", 0)),
            confirmed=GuidedStorySession._strict_bool(
                data.get("confirmed", False),
                "storyboard.confirmed",
            ),
            audio_path=str(data.get("audio_path", "")),
            subtitle_path=str(data.get("subtitle_path", "")),
            artifacts=[
                GuidedStorySession._artifact_from_data(item) for item in data.get("artifacts", [])
            ],
        )
        normalize_narration_timeline(plan)
        return plan

    @staticmethod
    def _manifest_from(data: dict[str, Any]) -> RenderManifest:
        return RenderManifest(
            status=str(data.get("status", "")),
            output_dir=str(data.get("output_dir", "")),
            render_run_id=str(data.get("render_run_id", "")),
            generated_shots=[int(item) for item in data.get("generated_shots", [])],
            reused_shots=[int(item) for item in data.get("reused_shots", [])],
            failed_shots=[int(item) for item in data.get("failed_shots", [])],
            dependency_failed_shots=[
                int(item) for item in data.get("dependency_failed_shots", [])
            ],
            unreferenced_fallback_shots=[
                int(item) for item in data.get("unreferenced_fallback_shots", [])
            ],
            artifacts=[
                GuidedStorySession._artifact_from_data(item) for item in data.get("artifacts", [])
            ],
            final_video_path=str(data.get("final_video_path", "")),
            audio_path=str(data.get("audio_path", "")),
            subtitle_path=str(data.get("subtitle_path", "")),
            error=str(data.get("error", "")),
        )

    @staticmethod
    def _artifact_from_data(data: dict[str, Any]) -> VideoArtifact:
        if not isinstance(data, dict):
            raise ValueError("视频产物必须是 JSON 对象。")
        payload = dict(data)
        payload["remote_url"] = sanitize_remote_url(str(payload.get("remote_url", "")))
        payload["reference_image_paths"] = [
            str(item) for item in payload.get("reference_image_paths", [])
        ]
        payload["confirmed_visual_inputs"] = [
            GuidedStorySession._visual_reference_from(item)
            for item in (payload.get("confirmed_visual_inputs") or [])
        ]
        payload["continuity_diagnostics"] = [
            str(item) for item in payload.get("continuity_diagnostics", [])
        ]
        return VideoArtifact(**payload)

    @staticmethod
    def _storyboard_shot_from(data: dict[str, Any]) -> StoryboardShot:
        if not isinstance(data, dict):
            raise ValueError("分镜必须是 JSON 对象。")
        payload = dict(data)
        payload["first_frame_prompt"] = str(
            payload.get("first_frame_prompt")
            or payload.get("start_frame")
            or payload.get("video_prompt", "")
        )
        payload["motion_prompt"] = str(
            payload.get("motion_prompt")
            or payload.get("action")
            or payload.get("video_prompt", "")
        )
        payload["end_frame_prompt"] = str(
            payload.get("end_frame_prompt")
            or payload.get("end_frame")
            or payload.get("video_prompt", "")
        )
        for field_name in (
            "reference_asset_ids",
            "reference_image_paths",
            "continuity_diagnostics",
        ):
            payload[field_name] = [
                str(value) for value in payload.get(field_name, [])
            ]
        payload["confirmed_visual_inputs"] = [
            GuidedStorySession._visual_reference_from(item)
            for item in (payload.get("confirmed_visual_inputs") or [])
        ]
        payload["continuity_state"] = dict(payload.get("continuity_state") or {})
        payload["continuity_start_state"] = GuidedStorySession._continuity_state_from(
            payload.get("continuity_start_state")
        )
        payload["continuity_end_state"] = GuidedStorySession._continuity_state_from(
            payload.get("continuity_end_state")
        )
        return StoryboardShot(**payload)

    @staticmethod
    def _visual_reference_from(data: Any) -> VisualReference:
        if isinstance(data, VisualReference):
            return data
        if not isinstance(data, dict):
            raise ValueError("视觉参考必须是 JSON 对象。")
        return VisualReference(
            reference_id=str(data.get("reference_id", "")),
            path=str(data.get("path", "")),
            usage=str(data.get("usage", "scene_reference")),
            content_digest=str(data.get("content_digest", "")),
            content_summary=str(data.get("content_summary", "")),
            confirmed=GuidedStorySession._strict_bool(
                data.get("confirmed", False),
                "visual_reference.confirmed",
            ),
            binding_kind=str(data.get("binding_kind", "")),
            binding_id=str(data.get("binding_id", "")),
        )

    @staticmethod
    def _continuity_state_from(data: Any) -> ContinuityState:
        if isinstance(data, ContinuityState):
            return data
        if not isinstance(data, dict):
            return ContinuityState()
        payload = dict(data)
        for field_name in (
            "character_appearance",
            "character_clothing",
            "character_positions",
            "character_emotions",
            "character_injuries",
            "prop_positions",
        ):
            payload[field_name] = {
                str(key): str(value)
                for key, value in dict(payload.get(field_name) or {}).items()
            }
        for field_name in ("character_knowledge", "character_held_props"):
            payload[field_name] = {
                str(key): [str(item) for item in value]
                for key, value in dict(payload.get(field_name) or {}).items()
                if isinstance(value, list)
            }
        for field_name in (
            "location",
            "time_of_day",
            "weather",
            "key_light_direction",
        ):
            payload[field_name] = str(payload.get(field_name, ""))
        return ContinuityState(**payload)
