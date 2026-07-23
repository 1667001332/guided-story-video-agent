from __future__ import annotations

import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

from .agent import IDEA_COUNT, RuleBasedStoryAgent, StoryAgent
from .models import (
    ArtifactReview,
    ArtifactRevision,
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
    to_plain_data,
)
from .storyboard import build_storyboard
from .timing import allocate_durations


class GuidedStorySession:
    """Single source of truth for the low-pressure idea garden."""

    schema_version = 4

    def __init__(
        self,
        brief: CreativeBrief | None = None,
        agent: StoryAgent | None = None,
    ) -> None:
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
        self.legacy_facts = StoryFacts(genre=self.brief.genre)
        self.revisions: dict[str, list[ArtifactRevision]] = {}
        self.revision_cursor: dict[str, int] = {}
        self.user_action_count = 0
        self.free_text_count = 0

    @property
    def current_batch(self) -> IdeaBatch | None:
        return self.idea_batches[-1] if self.idea_batches else None

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
    def current_question(self) -> str:
        if not self.direction:
            return "随便说一个方向就行，例如：校园里发生一件带点悬疑的事。"
        return "你可以选卡、换一批、混合，或者现在就生成完整故事。"

    def start_ideation(self, direction: str) -> IdeaBatch:
        cleaned = " ".join(direction.split())
        if not cleaned:
            raise ValueError("请先说一个大概方向，几个词也可以。")
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
        self.chat_history = [{"role": "user", "content": cleaned}]
        self.story = None
        self.story_history = []
        self.free_text_count = 1
        batch = self.agent.generate_ideas(cleaned, round_number=1)
        self._validate_batch(batch, previous_cards=self.all_ideas)
        self.idea_batches.append(batch)
        self.chat_history.append(
            {
                "role": "assistant",
                "content": "我先铺开8个不同方向。你不需要回答问题，只要挑喜欢的。",
            }
        )
        return batch

    def refresh_ideas(self, feedback: str = "") -> IdeaBatch:
        self._require_direction()
        if self.current_batch:
            self.rejected_idea_ids.extend(
                card.idea_id
                for card in self.current_batch.cards
                if card.idea_id not in self.selected_idea_ids
                and card.idea_id not in self.rejected_idea_ids
            )
        batch = self.agent.generate_ideas(
            self.direction,
            round_number=len(self.idea_batches) + 1,
            feedback=" ".join(feedback.split()),
            previous_cards=self.all_ideas,
        )
        self._validate_batch(batch, previous_cards=self.all_ideas)
        self.idea_batches.append(batch)
        self.user_action_count += 1
        return batch

    def select_ideas(self, idea_ids: list[str]) -> SelectionState:
        unique = list(dict.fromkeys(str(item) for item in idea_ids if str(item)))
        if len(unique) > 3:
            raise ValueError("最多同时保留3张创意卡。")
        known = {card.idea_id for card in self.all_ideas}
        unknown = [item for item in unique if item not in known]
        if unknown:
            raise ValueError("选择中包含已经过期的创意卡。")
        self.selected_idea_ids = unique
        self.user_action_count += 1
        return self.selection_state

    def more_like(self, idea_id: str) -> IdeaBatch:
        card = self._idea_by_id(idea_id)
        batch = self.agent.generate_ideas(
            self.direction,
            round_number=len(self.idea_batches) + 1,
            previous_cards=self.all_ideas,
            mode="similar",
            anchors=[card],
        )
        self._validate_batch(batch, previous_cards=self.all_ideas)
        for generated in batch.cards:
            generated.source_idea_ids = [card.idea_id]
        self.idea_batches.append(batch)
        self.user_action_count += 1
        return batch

    def mix_selected(self) -> IdeaBatch:
        if not 1 <= len(self.selected_idea_ids) <= 3:
            raise RuntimeError("请先选择1到3张想混合的卡。")
        batch = self.agent.generate_ideas(
            self.direction,
            round_number=len(self.idea_batches) + 1,
            previous_cards=self.all_ideas,
            mode="mix",
            anchors=self.selected_cards,
        )
        self._validate_batch(batch, previous_cards=self.all_ideas)
        required_sources = set(self.selected_idea_ids)
        for card in batch.cards:
            card.source_idea_ids = list(required_sources)
        self.idea_batches.append(batch)
        self.user_action_count += 1
        return batch

    def expand_selected(self) -> ElementPalette:
        self._require_direction()
        self.element_palette = self.agent.expand_elements(self.direction, self.selected_cards)
        for kind in ("character", "conflict", "turning_point", "ending"):
            if len(self.element_palette.options.get(kind, [])) != 4:
                raise ValueError(f"{kind} 必须提供4个选项。")
        self.user_action_count += 1
        return self.element_palette

    def choose_element(self, kind: str, option_id: str) -> SelectionState:
        option = self._element_by_id(kind, option_id)
        self.selected_elements[kind] = option.option_id
        self.user_action_count += 1
        return self.selection_state

    def auto_choose(self) -> SelectionState:
        self._require_direction()
        if self.current_batch is None:
            raise RuntimeError("还没有创意卡。")
        selected = self.current_batch.recommended_id or self.current_batch.cards[0].idea_id
        self.selected_idea_ids = [selected]
        self.user_action_count += 1
        return self.selection_state

    def chat_ideation(self, text: str) -> IdeationTurnResult:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("可以随便说一句想调整的方向。")
        if not self.direction:
            batch = self.start_ideation(cleaned)
        else:
            self.chat_history.append({"role": "user", "content": cleaned})
            self.free_text_count += 1
            batch = self.refresh_ideas(cleaned)
        message = f"已按“{cleaned}”换出8个新方向；你仍然可以直接生成完整故事。"
        self.chat_history.append({"role": "assistant", "content": message})
        return IdeationTurnResult(
            message=message,
            batch=batch,
            selection=self.selection_state,
            available_actions=["select", "more_like", "mix", "expand", "story"],
            used_fallback=bool(getattr(self.agent, "last_used_fallback", False)),
        )

    def generate_story(self) -> StoryDraft:
        self._require_direction()
        selected_options = {
            kind: self._element_by_id(kind, option_id)
            for kind, option_id in self.selected_elements.items()
        }
        story = self.agent.generate_story(
            self.direction,
            self.selected_cards,
            selected_options,
        )
        self._preserve_user_choices(story, selected_options)
        self._validate_story(story)
        story.version = len(self.story_history) + 1
        story.confirmed = False
        self.story = story
        self.story_history.append(deepcopy(story))
        self.script = None
        self.draft = None
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.STORY_REVIEW
        self.user_action_count += 1
        self._snapshot("story", to_plain_data(story))
        return story

    def revise_story(self, feedback: str) -> StoryDraft:
        if self.story is None:
            raise RuntimeError("请先生成一版完整故事。")
        revised = self.agent.revise_story(self.story, feedback)
        selected_options = {
            kind: self._element_by_id(kind, option_id)
            for kind, option_id in self.selected_elements.items()
        }
        self._preserve_user_choices(revised, selected_options)
        self._validate_story(revised)
        revised.version = len(self.story_history) + 1
        revised.confirmed = False
        self.story = revised
        self.story_history.append(deepcopy(revised))
        self.script = None
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.STORY_REVIEW
        self.free_text_count += 1
        self._snapshot("story", to_plain_data(revised), user_feedback=feedback)
        return revised

    def back_to_ideation(self) -> None:
        self._require_direction()
        self.stage = Stage.IDEATING

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

    def generate_script(self) -> StoryScript:
        if self.story is None or not self.story.confirmed:
            raise RuntimeError("请先确认完整故事。")
        self.script = self.agent.generate_script(self.story, self.brief.target_seconds)
        self._validate_or_repair_script(self.script)
        self.script.confirmed = False
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.SCRIPT_REVIEW
        self.user_action_count += 1
        self._snapshot("script", to_plain_data(self.script))
        return self.script

    def revise_script(self, feedback: str) -> StoryScript:
        if self.story is None or not self.story.confirmed or self.script is None:
            raise RuntimeError("请先确认故事并生成剧本。")
        self.script = self.agent.revise_script(self.story, self.script, feedback)
        self._validate_or_repair_script(self.script)
        self.script.confirmed = False
        self.storyboard = None
        self.render_manifest = None
        self.stage = Stage.SCRIPT_REVIEW
        self.free_text_count += 1
        self._snapshot("script", to_plain_data(self.script), user_feedback=feedback)
        return self.script

    def confirm_script(self) -> None:
        if self.script is None or self.stage != Stage.SCRIPT_REVIEW:
            raise RuntimeError("当前没有等待确认的剧本。")
        review = self.review_current_artifact("script")
        if not review.can_confirm:
            raise RuntimeError("剧本仍有必须修复的问题：" + "；".join(review.hard_errors))
        self.script.confirmed = True
        self._snapshot("script", to_plain_data(self.script), confirmed=True)

    def build_storyboard(self) -> StoryboardPlan:
        if self.script is None or not self.script.confirmed:
            raise RuntimeError("请先确认剧本。")
        self.storyboard = build_storyboard(self.script, self._story_facts())
        self.stage = Stage.STORYBOARD_REVIEW
        self._snapshot("storyboard", to_plain_data(self.storyboard))
        return self.storyboard

    # v0.3 compatibility wrappers -------------------------------------------------
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

    def revise_draft(self, feedback: str) -> DraftBundle:
        warnings.warn("revise_draft 已被 revise_story/revise_script 取代", DeprecationWarning)
        if self.draft is None:
            raise RuntimeError("请先生成剧本。")
        self.draft.script = self.revise_script(feedback)
        self.draft.version += 1
        return self.draft

    def confirm_draft(self) -> None:
        warnings.warn("confirm_draft 已被 confirm_script 取代", DeprecationWarning)
        self.confirm_script()

    def update_storyboard_shot(self, shot_id: int, patch: dict[str, Any]) -> StoryboardPlan:
        if self.storyboard is None:
            raise RuntimeError("尚未生成分镜。")
        shot = next((item for item in self.storyboard.shots if item.shot_id == int(shot_id)), None)
        if shot is None:
            raise ValueError("镜头不存在。")
        for field, value in patch.items():
            if not hasattr(shot, field) or field == "shot_id":
                raise ValueError(f"不支持的镜头字段：{field}")
            setattr(shot, field, value)
        self.storyboard.confirmed = False
        self.stage = Stage.STORYBOARD_REVIEW
        self._snapshot(
            "storyboard", to_plain_data(self.storyboard), user_feedback=f"retake {shot_id}"
        )
        return self.storyboard

    def confirm_storyboard(self) -> None:
        if self.storyboard is None or self.stage != Stage.STORYBOARD_REVIEW:
            raise RuntimeError("当前没有等待确认的分镜。")
        review = self.review_current_artifact("storyboard")
        if not review.can_confirm:
            raise RuntimeError("分镜仍有必须修复的问题：" + "；".join(review.hard_errors))
        self.storyboard.confirmed = True
        self.stage = Stage.RENDER_READY
        self._snapshot("storyboard", to_plain_data(self.storyboard), confirmed=True)

    def render_confirmed_plan(self, renderer, output_dir: str | Path) -> RenderManifest:
        if (
            self.stage != Stage.RENDER_READY
            or self.storyboard is None
            or not self.storyboard.confirmed
        ):
            raise RuntimeError("必须先确认完整分镜，才能调用视频生成。")
        self.render_manifest = renderer.render(self.storyboard, output_dir)
        if self.render_manifest.status == "succeeded":
            self.stage = Stage.COMPLETED
        return self.render_manifest

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
            if abs(self.script.total_duration - self.brief.target_seconds) > 1:
                review.hard_errors.append("剧本总时长不符合目标")
            if not self.script.scenes:
                review.hard_errors.append("剧本没有可转换的场景")
            if any(
                not (scene.visible_action or scene.action).strip()
                for scene in self.script.scenes
            ):
                review.hard_errors.append("存在无法拍摄的空动作场景")
            review.scores["filmability"] = 1.0 if not review.hard_errors else 0.5
        elif kind == "storyboard" and self.storyboard:
            if abs(self.storyboard.total_duration - self.brief.target_seconds) > 1:
                review.hard_errors.append("分镜总时长不符合目标")
            minimum = max(1, (self.brief.target_seconds + 14) // 15)
            maximum = max(1, self.brief.target_seconds // 3)
            if not minimum <= len(self.storyboard.shots) <= maximum:
                review.hard_errors.append(
                    f"当前时长下分镜数量必须在{minimum}到{maximum}之间"
                )
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
            review.scores["visual_identity_coverage"] = referenced / len(
                self.storyboard.shots
            )
        else:
            raise RuntimeError("当前没有可审查的产物。")
        return review

    # v0.2 compatibility wrappers -------------------------------------------------
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

    def answer_detail_question(self, text: str, *, source: str = "human") -> GuideTurnResult:
        return self.submit_user_turn(text, source=source)

    def request_suggestions(self) -> list[CreativeSuggestion]:
        if self.current_batch is None:
            return []
        return [
            CreativeSuggestion(card.idea_id, card.title, card.logline, "idea")
            for card in self.current_batch.cards[:3]
        ]

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

    def build_outline(self) -> StoryOutline:
        warnings.warn("build_outline 已被 generate_draft 取代", DeprecationWarning)
        return self.generate_draft().outline

    def confirm_outline(self) -> None:
        if self.story is None:
            raise RuntimeError("尚未生成故事。")
        self.confirm_story()

    def build_script(self) -> StoryScript:
        warnings.warn("build_script 已被 generate_script 取代", DeprecationWarning)
        if self.story is None:
            self.generate_story()
        if not self.story.confirmed:
            self.confirm_story()
        return self.generate_script()

    def update_script_scene(self, scene_id: int, patch: dict[str, Any]) -> StoryScript:
        if self.script is None:
            raise RuntimeError("尚未生成剧本。")
        scene = next(
            (item for item in self.script.scenes if item.scene_id == int(scene_id)), None
        )
        if scene is None:
            raise ValueError("场景不存在。")
        for field, value in patch.items():
            if not hasattr(scene, field) or field == "scene_id":
                raise ValueError(f"不支持的场景字段：{field}")
            setattr(scene, field, value)
        self.script.confirmed = False
        self.stage = Stage.SCRIPT_REVIEW
        self._snapshot("script", to_plain_data(self.script), user_feedback=f"edit scene {scene_id}")
        return self.script

    # persistence ----------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "brief": to_plain_data(self.brief),
            "direction": self.direction,
            "idea_batches": to_plain_data(self.idea_batches),
            "selected_idea_ids": self.selected_idea_ids,
            "rejected_idea_ids": self.rejected_idea_ids,
            "element_palette": to_plain_data(self.element_palette)
            if self.element_palette
            else None,
            "selected_elements": self.selected_elements,
            "chat_history": self.chat_history,
            "story": to_plain_data(self.story) if self.story else None,
            "story_history": to_plain_data(self.story_history),
            "script": to_plain_data(self.script) if self.script else None,
            "draft": to_plain_data(self.draft) if self.draft else None,
            "draft_history": to_plain_data(self.draft_history),
            "storyboard": to_plain_data(self.storyboard) if self.storyboard else None,
            "render_manifest": to_plain_data(self.render_manifest)
            if self.render_manifest
            else None,
            "legacy_facts": to_plain_data(self.legacy_facts),
            "revisions": to_plain_data(self.revisions),
            "revision_cursor": self.revision_cursor,
            "metrics": {
                "user_action_count": self.user_action_count,
                "free_text_count": self.free_text_count,
            },
        }

    @classmethod
    def load(cls, path: str | Path, *, agent: StoryAgent | None = None) -> GuidedStorySession:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        session = cls(CreativeBrief(**data.get("brief", {})), agent=agent)
        schema_version = int(data.get("schema_version", 1))
        if schema_version < 3:
            return session._load_v2(data)
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
        if data.get("render_manifest"):
            session.render_manifest = session._manifest_from(data["render_manifest"])
        session.legacy_facts = session._facts_from(data.get("legacy_facts", {}))
        session._load_revisions(data)
        metrics = data.get("metrics", {})
        session.user_action_count = int(metrics.get("user_action_count", 0))
        session.free_text_count = int(metrics.get("free_text_count", 0))
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
            self.stage = (
                Stage.RENDER_READY if old_stage == "render_ready" else Stage.STORYBOARD_REVIEW
            )
        if data.get("render_manifest"):
            self.render_manifest = self._manifest_from(data["render_manifest"])
        self._load_revisions(data)
        return self

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

    def _validate_or_repair_script(self, script: StoryScript) -> None:
        if not script.scenes:
            raise ValueError("剧本至少需要一个可拍摄场景。")
        total = script.total_duration
        if abs(total - self.brief.target_seconds) > 1:
            durations = allocate_durations(
                self.brief.target_seconds,
                len(script.scenes),
                minimum=1,
                maximum=self.brief.target_seconds,
            )
            for scene, duration in zip(script.scenes, durations):
                scene.duration = duration
        for scene in script.scenes:
            if not (scene.visible_action or scene.action).strip():
                scene.action = scene.narration or "主角完成一个清晰可见的动作"
                scene.visible_action = scene.action

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

    def undo_artifact(self, artifact_type: str | None = None) -> Any:
        kind = artifact_type or (
            "storyboard" if self.storyboard else "script" if self.script else "story"
        )
        cursor = self.revision_cursor.get(kind, -1)
        if cursor <= 0:
            raise RuntimeError("没有更早的版本可以撤销。")
        cursor -= 1
        self.revision_cursor[kind] = cursor
        return self._restore_revision(self.revisions[kind][cursor])

    def redo_artifact(self, artifact_type: str | None = None) -> Any:
        kind = artifact_type or (
            "storyboard" if self.storyboard else "script" if self.script else "story"
        )
        cursor = self.revision_cursor.get(kind, -1)
        history = self.revisions.get(kind, [])
        if cursor < 0 or cursor >= len(history) - 1:
            raise RuntimeError("没有更新的版本可以重做。")
        cursor += 1
        self.revision_cursor[kind] = cursor
        return self._restore_revision(history[cursor])

    def _restore_revision(self, revision: ArtifactRevision) -> Any:
        if revision.artifact_type == "story":
            self.story = self._story_from_data(revision.payload)
            self.stage = Stage.STORY_REVIEW
            self.script = None
            self.storyboard = None
            return self.story
        if revision.artifact_type == "script":
            self.script = self._script_from(revision.payload)
            self.stage = Stage.SCRIPT_REVIEW
            self.storyboard = None
            return self.script
        if revision.artifact_type == "draft":
            self.draft = self._draft_from(revision.payload)
            self.outline, self.script = self.draft.outline, self.draft.script
            self.stage = Stage.SCRIPT_REVIEW
            self.storyboard = None
            return self.draft
        if revision.artifact_type == "storyboard":
            self.storyboard = self._storyboard_from(revision.payload)
            self.stage = Stage.STORYBOARD_REVIEW
            return self.storyboard
        raise ValueError("未知版本类型。")

    def _load_revisions(self, data: dict[str, Any]) -> None:
        for kind, entries in data.get("revisions", {}).items():
            self.revisions[kind] = [ArtifactRevision(**item) for item in entries]
        self.revision_cursor = {str(k): int(v) for k, v in data.get("revision_cursor", {}).items()}

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
            confirmed=bool(data.get("confirmed", False)),
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
        return StoryOutline(**kwargs, beats=beats)

    @staticmethod
    def _script_from(data: dict[str, Any]) -> StoryScript:
        scenes = [StoryScene(**item) for item in data.get("scenes", [])]
        return StoryScript(
            title=str(data.get("title", "")),
            target_seconds=int(data.get("target_seconds", sum(item.duration for item in scenes))),
            scenes=scenes,
            confirmed=bool(data.get("confirmed", False)),
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
                    reference_images=[
                        str(path) for path in item.get("reference_images", [])
                    ],
                )
                for item in bible_data.get("assets", [])
            ],
            continuity_rules=[
                str(item) for item in bible_data.get("continuity_rules", [])
            ],
        )
        return StoryboardPlan(
            title=str(data.get("title", "")),
            target_seconds=int(data.get("target_seconds", 45)),
            shots=[
                StoryboardShot(
                    **{
                        **item,
                        "first_frame_prompt": str(
                            item.get("first_frame_prompt")
                            or item.get("start_frame")
                            or item.get("video_prompt", "")
                        ),
                        "motion_prompt": str(
                            item.get("motion_prompt")
                            or item.get("action")
                            or item.get("video_prompt", "")
                        ),
                        "end_frame_prompt": str(
                            item.get("end_frame_prompt")
                            or item.get("end_frame")
                            or item.get("video_prompt", "")
                        ),
                        "reference_asset_ids": [
                            str(value)
                            for value in item.get("reference_asset_ids", [])
                        ],
                    }
                )
                for item in data.get("shots", [])
            ],
            narration_text=str(data.get("narration_text", "")),
            visual_bible=visual_bible,
            confirmed=bool(data.get("confirmed", False)),
            audio_path=str(data.get("audio_path", "")),
            subtitle_path=str(data.get("subtitle_path", "")),
            artifacts=[VideoArtifact(**item) for item in data.get("artifacts", [])],
        )

    @staticmethod
    def _manifest_from(data: dict[str, Any]) -> RenderManifest:
        return RenderManifest(
            status=str(data.get("status", "")),
            output_dir=str(data.get("output_dir", "")),
            generated_shots=[int(item) for item in data.get("generated_shots", [])],
            reused_shots=[int(item) for item in data.get("reused_shots", [])],
            failed_shots=[int(item) for item in data.get("failed_shots", [])],
            artifacts=[VideoArtifact(**item) for item in data.get("artifacts", [])],
            final_video_path=str(data.get("final_video_path", "")),
            audio_path=str(data.get("audio_path", "")),
            subtitle_path=str(data.get("subtitle_path", "")),
            error=str(data.get("error", "")),
        )
