from __future__ import annotations

import json
import math
import warnings
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
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
    VideoJob,
    VideoArtifact,
    VisualAsset,
    VisualBible,
    VisualReference,
    to_plain_data,
)
from .narration import normalize_narration_timeline
from .quality import (
    evaluate_storyboard_quality,
    review_script_against_story,
    semantic_coverage,
)
from .storyboard import (
    assess_director_plan_timing,
    build_storyboard,
    derive_retake_seed,
    fit_scenes_to_duration,
    refresh_shot_prompts,
    shot_prompts_match_content,
)
from .video_job import build_video_job
from .timing import (
    ShotTimingDemand,
    ShotTimingProfile,
    assess_shot_readable_minimum,
    estimate_story_duration,
    plan_scene_durations,
)
from .video_provider import sanitize_remote_url
from .v2.director import DirectorAgent as V2DirectorAgent, DirectorOrchestrator
from .v2.compiler import VideoJobCompiler
from .v2.film_ir import FilmIR
from .v2.film_ir_builder import FilmIRBuilder
from .v2.execution import (
    CompilationOptions,
    CompileResult,
    ProviderCapabilities as V2ProviderCapabilities,
    VideoJob as V2VideoJob,
)
from .v2.ir import MovieIR
from .v2.ir_builder import MovieIRBuilder
from .v2.creative_passes import creative_pass_pipeline
from .v2.creative_analysis import creative_analysis_pipeline
from .v2.creative_optimizer import creative_optimizer
from .v2.revision_request import RevisionRequestBuilder
from .v2.optimizer import FilmIROptimizer, MovieIROptimizer
from .v2.passes import compiler_pass_pipeline, film_ir_pass_pipeline
from .v2.revision_loop import RuleBasedDirectorRevisionLoop
from .v2.revision_candidate import RevisionCandidate
from .v2.revision_diff import RevisionDiffBuilder
from .v2.revision_guard import RevisionGuard, RevisionGuardPolicy
from .v2.director_revision_adapter import (
    DirectorAgentRevisionAdapter,
    DirectorRevisionContext,
    GuardedRevisionResult,
    RuleBasedDirectorRevisionAdapter,
    run_director_revision_guarded,
)
from .v2.revision_apply import (
    ApplyRevisionCommand,
    RevisionApplyResult,
    RollbackRevisionCommand,
    RevisionRollbackResult,
    apply_revision_to_session,
    rollback_revision_to_session,
)
from .v2.lineage import LineageCheckResult, SourceLineageGuard
from .v2.fingerprint import ensure_movie_plan_provenance, content_fingerprint
from .v2.execution_plan import ExecutionPlan
from .v2.execution_bundle import ExecutionBundle
from .v2.execution_plan_builder import (
    ExecutionPlanCompileResult,
    ExecutionPlanCompiler,
)
from .v2.execution_plan_validation import validate_execution_bundle
from .v2.execution_state import ExecutionRun, ExecutionRunStatus, ExecutionState
from .v2.execution_runtime import ExecutionRuntime, ExecutionRuntimeError
from .v2.fake_provider_runtime import FakeProviderScenario
from .v2.provider_registry import ProviderRuntimeRegistry
from .v2.models import (
    CreativeBrief as V2CreativeBrief,
    MoviePlan as V2MoviePlan,
    as_plain_data as v2_plain_data,
)
from .v2.openai_director import RuleBasedDirectorAgent, movie_plan_from_data
from .v2.validation import (
    FilmIRValidator,
    MovieIRValidator,
    validate_movie_plan,
    validate_video_job,
)


MAX_USER_INPUT_CHARS = 4_000
CURRENT_STAGES = {
    Stage.IDEATING,
    Stage.STORY_REVIEW,
    Stage.SCRIPT_REVIEW,
    Stage.STORYBOARD_REVIEW,
    Stage.MOVIE_PLAN_REVIEW,
    Stage.MOVIE_PLAN_CONFIRMED,
    Stage.MOVIE_PLAN_REVISED,
    Stage.MOVIE_PLAN_ROLLED_BACK,
    Stage.FILM_IR_BUILT,
    Stage.MOVIE_IR_BUILT,
    Stage.VIDEO_JOB_COMPILED,
    Stage.EXECUTION_PLAN_BUILT,
    Stage.EXECUTION_READY,
    Stage.EXECUTION_RUNNING,
    Stage.EXECUTION_BLOCKED,
    Stage.EXECUTION_COMPLETED,
    Stage.EXECUTION_FAILED,
    Stage.RENDER_READY,
    Stage.COMPLETED,
}


def _optimizer_records(result: Any) -> list[dict[str, Any]]:
    """Serialize optimizer diagnostics and future transformation candidates."""

    records = [
        {"kind": "diagnostic", **v2_plain_data(item)}
        for item in result.diagnostics
    ]
    records.extend(
        {"kind": "transformation", **v2_plain_data(item)}
        for item in result.transformations
    )
    return records


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

    schema_version = 7

    def __init__(
        self,
        brief: CreativeBrief | None = None,
        agent: StoryAgent | None = None,
        *,
        director_agent: V2DirectorAgent | None = None,
        v2_enabled: bool = False,
        timing_profile: ShotTimingProfile | None = None,
    ) -> None:
        self._state_lock = RLock()
        self._render_in_progress = False
        self.brief = brief or CreativeBrief()
        self.brief.validate()
        self.agent = agent or RuleBasedStoryAgent()
        self.director_agent = director_agent
        self.v2_enabled = bool(v2_enabled)
        self.timing_profile = timing_profile or ShotTimingProfile()
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
        self.video_job: VideoJob | None = None
        self.movie_plan: V2MoviePlan | None = None
        self.movie_plan_revisions: list[V2MoviePlan] = []
        self.confirmed_movie_plan: V2MoviePlan | None = None
        self.film_ir: FilmIR | None = None
        self.film_ir_revisions: list[FilmIR] = []
        self.film_ir_build_diagnostics: list[dict[str, Any]] = []
        self.film_ir_build_metadata: dict[str, Any] = {}
        self.film_ir_validation_issues: list[dict[str, Any]] = []
        self.film_ir_pass_diagnostics: list[dict[str, Any]] = []
        self.creative_pass_diagnostics: list[dict[str, Any]] = []
        self.creative_analysis_results: list[dict[str, Any]] = []
        self.creative_analysis_diagnostics: list[dict[str, Any]] = []
        self.creative_analysis_artifacts: list[dict[str, Any]] = []
        self.creative_analysis_metrics: dict[str, float] = {}
        self.creative_optimizer_result: dict[str, Any] | None = None
        self.creative_optimizer_suggestions: list[dict[str, Any]] = []
        self.creative_optimizer_candidates: list[dict[str, Any]] = []
        self.creative_optimizer_diagnostics: list[dict[str, Any]] = []
        self.creative_revision_requests: list[dict[str, Any]] = []
        self.creative_revision_request_history: list[dict[str, Any]] = []
        self.creative_revision_stop_reason: str | None = None
        self.revision_candidates: list[dict[str, Any]] = []
        self.revision_diffs: list[dict[str, Any]] = []
        self.revision_decisions: list[dict[str, Any]] = []
        self.revision_guard_diagnostics: list[dict[str, Any]] = []
        self.revision_active_candidate_id: str | None = None
        self.revision_accepted_movie_plan_id: str | None = None
        self.revision_rollback_movie_plan_id: str | None = None
        self.director_revision_adapter_results: list[dict[str, Any]] = []
        self.director_revision_contexts: list[dict[str, Any]] = []
        self.guarded_revision_results: list[dict[str, Any]] = []
        self.director_revision_attempt_count = 0
        self.director_revision_last_stop_reason: str | None = None
        self.movie_plan_version_history: list[dict[str, Any]] = []
        self.revision_apply_history: list[dict[str, Any]] = []
        self.revision_rollback_history: list[dict[str, Any]] = []
        self.revision_apply_results: list[dict[str, Any]] = []
        self.revision_rollback_results: list[dict[str, Any]] = []
        self.current_movie_plan_id: str | None = None
        self.current_movie_plan_version: int | None = None
        self.current_movie_plan_fingerprint: str | None = None
        self.current_movie_plan_lineage_token: str | None = None
        self.previous_movie_plan_id: str | None = None
        self.stale_artifacts: list[dict[str, Any]] = []
        self.source_lineage_diagnostics: list[dict[str, Any]] = []
        self.stale_lineage_diagnostics: list[dict[str, Any]] = []
        self.current_film_ir_id: str | None = None
        self.current_movie_ir_id: str | None = None
        self.current_video_job_id: str | None = None
        self.film_ir_optimizer_diagnostics: list[dict[str, Any]] = []
        self.director_revision_history: list[dict[str, Any]] = []
        self.director_revision_stop_reason: str | None = None
        self.movie_ir: MovieIR | None = None
        self.movie_ir_revisions: list[MovieIR] = []
        self.movie_ir_build_diagnostics: list[dict[str, Any]] = []
        self.movie_ir_build_metadata: dict[str, Any] = {}
        self.movie_ir_validation_issues: list[dict[str, Any]] = []
        self.movie_ir_pass_diagnostics: list[dict[str, Any]] = []
        self.movie_ir_optimizer_diagnostics: list[dict[str, Any]] = []
        self.v2_video_job: V2VideoJob | None = None
        self.v2_compile_diagnostics: list[dict[str, Any]] = []
        self.v2_compile_metadata: dict[str, Any] = {}
        self.execution_plan: ExecutionPlan | None = None
        self.execution_bundle: ExecutionBundle | None = None
        self.current_execution_plan_id: str | None = None
        self.current_execution_plan_fingerprint: str | None = None
        self.current_execution_bundle_fingerprint: str | None = None
        self.execution_plan_diagnostics: list[dict[str, Any]] = []
        self.stale_execution_artifacts: list[dict[str, Any]] = []
        self.execution_run: ExecutionRun | None = None
        self.current_execution_run_id: str | None = None
        self.execution_runtime_status: str | None = None
        self.latest_execution_checkpoint_id: str | None = None
        self.execution_runtime_diagnostics: list[dict[str, Any]] = []
        self.provider_jobs: list[dict[str, Any]] = []
        self.runtime_artifacts: list[dict[str, Any]] = []
        self._execution_runtime: ExecutionRuntime | None = None
        self.render_manifest: RenderManifest | None = None
        self._confirmed_storyboard_signature = ""
        self.legacy_facts = StoryFacts(genre=self.brief.genre)
        self.revisions: dict[str, list[ArtifactRevision]] = {}
        self.revision_cursor: dict[str, int] = {}
        self.user_action_count = 0
        self.free_text_count = 0
        self.text_generation_events: list[dict[str, Any]] = []

    def _clear_execution_state(self) -> None:
        """Clear only the active Phase 4G artifacts; history remains external."""

        if self.execution_run is not None:
            self.stale_execution_artifacts.append(
                {
                    "artifact_type": "execution_run",
                    "artifact_id": self.execution_run.execution_run_id,
                    "reason": "upstream MoviePlan/IR changed; runtime invalidated",
                    "execution_bundle_fingerprint": self.execution_run.execution_bundle_fingerprint,
                }
            )
            if self._execution_runtime is not None:
                try:
                    self._execution_runtime.transition_service.mark_run_stale(
                        self.execution_run,
                        "upstream MoviePlan/IR changed; runtime invalidated",
                    )
                except Exception:
                    # The Session still records the stale audit entry even if
                    # an old external runtime store is unavailable.
                    pass

        self.execution_plan = None
        self.execution_bundle = None
        self.current_execution_plan_id = None
        self.current_execution_plan_fingerprint = None
        self.current_execution_bundle_fingerprint = None
        self.execution_plan_diagnostics = []
        self.execution_run = None
        self.current_execution_run_id = None
        self.execution_runtime_status = None
        self.latest_execution_checkpoint_id = None
        self.execution_runtime_diagnostics = []
        self.provider_jobs = []
        self.runtime_artifacts = []
        self._execution_runtime = None

    def _run_text_operation(
        self,
        operation: str,
        artifact_type: str,
        callback: Any,
    ) -> Any:
        """Run and audit a formal text operation, including failed attempts."""
        try:
            result = callback()
        except Exception as exc:
            reason = str(getattr(self.agent, "last_fallback_reason", "") or exc)
            self._record_text_generation_event(
                operation=operation,
                artifact_type=artifact_type,
                status="failed",
                reason=reason,
            )
            raise
        used_fallback = bool(getattr(self.agent, "last_used_fallback", False))
        is_remote_agent = hasattr(self.agent, "provider_name")
        self._record_text_generation_event(
            operation=operation,
            artifact_type=artifact_type,
            status=(
                "fallback"
                if used_fallback
                else "succeeded"
                if is_remote_agent
                else "offline"
            ),
            reason=str(getattr(self.agent, "last_fallback_reason", "")),
        )
        return result

    def _record_text_generation_event(
        self,
        *,
        operation: str,
        artifact_type: str,
        status: str,
        reason: str = "",
    ) -> None:
        provider = str(getattr(self.agent, "provider_name", "offline"))
        model = str(getattr(self.agent, "model", "rule-based"))
        self.text_generation_events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": operation,
                "artifact_type": artifact_type,
                "status": status,
                "provider": provider,
                "model": model,
                "fallback_kind": str(getattr(self.agent, "last_fallback_kind", "")),
                "reason": reason,
            }
        )

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
        self.video_job = None
        self.v2_video_job = None
        self.v2_compile_diagnostics = []
        self.v2_compile_metadata = {}
        self._clear_execution_state()
        self.film_ir = None
        self.film_ir_revisions = []
        self.film_ir_build_diagnostics = []
        self.film_ir_build_metadata = {}
        self.film_ir_validation_issues = []
        self.film_ir_pass_diagnostics = []
        self.creative_pass_diagnostics = []
        self.creative_analysis_results = []
        self.creative_analysis_diagnostics = []
        self.creative_analysis_artifacts = []
        self.creative_analysis_metrics = {}
        self.creative_optimizer_result = None
        self.creative_optimizer_suggestions = []
        self.creative_optimizer_candidates = []
        self.creative_optimizer_diagnostics = []
        self.creative_revision_requests = []
        self.creative_revision_request_history = []
        self.creative_revision_stop_reason = None
        self._clear_revision_guard_state()
        self.film_ir_optimizer_diagnostics = []
        self.director_revision_history = []
        self.director_revision_stop_reason = None
        self.source_lineage_diagnostics = []
        self.stale_lineage_diagnostics = []
        self.movie_ir = None
        self.movie_ir_revisions = []
        self.movie_ir_build_diagnostics = []
        self.movie_ir_build_metadata = {}
        self.movie_ir_validation_issues = []
        self.movie_ir_pass_diagnostics = []
        self.movie_ir_optimizer_diagnostics = []
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
    def generate_movie_plan(self, direction: str | None = None) -> V2MoviePlan:
        """Generate the V2 MoviePlan through the sole DirectorAgent port.

        This path is intentionally independent from the legacy idea/story/script
        garden.  It records the plan as a revision and does not synthesize any
        Provider or Storyboard fields.
        """
        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2 或 GUIDED_STORY_V2=1。")
        if self.director_agent is None:
            raise RuntimeError("V2 模式缺少 DirectorAgent。")
        cleaned_direction = self._clean_user_input(
            direction if direction is not None else self.direction,
            empty_message="V2 MoviePlan 需要创作方向。",
        )
        brief = self._v2_brief()
        plan = DirectorOrchestrator(self.director_agent, max_attempts=2).create_movie_plan(
            brief,
            cleaned_direction,
        )
        plan = replace(
            plan,
            revision=len(self.movie_plan_revisions) + 1,
            confirmed=False,
        )
        plan = ensure_movie_plan_provenance(plan, version=1)
        self.direction = cleaned_direction
        self.brief.resolved_target_seconds = brief.target_duration_seconds
        self.previous_movie_plan_id = self.current_movie_plan_id
        self.movie_plan = plan
        self.current_movie_plan_id = plan.plan_id
        self.current_movie_plan_version = plan.movie_plan_version
        self.current_movie_plan_fingerprint = plan.movie_plan_fingerprint
        self.current_movie_plan_lineage_token = plan.movie_plan_lineage_token
        self.current_film_ir_id = None
        self.current_movie_ir_id = None
        self.current_video_job_id = None
        self.source_lineage_diagnostics = []
        self.stale_lineage_diagnostics = []
        self.movie_plan_revisions.append(deepcopy(plan))
        self.confirmed_movie_plan = None
        self.v2_video_job = None
        self.v2_compile_diagnostics = []
        self.v2_compile_metadata = {}
        self._clear_execution_state()
        self.film_ir = None
        self.film_ir_revisions = []
        self.film_ir_build_diagnostics = []
        self.film_ir_build_metadata = {}
        self.film_ir_validation_issues = []
        self.film_ir_pass_diagnostics = []
        self.creative_pass_diagnostics = []
        self.creative_analysis_results = []
        self.creative_analysis_diagnostics = []
        self.creative_analysis_artifacts = []
        self.creative_analysis_metrics = {}
        self.creative_optimizer_result = None
        self.creative_optimizer_suggestions = []
        self.creative_optimizer_candidates = []
        self.creative_optimizer_diagnostics = []
        self.creative_revision_requests = []
        self.creative_revision_request_history = []
        self.creative_revision_stop_reason = None
        self._clear_revision_guard_state()
        self.film_ir_optimizer_diagnostics = []
        self.director_revision_history = []
        self.director_revision_stop_reason = None
        self.movie_ir = None
        self.movie_ir_revisions = []
        self.movie_ir_build_diagnostics = []
        self.movie_ir_build_metadata = {}
        self.movie_ir_validation_issues = []
        self.movie_ir_pass_diagnostics = []
        self.movie_ir_optimizer_diagnostics = []
        self.stage = Stage.MOVIE_PLAN_REVIEW
        self.user_action_count += 1
        self._snapshot("movie_plan", v2_plain_data(plan))
        return deepcopy(plan)

    @_state_mutation
    def confirm_movie_plan(self) -> None:
        """Confirm the V2 plan without changing any director-authored field."""
        if not self.v2_enabled or self.movie_plan is None:
            raise RuntimeError("当前没有待确认的 V2 MoviePlan。")
        if self.stage != Stage.MOVIE_PLAN_REVIEW:
            raise RuntimeError("当前阶段不是 V2 MoviePlan 审查阶段。")
        report = validate_movie_plan(self.movie_plan, self._v2_brief())
        if not report.valid:
            raise RuntimeError("MoviePlan 仍有必须修复的问题：" + "；".join(report.errors))
        confirmed = replace(self.movie_plan, confirmed=True)
        self.movie_plan = confirmed
        self.movie_plan_revisions[-1] = deepcopy(confirmed)
        self.confirmed_movie_plan = deepcopy(confirmed)
        self.stage = Stage.MOVIE_PLAN_CONFIRMED
        self._snapshot("movie_plan", v2_plain_data(confirmed), confirmed=True)

    @_state_mutation
    def build_film_ir_from_confirmed_movie_plan(self) -> FilmIR | None:
        """Build the film-language IR from the confirmed MoviePlan."""
        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        if self.confirmed_movie_plan is None:
            raise RuntimeError("必须先确认 MoviePlan，才能构建 FilmIR。")
        if self.stage not in {
            Stage.MOVIE_PLAN_CONFIRMED,
            Stage.MOVIE_PLAN_REVISED,
            Stage.MOVIE_PLAN_ROLLED_BACK,
            Stage.FILM_IR_BUILT,
            Stage.MOVIE_IR_BUILT,
            Stage.VIDEO_JOB_COMPILED,
            Stage.EXECUTION_PLAN_BUILT,
        }:
            raise RuntimeError("当前阶段不是 V2 MoviePlan 已确认阶段。")
        self.creative_pass_diagnostics = []
        self.creative_analysis_results = []
        self.creative_analysis_diagnostics = []
        self.creative_analysis_artifacts = []
        self.creative_analysis_metrics = {}
        self.creative_optimizer_result = None
        self.creative_optimizer_suggestions = []
        self.creative_optimizer_candidates = []
        self.creative_optimizer_diagnostics = []
        self.creative_revision_requests = []
        self.creative_revision_request_history = []
        self.creative_revision_stop_reason = None
        self._clear_revision_guard_state()
        self.film_ir_optimizer_diagnostics = []
        self.director_revision_history = []
        self.director_revision_stop_reason = None
        self.film_ir = None
        self.movie_ir = None
        self.v2_video_job = None
        self.current_film_ir_id = None
        self.current_movie_ir_id = None
        self.current_video_job_id = None
        self._clear_execution_state()
        self.stage = Stage.MOVIE_PLAN_CONFIRMED
        result = FilmIRBuilder().build(deepcopy(self.confirmed_movie_plan))
        self.film_ir_build_diagnostics = [
            {"code": item.code, "message": item.message, "path": item.path}
            for item in result.errors
        ] + [
            {"code": item.code, "message": item.message, "path": item.path}
            for item in result.diagnostics
        ]
        self.film_ir_build_metadata = {
            "source_movie_plan_id": self.confirmed_movie_plan.plan_id,
            "source_movie_plan_version": self.confirmed_movie_plan.movie_plan_version,
            "source_movie_plan_fingerprint": self.confirmed_movie_plan.movie_plan_fingerprint,
            "error_count": len(result.errors),
        }
        if not result.ok or result.film_ir is None:
            return None
        validation = FilmIRValidator().validate(result.film_ir)
        self.film_ir_validation_issues = [
            v2_plain_data(item) for item in validation.issues
        ]
        if not validation.ok:
            self.film_ir_pass_diagnostics = []
            self.creative_pass_diagnostics = []
            self.film_ir_optimizer_diagnostics = []
            self._record_director_revision(
                validation_issues=validation.issues,
            )
            return None
        pass_result = film_ir_pass_pipeline().run(result.film_ir)
        self.film_ir_pass_diagnostics = [
            v2_plain_data(item) for item in pass_result.diagnostics
        ]
        if not pass_result.ok or not isinstance(pass_result.ir, FilmIR):
            self.creative_pass_diagnostics = []
            self.film_ir_optimizer_diagnostics = []
            self._record_director_revision(
                validation_issues=validation.issues,
                creative_diagnostics=pass_result.diagnostics,
            )
            return None
        creative_result = creative_pass_pipeline().run(pass_result.ir)
        self.creative_pass_diagnostics = [
            v2_plain_data(item) for item in creative_result.diagnostics
        ]
        if not creative_result.ok or not isinstance(creative_result.ir, FilmIR):
            self.film_ir_optimizer_diagnostics = []
            self._record_director_revision(
                validation_issues=validation.issues,
                creative_diagnostics=creative_result.diagnostics,
            )
            return None
        optimizer_result = FilmIROptimizer().optimize(creative_result.ir)
        self.film_ir_optimizer_diagnostics = _optimizer_records(optimizer_result)
        self._record_director_revision(
            validation_issues=validation.issues,
            creative_diagnostics=creative_result.diagnostics,
            optimizer_diagnostics=optimizer_result.diagnostics,
        )
        if not optimizer_result.ok or not isinstance(optimizer_result.after_ir, FilmIR):
            return None
        self.film_ir = deepcopy(optimizer_result.after_ir)
        self.current_film_ir_id = self.film_ir.ir_id
        self.film_ir_revisions.append(deepcopy(optimizer_result.after_ir))
        self.movie_ir = None
        self.movie_ir_revisions = []
        self.movie_ir_build_diagnostics = []
        self.movie_ir_build_metadata = {}
        self.movie_ir_validation_issues = []
        self.movie_ir_pass_diagnostics = []
        self.movie_ir_optimizer_diagnostics = []
        self.v2_video_job = None
        self.v2_compile_diagnostics = []
        self.v2_compile_metadata = {}
        self._clear_execution_state()
        self.stage = Stage.FILM_IR_BUILT
        self._snapshot("film_ir", optimizer_result.after_ir.to_dict(), confirmed=True)
        self.refresh_source_lineage_diagnostics()
        return deepcopy(optimizer_result.after_ir)

    def _clear_revision_guard_state(self) -> None:
        """Drop active candidate state when upstream creative inputs change."""

        self.revision_candidates = []
        self.revision_diffs = []
        self.revision_decisions = []
        self.revision_guard_diagnostics = []
        self.revision_active_candidate_id = None
        self.revision_accepted_movie_plan_id = None
        self.revision_rollback_movie_plan_id = None
        self.director_revision_adapter_results = []
        self.director_revision_contexts = []
        self.guarded_revision_results = []
        self.director_revision_attempt_count = 0
        self.director_revision_last_stop_reason = None

    @_state_mutation
    def run_creative_analysis(self) -> tuple[dict[str, Any], ...]:
        """Run read-only StoryPlan/DirectorPlan/FilmIR analysis.

        Analysis is intentionally independent from lowering.  It may run with
        only a MoviePlan, or with the richer FilmIR when ``/build-film-ir`` has
        already completed.  It never changes either input object or the
        production stage.
        """
        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        movie_plan = self.confirmed_movie_plan or self.movie_plan
        if movie_plan is None:
            raise RuntimeError("必须先生成 MoviePlan，才能运行 Creative Analysis。")
        self.creative_optimizer_result = None
        self.creative_optimizer_suggestions = []
        self.creative_optimizer_candidates = []
        self.creative_optimizer_diagnostics = []
        self.creative_revision_requests = []
        self.creative_revision_request_history = []
        self.creative_revision_stop_reason = None
        self._clear_revision_guard_state()
        results = creative_analysis_pipeline().run(
            deepcopy(movie_plan),
            deepcopy(self.film_ir),
        )
        self.creative_analysis_results = [item.to_dict() for item in results]
        self.creative_analysis_diagnostics = [
            diagnostic.to_dict()
            for result in results
            for diagnostic in result.diagnostics
        ]
        self.creative_analysis_artifacts = [
            artifact.to_dict()
            for result in results
            for artifact in result.artifacts
        ]
        self.creative_analysis_metrics = {
            f"{result.analysis_type}.{key}": float(value)
            for result in results
            for key, value in result.metrics.items()
        }
        return tuple(deepcopy(self.creative_analysis_results))

    @_state_mutation
    def run_creative_optimization(self) -> dict[str, Any]:
        """Turn stored creative diagnostics into director-facing requests.

        This operation is read-only with respect to MoviePlan and all IR
        layers.  It only persists analysis-derived optimizer artifacts.
        """
        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        movie_plan = self.confirmed_movie_plan or self.movie_plan
        if movie_plan is None:
            raise RuntimeError("必须先生成 MoviePlan，才能运行 Creative Optimizer。")
        if not self.creative_analysis_results:
            raise RuntimeError("必须先执行 /analysis，才能运行 Creative Optimizer。")
        self._clear_revision_guard_state()
        optimizer_result = creative_optimizer().optimize(
            deepcopy(movie_plan),
            deepcopy(self.creative_analysis_results),
            deepcopy(self.film_ir),
        )
        request_result = RevisionRequestBuilder().build(
            deepcopy(movie_plan),
            optimizer_result,
            validation_issues=tuple(self.film_ir_validation_issues),
            analysis_diagnostics=tuple(self.creative_analysis_diagnostics),
        )
        director_result = RuleBasedDirectorRevisionLoop(max_revisions=1).run(
            deepcopy(movie_plan),
            creative_revision_requests=request_result.requests,
        )
        self.director_revision_history = deepcopy(list(director_result.revision_history))
        self.director_revision_stop_reason = director_result.stop_reason
        self.creative_optimizer_result = deepcopy(optimizer_result.to_dict())
        self.creative_optimizer_suggestions = [
            item.to_dict() for item in optimizer_result.suggestions
        ]
        self.creative_optimizer_candidates = [
            item.to_dict() for item in optimizer_result.transformation_candidates
        ]
        self.creative_optimizer_diagnostics = [
            item.to_dict() for item in optimizer_result.diagnostics
        ]
        self.creative_revision_requests = [
            item.to_dict() for item in request_result.requests
        ]
        self.creative_revision_request_history.append(
            {
                "source_movie_plan_id": request_result.source_movie_plan_id,
                "requests": deepcopy(self.creative_revision_requests),
                "deferred_suggestions": [
                    item.to_dict() for item in request_result.deferred_suggestions
                ],
                "stop_reason": request_result.stop_reason,
            }
        )
        self.creative_revision_stop_reason = request_result.stop_reason
        return {
            "optimizer": deepcopy(self.creative_optimizer_result),
            "revision_requests": deepcopy(self.creative_revision_requests),
            "revision_stop_reason": self.creative_revision_stop_reason,
        }

    @_state_mutation
    def run_revision_guard(
        self,
        candidate: RevisionCandidate | None = None,
        *,
        policy: RevisionGuardPolicy | None = None,
    ) -> dict[str, Any]:
        """Diff and guard a supplied candidate without applying it.

        With no candidate this records a safe ``pending_director`` decision.
        The default CLI path intentionally never creates a fake candidate.
        """
        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        movie_plan = self.confirmed_movie_plan or self.movie_plan
        if movie_plan is None:
            raise RuntimeError("必须先生成 MoviePlan，才能运行 RevisionGuard。")
        if candidate is None and self.revision_active_candidate_id:
            raw_candidate = next(
                (
                    item
                    for item in self.revision_candidates
                    if item.get("candidate_id") == self.revision_active_candidate_id
                ),
                None,
            )
            if raw_candidate is not None:
                raw_plan = raw_candidate.get("revised_movie_plan")
                revised_plan = movie_plan_from_data(raw_plan) if isinstance(raw_plan, dict) else None
                candidate = RevisionCandidate.from_dict(
                    raw_candidate,
                    revised_movie_plan=revised_plan,
                )
        requests = tuple(self.creative_revision_requests)
        diff = RevisionDiffBuilder().build_diff(movie_plan, candidate, requests)
        decision = RevisionGuard(policy=policy).evaluate(
            movie_plan,
            candidate,
            diff,
            validation_issues=tuple(self.film_ir_validation_issues),
            analysis_results=tuple(self.creative_analysis_results),
            requests=requests,
        )
        if candidate is not None:
            candidate_payload = candidate.to_dict()
            existing_index = next(
                (
                    index
                    for index, item in enumerate(self.revision_candidates)
                    if item.get("candidate_id") == candidate.candidate_id
                ),
                None,
            )
            if existing_index is None:
                self.revision_candidates.append(candidate_payload)
            else:
                self.revision_candidates[existing_index] = candidate_payload
            self.revision_active_candidate_id = candidate.candidate_id
        else:
            self.revision_active_candidate_id = None
        self.revision_diffs.append(diff.to_dict())
        self.revision_decisions.append(decision.to_dict())
        self.revision_guard_diagnostics.extend(
            deepcopy(list(decision.diagnostics))
        )
        if decision.decision in {"accept", "accept_with_warning"} and candidate is not None:
            self.revision_accepted_movie_plan_id = (
                candidate.revised_movie_plan.plan_id
                if candidate.revised_movie_plan is not None
                else None
            )
        elif decision.decision == "rollback":
            self.revision_rollback_movie_plan_id = decision.rollback_to_movie_plan_id
        return deepcopy(decision.to_dict())

    @_state_mutation
    def run_director_revision_guarded(
        self,
        adapter: Any | None = None,
        context: DirectorRevisionContext | None = None,
        *,
        policy: RevisionGuardPolicy | None = None,
    ) -> dict[str, Any]:
        """Generate one candidate and run it through validate, Diff, and Guard.

        This method intentionally persists only the proposal and its decision.
        Even an ``accept`` decision never replaces ``movie_plan``; explicit
        apply/rollback belongs to Phase 4D.3.
        """

        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        movie_plan = self.confirmed_movie_plan or self.movie_plan
        if movie_plan is None:
            raise RuntimeError("必须先生成 MoviePlan，才能运行 Director Revision。")
        requests = tuple(self.creative_revision_requests)
        revision_context = context or DirectorRevisionContext.from_requests(
            movie_plan,
            requests,
            max_revision_attempts=1,
        )
        if adapter is None:
            if self.director_agent is None or isinstance(self.director_agent, RuleBasedDirectorAgent):
                revision_adapter = RuleBasedDirectorRevisionAdapter()
            else:
                revision_adapter = DirectorAgentRevisionAdapter(
                    self.director_agent,
                    brief=self._v2_brief(),
                )
        else:
            revision_adapter = adapter
        result: GuardedRevisionResult = run_director_revision_guarded(
            deepcopy(movie_plan),
            requests,
            revision_adapter,
            revision_context,
            brief=self._v2_brief(),
            validation_issues=tuple(self.film_ir_validation_issues),
            analysis_results=tuple(self.creative_analysis_results),
            optimizer_result=self.creative_optimizer_result,
            policy=policy,
        )
        self.director_revision_attempt_count += 1
        self.director_revision_last_stop_reason = result.stop_reason
        self.director_revision_adapter_results.append(result.adapter_result.to_dict())
        self.director_revision_contexts.append(revision_context.to_dict())
        self.guarded_revision_results.append(result.to_dict())
        candidate = result.candidate
        if candidate is not None:
            candidate_payload = candidate.to_dict()
            existing_index = next(
                (
                    index
                    for index, item in enumerate(self.revision_candidates)
                    if item.get("candidate_id") == candidate.candidate_id
                ),
                None,
            )
            if existing_index is None:
                self.revision_candidates.append(candidate_payload)
            else:
                self.revision_candidates[existing_index] = candidate_payload
            self.revision_active_candidate_id = candidate.candidate_id
        else:
            self.revision_active_candidate_id = None
        if result.diff is not None:
            self.revision_diffs.append(result.diff.to_dict())
        if result.decision is not None:
            self.revision_decisions.append(result.decision.to_dict())
            self.revision_guard_diagnostics.extend(
                deepcopy(list(result.decision.diagnostics))
            )
            if result.decision.decision in {"accept", "accept_with_warning"} and candidate is not None:
                self.revision_accepted_movie_plan_id = (
                    candidate.revised_movie_plan.plan_id
                    if candidate.revised_movie_plan is not None
                    else None
                )
            elif result.decision.decision == "rollback":
                self.revision_rollback_movie_plan_id = result.decision.rollback_to_movie_plan_id
        return deepcopy(result.to_dict())

    @_state_mutation
    def apply_revision(self, command: ApplyRevisionCommand) -> RevisionApplyResult:
        """Explicitly apply an accepted candidate; never rebuild downstream IR."""

        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        if not isinstance(command, ApplyRevisionCommand):
            raise TypeError("apply_revision 需要 ApplyRevisionCommand。")
        return apply_revision_to_session(self, command)

    @_state_mutation
    def rollback_revision(self, command: RollbackRevisionCommand) -> RevisionRollbackResult:
        """Explicitly restore a version-history snapshot; never rebuild downstream IR."""

        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        if not isinstance(command, RollbackRevisionCommand):
            raise TypeError("rollback_revision 需要 RollbackRevisionCommand。")
        return rollback_revision_to_session(self, command)

    @_state_mutation
    def build_movie_ir_from_film_ir(self) -> MovieIR | None:
        """Lower an existing FilmIR into the executable MovieIR."""
        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        if self.film_ir is None:
            raise RuntimeError("Build FilmIR first with /build-film-ir.")
        self._require_film_ir_lineage()
        if self.stage not in {
            Stage.FILM_IR_BUILT,
            Stage.MOVIE_IR_BUILT,
            Stage.VIDEO_JOB_COMPILED,
            Stage.EXECUTION_PLAN_BUILT,
        }:
            raise RuntimeError("当前阶段不是 V2 FilmIR 已构建阶段。")
        self.movie_ir_optimizer_diagnostics = []
        self.movie_ir = None
        self.v2_video_job = None
        self.current_movie_ir_id = None
        self.current_video_job_id = None
        self._clear_execution_state()
        self.stage = Stage.FILM_IR_BUILT
        result = MovieIRBuilder().build(deepcopy(self.film_ir))
        self.movie_ir_build_diagnostics = [
            {"code": item.code, "message": item.message, "path": item.path}
            for item in result.errors
        ] + [
            {"code": item.code, "message": item.message, "path": item.path}
            for item in result.diagnostics
        ]
        self.movie_ir_build_metadata = {
            "source_movie_plan_id": self.film_ir.source_movie_plan_id,
            "source_movie_plan_version": self.film_ir.source_movie_plan_version,
            "source_movie_plan_fingerprint": self.film_ir.source_movie_plan_fingerprint,
            "source_film_ir_id": self.film_ir.ir_id,
            "source_film_ir_fingerprint": result.movie_ir.source_film_ir_fingerprint if result.movie_ir else "",
            "error_count": len(result.errors),
        }
        if not result.ok or result.movie_ir is None:
            return None
        validation = MovieIRValidator().validate(result.movie_ir)
        self.movie_ir_validation_issues = [
            v2_plain_data(item) for item in validation.issues
        ]
        if not validation.ok:
            self.movie_ir_pass_diagnostics = []
            self.movie_ir_optimizer_diagnostics = []
            return None
        pass_result = compiler_pass_pipeline().run(result.movie_ir)
        self.movie_ir_pass_diagnostics = [
            v2_plain_data(item) for item in pass_result.diagnostics
        ]
        if not pass_result.ok or not isinstance(pass_result.ir, MovieIR):
            self.movie_ir_optimizer_diagnostics = []
            return None
        optimizer_result = MovieIROptimizer().optimize(pass_result.ir)
        self.movie_ir_optimizer_diagnostics = _optimizer_records(optimizer_result)
        if not optimizer_result.ok or not isinstance(optimizer_result.after_ir, MovieIR):
            return None
        self.movie_ir = deepcopy(optimizer_result.after_ir)
        self.current_movie_ir_id = self.movie_ir.ir_id
        self.movie_ir_revisions.append(deepcopy(optimizer_result.after_ir))
        self.v2_video_job = None
        self.v2_compile_diagnostics = []
        self.v2_compile_metadata = {}
        self._clear_execution_state()
        self.stage = Stage.MOVIE_IR_BUILT
        self._snapshot("movie_ir", optimizer_result.after_ir.to_dict(), confirmed=True)
        self.refresh_source_lineage_diagnostics()
        return deepcopy(optimizer_result.after_ir)

    @_state_mutation
    def build_movie_ir_from_confirmed_movie_plan(self) -> MovieIR | None:
        """Legacy V2 facade that performs both explicit IR lowering steps."""
        if self.film_ir is None:
            if self.build_film_ir_from_confirmed_movie_plan() is None:
                return None
        return self.build_movie_ir_from_film_ir()

    @_state_mutation
    def compile_confirmed_movie_plan(
        self,
        capabilities: V2ProviderCapabilities | None = None,
        options: CompilationOptions | None = None,
    ) -> CompileResult:
        """Compile only the confirmed V2 plan; never invoke a Provider."""
        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        if self.confirmed_movie_plan is None:
            raise RuntimeError("必须先确认 MoviePlan，才能编译 VideoJob。")
        if self.film_ir is None:
            raise RuntimeError("必须先执行 /build-film-ir，才能编译 VideoJob。")
        if self.movie_ir is None:
            raise RuntimeError("必须先执行 /build-ir，才能编译 VideoJob。")
        self._require_movie_ir_lineage()
        if self.stage not in {
            Stage.MOVIE_IR_BUILT,
            Stage.VIDEO_JOB_COMPILED,
        }:
            raise RuntimeError("当前阶段不是 V2 MoviePlan 已确认阶段。")
        provider_capabilities = capabilities or V2ProviderCapabilities(
            provider_key="generic-v2",
            provider_profile="generic",
            supports_long_video=True,
            supports_multi_scene_prompt=True,
            supports_audio=True,
        )
        result = VideoJobCompiler().compile(
            deepcopy(self.movie_ir),
            provider_capabilities,
            options,
        )
        self.v2_compile_diagnostics = [
            {"kind": "error", **v2_plain_data(item)} for item in result.errors
        ] + [
            {"kind": "warning", **v2_plain_data(item)} for item in result.warnings
        ]
        self.v2_compile_metadata = deepcopy(dict(result.metadata))
        if not result.success or result.video_job is None:
            return result
        self.v2_video_job = deepcopy(result.video_job)
        self.current_video_job_id = self.v2_video_job.job_id
        self.stage = Stage.VIDEO_JOB_COMPILED
        self._snapshot("v2_video_job", v2_plain_data(result.video_job), confirmed=True)
        self.refresh_source_lineage_diagnostics()
        return result

    @_state_mutation
    def build_execution_plan(
        self,
        capabilities: V2ProviderCapabilities | None = None,
        options: CompilationOptions | None = None,
    ) -> ExecutionPlanCompileResult:
        """Lower MovieIR into an immutable ExecutionBundle only.

        This method never submits, polls, downloads, creates ProviderJob, or
        generates an artifact.  The historical ``/compile`` path remains a
        separate VideoJob compatibility entrypoint.
        """

        if not self.v2_enabled:
            raise RuntimeError("V2 DirectorAgent 未启用，请使用 --v2。")
        if self.confirmed_movie_plan is None:
            raise RuntimeError("必须先确认 MoviePlan，才能构建 ExecutionPlan。")
        if self.film_ir is None:
            raise RuntimeError("必须先执行 /build-film-ir，才能构建 ExecutionPlan。")
        if self.movie_ir is None:
            raise RuntimeError("必须先执行 /build-ir，才能构建 ExecutionPlan。")
        self._require_movie_ir_lineage()
        if self.stage not in {Stage.MOVIE_IR_BUILT, Stage.EXECUTION_PLAN_BUILT}:
            raise RuntimeError("当前阶段不是 V2 MovieIR 已构建阶段。")
        provider_capabilities = capabilities or V2ProviderCapabilities(
            provider_key="offline-v2",
            provider_profile="offline",
            supports_long_video=True,
            supports_multi_scene_prompt=True,
            supports_reference_images=True,
            supports_character_reference=True,
            supports_audio=True,
        )
        result = ExecutionPlanCompiler().compile(
            deepcopy(self.movie_ir),
            provider_capabilities,
            options,
        )
        self.execution_plan_diagnostics = [
            {"kind": "error", "code": item.code, "message": item.message, "path": item.path}
            for item in result.errors
        ] + [
            {"kind": "warning", "code": item.code, "message": item.message, "path": item.path}
            for item in result.warnings
        ]
        self.execution_plan = None
        self.execution_bundle = None
        self.current_execution_plan_id = None
        self.current_execution_plan_fingerprint = None
        self.current_execution_bundle_fingerprint = None
        if not result.success or result.bundle is None:
            self.refresh_source_lineage_diagnostics()
            return result
        self.execution_plan = result.bundle.execution_plan
        self.execution_bundle = result.bundle
        self.current_execution_plan_id = result.bundle.execution_plan.execution_plan_id
        self.current_execution_plan_fingerprint = result.bundle.execution_plan.execution_plan_fingerprint
        self.current_execution_bundle_fingerprint = result.bundle.bundle_fingerprint
        self.stage = Stage.EXECUTION_PLAN_BUILT
        self._snapshot("execution_bundle", result.bundle.to_dict(), confirmed=True)
        self.refresh_source_lineage_diagnostics()
        return result

    def validate_current_execution_plan(self):
        """Validate the active bundle without rebuilding or running it."""

        return validate_execution_bundle(self.execution_bundle)

    def _execution_runtime_for(
        self,
        *,
        runtime: ExecutionRuntime | None = None,
        runtime_root: str | Path | None = None,
        provider_registry: ProviderRuntimeRegistry | None = None,
        scenario: FakeProviderScenario | str | None = None,
    ) -> ExecutionRuntime:
        if runtime is not None:
            self._execution_runtime = runtime
            return runtime
        if self.execution_bundle is None:
            raise RuntimeError("必须先执行 /build-execution-plan，才能启动 ExecutionRuntime。")
        if self._execution_runtime is not None:
            if provider_registry is not None:
                self._execution_runtime.provider_registry = provider_registry
            return self._execution_runtime
        keys = tuple(
            assignment.provider_key
            for assignment in self.execution_bundle.execution_plan.provider_assignments
        )
        registry = provider_registry or ProviderRuntimeRegistry.with_fake(
            provider_keys=keys or ("fake",),
            scenario=scenario,
        )
        self._execution_runtime = ExecutionRuntime(
            self.execution_bundle,
            provider_registry=registry,
            artifact_root=runtime_root or Path(".guided-story-runtime"),
        )
        # A loaded Session may contain a summary while the independent JSON
        # store is unavailable.  Seed only that exact run; never rebuild it.
        if self.execution_run is not None:
            try:
                self._execution_runtime.state_store.load_run(self.execution_run.execution_run_id)
            except FileNotFoundError:
                self._execution_runtime.state_store.create_run(self.execution_run)
        return self._execution_runtime

    def _sync_execution_runtime(self, run: ExecutionRun) -> ExecutionRun:
        self.execution_run = run
        self.current_execution_run_id = run.execution_run_id
        self.execution_runtime_status = run.status.value
        self.latest_execution_checkpoint_id = run.latest_checkpoint_id
        self.provider_jobs = [job.to_dict() for job in run.provider_jobs.values()]
        self.runtime_artifacts = [dict(item) for item in run.artifacts.values()]
        self.execution_runtime_diagnostics = [
            {"code": "runtime_diagnostic", "message": item} for item in run.diagnostics
        ]
        if self._execution_runtime is not None:
            self.execution_runtime_diagnostics.extend(self._execution_runtime.capability_diagnostics)
        if any(state.state is ExecutionState.SUBMISSION_UNCERTAIN for state in run.unit_states.values()):
            self.execution_runtime_diagnostics.append(
                {
                    "code": "submission_uncertain",
                    "message": "Provider 是否受理无法确认；禁止自动重提，需要人工对账。",
                }
            )
        if run.status == ExecutionRunStatus.COMPLETED:
            self.stage = Stage.EXECUTION_COMPLETED
        elif run.status == ExecutionRunStatus.BLOCKED:
            self.stage = Stage.EXECUTION_BLOCKED
        elif run.status == ExecutionRunStatus.FAILED:
            self.stage = Stage.EXECUTION_FAILED
        elif run.status == ExecutionRunStatus.RUNNING:
            self.stage = Stage.EXECUTION_RUNNING
        elif run.status == ExecutionRunStatus.CANCELLED:
            self.stage = Stage.EXECUTION_BLOCKED
        else:
            self.stage = Stage.EXECUTION_READY
        return run

    @_state_mutation
    def start_execution(
        self,
        *,
        runtime: ExecutionRuntime | None = None,
        runtime_root: str | Path | None = None,
        provider_registry: ProviderRuntimeRegistry | None = None,
        scenario: FakeProviderScenario | str | None = None,
    ) -> ExecutionRun:
        """Create a durable offline run without submitting any Provider job."""

        validation = self.validate_current_execution_plan()
        if not validation.valid or self.execution_bundle is None:
            raise RuntimeError("ExecutionBundle 无效；请先重新执行 /build-execution-plan。")
        self.refresh_source_lineage_diagnostics()
        execution_lineage = SourceLineageGuard().check_execution_bundle(
            self.execution_bundle,
            current_movie_plan_id=self.current_movie_plan_id or "",
            current_movie_plan_version=self.current_movie_plan_version,
            current_movie_plan_fingerprint=self.current_movie_plan_fingerprint,
            current_movie_plan_lineage_token=self.current_movie_plan_lineage_token,
            current_film_ir_id=self.current_film_ir_id or "",
            current_film_ir_fingerprint=content_fingerprint(self.film_ir.to_dict()) if self.film_ir else None,
            current_movie_ir_id=self.current_movie_ir_id or "",
            current_movie_ir_fingerprint=content_fingerprint(self.movie_ir.to_dict()) if self.movie_ir else None,
        )
        if not execution_lineage.valid:
            raise RuntimeError("ExecutionBundle stale；请先重新执行 /build-execution-plan。")
        runner = self._execution_runtime_for(
            runtime=runtime,
            runtime_root=runtime_root,
            provider_registry=provider_registry,
            scenario=scenario,
        )
        try:
            run = runner.create_run(self.execution_bundle)
        except ExecutionRuntimeError as exc:
            self.execution_runtime_diagnostics = list(runner.capability_diagnostics) or [
                {"code": "execution_runtime_start_failed", "message": str(exc)}
            ]
            raise
        return self._sync_execution_runtime(run)

    @_state_mutation
    def step_execution(
        self,
        *,
        runtime: ExecutionRuntime | None = None,
        runtime_root: str | Path | None = None,
        provider_registry: ProviderRuntimeRegistry | None = None,
        scenario: FakeProviderScenario | str | None = None,
    ) -> ExecutionRun:
        if self.current_execution_run_id is None:
            raise RuntimeError("当前没有 ExecutionRun；请先执行 /start-execution。")
        runner = self._execution_runtime_for(
            runtime=runtime,
            runtime_root=runtime_root,
            provider_registry=provider_registry,
            scenario=scenario,
        )
        run = runner.step(self.current_execution_run_id)
        return self._sync_execution_runtime(run)

    @_state_mutation
    def run_execution(
        self,
        *,
        max_steps: int = 100,
        runtime: ExecutionRuntime | None = None,
        runtime_root: str | Path | None = None,
        provider_registry: ProviderRuntimeRegistry | None = None,
        scenario: FakeProviderScenario | str | None = None,
    ) -> ExecutionRun:
        if self.current_execution_run_id is None:
            raise RuntimeError("当前没有 ExecutionRun；请先执行 /start-execution。")
        runner = self._execution_runtime_for(
            runtime=runtime,
            runtime_root=runtime_root,
            provider_registry=provider_registry,
            scenario=scenario,
        )
        run = runner.run_until_blocked_or_complete(self.current_execution_run_id, max_steps=max_steps)
        return self._sync_execution_runtime(run)

    @_state_mutation
    def resume_execution(
        self,
        *,
        runtime: ExecutionRuntime | None = None,
        runtime_root: str | Path | None = None,
        provider_registry: ProviderRuntimeRegistry | None = None,
        scenario: FakeProviderScenario | str | None = None,
    ) -> ExecutionRun:
        if self.current_execution_run_id is None:
            raise RuntimeError("当前没有 ExecutionRun；请先执行 /start-execution。")
        runner = self._execution_runtime_for(
            runtime=runtime,
            runtime_root=runtime_root,
            provider_registry=provider_registry,
            scenario=scenario,
        )
        run = runner.resume(self.current_execution_run_id)
        return self._sync_execution_runtime(run)

    @_state_mutation
    def cancel_execution(
        self,
        *,
        unit_id: str | None = None,
        runtime: ExecutionRuntime | None = None,
        runtime_root: str | Path | None = None,
        provider_registry: ProviderRuntimeRegistry | None = None,
        scenario: FakeProviderScenario | str | None = None,
    ) -> ExecutionRun:
        if self.current_execution_run_id is None:
            raise RuntimeError("当前没有可取消的 ExecutionRun。")
        runner = self._execution_runtime_for(
            runtime=runtime,
            runtime_root=runtime_root,
            provider_registry=provider_registry,
            scenario=scenario,
        )
        if unit_id:
            run = runner.cancel_unit(self.current_execution_run_id, unit_id)
        else:
            run = runner.cancel(self.current_execution_run_id)
        return self._sync_execution_runtime(run)

    def execution_status(self, *, runtime_root: str | Path | None = None) -> dict[str, Any]:
        if self.current_execution_run_id is None:
            return {
                "execution_run_id": None,
                "status": None,
                "execution_bundle_fingerprint": self.current_execution_bundle_fingerprint,
            }
        runner = self._execution_runtime
        if runner is None and runtime_root is not None:
            runner = self._execution_runtime_for(runtime_root=runtime_root)
        if runner is not None:
            return runner.inspect(self.current_execution_run_id)
        run = self.execution_run
        counts: dict[str, int] = {}
        if run is not None:
            for state in run.unit_states.values():
                counts[state.state.value] = counts.get(state.state.value, 0) + 1
        return {
            "execution_run_id": self.current_execution_run_id,
            "execution_bundle_fingerprint": self.current_execution_bundle_fingerprint,
            "status": self.execution_runtime_status,
            "unit_state_counts": counts,
            "ready_units": [],
            "running_units": [],
            "blocked_units": [],
            "failed_units": [],
            "submission_uncertain_units": [],
            "latest_checkpoint_id": self.latest_execution_checkpoint_id,
            "provider_submit_counts": {},
            "provider_poll_counts": {},
            "artifact_count": len(self.runtime_artifacts),
            "diagnostics": list(self.execution_runtime_diagnostics),
        }

    def execution_events(
        self,
        *,
        limit: int | None = None,
        runtime_root: str | Path | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        if self.current_execution_run_id is None:
            return ()
        runner = self._execution_runtime
        if runner is None and runtime_root is not None:
            runner = self._execution_runtime_for(runtime_root=runtime_root)
        return runner.events(self.current_execution_run_id, limit=limit) if runner is not None else ()

    def refresh_source_lineage_diagnostics(self) -> LineageCheckResult:
        """Recompute persisted source/stale diagnostics without rebuilding anything."""

        result = SourceLineageGuard().check_session(self)
        records = [item.to_dict() for item in result.diagnostics]
        if self.execution_bundle is not None:
            plan_id, _, _ = self._lineage_ids()
            execution_result = SourceLineageGuard().check_execution_bundle(
                self.execution_bundle,
                current_movie_plan_id=plan_id or "",
                current_movie_plan_version=self.current_movie_plan_version,
                current_movie_plan_fingerprint=self.current_movie_plan_fingerprint,
                current_movie_plan_lineage_token=self.current_movie_plan_lineage_token,
                current_film_ir_id=self.current_film_ir_id or "",
                current_film_ir_fingerprint=(
                    content_fingerprint(self.film_ir.to_dict()) if self.film_ir is not None else None
                ),
                current_movie_ir_id=self.current_movie_ir_id or "",
                current_movie_ir_fingerprint=(
                    content_fingerprint(self.movie_ir.to_dict()) if self.movie_ir is not None else None
                ),
            )
            records.extend(item.to_dict() for item in execution_result.diagnostics)
        self.source_lineage_diagnostics = records
        self.stale_lineage_diagnostics = [
            item for item in records if item.get("severity") in {"error", "warning"}
        ]
        return result

    def _lineage_ids(self) -> tuple[str | None, str | None, str | None]:
        plan = self.confirmed_movie_plan or self.movie_plan
        plan_id = self.current_movie_plan_id or (plan.plan_id if plan else None)
        story_id = (
            f"{plan_id}:story_plan"
            if plan_id and plan is not None and plan.story_plan is not None
            else None
        )
        director_id = (
            f"{plan_id}:director_plan"
            if plan_id and plan is not None and plan.director_plan is not None
            else None
        )
        return plan_id, story_id, director_id

    def _store_lineage_result(self, results: tuple[LineageCheckResult, ...]) -> None:
        diagnostics = [
            item.to_dict()
            for result in results
            for item in result.diagnostics
        ]
        self.source_lineage_diagnostics = diagnostics
        self.stale_lineage_diagnostics = [
            item for item in diagnostics if item.get("severity") in {"error", "warning"}
        ]

    @staticmethod
    def _lineage_failure_message(results: tuple[LineageCheckResult, ...]) -> str:
        messages = []
        for result in results:
            for item in result.diagnostics:
                action = f"；请先执行 {item.action}" if item.action else ""
                messages.append(f"{item.message}{action}")
        return "；".join(messages) or "来源 lineage 校验失败。"

    def _require_film_ir_lineage(self) -> None:
        plan_id, story_id, director_id = self._lineage_ids()
        result = SourceLineageGuard().check_film_ir(
            self.film_ir,
            current_movie_plan_id=plan_id or "",
            current_story_plan_id=story_id or "",
            current_director_plan_id=director_id or "",
            current_film_ir_id=self.current_film_ir_id or "",
            current_movie_plan_version=self.current_movie_plan_version,
            current_movie_plan_fingerprint=self.current_movie_plan_fingerprint,
            current_movie_plan_lineage_token=self.current_movie_plan_lineage_token,
        )
        self._store_lineage_result((result,))
        if not result.valid:
            raise RuntimeError(self._lineage_failure_message((result,)))

    def _require_movie_ir_lineage(self) -> None:
        plan_id, story_id, director_id = self._lineage_ids()
        guard = SourceLineageGuard()
        film_result = guard.check_film_ir(
            self.film_ir,
            current_movie_plan_id=plan_id or "",
            current_story_plan_id=story_id or "",
            current_director_plan_id=director_id or "",
            current_film_ir_id=self.current_film_ir_id or "",
            current_movie_plan_version=self.current_movie_plan_version,
            current_movie_plan_fingerprint=self.current_movie_plan_fingerprint,
            current_movie_plan_lineage_token=self.current_movie_plan_lineage_token,
        )
        movie_result = guard.check_movie_ir(
            self.movie_ir,
            current_movie_plan_id=plan_id or "",
            current_film_ir_id=self.current_film_ir_id or "",
            current_movie_ir_id=self.current_movie_ir_id or "",
            current_movie_plan_version=self.current_movie_plan_version,
            current_movie_plan_fingerprint=self.current_movie_plan_fingerprint,
            current_movie_plan_lineage_token=self.current_movie_plan_lineage_token,
            current_film_ir_fingerprint=content_fingerprint(self.film_ir.to_dict()) if self.film_ir else None,
        )
        results = (film_result, movie_result)
        self._store_lineage_result(results)
        if not all(item.valid for item in results):
            raise RuntimeError(self._lineage_failure_message(results))

    def _require_video_job_lineage(self) -> None:
        plan_id, _, _ = self._lineage_ids()
        result = SourceLineageGuard().check_video_job(
            self.v2_video_job,
            current_movie_plan_id=plan_id or "",
            current_film_ir_id=self.current_film_ir_id or "",
            current_movie_ir_id=self.current_movie_ir_id or "",
            current_video_job_id=self.current_video_job_id or "",
            current_movie_plan_version=self.current_movie_plan_version,
            current_movie_plan_fingerprint=self.current_movie_plan_fingerprint,
            current_movie_plan_lineage_token=self.current_movie_plan_lineage_token,
            current_film_ir_fingerprint=content_fingerprint(self.film_ir.to_dict()) if self.film_ir else None,
            current_movie_ir_fingerprint=content_fingerprint(self.movie_ir.to_dict()) if self.movie_ir else None,
        )
        self._store_lineage_result((result,))
        if not result.valid:
            raise RuntimeError(self._lineage_failure_message((result,)))

    def _v2_brief(self) -> V2CreativeBrief:
        target = self.brief.resolved_target_seconds or self.brief.target_seconds
        if target is None:
            raise RuntimeError("V2 模式必须通过 --target-seconds 指定目标时长。")
        return V2CreativeBrief(
            target_duration_seconds=int(target),
            video_type=self.brief.genre or "short_film",
            visual_style=self.brief.visual_style or "cinematic",
            audience="general audience",
            narration_requirement="required" if self.brief.narration_enabled else "none",
            output_format="mp4",
        )

    def _record_director_revision(
        self,
        *,
        validation_issues: tuple[Any, ...] = (),
        creative_diagnostics: tuple[Any, ...] = (),
        optimizer_diagnostics: tuple[Any, ...] = (),
    ) -> None:
        """Persist the offline revision-loop decision without changing content."""

        if self.confirmed_movie_plan is None:
            self.director_revision_history = []
            self.director_revision_stop_reason = "missing_confirmed_movie_plan"
            return
        result = RuleBasedDirectorRevisionLoop(max_revisions=1).run(
            self.confirmed_movie_plan,
            validation_issues=validation_issues,
            creative_diagnostics=creative_diagnostics,
            optimizer_diagnostics=optimizer_diagnostics,
        )
        self.director_revision_history = deepcopy(list(result.revision_history))
        self.director_revision_stop_reason = result.stop_reason

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
            self._run_text_operation(
                "generate_story",
                "story",
                lambda: self.agent.generate_story(
                self.direction,
                deepcopy(self.selected_cards),
                deepcopy(selected_options),
                ),
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
        self.video_job = None
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
        revised = deepcopy(
            self._run_text_operation(
                "revise_story",
                "story",
                lambda: self.agent.revise_story(deepcopy(self.story), cleaned_feedback),
            )
        )
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
        self.video_job = None
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
        script = deepcopy(
            self._run_text_operation(
                "generate_script",
                "script",
                lambda: self.agent.generate_script(
                    deepcopy(self.story),
                    target_seconds,
                    timing_profile=self.timing_profile,
                ),
            )
        )
        self._validate_or_repair_script(script, target_seconds=target_seconds)
        script.confirmed = False
        self.brief.resolved_target_seconds = target_seconds
        self.brief.validate()
        self.script = script
        self.draft = None
        self.outline = None
        self.storyboard = None
        self.video_job = None
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
            self._run_text_operation(
                "revise_script",
                "script",
                lambda: self.agent.revise_script(
                    deepcopy(self.story),
                    deepcopy(self.script),
                    cleaned_feedback,
                    timing_profile=self.timing_profile,
                ),
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
        self.video_job = None
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
    def build_video_job(self) -> VideoJob:
        """Create the direct script-to-video request.

        This is the default path for the current product stage.  The old
        ``build_storyboard`` method remains available as a compatibility API,
        but it is no longer required to reach rendering.
        """
        if self.script is None or not self.script.confirmed:
            raise RuntimeError("请先确认剧本。")
        if self.stage not in {Stage.SCRIPT_REVIEW, Stage.STORYBOARD_REVIEW, Stage.RENDER_READY}:
            raise RuntimeError("当前阶段不能重新生成视频任务。")
        job = build_video_job(
            deepcopy(self.script),
            story=deepcopy(self.story),
            facts=deepcopy(self._story_facts()),
            visual_style=self.brief.visual_style,
        )
        self.video_job = job
        self.stage = Stage.RENDER_READY
        self.render_manifest = None
        self._snapshot("video_job", to_plain_data(job))
        return deepcopy(job)

    @_state_mutation
    def build_storyboard(self) -> StoryboardPlan:
        if self.script is None or not self.script.confirmed or self.stage != Stage.SCRIPT_REVIEW:
            raise RuntimeError("请先确认剧本。")
        facts = self._story_facts()
        planner = getattr(self.agent, "plan_storyboard", None)
        director_plan = self._run_text_operation(
            "plan_storyboard",
            "storyboard",
            lambda: (
                planner(
                    deepcopy(self.script),
                    deepcopy(facts),
                    timing_profile=self.timing_profile,
                )
                if callable(planner)
                else None
            ),
        )
        storyboard = build_storyboard(
            self.script,
            facts,
            director_plan=director_plan,
            timing_profile=self.timing_profile,
        )
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

        # Preserve the caller's lexical path for stable Windows temp-directory
        # containment checks. Security-sensitive callers validate canonical
        # paths separately before handing them to this state model.
        candidate = Path(path).expanduser().absolute()
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
        computed_timing_fields = {
            "dialogue",
            "duration_reason",
            "duration_weight",
            "end_frame_prompt",
            "estimated_duration",
            "first_frame_prompt",
            "minimum_readable_duration",
            "motion_prompt",
            "narration",
            "negative_prompt",
            "source_action",
            "video_prompt",
        }
        allowed = (
            set(StoryboardShot.__dataclass_fields__)
            - {"shot_id", "scene_id"}
            - computed_timing_fields
        )
        unsupported = [field for field in patch if field not in allowed]
        if unsupported:
            raise ValueError(f"不支持的镜头字段：{unsupported[0]}")

        candidate_plan = deepcopy(self.storyboard)
        candidate = candidate_plan.shots[shot_index]
        for field, value in patch.items():
            setattr(candidate, field, deepcopy(value))
        if "action" in patch and candidate.source_action:
            if candidate.action.strip() != candidate.source_action.strip():
                raise ValueError(
                    "已确认剧情动作不能在 Retake 中改写；请回到剧本阶段修改事件内容。"
                )
        if {"action", "shot_kind", "shot_purpose"}.intersection(patch):
            same_scene = [
                shot
                for shot in candidate_plan.shots
                if shot.scene_id == candidate.scene_id
            ]
            retake_floor = assess_shot_readable_minimum(
                ShotTimingDemand(
                    shot_kind=candidate.shot_kind,
                    purpose=candidate.shot_purpose,
                    priority=6 if candidate.shot_kind == "action" else 4,
                    action=(
                        f"{candidate.action}，{candidate.retake_instruction}"
                        if candidate.retake_instruction.strip()
                        else candidate.action
                    ),
                    dialogue="",
                    narration=candidate.narration,
                    emotional_change="",
                    scene_duration=sum(shot.duration for shot in same_scene),
                    scene_shot_count=max(1, len(same_scene)),
                    narration_is_per_shot=True,
                )
            )
            candidate.minimum_readable_duration = max(
                candidate.minimum_readable_duration,
                retake_floor.minimum_seconds,
            )
            candidate.duration_reason = (
                f"{candidate.duration_reason}；Retake后复核：{retake_floor.reason}"
            ).strip("；")
        if "seed" not in patch:
            candidate.seed = derive_retake_seed(
                candidate.seed,
                candidate.shot_id,
                patch,
            )
        refresh_shot_prompts(candidate)
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

    def resolve_submission_uncertainty(
        self,
        shot_id: int,
        *,
        accepted_by_provider: bool,
        provider_request_id: str = "",
    ) -> VideoArtifact:
        """Resolve one write-ahead submission intent after checking the provider console."""
        with self._state_lock:
            self._require_not_rendering()
            if self.storyboard is None:
                raise RuntimeError("当前没有可核对的视频任务。")
            candidates = [
                artifact
                for artifact in self.storyboard.artifacts
                if artifact.shot_id == shot_id
                and artifact.status == "submission_uncertain"
            ]
            if not candidates:
                raise RuntimeError(f"镜头 {shot_id} 没有等待人工核对的提交记录。")
            artifact = candidates[-1]
            cleaned_request_id = " ".join(provider_request_id.split())
            if accepted_by_provider:
                if not cleaned_request_id:
                    raise ValueError("Provider 已受理时，必须填写后台显示的真实任务 ID。")
                if cleaned_request_id.startswith("submit-intent-"):
                    raise ValueError("请填写 Provider 后台任务 ID，不能使用本地提交意图 ID。")
                artifact.status = "pending"
                artifact.request_id = cleaned_request_id
                artifact.error_message = (
                    "已由用户在 Provider 后台确认受理；下次生成只会查询该任务，"
                    "不会重新提交。"
                )
            else:
                artifact.status = "failed"
                artifact.request_id = None
                artifact.error_message = (
                    "已由用户在 Provider 后台确认未受理；允许下次生成重新提交该镜头。"
                )
            if self.render_manifest is not None:
                for index, manifest_artifact in enumerate(self.render_manifest.artifacts):
                    if manifest_artifact.artifact_id == artifact.artifact_id:
                        self.render_manifest.artifacts[index] = deepcopy(artifact)
                if accepted_by_provider:
                    self.render_manifest.status = "pending"
                    self.render_manifest.error = artifact.error_message
                else:
                    self.render_manifest.status = "failed"
                    if shot_id not in self.render_manifest.failed_shots:
                        self.render_manifest.failed_shots.append(shot_id)
                    self.render_manifest.error = artifact.error_message
            return deepcopy(artifact)

    @_state_mutation
    def resolve_video_submission_uncertainty(
        self,
        *,
        accepted_by_provider: bool,
        provider_request_id: str = "",
    ) -> VideoArtifact:
        """Resolve the single whole-video write-ahead intent after manual checks."""
        if self.video_job is None or self.render_manifest is None:
            raise RuntimeError("当前没有可核对的完整视频任务。")
        candidates = [
            artifact
            for artifact in self.render_manifest.artifacts
            if artifact.status == "submission_uncertain"
        ]
        if not candidates:
            raise RuntimeError("当前没有等待人工核对的完整视频提交记录。")
        artifact = candidates[-1]
        cleaned_request_id = " ".join(provider_request_id.split())
        if accepted_by_provider:
            if not cleaned_request_id or cleaned_request_id.startswith("submit-intent-"):
                raise ValueError("请填写 Provider 后台核实后的真实任务 ID。")
            artifact.status = "pending"
            artifact.request_id = cleaned_request_id
            artifact.error_message = (
                "已登记 Provider 真实任务 ID；下次生成只会查询该任务，不会重新提交。"
            )
            self.render_manifest.status = "pending"
        else:
            artifact.status = "failed"
            artifact.request_id = None
            artifact.error_message = "已确认 Provider 未受理；下次允许重新提交完整视频任务。"
            self.render_manifest.status = "failed"
            if 1 not in self.render_manifest.failed_shots:
                self.render_manifest.failed_shots.append(1)
        self.render_manifest.error = artifact.error_message
        self.render_manifest.artifacts[-1] = deepcopy(artifact)
        return deepcopy(artifact)

    def render_confirmed_video(self, renderer, output_dir: str | Path) -> RenderManifest:
        """Render the confirmed whole-video job in one provider call.

        Provider-specific duration limits and optional internal chunking are
        intentionally handled by the adapter behind ``renderer``.
        """
        with self._state_lock:
            if self._render_in_progress:
                raise RuntimeError("视频生成正在进行，请等待当前任务结束。")
            if self.stage != Stage.RENDER_READY or self.video_job is None:
                raise RuntimeError("必须先生成并确认完整视频任务，才能调用视频生成。")
            if not self.video_job.confirmed:
                raise RuntimeError("完整视频任务尚未确认，禁止调用视频生成。")
            if (
                self.story is None
                or not self.story.confirmed
                or self.script is None
                or not self.script.confirmed
            ):
                raise RuntimeError("故事与剧本确认状态不一致，禁止视频生成。")
            confirmed_job = deepcopy(self.video_job)
            previous_stage = self.stage
            previous_manifest = deepcopy(self.render_manifest)
            self._render_in_progress = True

        try:
            render_target = output_dir
            if (
                self.render_manifest is not None
                and self.render_manifest.status in {"pending", "submission_uncertain"}
                and self.render_manifest.output_dir
            ):
                # Reuse the original write-ahead directory so a retry can
                # resume/query the same remote task instead of submitting a
                # second whole-video request.
                render_target = self.render_manifest.output_dir
            manifest = renderer.render(confirmed_job, render_target)
            if not isinstance(manifest, RenderManifest):
                raise TypeError("渲染器必须返回 RenderManifest。")
            if manifest.status in {"succeeded", "succeeded_with_warnings"}:
                final = Path(manifest.final_video_path).expanduser()
                if not manifest.final_video_path or not final.is_file():
                    raise RuntimeError("渲染器报告成功，但没有可用的完整视频文件。")
            with self._state_lock:
                if self.video_job is None or self.stage != Stage.RENDER_READY:
                    raise RuntimeError("渲染期间完整视频任务发生变化，结果未写入当前会话。")
                self.render_manifest = deepcopy(manifest)
                if manifest.status in {"succeeded", "succeeded_with_warnings"}:
                    self.stage = Stage.COMPLETED
                return manifest
        except Exception:
            with self._state_lock:
                self.stage = previous_stage
                self.render_manifest = previous_manifest
            raise
        finally:
            with self._state_lock:
                self._render_in_progress = False

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
            "v2_video_job"
            if self.v2_video_job
            else "movie_ir"
            if self.movie_ir
            else "film_ir"
            if self.film_ir
            else "video_job"
            if self.video_job
            else "storyboard"
            if self.storyboard
            else "script"
            if self.script
            else "story"
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
            if any(
                scene.duration < self.timing_profile.min_duration_seconds
                for scene in self.script.scenes
            ):
                review.hard_errors.append("存在不足最短时长、无法稳定转换为镜头的场景")
            if len(self.script.scenes) > self.timing_profile.maximum_shot_count(
                target_seconds
            ):
                review.hard_errors.append("场景数量超过当前时长可容纳的分镜数量")
            if any(
                not (scene.visible_action or scene.action).strip() for scene in self.script.scenes
            ):
                review.hard_errors.append("存在无法拍摄的空动作场景")
            if self.story is not None:
                semantic = review_script_against_story(
                    self.story,
                    self.script,
                    required_character_names=self._required_story_characters(),
                )
                review.hard_errors.extend(semantic.hard_errors)
                review.warnings.extend(semantic.warnings)
                review.scores.update(semantic.scores)
            review.scores["filmability"] = 1.0 if not review.hard_errors else 0.5
        elif kind == "storyboard" and self.storyboard:
            target_seconds = self.effective_target_seconds
            if not self.storyboard.shots:
                review.hard_errors.append("分镜没有镜头")
                review.scores["shot_diversity"] = 0.0
                review.scores["visual_identity_coverage"] = 0.0
                return review
            try:
                self._validate_storyboard_plan(self.storyboard)
            except (TypeError, ValueError) as exc:
                review.hard_errors.append(f"分镜结构无效：{exc}")
                review.scores["shot_diversity"] = 0.0
                review.scores["visual_identity_coverage"] = 0.0
                review.scores["timing_budget_fit"] = 0.0
                return review
            if abs(self.storyboard.total_duration - target_seconds) > 1:
                review.hard_errors.append("分镜总时长不符合目标")
            minimum = self.timing_profile.minimum_shot_count(target_seconds)
            maximum = self.timing_profile.maximum_shot_count(target_seconds)
            if not minimum <= len(self.storyboard.shots) <= maximum:
                review.hard_errors.append(f"当前时长下分镜数量必须在{minimum}到{maximum}之间")
            if any(
                not self.timing_profile.min_duration_seconds
                <= shot.duration
                <= self.timing_profile.max_duration_seconds
                for shot in self.storyboard.shots
            ):
                review.hard_errors.append("存在超出时长限制的镜头")
            try:
                if self.script is None:
                    raise ValueError("缺少已确认剧本")
                timing_assessment = assess_director_plan_timing(
                    self.script,
                    [
                        {
                            "scene_id": shot.scene_id,
                            "kind": shot.shot_kind,
                            "action": (
                                f"{shot.action}，{shot.retake_instruction}"
                                if shot.retake_instruction.strip()
                                else shot.action
                            ),
                            "purpose": shot.shot_purpose,
                            "transition_type": shot.transition_type,
                            "transition_reason": shot.transition_reason,
                            "inherit_previous_frame": shot.inherit_previous_frame,
                            "camera": shot.camera,
                            "camera_movement": shot.camera_movement,
                            "composition": shot.composition,
                        }
                        for shot in self.storyboard.shots
                    ],
                    dialogue_overrides=[
                        shot.dialogue for shot in self.storyboard.shots
                    ],
                    narration_overrides=[
                        shot.narration for shot in self.storyboard.shots
                    ],
                    timing_profile=self.timing_profile,
                )
            except (TypeError, ValueError) as exc:
                review.hard_errors.append(
                    f"无法根据当前剧本重新核验分镜时长预算：{exc}"
                )
                review.scores["timing_budget_fit"] = 0.0
            else:
                readable_minimums = list(timing_assessment.minimum_durations)
                minimum_total = sum(readable_minimums)
                over_capacity = [
                    shot.shot_id
                    for shot, minimum_duration in zip(
                        self.storyboard.shots,
                        readable_minimums,
                    )
                    if minimum_duration > self.timing_profile.max_duration_seconds
                ]
                if over_capacity:
                    review.hard_errors.append(
                        "存在单镜内容可读下限超过时长上限、必须拆分的镜头："
                        + "、".join(str(shot_id) for shot_id in over_capacity)
                    )
                underfunded = [
                    shot.shot_id
                    for shot, minimum_duration in zip(
                        self.storyboard.shots,
                        readable_minimums,
                    )
                    if shot.duration < minimum_duration
                ]
                if underfunded:
                    review.hard_errors.append(
                        "存在实际时长低于内容可读下限的镜头："
                        + "、".join(str(shot_id) for shot_id in underfunded)
                    )
                if minimum_total > target_seconds:
                    review.hard_errors.append(
                        f"分镜内容至少需要 {minimum_total} 秒，"
                        f"超过目标时长 {target_seconds} 秒"
                    )
                review.scores["timing_budget_fit"] = round(
                    min(1.0, target_seconds / max(1, minimum_total)),
                    3,
                )
                stored_minimums = [
                    shot.minimum_readable_duration
                    for shot in self.storyboard.shots
                ]
                if stored_minimums != readable_minimums:
                    review.warnings.append(
                        "已按当前剧本和镜头内容重新计算可读时长下限；"
                        "持久化的旧时长元数据未被用作确认依据"
                    )
            preferred_total = sum(
                shot.estimated_duration
                for shot in self.storyboard.shots
                if math.isfinite(float(shot.estimated_duration))
                and shot.estimated_duration > 0
            )
            review.scores["timing_preferred_fit"] = round(
                min(1.0, target_seconds / preferred_total),
                3,
            ) if preferred_total else 0.0
            if any(
                not shot.first_frame_prompt.strip()
                or not shot.motion_prompt.strip()
                or not shot.end_frame_prompt.strip()
                for shot in self.storyboard.shots
            ):
                review.hard_errors.append("存在缺少首帧、动作或结束帧描述的镜头")
            inconsistent_prompts = [
                shot.shot_id
                for shot in self.storyboard.shots
                if not shot_prompts_match_content(shot)
            ]
            if inconsistent_prompts:
                review.hard_errors.append(
                    "Provider 最终提示词与已审查的结构化镜头内容不一致：镜头"
                    + "、".join(str(shot_id) for shot_id in inconsistent_prompts)
                )
            cameras = {shot.camera for shot in self.storyboard.shots}
            review.scores["shot_diversity"] = min(1.0, len(cameras) / 4)
            referenced = sum(bool(shot.reference_asset_ids) for shot in self.storyboard.shots)
            review.scores["visual_identity_coverage"] = referenced / len(self.storyboard.shots)
            deterministic_quality = evaluate_storyboard_quality(self.storyboard)
            for key in (
                "storyboard_action_uniqueness",
                "storyboard_transition_explicitness",
                "storyboard_atomic_action_rate",
            ):
                review.scores[key] = float(deterministic_quality[key])
            atomic_rate = float(
                deterministic_quality["storyboard_atomic_action_rate"]
            )
            if atomic_rate < 0.8:
                review.hard_errors.append(
                    "过多镜头包含复合动作，必须拆成原子动作或压缩剧本内容"
                )
            elif atomic_rate < 1.0:
                review.warnings.append("少量镜头仍包含较多动作阶段，请重点预览")
        elif kind == "video_job" and self.video_job:
            if not self.video_job.title.strip() or not self.video_job.prompt.strip():
                review.hard_errors.append("完整视频任务缺少标题或提示词")
            if self.video_job.target_seconds <= 0:
                review.hard_errors.append("完整视频任务目标时长无效")
            review.scores["filmability"] = 1.0 if not review.hard_errors else 0.5
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
        self.video_job = None
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
                "text_generation_events": deepcopy(self.text_generation_events),
                "story": to_plain_data(self.story) if self.story else None,
                "story_history": to_plain_data(self.story_history),
                "script": to_plain_data(self.script) if self.script else None,
                "draft": to_plain_data(self.draft) if self.draft else None,
                "draft_history": to_plain_data(self.draft_history),
                "storyboard": to_plain_data(self.storyboard) if self.storyboard else None,
                "video_job": to_plain_data(self.video_job) if self.video_job else None,
                "v2_enabled": self.v2_enabled,
                # Phase 4A keeps StoryPlan/DirectorPlan nested in MoviePlan so
                # Session has one source of truth instead of duplicate layers.
                "movie_plan": v2_plain_data(self.movie_plan) if self.movie_plan else None,
                "movie_plan_revisions": v2_plain_data(self.movie_plan_revisions),
                "confirmed_movie_plan": (
                    v2_plain_data(self.confirmed_movie_plan)
                    if self.confirmed_movie_plan
                    else None
                ),
                "film_ir": self.film_ir.to_dict() if self.film_ir else None,
                "film_ir_revisions": [item.to_dict() for item in self.film_ir_revisions],
                "film_ir_build_diagnostics": deepcopy(self.film_ir_build_diagnostics),
                "film_ir_build_metadata": deepcopy(self.film_ir_build_metadata),
                "film_ir_validation_issues": deepcopy(self.film_ir_validation_issues),
                "film_ir_pass_diagnostics": deepcopy(self.film_ir_pass_diagnostics),
                "creative_pass_diagnostics": deepcopy(self.creative_pass_diagnostics),
                "creative_analysis_results": deepcopy(self.creative_analysis_results),
                "creative_analysis_diagnostics": deepcopy(self.creative_analysis_diagnostics),
                "creative_analysis_artifacts": deepcopy(self.creative_analysis_artifacts),
                "creative_analysis_metrics": deepcopy(self.creative_analysis_metrics),
                "creative_optimizer_result": deepcopy(self.creative_optimizer_result),
                "creative_optimizer_suggestions": deepcopy(self.creative_optimizer_suggestions),
                "creative_optimizer_candidates": deepcopy(self.creative_optimizer_candidates),
                "creative_optimizer_diagnostics": deepcopy(self.creative_optimizer_diagnostics),
                "creative_revision_requests": deepcopy(self.creative_revision_requests),
                "creative_revision_request_history": deepcopy(self.creative_revision_request_history),
                "creative_revision_stop_reason": self.creative_revision_stop_reason,
                "revision_candidates": deepcopy(self.revision_candidates),
                "revision_diffs": deepcopy(self.revision_diffs),
                "revision_decisions": deepcopy(self.revision_decisions),
                "revision_guard_diagnostics": deepcopy(self.revision_guard_diagnostics),
                "revision_active_candidate_id": self.revision_active_candidate_id,
                "revision_accepted_movie_plan_id": self.revision_accepted_movie_plan_id,
                "revision_rollback_movie_plan_id": self.revision_rollback_movie_plan_id,
                "director_revision_adapter_results": deepcopy(
                    self.director_revision_adapter_results
                ),
                "director_revision_contexts": deepcopy(self.director_revision_contexts),
                "guarded_revision_results": deepcopy(self.guarded_revision_results),
                "director_revision_attempt_count": self.director_revision_attempt_count,
                "director_revision_last_stop_reason": self.director_revision_last_stop_reason,
                "movie_plan_version_history": deepcopy(self.movie_plan_version_history),
                "revision_apply_history": deepcopy(self.revision_apply_history),
                "revision_rollback_history": deepcopy(self.revision_rollback_history),
                "revision_apply_results": deepcopy(self.revision_apply_results),
                "revision_rollback_results": deepcopy(self.revision_rollback_results),
                "current_movie_plan_id": self.current_movie_plan_id,
                "current_movie_plan_version": self.current_movie_plan_version,
                "current_movie_plan_fingerprint": self.current_movie_plan_fingerprint,
                "current_movie_plan_lineage_token": self.current_movie_plan_lineage_token,
                "previous_movie_plan_id": self.previous_movie_plan_id,
                "stale_artifacts": deepcopy(self.stale_artifacts),
                "source_lineage_diagnostics": deepcopy(self.source_lineage_diagnostics),
                "stale_lineage_diagnostics": deepcopy(self.stale_lineage_diagnostics),
                "current_film_ir_id": self.current_film_ir_id,
                "current_movie_ir_id": self.current_movie_ir_id,
                "current_video_job_id": self.current_video_job_id,
                "film_ir_optimizer_diagnostics": deepcopy(
                    self.film_ir_optimizer_diagnostics
                ),
                "director_revision_history": deepcopy(self.director_revision_history),
                "director_revision_stop_reason": self.director_revision_stop_reason,
                "movie_ir": self.movie_ir.to_dict() if self.movie_ir else None,
                "movie_ir_revisions": [item.to_dict() for item in self.movie_ir_revisions],
                "movie_ir_build_diagnostics": deepcopy(self.movie_ir_build_diagnostics),
                "movie_ir_build_metadata": deepcopy(self.movie_ir_build_metadata),
                "movie_ir_validation_issues": deepcopy(self.movie_ir_validation_issues),
                "movie_ir_pass_diagnostics": deepcopy(self.movie_ir_pass_diagnostics),
                "movie_ir_optimizer_diagnostics": deepcopy(
                    self.movie_ir_optimizer_diagnostics
                ),
                "v2_video_job": v2_plain_data(self.v2_video_job) if self.v2_video_job else None,
                "v2_compile_diagnostics": deepcopy(self.v2_compile_diagnostics),
                "v2_compile_metadata": deepcopy(self.v2_compile_metadata),
                "execution_plan": self.execution_plan.to_dict() if self.execution_plan else None,
                "execution_bundle": self.execution_bundle.to_dict() if self.execution_bundle else None,
                "current_execution_plan_id": self.current_execution_plan_id,
                "current_execution_plan_fingerprint": self.current_execution_plan_fingerprint,
                "current_execution_bundle_fingerprint": self.current_execution_bundle_fingerprint,
                "execution_plan_diagnostics": deepcopy(self.execution_plan_diagnostics),
                "stale_execution_artifacts": deepcopy(self.stale_execution_artifacts),
                "execution_run": self.execution_run.to_dict() if self.execution_run else None,
                "current_execution_run_id": self.current_execution_run_id,
                "execution_runtime_status": self.execution_runtime_status,
                "latest_execution_checkpoint_id": self.latest_execution_checkpoint_id,
                "execution_runtime_diagnostics": deepcopy(self.execution_runtime_diagnostics),
                "provider_jobs": deepcopy(self.provider_jobs),
                "runtime_artifacts": deepcopy(self.runtime_artifacts),
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
    def load(
        cls,
        path: str | Path,
        *,
        agent: StoryAgent | None = None,
        director_agent: V2DirectorAgent | None = None,
        v2_enabled: bool | None = None,
    ) -> GuidedStorySession:
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
        session = cls(
            CreativeBrief(**brief_data),
            agent=agent,
            director_agent=director_agent,
            v2_enabled=bool(
                data.get("v2_enabled", v2_enabled or False)
                or data.get("movie_plan")
                or data.get("confirmed_movie_plan")
                or data.get("film_ir")
                or data.get("movie_ir")
                or data.get("v2_video_job")
            ),
        )
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
        raw_text_events = data.get("text_generation_events", [])
        if not isinstance(raw_text_events, list) or not all(
            isinstance(item, dict) for item in raw_text_events
        ):
            raise ValueError("text_generation_events 必须是 JSON 对象数组。")
        session.text_generation_events = deepcopy(raw_text_events)
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
            session.storyboard = session._storyboard_from(
                data["storyboard"],
                normalize_legacy_narration=schema_version < 5,
            )
            stored_signature = str(data.get("confirmed_storyboard_signature", ""))
            session._confirmed_storyboard_signature = (
                stored_signature
                or (
                    session._storyboard_confirmation_signature(session.storyboard)
                    if session.storyboard.confirmed
                    else ""
                )
            )
        if data.get("video_job"):
            session.video_job = session._video_job_from(data["video_job"])
        if data.get("movie_plan"):
            # movie_plan_from_data migrates pre-Phase-4A JSON that lacks the
            # nested story_plan/director_plan fields.
            session.movie_plan = movie_plan_from_data(data["movie_plan"])
        session.movie_plan_revisions = [
            movie_plan_from_data(item) for item in data.get("movie_plan_revisions", [])
        ]
        if data.get("confirmed_movie_plan"):
            session.confirmed_movie_plan = movie_plan_from_data(data["confirmed_movie_plan"])
        for field_name in ("current_movie_plan_id", "previous_movie_plan_id"):
            raw_id = data.get(field_name)
            if raw_id is not None and not isinstance(raw_id, str):
                raise ValueError(f"{field_name} 必须是字符串或 null。")
            setattr(session, field_name, raw_id)
        raw_version = data.get("current_movie_plan_version")
        if raw_version is not None and (isinstance(raw_version, bool) or not isinstance(raw_version, int) or raw_version < 1):
            raise ValueError("current_movie_plan_version 必须是正整数或 null。")
        session.current_movie_plan_version = raw_version
        for field_name in ("current_movie_plan_fingerprint", "current_movie_plan_lineage_token"):
            raw_value = data.get(field_name)
            if raw_value is not None and not isinstance(raw_value, str):
                raise ValueError(f"{field_name} 必须是字符串或 null。")
            setattr(session, field_name, raw_value)
        if session.current_movie_plan_id is None:
            current_plan = session.confirmed_movie_plan or session.movie_plan
            session.current_movie_plan_id = current_plan.plan_id if current_plan else None
        current_plan = session.confirmed_movie_plan or session.movie_plan
        if current_plan is not None:
            # Phase 4F migration is deterministic and does not regenerate the
            # plan or rewrite historical snapshots.
            migrated_plan = ensure_movie_plan_provenance(current_plan)
            if session.confirmed_movie_plan is not None:
                session.confirmed_movie_plan = replace(
                    session.confirmed_movie_plan,
                    movie_plan_version=migrated_plan.movie_plan_version,
                    movie_plan_fingerprint=migrated_plan.movie_plan_fingerprint,
                    movie_plan_lineage_token=migrated_plan.movie_plan_lineage_token,
                )
            if session.movie_plan is not None:
                session.movie_plan = replace(
                    session.movie_plan,
                    movie_plan_version=migrated_plan.movie_plan_version,
                    movie_plan_fingerprint=migrated_plan.movie_plan_fingerprint,
                    movie_plan_lineage_token=migrated_plan.movie_plan_lineage_token,
                )
            session.current_movie_plan_version = session.current_movie_plan_version or migrated_plan.movie_plan_version
            session.current_movie_plan_fingerprint = session.current_movie_plan_fingerprint or migrated_plan.movie_plan_fingerprint
            session.current_movie_plan_lineage_token = session.current_movie_plan_lineage_token or migrated_plan.movie_plan_lineage_token
        if data.get("film_ir"):
            session.film_ir = FilmIR.from_dict(data["film_ir"])
        session.film_ir_revisions = [
            FilmIR.from_dict(item) for item in data.get("film_ir_revisions", [])
        ]
        raw_film_diagnostics = data.get("film_ir_build_diagnostics", [])
        if not isinstance(raw_film_diagnostics, list) or any(
            not isinstance(item, dict) for item in raw_film_diagnostics
        ):
            raise ValueError("film_ir_build_diagnostics 必须是 JSON 对象数组。")
        session.film_ir_build_diagnostics = deepcopy(raw_film_diagnostics)
        raw_film_metadata = data.get("film_ir_build_metadata", {})
        if not isinstance(raw_film_metadata, dict):
            raise ValueError("film_ir_build_metadata 必须是 JSON 对象。")
        session.film_ir_build_metadata = deepcopy(raw_film_metadata)
        raw_film_issues = data.get("film_ir_validation_issues", [])
        if not isinstance(raw_film_issues, list) or any(
            not isinstance(item, dict) for item in raw_film_issues
        ):
            raise ValueError("film_ir_validation_issues 必须是 JSON 对象数组。")
        session.film_ir_validation_issues = deepcopy(raw_film_issues)
        raw_film_passes = data.get("film_ir_pass_diagnostics", [])
        if not isinstance(raw_film_passes, list) or any(
            not isinstance(item, dict) for item in raw_film_passes
        ):
            raise ValueError("film_ir_pass_diagnostics 必须是 JSON 对象数组。")
        session.film_ir_pass_diagnostics = deepcopy(raw_film_passes)
        raw_creative_diagnostics = data.get("creative_pass_diagnostics", [])
        if not isinstance(raw_creative_diagnostics, list) or any(
            not isinstance(item, dict) for item in raw_creative_diagnostics
        ):
            raise ValueError("creative_pass_diagnostics 必须是 JSON 对象数组。")
        session.creative_pass_diagnostics = deepcopy(raw_creative_diagnostics)
        raw_analysis_results = data.get("creative_analysis_results", [])
        if not isinstance(raw_analysis_results, list) or any(
            not isinstance(item, dict) for item in raw_analysis_results
        ):
            raise ValueError("creative_analysis_results 必须是 JSON 对象数组。")
        session.creative_analysis_results = deepcopy(raw_analysis_results)
        raw_analysis_diagnostics = data.get("creative_analysis_diagnostics", [])
        if not isinstance(raw_analysis_diagnostics, list) or any(
            not isinstance(item, dict) for item in raw_analysis_diagnostics
        ):
            raise ValueError("creative_analysis_diagnostics 必须是 JSON 对象数组。")
        session.creative_analysis_diagnostics = deepcopy(raw_analysis_diagnostics)
        raw_analysis_artifacts = data.get("creative_analysis_artifacts", [])
        if not isinstance(raw_analysis_artifacts, list) or any(
            not isinstance(item, dict) for item in raw_analysis_artifacts
        ):
            raise ValueError("creative_analysis_artifacts 必须是 JSON 对象数组。")
        session.creative_analysis_artifacts = deepcopy(raw_analysis_artifacts)
        raw_analysis_metrics = data.get("creative_analysis_metrics", {})
        if not isinstance(raw_analysis_metrics, dict):
            raise ValueError("creative_analysis_metrics 必须是 JSON 对象。")
        session.creative_analysis_metrics = {
            str(key): float(value) for key, value in raw_analysis_metrics.items()
        }
        raw_creative_optimizer_result = data.get("creative_optimizer_result")
        if raw_creative_optimizer_result is not None and not isinstance(raw_creative_optimizer_result, dict):
            raise ValueError("creative_optimizer_result 必须是 JSON 对象或 null。")
        session.creative_optimizer_result = deepcopy(raw_creative_optimizer_result)
        for field_name in (
            "creative_optimizer_suggestions",
            "creative_optimizer_candidates",
            "creative_optimizer_diagnostics",
            "creative_revision_requests",
            "creative_revision_request_history",
        ):
            raw_items = data.get(field_name, [])
            if not isinstance(raw_items, list) or any(not isinstance(item, dict) for item in raw_items):
                raise ValueError(f"{field_name} 必须是 JSON 对象数组。")
            setattr(session, field_name, deepcopy(raw_items))
        raw_creative_revision_stop = data.get("creative_revision_stop_reason")
        if raw_creative_revision_stop is not None and not isinstance(raw_creative_revision_stop, str):
            raise ValueError("creative_revision_stop_reason 必须是字符串或 null。")
        session.creative_revision_stop_reason = raw_creative_revision_stop
        for field_name in (
            "revision_candidates",
            "revision_diffs",
            "revision_decisions",
            "revision_guard_diagnostics",
        ):
            raw_items = data.get(field_name, [])
            if not isinstance(raw_items, list) or any(not isinstance(item, dict) for item in raw_items):
                raise ValueError(f"{field_name} 必须是 JSON 对象数组。")
            setattr(session, field_name, deepcopy(raw_items))
        for field_name in (
            "revision_active_candidate_id",
            "revision_accepted_movie_plan_id",
            "revision_rollback_movie_plan_id",
        ):
            raw_id = data.get(field_name)
            if raw_id is not None and not isinstance(raw_id, str):
                raise ValueError(f"{field_name} 必须是字符串或 null。")
            setattr(session, field_name, raw_id)
        for field_name in (
            "director_revision_adapter_results",
            "director_revision_contexts",
            "guarded_revision_results",
        ):
            raw_items = data.get(field_name, [])
            if not isinstance(raw_items, list) or any(
                not isinstance(item, dict) for item in raw_items
            ):
                raise ValueError(f"{field_name} 必须是 JSON 对象数组。")
            setattr(session, field_name, deepcopy(raw_items))
        raw_attempt_count = data.get("director_revision_attempt_count", 0)
        if isinstance(raw_attempt_count, bool) or not isinstance(raw_attempt_count, int) or raw_attempt_count < 0:
            raise ValueError("director_revision_attempt_count 必须是非负整数。")
        session.director_revision_attempt_count = raw_attempt_count
        raw_last_stop = data.get("director_revision_last_stop_reason")
        if raw_last_stop is not None and not isinstance(raw_last_stop, str):
            raise ValueError("director_revision_last_stop_reason 必须是字符串或 null。")
        session.director_revision_last_stop_reason = raw_last_stop
        for field_name in (
            "movie_plan_version_history",
            "revision_apply_history",
            "revision_rollback_history",
            "revision_apply_results",
            "revision_rollback_results",
            "stale_artifacts",
        ):
            raw_items = data.get(field_name, [])
            if not isinstance(raw_items, list) or any(
                not isinstance(item, dict) for item in raw_items
            ):
                raise ValueError(f"{field_name} 必须是 JSON 对象数组。")
            setattr(session, field_name, deepcopy(raw_items))
        for field_name in ("source_lineage_diagnostics", "stale_lineage_diagnostics"):
            raw_items = data.get(field_name, [])
            if not isinstance(raw_items, list) or any(
                not isinstance(item, dict) for item in raw_items
            ):
                raise ValueError(f"{field_name} 必须是 JSON 对象数组。")
            setattr(session, field_name, deepcopy(raw_items))
        for field_name in (
            "current_film_ir_id",
            "current_movie_ir_id",
            "current_video_job_id",
        ):
            raw_id = data.get(field_name)
            if raw_id is not None and not isinstance(raw_id, str):
                raise ValueError(f"{field_name} 必须是字符串或 null。")
            setattr(session, field_name, raw_id)
        raw_film_optimizer = data.get("film_ir_optimizer_diagnostics", [])
        if not isinstance(raw_film_optimizer, list) or any(
            not isinstance(item, dict) for item in raw_film_optimizer
        ):
            raise ValueError("film_ir_optimizer_diagnostics 必须是 JSON 对象数组。")
        session.film_ir_optimizer_diagnostics = deepcopy(raw_film_optimizer)
        raw_revision_history = data.get("director_revision_history", [])
        if not isinstance(raw_revision_history, list) or any(
            not isinstance(item, dict) for item in raw_revision_history
        ):
            raise ValueError("director_revision_history 必须是 JSON 对象数组。")
        session.director_revision_history = deepcopy(raw_revision_history)
        raw_stop_reason = data.get("director_revision_stop_reason")
        if raw_stop_reason is not None and not isinstance(raw_stop_reason, str):
            raise ValueError("director_revision_stop_reason 必须是字符串或 null。")
        session.director_revision_stop_reason = raw_stop_reason
        if data.get("movie_ir"):
            session.movie_ir = MovieIR.from_dict(data["movie_ir"])
        session.movie_ir_revisions = [
            MovieIR.from_dict(item) for item in data.get("movie_ir_revisions", [])
        ]
        raw_ir_diagnostics = data.get("movie_ir_build_diagnostics", [])
        if not isinstance(raw_ir_diagnostics, list) or any(
            not isinstance(item, dict) for item in raw_ir_diagnostics
        ):
            raise ValueError("movie_ir_build_diagnostics 必须是 JSON 对象数组。")
        session.movie_ir_build_diagnostics = deepcopy(raw_ir_diagnostics)
        raw_ir_metadata = data.get("movie_ir_build_metadata", {})
        if not isinstance(raw_ir_metadata, dict):
            raise ValueError("movie_ir_build_metadata 必须是 JSON 对象。")
        session.movie_ir_build_metadata = deepcopy(raw_ir_metadata)
        raw_movie_issues = data.get("movie_ir_validation_issues", [])
        if not isinstance(raw_movie_issues, list) or any(
            not isinstance(item, dict) for item in raw_movie_issues
        ):
            raise ValueError("movie_ir_validation_issues 必须是 JSON 对象数组。")
        session.movie_ir_validation_issues = deepcopy(raw_movie_issues)
        raw_movie_passes = data.get("movie_ir_pass_diagnostics", [])
        if not isinstance(raw_movie_passes, list) or any(
            not isinstance(item, dict) for item in raw_movie_passes
        ):
            raise ValueError("movie_ir_pass_diagnostics 必须是 JSON 对象数组。")
        session.movie_ir_pass_diagnostics = deepcopy(raw_movie_passes)
        raw_movie_optimizer = data.get("movie_ir_optimizer_diagnostics", [])
        if not isinstance(raw_movie_optimizer, list) or any(
            not isinstance(item, dict) for item in raw_movie_optimizer
        ):
            raise ValueError("movie_ir_optimizer_diagnostics 必须是 JSON 对象数组。")
        session.movie_ir_optimizer_diagnostics = deepcopy(raw_movie_optimizer)
        if data.get("v2_video_job"):
            session.v2_video_job = session._v2_video_job_from(data["v2_video_job"])
        raw_compile_diagnostics = data.get("v2_compile_diagnostics", [])
        if not isinstance(raw_compile_diagnostics, list) or any(
            not isinstance(item, dict) for item in raw_compile_diagnostics
        ):
            raise ValueError("v2_compile_diagnostics 必须是 JSON 对象数组。")
        session.v2_compile_diagnostics = deepcopy(raw_compile_diagnostics)
        raw_compile_metadata = data.get("v2_compile_metadata", {})
        if not isinstance(raw_compile_metadata, dict):
            raise ValueError("v2_compile_metadata 必须是 JSON 对象。")
        session.v2_compile_metadata = deepcopy(raw_compile_metadata)
        if data.get("execution_plan"):
            session.execution_plan = ExecutionPlan.from_dict(data["execution_plan"])
        if data.get("execution_bundle"):
            session.execution_bundle = ExecutionBundle.from_dict(data["execution_bundle"])
            if session.execution_plan is None:
                session.execution_plan = session.execution_bundle.execution_plan
        for field_name in (
            "current_execution_plan_id",
            "current_execution_plan_fingerprint",
            "current_execution_bundle_fingerprint",
        ):
            raw_value = data.get(field_name)
            if raw_value is not None and not isinstance(raw_value, str):
                raise ValueError(f"{field_name} 必须是字符串或 null。")
            setattr(session, field_name, raw_value)
        raw_execution_diagnostics = data.get("execution_plan_diagnostics", [])
        if not isinstance(raw_execution_diagnostics, list) or any(
            not isinstance(item, dict) for item in raw_execution_diagnostics
        ):
            raise ValueError("execution_plan_diagnostics 必须是 JSON 对象数组。")
        session.execution_plan_diagnostics = deepcopy(raw_execution_diagnostics)
        raw_stale_execution = data.get("stale_execution_artifacts", [])
        if not isinstance(raw_stale_execution, list) or any(
            not isinstance(item, dict) for item in raw_stale_execution
        ):
            raise ValueError("stale_execution_artifacts 必须是 JSON 对象数组。")
        session.stale_execution_artifacts = deepcopy(raw_stale_execution)
        if data.get("execution_run"):
            if not isinstance(data["execution_run"], dict):
                raise ValueError("execution_run 必须是 JSON 对象或 null。")
            session.execution_run = ExecutionRun.from_dict(data["execution_run"])
        raw_run_id = data.get("current_execution_run_id")
        if raw_run_id is not None and not isinstance(raw_run_id, str):
            raise ValueError("current_execution_run_id 必须是字符串或 null。")
        session.current_execution_run_id = raw_run_id
        raw_runtime_status = data.get("execution_runtime_status")
        if raw_runtime_status is not None and not isinstance(raw_runtime_status, str):
            raise ValueError("execution_runtime_status 必须是字符串或 null。")
        session.execution_runtime_status = raw_runtime_status
        raw_checkpoint_id = data.get("latest_execution_checkpoint_id")
        if raw_checkpoint_id is not None and not isinstance(raw_checkpoint_id, str):
            raise ValueError("latest_execution_checkpoint_id 必须是字符串或 null。")
        session.latest_execution_checkpoint_id = raw_checkpoint_id
        raw_runtime_diagnostics = data.get("execution_runtime_diagnostics", [])
        if not isinstance(raw_runtime_diagnostics, list) or any(
            not isinstance(item, dict) for item in raw_runtime_diagnostics
        ):
            raise ValueError("execution_runtime_diagnostics 必须是 JSON 对象数组。")
        session.execution_runtime_diagnostics = deepcopy(raw_runtime_diagnostics)
        raw_provider_jobs = data.get("provider_jobs", [])
        if not isinstance(raw_provider_jobs, list) or any(
            not isinstance(item, dict) for item in raw_provider_jobs
        ):
            raise ValueError("provider_jobs 必须是 JSON 对象数组。")
        session.provider_jobs = deepcopy(raw_provider_jobs)
        raw_runtime_artifacts = data.get("runtime_artifacts", [])
        if not isinstance(raw_runtime_artifacts, list) or any(
            not isinstance(item, dict) for item in raw_runtime_artifacts
        ):
            raise ValueError("runtime_artifacts 必须是 JSON 对象数组。")
        session.runtime_artifacts = deepcopy(raw_runtime_artifacts)
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
        # Migrate old sessions that already have artifact IDs but did not yet
        # persist the explicit current_* lineage pointers.
        if session.current_film_ir_id is None and session.film_ir is not None:
            session.current_film_ir_id = session.film_ir.ir_id
        if session.current_movie_ir_id is None and session.movie_ir is not None:
            session.current_movie_ir_id = session.movie_ir.ir_id
        if session.current_video_job_id is None and session.v2_video_job is not None:
            session.current_video_job_id = session.v2_video_job.job_id
        if session.execution_bundle is not None:
            if session.execution_plan is None:
                session.execution_plan = session.execution_bundle.execution_plan
            if session.current_execution_plan_id is None:
                session.current_execution_plan_id = session.execution_bundle.execution_plan.execution_plan_id
            if session.current_execution_plan_fingerprint is None:
                session.current_execution_plan_fingerprint = session.execution_bundle.execution_plan.execution_plan_fingerprint
            if session.current_execution_bundle_fingerprint is None:
                session.current_execution_bundle_fingerprint = session.execution_bundle.bundle_fingerprint
        if session.execution_run is not None:
            if session.current_execution_run_id is None:
                session.current_execution_run_id = session.execution_run.execution_run_id
            if session.execution_runtime_status is None:
                session.execution_runtime_status = session.execution_run.status.value
            if session.latest_execution_checkpoint_id is None:
                session.latest_execution_checkpoint_id = session.execution_run.latest_checkpoint_id
            if not session.provider_jobs:
                session.provider_jobs = [job.to_dict() for job in session.execution_run.provider_jobs.values()]
            if not session.runtime_artifacts:
                session.runtime_artifacts = [dict(item) for item in session.execution_run.artifacts.values()]
        session._validate_loaded_state(schema_version=schema_version)
        session.refresh_source_lineage_diagnostics()
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
            self.storyboard = self._storyboard_from(
                data["storyboard"],
                normalize_legacy_narration=True,
            )
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
        if self.render_manifest is not None and self.storyboard is None and self.video_job is None:
            raise ValueError("存在渲染记录，但视频任务缺失。")
        if self.render_manifest is not None and self.storyboard is not None:
            self._validate_render_outputs(self.storyboard, self.render_manifest)
        if self.video_job is not None:
            if not self.video_job.title.strip() or not self.video_job.prompt.strip():
                raise ValueError("保存的视频任务缺少标题或提示词。")
            if self.video_job.target_seconds <= 0:
                raise ValueError("保存的视频任务目标时长无效。")

        if self.movie_plan is not None:
            report = validate_movie_plan(self.movie_plan, self._v2_brief())
            if not report.valid:
                raise ValueError("保存的 V2 MoviePlan 无效：" + "；".join(report.errors))
            if self.current_movie_plan_id not in {None, self.movie_plan.plan_id}:
                raise ValueError("current_movie_plan_id 与当前 MoviePlan 不一致。")
            if self.current_movie_plan_version not in {None, self.movie_plan.movie_plan_version}:
                raise ValueError("current_movie_plan_version 与当前 MoviePlan 不一致。")
            if self.current_movie_plan_fingerprint not in {None, self.movie_plan.movie_plan_fingerprint}:
                raise ValueError("current_movie_plan_fingerprint 与当前 MoviePlan 不一致。")
            if self.current_movie_plan_lineage_token not in {None, self.movie_plan.movie_plan_lineage_token}:
                raise ValueError("current_movie_plan_lineage_token 与当前 MoviePlan 不一致。")
        if any(not isinstance(item, V2MoviePlan) for item in self.movie_plan_revisions):
            raise ValueError("保存的 V2 MoviePlan revisions 无效。")
        if self.confirmed_movie_plan is not None:
            report = validate_movie_plan(self.confirmed_movie_plan, self._v2_brief())
            if not report.valid or not self.confirmed_movie_plan.confirmed:
                raise ValueError("保存的 confirmed_movie_plan 无效。")
        if self.film_ir is not None:
            if not self.film_ir.beats or not self.film_ir.shots:
                raise ValueError("保存的 FilmIR 缺少 beats 或 shots。")
        if any(not isinstance(item, FilmIR) for item in self.film_ir_revisions):
            raise ValueError("保存的 FilmIR revisions 无效。")
        if self.movie_ir is not None:
            if not self.movie_ir.shots:
                raise ValueError("保存的 MovieIR 缺少 shots。")
        if any(not isinstance(item, MovieIR) for item in self.movie_ir_revisions):
            raise ValueError("保存的 MovieIR revisions 无效。")
        if self.v2_video_job is not None:
            report = validate_video_job(self.v2_video_job)
            if not report.valid or not self.v2_video_job.confirmed:
                raise ValueError("保存的 V2 VideoJob 无效。")
        if self.execution_plan is not None and self.execution_bundle is None:
            raise ValueError("保存的 ExecutionPlan 缺少 ExecutionBundle。")
        if self.execution_bundle is not None:
            report = validate_execution_bundle(self.execution_bundle)
            if not report.valid:
                raise ValueError(
                    "保存的 ExecutionBundle 无效："
                    + "；".join(item.message for item in report.diagnostics)
                )
            if self.execution_plan is not None and self.execution_plan != self.execution_bundle.execution_plan:
                raise ValueError("ExecutionPlan 与 ExecutionBundle 中的计划不一致。")
            if self.current_execution_plan_id not in {None, self.execution_bundle.execution_plan.execution_plan_id}:
                raise ValueError("current_execution_plan_id 与当前 ExecutionPlan 不一致。")
            if self.current_execution_plan_fingerprint not in {None, self.execution_bundle.execution_plan.execution_plan_fingerprint}:
                raise ValueError("current_execution_plan_fingerprint 与当前 ExecutionPlan 不一致。")
            if self.current_execution_bundle_fingerprint not in {None, self.execution_bundle.bundle_fingerprint}:
                raise ValueError("current_execution_bundle_fingerprint 与当前 ExecutionBundle 不一致。")
        if self.execution_run is not None:
            if self.execution_bundle is None:
                raise ValueError("保存的 ExecutionRun 缺少 ExecutionBundle。")
            if self.current_execution_run_id not in {None, self.execution_run.execution_run_id}:
                raise ValueError("current_execution_run_id 与 ExecutionRun 不一致。")
            if self.execution_runtime_status not in {None, self.execution_run.status.value}:
                raise ValueError("execution_runtime_status 与 ExecutionRun 不一致。")
            # A stale run is retained as audit history.  It is not allowed to
            # silently become the active input for a newer Bundle.
            if self.execution_run.execution_bundle_fingerprint != self.execution_bundle.bundle_fingerprint:
                self.execution_runtime_diagnostics.append(
                    {
                        "code": "stale_execution_run",
                        "message": "ExecutionRun 与当前 ExecutionBundle fingerprint 不一致；禁止 resume/step/run。",
                    }
                )

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
            or not (
                (self.storyboard is not None and self.storyboard.confirmed)
                or (self.video_job is not None and self.video_job.confirmed)
            )
        ):
            raise ValueError("可渲染阶段的故事、剧本或视频任务确认链不完整。")
        if self.stage == Stage.COMPLETED and (
            self.render_manifest is None
            or self.render_manifest.status not in {"succeeded", "succeeded_with_warnings"}
        ):
            raise ValueError("已完成阶段缺少成功的渲染记录。")
        if self.stage == Stage.MOVIE_PLAN_REVIEW and self.movie_plan is None:
            raise ValueError("V2 MoviePlan 审查阶段缺少 MoviePlan。")
        if self.stage == Stage.MOVIE_PLAN_CONFIRMED and (
            self.confirmed_movie_plan is None
            or not self.confirmed_movie_plan.confirmed
        ):
            raise ValueError("V2 MoviePlan 已确认阶段缺少 confirmed_movie_plan。")
        if self.stage in {Stage.MOVIE_PLAN_REVISED, Stage.MOVIE_PLAN_ROLLED_BACK} and (
            self.movie_plan is None
            or self.confirmed_movie_plan is None
            or not self.confirmed_movie_plan.confirmed
            or self.film_ir is not None
            or self.movie_ir is not None
            or self.v2_video_job is not None
        ):
            raise ValueError("V2 MoviePlan 修订/回滚阶段必须只有重新确认的 MoviePlan，旧下游产物必须失效。")
        if self.stage == Stage.FILM_IR_BUILT and (
            self.confirmed_movie_plan is None
            or not self.confirmed_movie_plan.confirmed
            or self.film_ir is None
        ):
            raise ValueError("V2 FilmIR 已构建阶段缺少 confirmed_movie_plan 或 film_ir。")
        if self.stage == Stage.MOVIE_IR_BUILT and (
            self.confirmed_movie_plan is None
            or not self.confirmed_movie_plan.confirmed
            or self.movie_ir is None
            or (self.movie_ir.source_film_ir_id and self.film_ir is None)
        ):
            raise ValueError("V2 MovieIR 已构建阶段缺少 confirmed_movie_plan、film_ir 或 movie_ir。")
        if self.stage == Stage.VIDEO_JOB_COMPILED and (
            self.confirmed_movie_plan is None
            or not self.confirmed_movie_plan.confirmed
            or self.v2_video_job is None
            or not self.v2_video_job.confirmed
            or self.movie_ir is None
            or (self.v2_video_job.source_film_ir_id and self.film_ir is None)
        ):
            raise ValueError("V2 VideoJob 已编译阶段缺少 confirmed_movie_plan、film_ir 或 v2_video_job。")
        if self.stage == Stage.EXECUTION_PLAN_BUILT and (
            self.confirmed_movie_plan is None
            or not self.confirmed_movie_plan.confirmed
            or self.film_ir is None
            or self.movie_ir is None
            or self.execution_plan is None
            or self.execution_bundle is None
        ):
            raise ValueError("V2 ExecutionPlan 已构建阶段缺少完整 MoviePlan/FilmIR/MovieIR/ExecutionBundle。")
        if self.stage in {
            Stage.EXECUTION_READY,
            Stage.EXECUTION_RUNNING,
            Stage.EXECUTION_BLOCKED,
            Stage.EXECUTION_COMPLETED,
            Stage.EXECUTION_FAILED,
        } and self.execution_bundle is None:
            raise ValueError("Execution Runtime 阶段缺少 ExecutionBundle。")

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
        if len(script.scenes) > self.timing_profile.maximum_shot_count(
            script.target_seconds
        ):
            raise ValueError("保存的剧本场景过密，无法转换为最短时长以上的镜头。")
        seen_ids: set[int] = set()
        for scene in script.scenes:
            self._validate_scene_fields(scene)
            if scene.scene_id in seen_ids:
                raise ValueError("保存的剧本场景 ID 不能重复。")
            seen_ids.add(scene.scene_id)
            if scene.duration < self.timing_profile.min_duration_seconds:
                raise ValueError("保存的剧本包含不足最短时长的场景。")
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
        previous_end_state = ""
        for index, scene in enumerate(script.scenes):
            self._validate_scene_fields(scene)
            if not (scene.visible_action or scene.action).strip():
                scene.action = scene.narration or "主角完成一个清晰可见的动作"
                scene.visible_action = scene.action
            visible_action = (scene.visible_action or scene.action).strip()
            if not scene.start_state.strip():
                if previous_end_state:
                    scene.start_state = f"承接上一场结果：{previous_end_state}"
                elif self.story is not None:
                    scene.start_state = (
                        f"已确认的核心冲突是“{self.story.core_conflict}”；"
                        f"人物位于{scene.location or '故事核心场景'}，准备开始行动"
                    )
                else:
                    scene.start_state = (
                        f"人物位于{scene.location or '故事核心场景'}，准备开始行动"
                    )
            if not scene.end_state.strip():
                scene.end_state = visible_action
                if index == len(script.scenes) - 1 and self.story is not None:
                    scene.end_state = (
                        f"{visible_action}；动作结果让故事走向“{self.story.ending}”"
                    )
            previous_end_state = scene.end_state
        script.scenes = fit_scenes_to_duration(script.scenes, target, minimum=3)
        durations, weights, reasons = plan_scene_durations(
            script.scenes,
            target,
            minimum=3,
            maximum=target,
        )
        for scene, duration, weight, reason in zip(
            script.scenes,
            durations,
            weights,
            reasons,
        ):
            scene.duration = duration
            scene.duration_weight = weight
            scene.duration_reason = reason
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
        """Record immutable provenance without rewriting a reviewed story after the model call."""
        cards = self.selected_cards
        source_ids = [card.idea_id for card in cards]
        if cards:
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
            character_material = f"{character_text}\n{story.story_text}"
            if semantic_coverage(card.protagonist, character_material) < 0.45:
                raise ValueError("所选创意卡的主角没有落实到故事结构中。")
            source = f"{card.central_conflict} {card.ending_direction}"
            target = f"{story.core_conflict} {story.ending} {story.story_text}"
            if semantic_coverage(source, target) < 0.18:
                raise ValueError("所选创意卡的冲突或结局没有被完整融合进故事。")
        for kind, option in selected_options.items():
            if kind == "character":
                material = f"{character_text}\n{story.story_text}"
                if semantic_coverage(option.content, material) < 0.45:
                    raise ValueError("所选角色设定没有落实到故事结构中。")
            elif kind == "turning_point" and semantic_coverage(
                option.content,
                story.story_text,
            ) < 0.45:
                raise ValueError("所选转折没有落实到故事正文中。")
            elif kind == "conflict" and semantic_coverage(
                option.content,
                f"{story.core_conflict} {story.story_text}",
            ) < 0.45:
                raise ValueError("所选冲突没有保留。")
            elif kind == "ending" and semantic_coverage(
                option.content,
                f"{story.ending} {story.story_text}",
            ) < 0.45:
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
        self.video_job = None
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
        candidate = Path(reference.path).expanduser().absolute()
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
        review = review_script_against_story(
            self.story,
            script,
            required_character_names=self._required_story_characters(),
        )
        if review.hard_errors:
            raise ValueError("；".join(review.hard_errors))

    def _required_story_characters(self) -> list[str]:
        if self.story is not None:
            confirmed = [
                character.name.strip()
                for character in self.story.characters[:1]
                if character.name.strip()
            ]
            if confirmed:
                return confirmed
        selected = [
            card.protagonist.strip()
            for card in self.selected_cards
            if card.protagonist.strip()
        ]
        return list(dict.fromkeys(selected))

    def _validate_storyboard_plan(self, plan: StoryboardPlan) -> None:
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
            "dialogue",
            "source_action",
            "retake_instruction",
            "time_of_day",
            "visual_style",
            "color_palette",
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
            "transition_type",
            "transition_reason",
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
            if not (
                self.timing_profile.min_duration_seconds
                <= shot.duration
                <= self.timing_profile.max_duration_seconds
            ):
                raise ValueError(
                    "镜头时长必须在 "
                    f"{self.timing_profile.min_duration_seconds} 到 "
                    f"{self.timing_profile.max_duration_seconds} 秒之间。"
                )
            if (
                isinstance(shot.minimum_readable_duration, bool)
                or not isinstance(shot.minimum_readable_duration, int)
                or shot.minimum_readable_duration < 0
            ):
                raise ValueError("镜头内容可读下限必须是非负整数。")
            for field in text_fields:
                if not isinstance(getattr(shot, field), str):
                    raise ValueError(f"镜头字段 {field} 必须是字符串。")
            for field in required_text:
                value = getattr(shot, field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"镜头缺少有效字段：{field}")
            if shot.aspect_ratio not in {"16:9", "9:16", "1:1"}:
                raise ValueError("镜头画幅比例只支持 16:9、9:16 或 1:1。")
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
                if not math.isfinite(float(value)):
                    raise ValueError(f"镜头字段 {field} 必须是有限数字。")
                if value < 0:
                    raise ValueError(f"镜头字段 {field} 不能为负数。")
            if shot.continuity_mode not in {
                "independent",
                "same_scene_chain",
                "same_scene_reference",
                "new_scene_reference",
            }:
                raise ValueError("镜头 continuity_mode 无法识别。")
            if shot.previous_shot_id is not None and (
                isinstance(shot.previous_shot_id, bool)
                or not isinstance(shot.previous_shot_id, int)
            ):
                raise ValueError("镜头 previous_shot_id 必须是整数或 null。")
            if shot.continuity_mode in {
                "same_scene_chain",
                "same_scene_reference",
            }:
                if (
                    shot.previous_shot_id is None
                    or shot.previous_shot_id == shot.shot_id
                    or shot.previous_shot_id not in seen_ids
                ):
                    raise ValueError("同场景镜头必须引用前面已存在的镜头。")
            elif shot.previous_shot_id is not None:
                raise ValueError("独立或新场景镜头不能继承上一镜头。")
            if not isinstance(shot.inherit_previous_frame, bool):
                raise ValueError("镜头 inherit_previous_frame 必须是布尔值。")
            if (
                shot.continuity_mode == "same_scene_chain"
            ) != shot.inherit_previous_frame:
                raise ValueError(
                    "只有 same_scene_chain 可以继承上一镜头末帧。"
                )
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
            "transition_type",
            "transition_reason",
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
            if not isinstance(artifact.inherit_previous_frame, bool):
                raise TypeError("视频产物继承末帧标志必须是布尔值。")
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
        job = payload.get("video_job")
        if isinstance(job, dict):
            job["initial_frame_url"] = sanitize_remote_url(
                str(job.get("initial_frame_url", ""))
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
            "v2_video_job"
            if self.v2_video_job
            else "movie_ir"
            if self.movie_ir
            else "film_ir"
            if self.film_ir
            else "video_job"
            if self.video_job
            else "storyboard"
            if self.storyboard
            else "script"
            if self.script
            else "story"
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
            "v2_video_job"
            if self.v2_video_job
            else "movie_ir"
            if self.movie_ir
            else "film_ir"
            if self.film_ir
            else "video_job"
            if self.video_job
            else "storyboard"
            if self.storyboard
            else "script"
            if self.script
            else "story"
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
        if revision.artifact_type == "video_job":
            if self.story is None or not self.story.confirmed or self.script is None or not self.script.confirmed:
                raise ValueError("恢复视频任务版本前必须存在已确认故事和剧本。")
            candidate = self._video_job_from(deepcopy(revision.payload))
            if not candidate.title.strip() or not candidate.prompt.strip() or candidate.target_seconds <= 0:
                raise ValueError("恢复的视频任务版本无效。")
            return candidate
        if revision.artifact_type == "v2_video_job":
            if self.confirmed_movie_plan is None or not self.confirmed_movie_plan.confirmed:
                raise ValueError("恢复 V2 VideoJob 前必须存在已确认 MoviePlan。")
            candidate = self._v2_video_job_from(deepcopy(revision.payload))
            if not validate_video_job(candidate).valid or not candidate.confirmed:
                raise ValueError("恢复的 V2 VideoJob 版本无效。")
            return candidate
        if revision.artifact_type == "movie_ir":
            if self.confirmed_movie_plan is None or not self.confirmed_movie_plan.confirmed:
                raise ValueError("恢复 MovieIR 前必须存在已确认 MoviePlan。")
            candidate = MovieIR.from_dict(deepcopy(revision.payload))
            if candidate.source_movie_plan_id != self.confirmed_movie_plan.plan_id:
                raise ValueError("恢复的 MovieIR 来源不一致。")
            if candidate.source_film_ir_id and (
                self.film_ir is None or candidate.source_film_ir_id != self.film_ir.ir_id
            ):
                raise ValueError("恢复的 MovieIR 来源 FilmIR 不一致。")
            return candidate
        if revision.artifact_type == "film_ir":
            if self.confirmed_movie_plan is None or not self.confirmed_movie_plan.confirmed:
                raise ValueError("恢复 FilmIR 前必须存在已确认 MoviePlan。")
            candidate = FilmIR.from_dict(deepcopy(revision.payload))
            if candidate.source_movie_plan_id != self.confirmed_movie_plan.plan_id:
                raise ValueError("恢复的 FilmIR 来源 MoviePlan 不一致。")
            return candidate
        if revision.artifact_type == "execution_bundle":
            candidate = ExecutionBundle.from_dict(deepcopy(revision.payload))
            if not validate_execution_bundle(candidate).valid:
                raise ValueError("恢复的 ExecutionBundle 版本无效。")
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
            self.video_job = None
            self.render_manifest = None
            return self.story
        if artifact_type == "script":
            self.script = candidate
            self.stage = Stage.SCRIPT_REVIEW
            self.draft = None
            self.outline = None
            self.storyboard = None
            self.video_job = None
            self.render_manifest = None
            return self.script
        if artifact_type == "draft":
            self.draft = candidate
            self.outline, self.script = self.draft.outline, self.draft.script
            self.stage = Stage.SCRIPT_REVIEW
            self.storyboard = None
            self.video_job = None
            self.render_manifest = None
            return self.draft
        if artifact_type == "storyboard":
            self.storyboard = candidate
            self.stage = Stage.STORYBOARD_REVIEW
            self.render_manifest = None
            return self.storyboard
        if artifact_type == "video_job":
            self.video_job = candidate
            self.stage = Stage.RENDER_READY
            self.render_manifest = None
            return self.video_job
        if artifact_type == "v2_video_job":
            self.v2_video_job = candidate
            self.stage = Stage.VIDEO_JOB_COMPILED
            return self.v2_video_job
        if artifact_type == "movie_ir":
            self.movie_ir = candidate
            self.stage = Stage.MOVIE_IR_BUILT
            self.v2_video_job = None
            self._clear_execution_state()
            return self.movie_ir
        if artifact_type == "film_ir":
            self.film_ir = candidate
            self.stage = Stage.FILM_IR_BUILT
            self.movie_ir = None
            self.v2_video_job = None
            self._clear_execution_state()
            return self.film_ir
        if artifact_type == "execution_bundle":
            self.execution_bundle = candidate
            self.execution_plan = candidate.execution_plan
            self.current_execution_plan_id = candidate.execution_plan.execution_plan_id
            self.current_execution_plan_fingerprint = candidate.execution_plan.execution_plan_fingerprint
            self.current_execution_bundle_fingerprint = candidate.bundle_fingerprint
            self.stage = Stage.EXECUTION_PLAN_BUILT
            return self.execution_bundle
        raise ValueError("未知版本类型。")

    def _load_revisions(self, data: dict[str, Any]) -> None:
        raw_revisions = data.get("revisions", {})
        raw_cursors = data.get("revision_cursor", {})
        if not isinstance(raw_revisions, dict) or not isinstance(raw_cursors, dict):
            raise ValueError("revisions 和 revision_cursor 必须是 JSON 对象。")
        allowed_kinds = {
            "story",
            "script",
            "draft",
            "storyboard",
            "video_job",
            "movie_plan",
            "film_ir",
            "movie_ir",
            "v2_video_job",
            "execution_bundle",
        }
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
    def _v2_video_job_from(data: dict[str, Any]) -> V2VideoJob:
        if not isinstance(data, dict):
            raise ValueError("v2_video_job 必须是 JSON 对象。")
        allowed = {
            "job_id",
            "provider_key",
            "provider_prompt",
            "negative_prompt",
            "duration_seconds",
            "output_format",
            "aspect_ratio",
            "resolution",
            "fps",
            "references",
            "character_references",
            "continuity_references",
            "source_movie_plan_id",
            "source_movie_ir_id",
            "source_film_ir_id",
            "source_movie_plan_version",
            "source_movie_plan_fingerprint",
            "source_movie_plan_lineage_token",
            "source_film_ir_fingerprint",
            "source_movie_ir_fingerprint",
            "video_job_fingerprint",
            "schema_version",
            "compiler_version",
            "provider_profile",
            "execution_units",
            "metadata",
            "created_at",
            "confirmed",
        }
        unexpected = sorted(set(data) - allowed)
        if unexpected:
            raise ValueError("v2_video_job 包含未允许字段：" + ", ".join(unexpected))
        payload = {key: deepcopy(value) for key, value in data.items() if key in allowed}
        payload["job_id"] = str(payload.get("job_id", ""))
        payload["provider_key"] = str(payload.get("provider_key", ""))
        payload["provider_prompt"] = str(payload.get("provider_prompt", ""))
        payload["negative_prompt"] = str(payload.get("negative_prompt", ""))
        payload["duration_seconds"] = float(payload.get("duration_seconds", 0))
        payload["output_format"] = str(payload.get("output_format", ""))
        payload["aspect_ratio"] = str(payload.get("aspect_ratio", "16:9"))
        payload["resolution"] = str(payload.get("resolution", ""))
        payload["fps"] = (
            None if payload.get("fps") is None else float(payload.get("fps"))
        )
        for key in ("references", "character_references", "continuity_references"):
            payload[key] = tuple(str(item) for item in payload.get(key, []))
        payload["source_movie_plan_id"] = str(payload.get("source_movie_plan_id", ""))
        payload["source_movie_ir_id"] = str(payload.get("source_movie_ir_id", ""))
        payload["source_film_ir_id"] = str(payload.get("source_film_ir_id", ""))
        payload["source_movie_plan_version"] = int(payload.get("source_movie_plan_version", 0) or 0)
        payload["source_movie_plan_fingerprint"] = str(payload.get("source_movie_plan_fingerprint", ""))
        payload["source_movie_plan_lineage_token"] = str(payload.get("source_movie_plan_lineage_token", ""))
        payload["source_film_ir_fingerprint"] = str(payload.get("source_film_ir_fingerprint", ""))
        payload["source_movie_ir_fingerprint"] = str(payload.get("source_movie_ir_fingerprint", ""))
        payload["video_job_fingerprint"] = str(payload.get("video_job_fingerprint", ""))
        payload["schema_version"] = str(payload.get("schema_version", "v2-video-job/1"))
        payload["compiler_version"] = str(payload.get("compiler_version", ""))
        payload["provider_profile"] = str(payload.get("provider_profile", ""))
        payload["execution_units"] = tuple(
            dict(item) for item in payload.get("execution_units", [])
        )
        payload["metadata"] = dict(payload.get("metadata", {}))
        payload["created_at"] = str(payload.get("created_at", ""))
        payload["confirmed"] = GuidedStorySession._strict_bool(
            payload.get("confirmed", False),
            "v2_video_job.confirmed",
        )
        return V2VideoJob(**payload)

    @staticmethod
    def _video_job_from(data: dict[str, Any]) -> VideoJob:
        if not isinstance(data, dict):
            raise ValueError("video_job 必须是 JSON 对象。")
        fields = VideoJob.__dataclass_fields__
        payload = {key: deepcopy(value) for key, value in data.items() if key in fields}
        payload["title"] = str(payload.get("title", ""))
        payload["prompt"] = str(payload.get("prompt", ""))
        payload["target_seconds"] = int(payload.get("target_seconds", 0))
        payload["negative_prompt"] = str(payload.get("negative_prompt", ""))
        payload["aspect_ratio"] = str(payload.get("aspect_ratio", "16:9"))
        payload["dialogue"] = str(payload.get("dialogue", ""))
        payload["narration"] = str(payload.get("narration", ""))
        payload["visual_style"] = str(payload.get("visual_style", ""))
        payload["reference_image_paths"] = [
            str(item) for item in payload.get("reference_image_paths", [])
        ]
        payload["initial_frame_path"] = str(payload.get("initial_frame_path", ""))
        payload["initial_frame_url"] = sanitize_remote_url(
            str(payload.get("initial_frame_url", ""))
        )
        payload["metadata"] = dict(payload.get("metadata", {}))
        payload["job_id"] = str(payload.get("job_id", ""))
        payload["confirmed"] = GuidedStorySession._strict_bool(
            payload.get("confirmed", False),
            "video_job.confirmed",
        )
        return VideoJob(**payload)

    @staticmethod
    def _storyboard_from(
        data: dict[str, Any],
        *,
        normalize_legacy_narration: bool = False,
    ) -> StoryboardPlan:
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
        normalize_narration_timeline(
            plan,
            deduplicate_legacy=normalize_legacy_narration,
        )
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
        mode = str(payload.get("continuity_mode", "independent"))
        payload["inherit_previous_frame"] = GuidedStorySession._strict_bool(
            payload.get("inherit_previous_frame", mode == "same_scene_chain"),
            "video_artifact.inherit_previous_frame",
        )
        payload["transition_type"] = str(
            payload.get("transition_type")
            or (
                "continuous_action"
                if mode == "same_scene_chain"
                else "scene_change"
                if mode == "new_scene_reference"
                else "independent"
            )
        )
        payload["transition_reason"] = str(payload.get("transition_reason", ""))
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
        payload["source_action"] = str(
            payload.get("source_action") or payload.get("action", "")
        )
        payload["retake_instruction"] = str(payload.get("retake_instruction", ""))
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
        mode = str(payload.get("continuity_mode", "independent"))
        payload["inherit_previous_frame"] = GuidedStorySession._strict_bool(
            payload.get("inherit_previous_frame", mode == "same_scene_chain"),
            "storyboard_shot.inherit_previous_frame",
        )
        payload["transition_type"] = str(
            payload.get("transition_type")
            or (
                "continuous_action"
                if mode == "same_scene_chain"
                else "scene_change"
                if mode == "new_scene_reference"
                else "independent"
            )
        )
        payload["transition_reason"] = str(payload.get("transition_reason", ""))
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
