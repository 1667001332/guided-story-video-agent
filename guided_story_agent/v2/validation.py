"""Pure validation for V2 plans and execution requests.

The functions in this module are intentionally side-effect free.  They return
errors; they never repair, normalize, allocate, or rewrite a plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

from .execution import ProviderCapabilities, VideoJob
from .film_ir import FilmIR
from .ir import MovieIR
from .models import CreativeBrief, MoviePlan, as_plain_data


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    context: str = ""

    @property
    def valid(self) -> bool:
        return not self.errors

    def feedback(self) -> str:
        if self.valid:
            return ""
        prefix = f"{self.context}: " if self.context else ""
        return prefix + "; ".join(self.errors)


ValidationSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Structured validation feedback for IR boundaries."""

    code: str
    message: str
    path: str = ""
    severity: ValidationSeverity = "error"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Validation result that never mutates the input IR."""

    ok: bool
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


class FilmIRValidator:
    """Validate FilmIR without lowering, repairing, or inventing content."""

    def validate(self, film_ir: FilmIR) -> ValidationResult:
        if not isinstance(film_ir, FilmIR):
            return ValidationResult(
                ok=False,
                issues=(ValidationIssue("invalid_film_ir", "输入不是 FilmIR。"),),
            )
        issues: list[ValidationIssue] = []
        _provider_field_issues(issues, film_ir.to_dict(), "film_ir")
        issues.extend(
            _prompt_leakage_issues(
                film_ir.to_dict(),
                "film_ir",
                style_warning=False,
            )
        )
        if not film_ir.ir_id.strip():
            issues.append(ValidationIssue("missing_ir_id", "FilmIR 缺少 ir_id。", "ir_id"))
        if not film_ir.source_movie_plan_id.strip():
            issues.append(
                ValidationIssue(
                    "missing_source_movie_plan_id",
                    "FilmIR 必须包含 source_movie_plan_id。",
                    "source_movie_plan_id",
                )
            )
        if not film_ir.source_story_plan_id.strip():
            issues.append(
                ValidationIssue(
                    "missing_source_story_plan_id",
                    "FilmIR 必须包含 source_story_plan_id。",
                    "source_story_plan_id",
                )
            )
        if not film_ir.source_director_plan_id.strip():
            issues.append(
                ValidationIssue(
                    "missing_source_director_plan_id",
                    "FilmIR 必须包含 source_director_plan_id。",
                    "source_director_plan_id",
                )
            )
        if not film_ir.beats:
            issues.append(
                ValidationIssue(
                    "missing_film_beats",
                    "FilmIR 必须包含 film-level beats。",
                    "beats",
                )
            )
        if not film_ir.shots:
            issues.append(
                ValidationIssue("missing_shots", "FilmIR 必须包含可追踪的 shots。", "shots")
            )
        shot_by_id = {shot.shot_id: shot for shot in film_ir.shots}
        covered_shots: list[str] = []
        shot_orders = [shot.order for shot in film_ir.shots]
        if shot_orders != list(range(1, len(shot_orders) + 1)):
            issues.append(
                ValidationIssue(
                    "invalid_shot_order",
                    "FilmIR shot order 必须从 1 连续递增。",
                    "shots",
                )
            )
        shot_total = 0.0
        for index, shot in enumerate(film_ir.shots):
            path = f"shots[{index}]"
            shot_total += shot.duration_seconds
            for field_name, value in (
                ("shot_id", shot.shot_id),
                ("scene_id", shot.scene_id),
                ("purpose", shot.purpose),
                ("visible_action", shot.visible_action),
                ("subject", shot.subject),
                ("camera", shot.camera),
                ("motion", shot.motion),
                ("lighting", shot.lighting),
                ("composition", shot.composition),
            ):
                if not value.strip():
                    issues.append(
                        ValidationIssue(
                            "missing_shot_field",
                            f"{field_name} 不能为空。",
                            f"{path}.{field_name}",
                        )
                    )
            if not shot.characters:
                issues.append(
                    ValidationIssue(
                        "missing_character_anchor",
                        "shot 必须声明 characters。",
                        f"{path}.characters",
                    )
                )
            if not shot.continuity_anchors:
                issues.append(
                    ValidationIssue(
                        "missing_continuity_anchor",
                        "shot 必须声明 continuity_anchors。",
                        f"{path}.continuity_anchors",
                    )
                )
            if not shot.required_visual_evidence:
                issues.append(
                    ValidationIssue(
                        "missing_required_evidence",
                        "shot 必须声明 required_visual_evidence。",
                        f"{path}.required_visual_evidence",
                    )
                )
            if not shot.acceptance_criteria:
                issues.append(
                    ValidationIssue(
                        "missing_acceptance_criteria",
                        "shot 必须声明 acceptance_criteria。",
                        f"{path}.acceptance_criteria",
                    )
                )
        if not math.isclose(shot_total, film_ir.target_duration_seconds, rel_tol=0.0, abs_tol=1e-6):
            issues.append(
                ValidationIssue(
                    "invalid_total_duration",
                    "FilmIR shot 总时长必须等于 target_duration_seconds。",
                    "shots",
                )
            )
        beat_orders = [beat.order for beat in film_ir.beats]
        if beat_orders != list(range(1, len(beat_orders) + 1)):
            issues.append(
                ValidationIssue(
                    "invalid_beat_order",
                    "FilmIR beat order 必须从 1 连续递增。",
                    "beats",
                )
            )
        for index, beat in enumerate(film_ir.beats):
            path = f"beats[{index}]"
            for field_name, value in (
                ("beat_id", beat.beat_id),
                ("scene_id", beat.scene_id),
                ("dramatic_purpose", beat.dramatic_purpose),
                ("narrative_function", beat.narrative_function),
                ("viewer_state_before", beat.viewer_state_before),
                ("viewer_state_after", beat.viewer_state_after),
                ("required_audience_understanding", beat.required_audience_understanding),
                ("visual_focus", beat.visual_focus),
            ):
                if not value.strip():
                    issues.append(
                        ValidationIssue(
                            "missing_film_decision",
                            f"{field_name} 不能为空。",
                            f"{path}.{field_name}",
                        )
                    )
            if not beat.shot_ids:
                issues.append(
                    ValidationIssue(
                        "missing_shot_reference",
                        "每个 FilmIR beat 必须引用 shot_id。",
                        f"{path}.shot_ids",
                    )
                )
            if not beat.required_evidence:
                issues.append(
                    ValidationIssue(
                        "missing_required_evidence",
                        "每个 FilmIR beat 必须包含 required_evidence。",
                        f"{path}.required_evidence",
                    )
                )
            if not beat.acceptance_criteria:
                issues.append(
                    ValidationIssue(
                        "missing_acceptance_criteria",
                        "每个 FilmIR beat 必须包含 acceptance_criteria。",
                        f"{path}.acceptance_criteria",
                    )
                )
            if not _positive_finite(beat.duration_seconds):
                issues.append(
                    ValidationIssue(
                        "invalid_beat_duration",
                        "beat duration 必须是正数。",
                        f"{path}.duration_seconds",
                    )
                )
            for shot_id in beat.shot_ids:
                covered_shots.append(shot_id)
                shot = shot_by_id.get(shot_id)
                if shot is None:
                    issues.append(
                        ValidationIssue(
                            "untracked_shot_reference",
                            "beat 引用了不存在的 shot_id。",
                            f"{path}.shot_ids",
                        )
                    )
                elif shot.scene_id != beat.scene_id:
                    issues.append(
                        ValidationIssue(
                            "scene_reference_mismatch",
                            "beat.scene_id 必须与引用 shot 的 scene_id 一致。",
                            f"{path}.scene_id",
                        )
                    )
        if covered_shots and (
            set(covered_shots) != set(shot_by_id) or len(covered_shots) != len(set(covered_shots))
        ):
            issues.append(
                ValidationIssue(
                    "invalid_beat_coverage",
                    "FilmIR beats 必须恰好覆盖全部 shot。",
                    "beats",
                )
            )
        issues.extend(_validate_film_timeline(film_ir))
        unique_issues = tuple(_unique_issues(issues))
        return ValidationResult(
            ok=not any(issue.severity == "error" for issue in unique_issues),
            issues=unique_issues,
        )


class MovieIRValidator:
    """Validate executable, provider-neutral MovieIR structure."""

    def validate(self, movie_ir: MovieIR) -> ValidationResult:
        if not isinstance(movie_ir, MovieIR):
            return ValidationResult(
                ok=False,
                issues=(ValidationIssue("invalid_movie_ir", "输入不是 MovieIR。"),),
            )
        issues: list[ValidationIssue] = []
        payload = movie_ir.to_dict()
        _provider_field_issues(issues, payload, "movie_ir")
        if not movie_ir.source_movie_plan_id.strip():
            issues.append(
                ValidationIssue(
                    "missing_source_movie_plan_id",
                    "MovieIR 必须包含 source_movie_plan_id。",
                    "source_movie_plan_id",
                )
            )
        if not movie_ir.source_film_ir_id.strip():
            issues.append(
                ValidationIssue(
                    "missing_source_film_ir_id",
                    "MovieIR 必须包含 source_film_ir_id。",
                    "source_film_ir_id",
                )
            )
        if not movie_ir.shots:
            issues.append(
                ValidationIssue(
                    "missing_execution_units",
                    "MovieIR 必须包含 shot-level execution units。",
                    "shots",
                )
            )
        shot_by_id = {shot.shot_id: shot for shot in movie_ir.shots}
        orders = [shot.order for shot in movie_ir.shots]
        if orders != list(range(1, len(orders) + 1)):
            issues.append(
                ValidationIssue(
                    "invalid_shot_order",
                    "MovieIR shot order 必须从 1 连续递增。",
                    "shots",
                )
            )
        for index, shot in enumerate(movie_ir.shots):
            path = f"shots[{index}]"
            if not shot.visible_action.strip():
                issues.append(
                    ValidationIssue(
                        "missing_visible_action",
                        "每个 shot 必须有 visible_action。",
                        f"{path}.visible_action",
                    )
                )
            if not _positive_finite(shot.duration_seconds):
                issues.append(
                    ValidationIssue(
                        "invalid_shot_duration",
                        "shot duration 必须大于 0。",
                        f"{path}.duration_seconds",
                    )
                )
            if not shot.required_visual_evidence:
                issues.append(
                    ValidationIssue(
                        "missing_visual_evidence",
                        "每个 shot 必须有 required_visual_evidence。",
                        f"{path}.required_visual_evidence",
                    )
                )
            if not shot.continuity_anchors:
                issues.append(
                    ValidationIssue(
                        "missing_continuity_anchor",
                        "每个 shot 必须有可追踪的 continuity anchor。",
                        f"{path}.continuity_anchors",
                    )
                )
            if not str(shot.metadata.get("source_film_beat_id", "")).strip():
                issues.append(
                    ValidationIssue(
                        "missing_film_trace",
                        "每个 shot 必须能追踪到 FilmIR beat。",
                        f"{path}.metadata.source_film_beat_id",
                    )
                )
        issues.extend(_validate_movie_timeline(movie_ir, shot_by_id))
        issues.extend(_prompt_leakage_issues(payload))
        unique_issues = tuple(_unique_issues(issues))
        return ValidationResult(
            ok=not any(issue.severity == "error" for issue in unique_issues),
            issues=unique_issues,
        )


_FORBIDDEN_IR_KEYS = {
    "provider",
    "provider_key",
    "provider_name",
    "provider_profile",
    "model",
    "prompt",
    "provider_prompt",
    "negative_prompt",
    "api",
    "api_key",
    "endpoint",
    "payload",
    "api_payload",
    "http_payload",
    "task_id",
    "task",
    "task_url",
    "provider_task_id",
    "poll_url",
    "download_url",
    "billing",
}
_PROMPT_STUFFING_TERMS = (
    "masterpiece",
    "best quality",
    "ultra realistic",
    "ultra-realistic",
    "8k",
    "4k highly detailed",
)


def _provider_field_issues(
    issues: list[ValidationIssue],
    value: Any,
    path: str,
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_IR_KEYS:
                issues.append(
                    ValidationIssue(
                        "provider_field_in_ir",
                        f"IR 不允许 Provider/API 字段：{key_text}。",
                        f"{path}.{key_text}",
                    )
                )
            _provider_field_issues(issues, child, f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _provider_field_issues(issues, child, f"{path}[{index}]")


def _prompt_leakage_issues(
    value: Any,
    path: str = "movie_ir",
    *,
    style_warning: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if isinstance(value, dict):
        for key, child in value.items():
            issues.extend(
                _prompt_leakage_issues(
                    child,
                    f"{path}.{key}",
                    style_warning=style_warning,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(
                _prompt_leakage_issues(
                    child,
                    f"{path}[{index}]",
                    style_warning=style_warning,
                )
            )
    elif isinstance(value, str):
        lowered = value.lower()
        for term in _PROMPT_STUFFING_TERMS:
            if term in lowered:
                severity: ValidationSeverity = (
                    "warning"
                    if style_warning and path.endswith(".visual_style")
                    else "error"
                )
                issues.append(
                    ValidationIssue(
                        "prompt_leakage",
                        f"发现疑似 Provider prompt stuffing：{term}。",
                        path,
                        severity,
                    )
                )
                break
    return issues


def _validate_film_timeline(film_ir: FilmIR) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    timeline = film_ir.beat_timeline
    if len(timeline) != len(film_ir.beats):
        issues.append(
            ValidationIssue(
                "timeline_coverage_mismatch",
                "FilmIR beat_timeline 必须覆盖全部 beats。",
                "beat_timeline",
            )
        )
        return issues
    cursor = 0.0
    for index, entry in enumerate(timeline):
        path = f"beat_timeline[{index}]"
        if entry.order != index + 1:
            issues.append(
                ValidationIssue(
                    "invalid_timeline_order",
                    "beat_timeline order 必须连续。",
                    f"{path}.order",
                )
            )
        if not _positive_finite(entry.duration_seconds):
            issues.append(
                ValidationIssue(
                    "invalid_timeline_duration",
                    "beat_timeline duration 必须为正数。",
                    f"{path}.duration_seconds",
                )
            )
        if not math.isclose(entry.start_seconds, cursor, rel_tol=0.0, abs_tol=1e-6):
            issues.append(
                ValidationIssue(
                    "invalid_timeline_continuity",
                    "beat_timeline 不能重叠或断裂。",
                    f"{path}.start_seconds",
                )
            )
        cursor = entry.start_seconds + entry.duration_seconds
    if not math.isclose(cursor, film_ir.target_duration_seconds, rel_tol=0.0, abs_tol=1e-6):
        issues.append(
            ValidationIssue(
                "invalid_total_duration",
                "FilmIR beat_timeline 总时长必须等于 target_duration_seconds。",
                "beat_timeline",
            )
        )
    return issues


def _validate_movie_timeline(
    movie_ir: MovieIR,
    shot_by_id: dict[str, Any],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    timeline = movie_ir.timeline
    if len(timeline) != len(movie_ir.shots):
        issues.append(
            ValidationIssue(
                "timeline_coverage_mismatch",
                "MovieIR timeline 必须覆盖全部 shots。",
                "timeline",
            )
        )
        return issues
    cursor = 0.0
    seen_shots: set[str] = set()
    for index, entry in enumerate(timeline):
        path = f"timeline[{index}]"
        if entry.order != index + 1:
            issues.append(
                ValidationIssue(
                    "invalid_timeline_order",
                    "MovieIR timeline order 必须连续。",
                    f"{path}.order",
                )
            )
        if entry.shot_id in seen_shots or entry.shot_id not in shot_by_id:
            issues.append(
                ValidationIssue(
                    "invalid_timeline_shot_reference",
                    "timeline 必须为每个 shot 提供唯一可追踪条目。",
                    f"{path}.shot_id",
                )
            )
        seen_shots.add(entry.shot_id)
        if not _positive_finite(entry.duration_seconds):
            issues.append(
                ValidationIssue(
                    "invalid_timeline_duration",
                    "timeline duration 必须为正数。",
                    f"{path}.duration_seconds",
                )
            )
        if not math.isclose(entry.start_seconds, cursor, rel_tol=0.0, abs_tol=1e-6):
            issues.append(
                ValidationIssue(
                    "invalid_timeline_continuity",
                    "MovieIR timeline 不能重叠或断裂。",
                    f"{path}.start_seconds",
                )
            )
        shot = shot_by_id.get(entry.shot_id)
        if shot is not None and not math.isclose(
            entry.duration_seconds,
            shot.duration_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            issues.append(
                ValidationIssue(
                    "timeline_duration_mismatch",
                    "timeline duration 必须与对应 shot duration 一致。",
                    f"{path}.duration_seconds",
                )
            )
        cursor = entry.start_seconds + entry.duration_seconds
    if seen_shots != set(shot_by_id):
        issues.append(
            ValidationIssue(
                "timeline_coverage_mismatch",
                "MovieIR timeline 必须覆盖每个 shot 一次。",
                "timeline",
            )
        )
    if not math.isclose(cursor, movie_ir.target_duration_seconds, rel_tol=0.0, abs_tol=1e-6):
        issues.append(
            ValidationIssue(
                "invalid_total_duration",
                "MovieIR timeline 总时长必须等于 target_duration_seconds。",
                "timeline",
            )
        )
    return issues


def _positive_finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _unique_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.code, issue.message, issue.path, issue.severity)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def validate_movie_plan(
    plan: MoviePlan,
    brief: CreativeBrief,
) -> ValidationReport:
    errors: list[str] = []
    try:
        brief.validate()
    except ValueError as exc:
        errors.append(f"brief invalid: {exc}")
    if not isinstance(plan, MoviePlan):
        return ValidationReport(("director output is not a MoviePlan",), context="movie_plan")

    # Phase 4A keeps the nested plans provider-neutral as well.  Reuse the
    # same fail-closed boundary used by IR validators, without repairing the
    # legacy aggregate or inventing missing creative decisions.
    nested_payload = {
        "story_plan": as_plain_data(plan.story_plan),
        "director_plan": as_plain_data(plan.director_plan),
    }
    nested_issues: list[ValidationIssue] = []
    _provider_field_issues(nested_issues, nested_payload, "movie_plan")
    nested_issues.extend(_prompt_leakage_issues(nested_payload, "movie_plan"))
    errors.extend(
        f"{issue.path}: {issue.message}"
        for issue in nested_issues
        if issue.severity == "error"
    )

    if not plan.plan_id.strip():
        errors.append("plan_id is required")
    if not plan.visual_style.strip():
        errors.append("visual_style is required")
    for label, value in (
        ("story.title", plan.story.title),
        ("story.logline", plan.story.logline),
        ("story.synopsis", plan.story.synopsis),
        ("script.title", plan.script.title),
    ):
        if not str(value or "").strip():
            errors.append(f"{label} is required")

    scenes = plan.script.scenes
    scene_ids = [scene.scene_id.strip() for scene in scenes]
    if not scenes:
        errors.append("script must contain at least one scene")
    if any(not scene_id for scene_id in scene_ids):
        errors.append("every scene must have a non-empty scene_id")
    if len(scene_ids) != len(set(scene_ids)):
        errors.append("scene_id values must be unique")
    expected = set(scene_ids)

    for scene in scenes:
        for field_name, value in (
            ("goal", scene.goal),
            ("emotion", scene.emotion),
            ("importance", scene.importance),
            ("camera_language", scene.camera_language),
            ("motion_type", scene.motion_type),
            ("location", scene.location),
            ("transition", scene.transition),
            ("timing_reason", scene.timing_reason),
        ):
            if not str(value or "").strip():
                errors.append(f"scene {scene.scene_id}: {field_name} is required")
        if not _is_finite_number(scene.estimated_duration_weight) or (
            float(scene.estimated_duration_weight) < 0
        ):
            errors.append(f"scene {scene.scene_id}: estimated_duration_weight is invalid")
        if not _is_finite_number(scene.minimum_duration) or float(scene.minimum_duration) <= 0:
            errors.append(f"scene {scene.scene_id}: minimum_duration must be positive")


    timing_entries = plan.timing_plan.entries
    timing_ids = [entry.scene_id.strip() for entry in timing_entries]
    if set(timing_ids) != expected or len(timing_ids) != len(expected):
        errors.append("TimingPlan scene IDs must match Script scene IDs exactly")
    if not _is_finite_number(plan.timing_plan.target_duration_seconds) or not math.isclose(
        float(plan.timing_plan.target_duration_seconds),
        float(brief.target_duration_seconds),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        errors.append("TimingPlan target duration must equal CreativeBrief target duration")
    if not math.isclose(
        plan.timing_plan.declared_total_seconds,
        plan.timing_plan.target_duration_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        errors.append("TimingPlan declared scene durations do not equal its target duration")
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    for entry in timing_entries:
        if not _is_finite_number(entry.duration_seconds) or float(entry.duration_seconds) <= 0:
            errors.append(f"scene {entry.scene_id}: duration_seconds must be positive")
            continue
        scene = scene_by_id.get(entry.scene_id)
        if scene is not None and float(entry.duration_seconds) < float(scene.minimum_duration):
            errors.append(
                f"scene {entry.scene_id}: declared duration is below the LLM minimum_duration"
            )
        if not entry.reason.strip():
            errors.append(f"scene {entry.scene_id}: timing reason is required")

    _check_references(errors, "ScenePlan", [item.scene_id for item in plan.scene_plan.scenes], expected)
    _check_references(
        errors,
        "CameraPlan",
        [item.scene_id for item in plan.camera_plan.instructions],
        expected,
    )
    _check_references(
        errors,
        "ContinuityPlan",
        [item.scene_id for item in plan.continuity_plan.entries],
        expected,
    )
    _check_references(
        errors,
        "NarrationPlan",
        [item.scene_id for item in plan.narration_plan.segments],
        expected,
        allow_empty=True,
    )
    return ValidationReport(tuple(errors), context="movie_plan")


def validate_video_job(
    job: VideoJob,
    capabilities: ProviderCapabilities | None = None,
) -> ValidationReport:
    errors: list[str] = []
    if not isinstance(job, VideoJob):
        return ValidationReport(("execution input is not a VideoJob",), context="video_job")
    for name, value in (("job_id", job.job_id), ("provider_key", job.provider_key)):
        if not str(value or "").strip():
            errors.append(f"{name} is required")
    if not job.provider_prompt.strip():
        errors.append("provider_prompt must come from the Compiler")
    if not _is_finite_number(job.duration_seconds) or float(job.duration_seconds) <= 0:
        errors.append("duration_seconds must be positive")
    if not job.output_format.strip():
        errors.append("output_format is required")
    if capabilities is not None:
        if capabilities.provider_key != job.provider_key:
            errors.append("VideoJob provider_key does not match ProviderCapabilities")
        if (
            capabilities.min_duration_seconds is not None
            and job.duration_seconds < capabilities.min_duration_seconds
        ):
            errors.append("VideoJob duration is below Provider minimum")
        if (
            capabilities.max_duration_seconds is not None
            and job.duration_seconds > capabilities.max_duration_seconds
        ):
            errors.append("VideoJob duration exceeds Provider maximum")
        if (
            capabilities.output_formats
            and job.output_format not in capabilities.output_formats
        ):
            errors.append("VideoJob output_format is not supported by Provider")
        if (
            capabilities.supported_aspect_ratios
            and job.aspect_ratio not in capabilities.supported_aspect_ratios
        ):
            errors.append("VideoJob aspect_ratio is not supported by Provider")
        if (
            capabilities.supported_resolutions
            and job.resolution
            and job.resolution not in capabilities.supported_resolutions
        ):
            errors.append("VideoJob resolution is not supported by Provider")
        if (
            capabilities.supported_fps
            and job.fps is not None
            and job.fps not in capabilities.supported_fps
        ):
            errors.append("VideoJob fps is not supported by Provider")
    return ValidationReport(tuple(errors), context="video_job")


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check_references(
    errors: list[str],
    label: str,
    values: list[str],
    expected: set[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not values and allow_empty:
        return
    if set(values) != expected or len(values) != len(set(values)):
        errors.append(f"{label} scene IDs must match Script scene IDs")
