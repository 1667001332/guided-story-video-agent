"""Invalidate active downstream state after a MoviePlan change."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_NONE_FIELDS = {
    "film_ir",
    "movie_ir",
    "v2_video_job",
    "render_manifest",
    "execution_plan",
    "execution_bundle",
    "execution_run",
}
_LIST_FIELDS = {
    "film_ir_revisions",
    "film_ir_build_diagnostics",
    "film_ir_validation_issues",
    "film_ir_pass_diagnostics",
    "creative_pass_diagnostics",
    "creative_analysis_results",
    "creative_analysis_diagnostics",
    "creative_analysis_artifacts",
    "creative_optimizer_suggestions",
    "creative_optimizer_candidates",
    "creative_optimizer_diagnostics",
    "creative_revision_requests",
    "creative_revision_request_history",
    "revision_candidates",
    "revision_diffs",
    "revision_decisions",
    "revision_guard_diagnostics",
    "director_revision_adapter_results",
    "director_revision_contexts",
    "guarded_revision_results",
    "movie_ir_revisions",
    "movie_ir_build_diagnostics",
    "movie_ir_validation_issues",
    "movie_ir_pass_diagnostics",
    "movie_ir_optimizer_diagnostics",
    "film_ir_optimizer_diagnostics",
    "director_revision_history",
    "source_lineage_diagnostics",
    "stale_lineage_diagnostics",
    "execution_runtime_diagnostics",
    "provider_jobs",
    "runtime_artifacts",
}
_DICT_FIELDS = {
    "film_ir_build_metadata",
    "creative_analysis_metrics",
    "creative_optimizer_result",
    "movie_ir_build_metadata",
    "v2_compile_metadata",
}
_SCALAR_FIELDS = {
    "creative_revision_stop_reason",
    "director_revision_stop_reason",
    "director_revision_last_stop_reason",
    "revision_active_candidate_id",
    "revision_accepted_movie_plan_id",
    "revision_rollback_movie_plan_id",
    "director_revision_attempt_count",
    "current_film_ir_id",
    "current_movie_ir_id",
    "current_video_job_id",
    "current_execution_plan_id",
    "current_execution_plan_fingerprint",
    "current_execution_bundle_fingerprint",
    "current_execution_run_id",
    "execution_runtime_status",
    "latest_execution_checkpoint_id",
}
_DIAGNOSTIC_FIELDS = {
    "v2_compile_diagnostics",
    "execution_plan_diagnostics",
}


@dataclass(frozen=True, slots=True)
class DownstreamInvalidationResult:
    invalidated: tuple[str, ...] = ()
    preserved: tuple[str, ...] = ()
    reason: str = ""
    succeeded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "invalidated": list(self.invalidated),
            "preserved": list(self.preserved),
            "reason": self.reason,
            "succeeded": bool(self.succeeded),
        }


def _artifact_identifier(value: Any, field_name: str) -> str:
    if value is None:
        return field_name
    for key in ("ir_id", "job_id", "artifact_id", "manifest_id", "execution_run_id", "bundle_fingerprint", "id"):
        item = getattr(value, key, None)
        if item:
            return str(item)
        if isinstance(value, dict) and value.get(key):
            return str(value[key])
    if isinstance(value, (list, tuple)):
        return f"{field_name}[{len(value)}]"
    return field_name


def _mark_stale(session: Any, field_name: str, value: Any, reason: str, source_id: str | None) -> None:
    if value is None or value == [] or value == {}:
        return
    stale = getattr(session, "stale_artifacts", None)
    if stale is None:
        stale = []
        setattr(session, "stale_artifacts", stale)
    stale.append(
        {
            "artifact_type": field_name,
            "artifact_id": _artifact_identifier(value, field_name),
            "reason": reason,
            "source_movie_plan_id": source_id,
        }
    )


def _mark_execution_stale(session: Any, field_name: str, value: Any, reason: str, source_id: str | None) -> None:
    if value is None or value == [] or value == {}:
        return
    stale = getattr(session, "stale_execution_artifacts", None)
    if stale is None:
        stale = []
        setattr(session, "stale_execution_artifacts", stale)
    identifier = _artifact_identifier(value, field_name)
    if isinstance(value, dict):
        identifier = str(value.get("execution_plan_id") or value.get("bundle_fingerprint") or identifier)
    stale.append(
        {
            "artifact_type": field_name,
            "artifact_id": identifier,
            "reason": reason,
            "source_movie_plan_id": source_id,
        }
    )


def invalidate_downstream_after_movie_plan_change(
    session: Any,
    reason: str,
    *,
    source_movie_plan_id: str | None = None,
) -> DownstreamInvalidationResult:
    """Clear active V2 outputs while retaining explicit history records."""

    invalidated: list[str] = []
    preserved: list[str] = []
    for field_name in sorted(_NONE_FIELDS | _LIST_FIELDS | _DICT_FIELDS | _SCALAR_FIELDS | _DIAGNOSTIC_FIELDS):
        if not hasattr(session, field_name):
            preserved.append(field_name)
            continue
        value = getattr(session, field_name)
        if field_name in {"execution_plan", "execution_bundle", "execution_run"}:
            _mark_execution_stale(session, field_name, value, reason, source_movie_plan_id)
        if field_name not in _SCALAR_FIELDS and field_name not in {
            "source_lineage_diagnostics",
            "stale_lineage_diagnostics",
        }:
            _mark_stale(session, field_name, value, reason, source_movie_plan_id)
        if field_name in _NONE_FIELDS:
            setattr(session, field_name, None)
        elif field_name in _LIST_FIELDS or field_name in _DIAGNOSTIC_FIELDS:
            setattr(session, field_name, [])
        elif field_name in _DICT_FIELDS:
            setattr(session, field_name, None if field_name == "creative_optimizer_result" else {})
        elif field_name == "director_revision_attempt_count":
            setattr(session, field_name, 0)
        else:
            setattr(session, field_name, None)
        invalidated.append(field_name)

    # Legacy Provider state is not owned by the V2 compiler.  Preserve it, but
    # mark it stale if an embedding application has attached such fields.
    for field_name in ("provider_job", "artifact", "artifacts", "video_job"):
        if hasattr(session, field_name):
            value = getattr(session, field_name)
            if value is not None and value != []:
                _mark_stale(session, field_name, value, reason, source_movie_plan_id)
                preserved.append(field_name)
    return DownstreamInvalidationResult(
        invalidated=tuple(invalidated),
        preserved=tuple(preserved),
        reason=reason,
        succeeded=True,
    )


__all__ = [
    "DownstreamInvalidationResult",
    "invalidate_downstream_after_movie_plan_change",
]
