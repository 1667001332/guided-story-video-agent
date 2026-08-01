"""Source lineage and stale-artifact checks for the V2 compiler chain.

Lineage is deliberately provider-neutral.  It proves that an artifact was
lowered from the current MoviePlan/FilmIR/MovieIR without inspecting or
constructing any Provider payload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceLineage:
    """Serializable source identifiers carried by a V2 artifact."""

    artifact_type: str = ""
    artifact_id: str = ""
    source_movie_plan_id: str = ""
    source_story_plan_id: str = ""
    source_director_plan_id: str = ""
    source_film_ir_id: str = ""
    source_movie_ir_id: str = ""

    @property
    def movie_plan_id(self) -> str:
        """Compatibility alias for callers that use semantic names."""

        return self.source_movie_plan_id

    @property
    def film_ir_id(self) -> str:
        return self.source_film_ir_id

    @property
    def movie_ir_id(self) -> str:
        return self.source_movie_ir_id

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class StaleArtifactDiagnostic:
    """One fail-closed lineage problem and its required recovery action."""

    code: str
    message: str
    artifact_type: str
    path: str = ""
    severity: str = "error"
    action: str = ""
    expected: str = ""
    actual: str = ""

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class LineageCheckResult:
    """Result of checking one artifact or a complete V2 Session."""

    artifact_type: str
    valid: bool
    diagnostics: tuple[StaleArtifactDiagnostic, ...] = ()
    lineage: SourceLineage | None = None

    @property
    def ok(self) -> bool:
        return self.valid

    @property
    def status(self) -> str:
        if self.valid:
            return "fresh"
        if any(item.code.endswith("unknown_lineage") for item in self.diagnostics):
            return "unknown"
        return "stale"

    @property
    def action(self) -> str:
        return next((item.action for item in self.diagnostics if item.action), "")

    @property
    def is_stale(self) -> bool:
        return self.status == "stale"

    @property
    def is_unknown(self) -> bool:
        return self.status == "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "valid": self.valid,
            "status": self.status,
            "action": self.action,
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


class SourceLineageGuard:
    """Fail-closed checks for source identity across V2 lowering stages."""

    def check_film_ir(
        self,
        film_ir: Any,
        current_movie_plan_id: str | None = None,
        current_story_plan_id: str | None = None,
        current_director_plan_id: str | None = None,
        current_film_ir_id: str | None = None,
    ) -> LineageCheckResult:
        lineage = _lineage_from(film_ir, "film_ir")
        diagnostics: list[StaleArtifactDiagnostic] = []
        if film_ir is None:
            diagnostics.append(
                _diag(
                    "missing_film_ir",
                    "当前没有 FilmIR；请先执行 /build-film-ir。",
                    "film_ir",
                    action="/build-film-ir",
                )
            )
            return LineageCheckResult("film_ir", False, tuple(diagnostics), lineage)
        _compare_required(
            diagnostics,
            artifact_type="film_ir",
            field="source_movie_plan_id",
            actual=lineage.source_movie_plan_id,
            expected=current_movie_plan_id,
            action="/build-film-ir",
        )
        _compare_artifact_identity(
            diagnostics,
            artifact_type="film_ir",
            actual=lineage.artifact_id,
            expected=current_film_ir_id,
            action="/build-film-ir",
        )
        _compare_required(
            diagnostics,
            artifact_type="film_ir",
            field="source_story_plan_id",
            actual=lineage.source_story_plan_id,
            expected=current_story_plan_id,
            action="/build-film-ir",
        )
        _compare_required(
            diagnostics,
            artifact_type="film_ir",
            field="source_director_plan_id",
            actual=lineage.source_director_plan_id,
            expected=current_director_plan_id,
            action="/build-film-ir",
        )
        return LineageCheckResult("film_ir", not diagnostics, tuple(diagnostics), lineage)

    def check_movie_ir(
        self,
        movie_ir: Any,
        current_movie_plan_id: str | None = None,
        current_film_ir_id: str | None = "",
        current_movie_ir_id: str | None = "",
    ) -> LineageCheckResult:
        lineage = _lineage_from(movie_ir, "movie_ir")
        diagnostics: list[StaleArtifactDiagnostic] = []
        if movie_ir is None:
            diagnostics.append(
                _diag(
                    "missing_movie_ir",
                    "当前没有 MovieIR；请先执行 /build-ir。",
                    "movie_ir",
                    action="/build-ir",
                )
            )
            return LineageCheckResult("movie_ir", False, tuple(diagnostics), lineage)
        _compare_required(
            diagnostics,
            artifact_type="movie_ir",
            field="source_movie_plan_id",
            actual=lineage.source_movie_plan_id,
            expected=current_movie_plan_id,
            action="/build-ir",
        )
        _compare_artifact_identity(
            diagnostics,
            artifact_type="movie_ir",
            actual=lineage.artifact_id,
            expected=current_movie_ir_id,
            action="/build-ir",
        )
        _compare_required(
            diagnostics,
            artifact_type="movie_ir",
            field="source_film_ir_id",
            actual=lineage.source_film_ir_id,
            expected=current_film_ir_id,
            action="/build-ir",
        )
        return LineageCheckResult("movie_ir", not diagnostics, tuple(diagnostics), lineage)

    def check_video_job(
        self,
        video_job: Any,
        current_movie_plan_id: str | None = None,
        current_film_ir_id: str | None = "",
        current_movie_ir_id: str | None = "",
        current_video_job_id: str | None = "",
    ) -> LineageCheckResult:
        lineage = _lineage_from(video_job, "video_job")
        diagnostics: list[StaleArtifactDiagnostic] = []
        if video_job is None:
            diagnostics.append(
                _diag(
                    "missing_video_job",
                    "当前没有 VideoJob；请先执行 /compile。",
                    "video_job",
                    action="/compile",
                )
            )
            return LineageCheckResult("video_job", False, tuple(diagnostics), lineage)
        for field, expected in (
            ("source_movie_plan_id", current_movie_plan_id),
            ("source_film_ir_id", current_film_ir_id),
            ("source_movie_ir_id", current_movie_ir_id),
        ):
            _compare_required(
                diagnostics,
                artifact_type="video_job",
                field=field,
                actual=getattr(lineage, field),
                expected=expected,
                action="/compile",
            )
        _compare_artifact_identity(
            diagnostics,
            artifact_type="video_job",
            actual=lineage.artifact_id,
            expected=current_video_job_id,
            action="/compile",
        )
        return LineageCheckResult("video_job", not diagnostics, tuple(diagnostics), lineage)

    def check_provider_artifact(
        self,
        provider_job: Any = None,
        artifact: Any = None,
        current_video_job_id: str | None = None,
    ) -> LineageCheckResult:
        diagnostics: list[StaleArtifactDiagnostic] = []
        values: list[tuple[str, Any]] = []
        if provider_job is not None:
            values.append(("provider_job", provider_job))
        if artifact is not None:
            values.append(("artifact", artifact))
        for artifact_type, value in values:
            actual = _value(value, "source_video_job_id", "video_job_id", "job_id")
            if not actual:
                diagnostics.append(
                    _diag(
                        "provider_artifact_unknown_lineage",
                        f"{artifact_type} 缺少 source VideoJob 标识，只能诊断，不能证明来源。",
                        artifact_type,
                        action="/compile",
                    )
                )
            elif not current_video_job_id:
                diagnostics.append(
                    _diag(
                        "provider_artifact_unknown_lineage",
                        f"当前没有 current_video_job_id，无法验证 {artifact_type}。",
                        artifact_type,
                        action="/compile",
                        actual=actual,
                    )
                )
            elif actual != current_video_job_id:
                diagnostics.append(
                    _diag(
                        "provider_artifact_stale",
                        f"{artifact_type} 来源 VideoJob 与当前 VideoJob 不一致。",
                        artifact_type,
                        action="/compile",
                        expected=current_video_job_id,
                        actual=actual,
                    )
                )
        lineage = SourceLineage(artifact_type="provider_runtime")
        return LineageCheckResult(
            "provider_runtime",
            not diagnostics,
            tuple(diagnostics),
            lineage,
        )

    def check_session(self, session: Any) -> LineageCheckResult:
        plan = getattr(session, "confirmed_movie_plan", None) or getattr(session, "movie_plan", None)
        current_plan_id = getattr(session, "current_movie_plan_id", None) or getattr(plan, "plan_id", None)
        story_id = f"{current_plan_id}:story_plan" if plan is not None and getattr(plan, "story_plan", None) else None
        director_id = f"{current_plan_id}:director_plan" if plan is not None and getattr(plan, "director_plan", None) else None
        diagnostics: list[StaleArtifactDiagnostic] = []
        for result in (
            self.check_film_ir(
                getattr(session, "film_ir", None),
                current_movie_plan_id=current_plan_id or "",
                current_story_plan_id=story_id or "",
                current_director_plan_id=director_id or "",
                current_film_ir_id=getattr(session, "current_film_ir_id", None) or "",
            ),
            self.check_movie_ir(
                getattr(session, "movie_ir", None),
                current_movie_plan_id=current_plan_id or "",
                current_film_ir_id=getattr(session, "current_film_ir_id", None) or "",
                current_movie_ir_id=getattr(session, "current_movie_ir_id", None) or "",
            ),
            self.check_video_job(
                getattr(session, "v2_video_job", None),
                current_movie_plan_id=current_plan_id or "",
                current_film_ir_id=getattr(session, "current_film_ir_id", None) or "",
                current_movie_ir_id=getattr(session, "current_movie_ir_id", None) or "",
                current_video_job_id=getattr(session, "current_video_job_id", None) or "",
            ),
        ):
            if getattr(session, _session_artifact_field(result.artifact_type), None) is not None:
                diagnostics.extend(result.diagnostics)
        provider_result = self.check_provider_artifact(
            getattr(session, "provider_job", None),
            getattr(session, "artifact", None),
            current_video_job_id=getattr(session, "current_video_job_id", None),
        )
        diagnostics.extend(provider_result.diagnostics)
        return LineageCheckResult(
            "session",
            not diagnostics,
            tuple(diagnostics),
            SourceLineage(source_movie_plan_id=str(current_plan_id or "")),
        )


def _session_artifact_field(artifact_type: str) -> str:
    return {
        "film_ir": "film_ir",
        "movie_ir": "movie_ir",
        "video_job": "v2_video_job",
    }.get(artifact_type, artifact_type)


def _lineage_from(value: Any, artifact_type: str) -> SourceLineage:
    if value is None:
        return SourceLineage(artifact_type=artifact_type)
    return SourceLineage(
        artifact_type=artifact_type,
        artifact_id=str(_value(value, "ir_id", "job_id", "id") or ""),
        source_movie_plan_id=str(_value(value, "source_movie_plan_id") or "").strip(),
        source_story_plan_id=str(_value(value, "source_story_plan_id") or "").strip(),
        source_director_plan_id=str(_value(value, "source_director_plan_id") or "").strip(),
        source_film_ir_id=str(_value(value, "source_film_ir_id") or "").strip(),
        source_movie_ir_id=str(_value(value, "source_movie_ir_id") or "").strip(),
    )


def _value(value: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(value, dict) and value.get(key) is not None:
            return value.get(key)
        candidate = getattr(value, key, None)
        if candidate is not None:
            return candidate
    return None


def _compare_required(
    diagnostics: list[StaleArtifactDiagnostic],
    *,
    artifact_type: str,
    field: str,
    actual: str,
    expected: str | None,
    action: str,
) -> None:
    # ``None`` means the caller intentionally omitted an optional comparison;
    # an empty string means the current source was expected but is unknown.
    if expected is None:
        return
    if not expected or not actual:
        code = f"{artifact_type}_unknown_lineage"
        message = f"{artifact_type} 缺少 {field} 或当前来源 ID，无法证明来源。"
    elif actual != expected:
        code = f"{artifact_type}_source_mismatch"
        message = f"{artifact_type}.{field} 与当前来源不一致。"
    else:
        return
    diagnostics.append(
        _diag(
            code,
            message,
            artifact_type,
            path=field,
            action=action,
            expected=str(expected or ""),
            actual=actual,
        )
    )


def _compare_artifact_identity(
    diagnostics: list[StaleArtifactDiagnostic],
    *,
    artifact_type: str,
    actual: str,
    expected: str | None,
    action: str,
) -> None:
    if expected is None:
        return
    if not actual or not expected:
        code = f"{artifact_type}_unknown_lineage"
        message = f"{artifact_type} 缺少 artifact ID 或 current ID，无法证明当前实例。"
    elif actual != expected:
        code = f"{artifact_type}_identity_mismatch"
        message = f"{artifact_type} artifact ID 与当前 active ID 不一致。"
    else:
        return
    diagnostics.append(
        _diag(
            code,
            message,
            artifact_type,
            path="artifact_id",
            action=action,
            expected=str(expected or ""),
            actual=actual,
        )
    )


def _diag(
    code: str,
    message: str,
    artifact_type: str,
    *,
    path: str = "",
    action: str = "",
    expected: str = "",
    actual: str = "",
) -> StaleArtifactDiagnostic:
    return StaleArtifactDiagnostic(
        code=code,
        message=message,
        artifact_type=artifact_type,
        path=path,
        action=action,
        expected=expected,
        actual=actual,
    )


__all__ = [
    "LineageCheckResult",
    "SourceLineage",
    "SourceLineageGuard",
    "StaleArtifactDiagnostic",
]
