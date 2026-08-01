"""Pluggable Director revision loop for the V2 creative compiler.

This module does not call an LLM by itself.  The rule-based implementation
records why a MoviePlan is accepted or rejected and deliberately refuses to
invent missing beats, viewer states, or pacing decisions.  A future adapter
can subclass the loop and use the existing DirectorAgent revision port.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .models import MoviePlan
from .revision_request import CreativeRevisionRequest


@dataclass(frozen=True, slots=True)
class DirectorRevisionRequest:
    plan_id: str
    validation_issues: tuple[Any, ...] = ()
    creative_diagnostics: tuple[Any, ...] = ()
    optimizer_diagnostics: tuple[Any, ...] = ()
    revision_index: int = 1
    max_revisions: int = 1
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class DirectorRevisionResult:
    revised_movie_plan: MoviePlan | None
    revision_history: tuple[dict[str, Any], ...] = ()
    accepted: bool = False
    stop_reason: str = ""

    @property
    def movie_plan(self) -> MoviePlan | None:
        """Compatibility alias for callers that call the output simply plan."""

        return self.revised_movie_plan


class DirectorRevisionLoop:
    """Interface for bounded director revision orchestration."""

    def __init__(self, *, max_revisions: int = 1) -> None:
        if max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        self.max_revisions = max_revisions

    def run(
        self,
        movie_plan: MoviePlan,
        validation_issues: Iterable[Any] = (),
        creative_diagnostics: Iterable[Any] = (),
        optimizer_diagnostics: Iterable[Any] = (),
        max_revisions: int | None = None,
        creative_revision_requests: Iterable[CreativeRevisionRequest] = (),
    ) -> DirectorRevisionResult:
        raise NotImplementedError

    def revise(
        self,
        movie_plan: MoviePlan,
        validation_issues: Iterable[Any] = (),
        creative_diagnostics: Iterable[Any] = (),
        optimizer_diagnostics: Iterable[Any] = (),
        max_revisions: int | None = None,
        creative_revision_requests: Iterable[CreativeRevisionRequest] = (),
    ) -> DirectorRevisionResult:
        return self.run(
            movie_plan,
            validation_issues,
            creative_diagnostics,
            optimizer_diagnostics,
            max_revisions,
            creative_revision_requests,
        )


class RuleBasedDirectorRevisionLoop(DirectorRevisionLoop):
    """Offline implementation that never fabricates director content."""

    def run(
        self,
        movie_plan: MoviePlan,
        validation_issues: Iterable[Any] = (),
        creative_diagnostics: Iterable[Any] = (),
        optimizer_diagnostics: Iterable[Any] = (),
        max_revisions: int | None = None,
        creative_revision_requests: Iterable[CreativeRevisionRequest] = (),
    ) -> DirectorRevisionResult:
        if not isinstance(movie_plan, MoviePlan):
            raise TypeError("RuleBasedDirectorRevisionLoop 只接受 MoviePlan。")
        limit = self.max_revisions if max_revisions is None else max_revisions
        if limit < 0:
            raise ValueError("max_revisions must be non-negative")
        intrinsic_issues: list[dict[str, str]] = []
        if not movie_plan.film_beats:
            intrinsic_issues.append(
                {
                    "code": "missing_film_beats",
                    "message": "MoviePlan 缺少 film-level beats。",
                    "path": "film_beats",
                    "severity": "error",
                }
            )
        issues = tuple((*validation_issues, *intrinsic_issues))
        creative = tuple(creative_diagnostics)
        optimizer = tuple(optimizer_diagnostics)
        hard_errors = tuple(
            item
            for item in (*issues, *creative, *optimizer)
            if _severity(item) == "error"
        )
        requests = tuple(creative_revision_requests)
        request = DirectorRevisionRequest(
            plan_id=movie_plan.plan_id,
            validation_issues=issues,
            creative_diagnostics=creative,
            optimizer_diagnostics=optimizer,
            revision_index=1,
            max_revisions=limit,
            reason=(
                "存在硬错误，需要 DirectorAgent 重新生成 MoviePlan。"
                if hard_errors
                else "当前只有建议性诊断，不自动改变导演决策。"
            ),
        )
        if hard_errors or requests:
            history_items: list[dict[str, Any]] = [
                {
                    "revision_index": 1,
                    "status": "rejected" if hard_errors else "pending_director",
                    "plan_id": movie_plan.plan_id,
                    "reason": (
                        "hard_error_requires_director_revision"
                        if hard_errors
                        else "creative_revision_request_requires_director"
                    ),
                    "request": request.to_dict(),
                },
            ]
            history_items.extend(
                {
                    "revision_index": index,
                    "status": "pending_director",
                    "plan_id": movie_plan.plan_id,
                    "reason": "creative_revision_request_requires_director",
                    "request": item.to_dict() if hasattr(item, "to_dict") else item,
                }
                for index, item in enumerate(requests, 2)
            )
            return DirectorRevisionResult(
                revised_movie_plan=None,
                revision_history=tuple(history_items),
                accepted=False,
                stop_reason=(
                    "hard_error_requires_director_revision"
                    if hard_errors
                    else "pending_director_revision"
                ),
            )
        stop_reason = (
            "suggestions_recorded_no_revision"
            if (creative or optimizer)
            else "no_revision_required"
        )
        history = (
            {
                "revision_index": 0,
                "status": "accepted",
                "plan_id": movie_plan.plan_id,
                "reason": stop_reason,
                "request": request.to_dict(),
            },
        )
        return DirectorRevisionResult(
            revised_movie_plan=movie_plan,
            revision_history=history,
            accepted=True,
            stop_reason=stop_reason,
        )


def _severity(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("severity", "warning"))
    return str(getattr(value, "severity", "warning"))


def _plain(value: Any) -> Any:
    if isinstance(value, MoviePlan):
        return _plain(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value
