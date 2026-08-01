"""Candidate-only adapters for DirectorAgent revisions.

Phase 4D.2 deliberately keeps creative revision separate from Session state.
An adapter may ask a DirectorAgent for a complete replacement ``MoviePlan``;
it can only return that plan wrapped in :class:`RevisionCandidate`.  The
candidate is then validated, diffed, and guarded by the explicit helper in
this module.  No adapter applies a plan or touches a Provider.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Protocol, Sequence

from .models import CreativeBrief, MoviePlan, as_plain_data
from .revision_candidate import RevisionCandidate, RevisionCandidateFactory
from .revision_diff import RevisionDiff, RevisionDiffBuilder
from .revision_guard import RevisionDecision, RevisionGuard, RevisionGuardPolicy
from .revision_request import CreativeRevisionRequest
from .validation import ValidationIssue, validate_movie_plan


_STATUSES = {
    "candidate_created",
    "pending_director",
    "director_failed",
    "invalid_director_output",
    "rejected_before_guard",
}
_FORBIDDEN_KEYS = {
    "provider",
    "provider_key",
    "provider_name",
    "provider_profile",
    "api",
    "api_key",
    "payload",
    "provider_payload",
    "request_payload",
    "video_payload",
    "api_payload",
    "http_payload",
    "endpoint",
    "model",
    "task",
    "task_id",
    "video_id",
    "submit",
    "poll",
    "download",
}
_PROMPT_STUFFING_TERMS = (
    "masterpiece",
    "best quality",
    "ultra realistic",
    "ultra-realistic",
    "8k",
)
_DEFAULT_FORBIDDEN_SCOPE = (
    "不改主要人物",
    "不改核心冲突",
    "不改结局基调",
    "不新增 Provider prompt",
    "不新增 Provider/API 字段",
)
_LOCAL_PATH_PATTERN = re.compile(r"^(?:[a-zA-Z]:[\\/]|~[\\/]|/{1,2})")
_LOCAL_PATH_ANYWHERE = re.compile(
    r"(?i)(?:[a-z]:[\\/]|~[\\/]|/(?:users|home|tmp)/|\\\\)[^\s\"']+"
)


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _request_data(request: CreativeRevisionRequest | dict[str, Any]) -> dict[str, Any]:
    if isinstance(request, CreativeRevisionRequest):
        return request.to_dict()
    if isinstance(request, dict):
        return deepcopy(request)
    raise TypeError("revision requests must contain CreativeRevisionRequest objects or dictionaries")


def _request_ids(requests: Sequence[CreativeRevisionRequest | dict[str, Any]]) -> tuple[str, ...]:
    result: list[str] = []
    for request in requests:
        value = _request_data(request).get("request_id", "")
        if str(value).strip():
            result.append(str(value))
    return tuple(result)


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return tuple(result)


def _ensure_safe(value: Any, path: str = "director_revision") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"Director revision contains forbidden field: {path}.{key}")
            _ensure_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for term in _PROMPT_STUFFING_TERMS:
            if term in lowered:
                raise ValueError(f"Director revision contains prompt stuffing at {path}: {term}")


def _sanitize_prompt_value(value: Any) -> Any:
    """Keep creative context while omitting accidental local path values."""

    if isinstance(value, dict):
        return {str(key): _sanitize_prompt_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_prompt_value(item) for item in value]
    if isinstance(value, str):
        cleaned = value.strip()
        if _LOCAL_PATH_PATTERN.match(cleaned):
            return "[omitted local path]"
        if _LOCAL_PATH_ANYWHERE.search(value):
            return _LOCAL_PATH_ANYWHERE.sub("[omitted local path]", value)
    return value


@dataclass(frozen=True, slots=True)
class DirectorRevisionContext:
    """Safe, provider-neutral context passed to a revision adapter."""

    original_movie_plan_id: str
    request_ids: tuple[str, ...] = ()
    preserve: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()
    allowed_change_scope: tuple[str, ...] = ()
    forbidden_change_scope: tuple[str, ...] = _DEFAULT_FORBIDDEN_SCOPE
    max_revision_attempts: int = 1
    require_candidate_only: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.original_movie_plan_id).strip():
            raise ValueError("DirectorRevisionContext.original_movie_plan_id is required")
        if isinstance(self.max_revision_attempts, bool) or self.max_revision_attempts < 1:
            raise ValueError("max_revision_attempts must be at least one")
        _ensure_safe(self.metadata, "director_revision_context.metadata")

    @classmethod
    def from_requests(
        cls,
        movie_plan: MoviePlan,
        requests: Sequence[CreativeRevisionRequest | dict[str, Any]] = (),
        *,
        allowed_change_scope: Sequence[str] = (),
        forbidden_change_scope: Sequence[str] = _DEFAULT_FORBIDDEN_SCOPE,
        max_revision_attempts: int = 1,
        metadata: dict[str, object] | None = None,
    ) -> "DirectorRevisionContext":
        if not isinstance(movie_plan, MoviePlan):
            raise TypeError("DirectorRevisionContext 只接受 MoviePlan")
        request_data = tuple(_request_data(item) for item in requests)
        _ensure_safe(request_data, "director_revision_context.requests")
        preserve = _unique_strings(
            tuple(str(value) for item in request_data for value in item.get("preserve", ()))
        )
        avoid = _unique_strings(
            tuple(str(value) for item in request_data for value in item.get("avoid", ()))
        )
        scopes = tuple(
            str(item.get("target", "")) for item in request_data if str(item.get("target", "")).strip()
        )
        return cls(
            original_movie_plan_id=movie_plan.plan_id,
            request_ids=_request_ids(tuple(requests)),
            preserve=preserve,
            avoid=avoid,
            allowed_change_scope=_unique_strings(tuple(allowed_change_scope) or scopes),
            forbidden_change_scope=_unique_strings(tuple(forbidden_change_scope)),
            max_revision_attempts=max_revision_attempts,
            metadata=deepcopy(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class DirectorRevisionAdapterResult:
    """The adapter result, before a final Guard decision."""

    source_movie_plan_id: str
    candidate: RevisionCandidate | None = None
    status: str = "pending_director"
    diagnostics: tuple[dict[str, object], ...] = ()
    request_ids: tuple[str, ...] = ()
    adapter_name: str = "unknown"
    stop_reason: str | None = None
    succeeded: bool = False

    def __post_init__(self) -> None:
        if not str(self.source_movie_plan_id).strip():
            raise ValueError("DirectorRevisionAdapterResult.source_movie_plan_id is required")
        if self.status not in _STATUSES:
            raise ValueError(f"Unsupported DirectorRevisionAdapterResult.status: {self.status}")
        if self.candidate is not None and not isinstance(self.candidate, RevisionCandidate):
            raise TypeError("DirectorRevisionAdapterResult.candidate must be RevisionCandidate or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_movie_plan_id": self.source_movie_plan_id,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "status": self.status,
            "diagnostics": _plain(self.diagnostics),
            "request_ids": list(self.request_ids),
            "adapter_name": self.adapter_name,
            "stop_reason": self.stop_reason,
            "succeeded": bool(self.succeeded),
        }


@dataclass(frozen=True, slots=True)
class GuardedRevisionResult:
    """The complete candidate-only adapter → validation → diff → Guard result."""

    adapter_result: DirectorRevisionAdapterResult
    validation_issues: tuple[dict[str, object], ...] = ()
    diff: RevisionDiff | None = None
    decision: RevisionDecision | None = None
    candidate: RevisionCandidate | None = None
    succeeded: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_result": self.adapter_result.to_dict(),
            "validation_issues": _plain(self.validation_issues),
            "diff": self.diff.to_dict() if self.diff else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "succeeded": bool(self.succeeded),
            "stop_reason": self.stop_reason,
        }


class DirectorRevisionAdapter(Protocol):
    """Port for candidate-producing Director revision implementations."""

    adapter_name: str

    def revise(
        self,
        movie_plan: MoviePlan,
        requests: Sequence[CreativeRevisionRequest | dict[str, Any]],
        context: DirectorRevisionContext,
    ) -> DirectorRevisionAdapterResult: ...


def build_director_revision_prompt(
    movie_plan: MoviePlan,
    requests: Sequence[CreativeRevisionRequest | dict[str, Any]],
    context: DirectorRevisionContext,
) -> str:
    """Pack only safe creative context for a DirectorAgent revision call."""

    if not isinstance(movie_plan, MoviePlan):
        raise TypeError("build_director_revision_prompt 只接受 MoviePlan")
    if not context.require_candidate_only:
        raise ValueError("revision prompt 只支持 candidate-only 模式")
    if context.original_movie_plan_id != movie_plan.plan_id:
        raise ValueError("revision context 必须引用当前 MoviePlan")
    request_data = [_request_data(item) for item in requests]
    _ensure_safe(request_data, "director_revision_prompt.requests")
    payload = {
        "original_movie_plan_id": context.original_movie_plan_id,
        "request_ids": list(context.request_ids),
        "preserve": list(context.preserve),
        "avoid": list(context.avoid),
        "allowed_change_scope": list(context.allowed_change_scope),
        "forbidden_change_scope": list(context.forbidden_change_scope),
        "candidate_only": True,
        "revision_requests": request_data,
        "current_movie_plan": _sanitize_prompt_value(as_plain_data(movie_plan)),
    }
    _ensure_safe(payload, "director_revision_prompt")
    return (
        "你是负责电影创作的 DirectorAgent。请只针对 revision_requests 修订当前 MoviePlan。\n"
        "只能输出一个完整的 MoviePlan JSON 对象，不能输出解释文字。\n"
        "这是 candidate-only 流程：输出只会成为 RevisionCandidate，不能直接写回当前 MoviePlan。\n"
        "必须保留 preserve 约束，遵守 avoid 约束，只在 allowed_change_scope 内工作。\n"
        "不得改变 forbidden_change_scope；不得输出 Provider prompt、Provider/API payload、endpoint、model、task 或 video_id；"
        "不得生成 MP4，也不得进行 Provider 操作。\n"
        "不要使用 prompt stuffing；除非 revision request 明确允许，不要重写整个故事、增加主要人物或改变结局基调。\n"
        "安全的结构化输入如下：\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


class RuleBasedDirectorRevisionAdapter:
    """Offline adapter with explicit, deterministic fixture modes."""

    adapter_name = "rule_based_revision"
    _MODES = {"pending_only", "noop_candidate", "safe_metadata_candidate", "policy_violation_candidate"}

    def __init__(
        self,
        mode: str = "pending_only",
        *,
        factory: RevisionCandidateFactory | None = None,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"Unsupported RuleBasedDirectorRevisionAdapter mode: {mode}")
        self.mode = mode
        self.factory = factory or RevisionCandidateFactory()

    def revise(
        self,
        movie_plan: MoviePlan,
        requests: Sequence[CreativeRevisionRequest | dict[str, Any]],
        context: DirectorRevisionContext,
    ) -> DirectorRevisionAdapterResult:
        if not isinstance(movie_plan, MoviePlan):
            raise TypeError("RuleBasedDirectorRevisionAdapter 只接受 MoviePlan")
        if not context.require_candidate_only:
            return _adapter_failure(
                movie_plan,
                requests,
                self.adapter_name,
                "rejected_before_guard",
                "revision context 未启用 candidate-only。",
            )
        if context.original_movie_plan_id != movie_plan.plan_id:
            return _adapter_failure(
                movie_plan,
                requests,
                self.adapter_name,
                "invalid_director_output",
                "revision context 与当前 MoviePlan 不一致。",
            )
        if self.mode == "pending_only":
            candidate = self.factory.create_pending_director_candidate(movie_plan, requests)
            return DirectorRevisionAdapterResult(
                movie_plan.plan_id,
                candidate,
                "pending_director",
                request_ids=_request_ids(requests),
                adapter_name=self.adapter_name,
                stop_reason="pending_director_revision",
                succeeded=False,
            )
        if self.mode == "noop_candidate":
            candidate = self.factory.create_noop_candidate(movie_plan, requests)
        elif self.mode == "safe_metadata_candidate":
            candidate = self.factory.create_noop_candidate(movie_plan, requests)
            candidate = RevisionCandidate(
                candidate_id=candidate.candidate_id,
                source_movie_plan_id=candidate.source_movie_plan_id,
                source_revision_request_ids=candidate.source_revision_request_ids,
                revised_movie_plan=candidate.revised_movie_plan,
                candidate_type="fake_targeted_patch",
                created_by=self.adapter_name,
                status="candidate",
                rationale="没有安全的现有 metadata 字段可修改，因此保持 MoviePlan 内容不变。",
                metadata={"mode": "safe_metadata_candidate"},
            )
        else:
            candidate = self.factory.create_policy_violation_candidate(movie_plan, requests)
        return DirectorRevisionAdapterResult(
            movie_plan.plan_id,
            candidate,
            "candidate_created",
            request_ids=_request_ids(requests),
            adapter_name=self.adapter_name,
            stop_reason="candidate_created",
            succeeded=True,
        )


class DirectorAgentRevisionAdapter:
    """Wrap an existing DirectorAgent without allowing it to mutate Session."""

    adapter_name = "director_agent_revision"

    def __init__(
        self,
        director_agent: Any,
        brief: CreativeBrief | None = None,
        *,
        adapter_name: str | None = None,
    ) -> None:
        if director_agent is None or not callable(getattr(director_agent, "revise_movie_plan", None)):
            raise TypeError("director_agent 必须实现 revise_movie_plan()")
        self.director_agent = director_agent
        self.brief = brief
        if adapter_name:
            self.adapter_name = adapter_name

    def revise(
        self,
        movie_plan: MoviePlan,
        requests: Sequence[CreativeRevisionRequest | dict[str, Any]],
        context: DirectorRevisionContext,
    ) -> DirectorRevisionAdapterResult:
        if not isinstance(movie_plan, MoviePlan):
            raise TypeError("DirectorAgentRevisionAdapter 只接受 MoviePlan")
        if not context.require_candidate_only:
            return _adapter_failure(
                movie_plan,
                requests,
                self.adapter_name,
                "rejected_before_guard",
                "revision context 未启用 candidate-only。",
            )
        if context.original_movie_plan_id != movie_plan.plan_id:
            return _adapter_failure(
                movie_plan,
                requests,
                self.adapter_name,
                "invalid_director_output",
                "revision context 与当前 MoviePlan 不一致。",
            )
        try:
            prompt = build_director_revision_prompt(movie_plan, requests, context)
        except (TypeError, ValueError) as exc:
            return _adapter_failure(
                movie_plan,
                requests,
                self.adapter_name,
                "rejected_before_guard",
                f"修订请求未通过安全打包：{exc}",
            )
        try:
            # Never hand the active object to an untrusted DirectorAgent.  A
            # buggy fake or SDK wrapper may mutate its argument in place.
            revised = self.director_agent.revise_movie_plan(
                self.brief,
                deepcopy(movie_plan),
                prompt,
            )
        except Exception as exc:  # adapter boundary turns Director failures into diagnostics
            return _adapter_failure(
                movie_plan,
                requests,
                self.adapter_name,
                "director_failed",
                f"DirectorAgent 修订失败：{exc}",
            )
        if isinstance(revised, dict):
            try:
                from .openai_director import movie_plan_from_data

                revised = movie_plan_from_data(revised)
            except Exception as exc:
                return _adapter_failure(
                    movie_plan,
                    requests,
                    self.adapter_name,
                    "invalid_director_output",
                    f"DirectorAgent JSON 修订输出无法解析：{exc}",
                )
        if not isinstance(revised, MoviePlan):
            return _adapter_failure(
                movie_plan,
                requests,
                self.adapter_name,
                "invalid_director_output",
                "DirectorAgent 修订输出不是 MoviePlan。",
            )
        candidate = RevisionCandidate(
            candidate_id=f"external-director-{revised.plan_id}-{context.request_ids[0] if context.request_ids else 'revision'}",
            source_movie_plan_id=movie_plan.plan_id,
            source_revision_request_ids=context.request_ids,
            revised_movie_plan=deepcopy(revised),
            candidate_type="external_director_candidate",
            created_by=self.adapter_name,
            status="candidate",
            rationale="由 DirectorAgent 生成，待 validate / diff / guard。",
            metadata={"adapter_name": self.adapter_name},
        )
        return DirectorRevisionAdapterResult(
            movie_plan.plan_id,
            candidate,
            "candidate_created",
            request_ids=context.request_ids,
            adapter_name=self.adapter_name,
            stop_reason="candidate_created",
            succeeded=True,
        )


class OpenAIDirectorRevisionAdapter(DirectorAgentRevisionAdapter):
    """Named adapter for OpenAIDirectorAgent; no network is called by itself."""

    adapter_name = "openai_director_revision"


def _adapter_failure(
    movie_plan: MoviePlan,
    requests: Sequence[CreativeRevisionRequest | dict[str, Any]],
    adapter_name: str,
    status: str,
    message: str,
) -> DirectorRevisionAdapterResult:
    return DirectorRevisionAdapterResult(
        source_movie_plan_id=movie_plan.plan_id,
        candidate=None,
        status=status,
        diagnostics=({"code": status, "message": message, "severity": "error"},),
        request_ids=_request_ids(requests),
        adapter_name=adapter_name,
        stop_reason=status,
        succeeded=False,
    )


def _validation_records(
    movie_plan: MoviePlan,
    candidate: RevisionCandidate | None,
    brief: CreativeBrief | None,
    provided: Sequence[ValidationIssue | dict[str, Any]],
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = [_plain(item) for item in provided]
    if candidate is None or candidate.revised_movie_plan is None or brief is None:
        return tuple(records)
    report = validate_movie_plan(candidate.revised_movie_plan, brief)
    records.extend(
        {
            "code": "movie_plan_validation_error",
            "message": error,
            "path": "movie_plan",
            "severity": "error",
        }
        for error in report.errors
    )
    return tuple(records)


def run_director_revision_guarded(
    movie_plan: MoviePlan,
    requests: Sequence[CreativeRevisionRequest | dict[str, Any]],
    adapter: DirectorRevisionAdapter,
    context: DirectorRevisionContext,
    *,
    brief: CreativeBrief | None = None,
    validation_issues: Sequence[ValidationIssue | dict[str, Any]] = (),
    analysis_results: Sequence[Any] = (),
    optimizer_result: Any | None = None,
    policy: RevisionGuardPolicy | None = None,
) -> GuardedRevisionResult:
    """Run adapter output through validation, Diff, and Guard, without apply."""

    if not isinstance(movie_plan, MoviePlan):
        raise TypeError("run_director_revision_guarded 只接受 MoviePlan")
    if context.original_movie_plan_id != movie_plan.plan_id:
        raise ValueError("revision context 必须引用当前 MoviePlan")
    adapter_result = adapter.revise(movie_plan, requests, context)
    candidate = adapter_result.candidate
    records = _validation_records(movie_plan, candidate, brief, validation_issues)
    diff = RevisionDiffBuilder().build_diff(movie_plan, candidate, requests)
    decision = RevisionGuard(policy=policy).evaluate(
        movie_plan,
        candidate,
        diff,
        validation_issues=records,
        analysis_results=analysis_results,
        optimizer_result=optimizer_result,
        requests=requests,
    )
    if adapter_result.status in {"director_failed", "invalid_director_output", "rejected_before_guard"}:
        stop_reason = adapter_result.stop_reason or adapter_result.status
    elif decision.decision == "pending_director":
        stop_reason = "pending_director"
    else:
        stop_reason = decision.decision
    return GuardedRevisionResult(
        adapter_result=adapter_result,
        validation_issues=records,
        diff=diff,
        decision=decision,
        candidate=candidate,
        succeeded=decision.decision in {"accept", "accept_with_warning"},
        stop_reason=stop_reason,
    )


__all__ = [
    "DirectorRevisionContext",
    "DirectorRevisionAdapterResult",
    "GuardedRevisionResult",
    "DirectorRevisionAdapter",
    "RuleBasedDirectorRevisionAdapter",
    "DirectorAgentRevisionAdapter",
    "OpenAIDirectorRevisionAdapter",
    "build_director_revision_prompt",
    "run_director_revision_guarded",
]
