"""Safe revision candidates produced before a real Director revision adapter.

Candidates are proposals, not replacements for the active MoviePlan.  The
factory in this module only creates deterministic fixtures for Diff/Guard
tests; production CLI paths create no fake revised plan.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
from typing import Any, Iterable

from .models import MoviePlan
from .revision_request import CreativeRevisionRequest


_CANDIDATE_TYPES = {
    "fake_noop",
    "fake_policy_violation",
    "fake_targeted_patch",
    "external_director_candidate",
}
_STATUSES = {"candidate", "pending_director", "accepted", "rejected", "rollback"}


def _candidate_id(plan_id: str, request_ids: Iterable[str], suffix: str) -> str:
    source = "|".join((plan_id, *sorted(str(item) for item in request_ids), suffix))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"revision-candidate-{digest}"


def _plain(value: Any) -> Any:
    if isinstance(value, MoviePlan):
        return {
            str(key): _plain(item)
            for key, item in asdict(value).items()
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return {
            str(key): _plain(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RevisionCandidate:
    candidate_id: str
    source_movie_plan_id: str
    source_revision_request_ids: tuple[str, ...] = ()
    revised_movie_plan: MoviePlan | None = None
    candidate_type: str = "external_director_candidate"
    created_by: str = "unknown"
    status: str = "candidate"
    rationale: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise ValueError("RevisionCandidate.candidate_id is required")
        if not str(self.source_movie_plan_id).strip():
            raise ValueError("RevisionCandidate.source_movie_plan_id is required")
        if self.candidate_type not in _CANDIDATE_TYPES:
            raise ValueError(f"Unsupported RevisionCandidate.candidate_type: {self.candidate_type}")
        if self.status not in _STATUSES:
            raise ValueError(f"Unsupported RevisionCandidate.status: {self.status}")
        if self.revised_movie_plan is not None and not isinstance(self.revised_movie_plan, MoviePlan):
            raise TypeError("RevisionCandidate.revised_movie_plan must be MoviePlan or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_movie_plan_id": self.source_movie_plan_id,
            "source_revision_request_ids": list(self.source_revision_request_ids),
            "revised_movie_plan": _plain(self.revised_movie_plan),
            "candidate_type": self.candidate_type,
            "created_by": self.created_by,
            "status": self.status,
            "rationale": self.rationale,
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        revised_movie_plan: MoviePlan | None = None,
    ) -> "RevisionCandidate":
        if not isinstance(data, dict):
            raise ValueError("RevisionCandidate must be a JSON object")
        raw_requests = data.get("source_revision_request_ids", [])
        raw_metadata = data.get("metadata", {})
        if not isinstance(raw_requests, list) or not all(isinstance(item, str) for item in raw_requests):
            raise ValueError("source_revision_request_ids must be a string array")
        if not isinstance(raw_metadata, dict):
            raise ValueError("RevisionCandidate.metadata must be an object")
        raw_plan = data.get("revised_movie_plan")
        if raw_plan is not None and revised_movie_plan is None:
            raise ValueError("revised_movie_plan requires a MoviePlan loader")
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            source_movie_plan_id=str(data.get("source_movie_plan_id", "")),
            source_revision_request_ids=tuple(raw_requests),
            revised_movie_plan=revised_movie_plan,
            candidate_type=str(data.get("candidate_type", "external_director_candidate")),
            created_by=str(data.get("created_by", "unknown")),
            status=str(data.get("status", "candidate")),
            rationale=str(data.get("rationale", "")),
            metadata=deepcopy(raw_metadata),
        )


class RevisionCandidateFactory:
    """Create only deterministic fixtures; never used by default CLI flow."""

    def create_noop_candidate(
        self,
        original: MoviePlan,
        requests: Iterable[CreativeRevisionRequest] = (),
    ) -> RevisionCandidate:
        _require_plan(original)
        request_ids = _request_ids(requests)
        return RevisionCandidate(
            candidate_id=_candidate_id(original.plan_id, request_ids, "noop"),
            source_movie_plan_id=original.plan_id,
            source_revision_request_ids=request_ids,
            revised_movie_plan=deepcopy(original),
            candidate_type="fake_noop",
            created_by="deterministic_fake",
            status="candidate",
            rationale="测试 no-change Guard/Diff，不代表导演修订。",
            metadata={"fixture": "noop"},
        )

    def create_policy_violation_candidate(
        self,
        original: MoviePlan,
        requests: Iterable[CreativeRevisionRequest] = (),
    ) -> RevisionCandidate:
        _require_plan(original)
        request_ids = _request_ids(requests)
        story_plan = original.story_plan
        characters = tuple(story_plan.characters)
        revised_story_plan = replace(
            story_plan,
            characters=characters[1:] if characters else (),
        )
        revised = replace(original, story_plan=revised_story_plan)
        return RevisionCandidate(
            candidate_id=_candidate_id(original.plan_id, request_ids, "policy-violation"),
            source_movie_plan_id=original.plan_id,
            source_revision_request_ids=request_ids,
            revised_movie_plan=revised,
            candidate_type="fake_policy_violation",
            created_by="deterministic_fake",
            status="candidate",
            rationale="测试 preserve/provider/prompt Guard 拒绝路径，不得进入生产流程。",
            metadata={
                "fixture": "policy_violation",
                "unsafe_fields": ["provider_payload"],
                "unsafe_terms": ["best quality"],
            },
        )

    def create_pending_director_candidate(
        self,
        original: MoviePlan,
        requests: Iterable[CreativeRevisionRequest] = (),
    ) -> RevisionCandidate:
        _require_plan(original)
        request_ids = _request_ids(requests)
        return RevisionCandidate(
            candidate_id=_candidate_id(original.plan_id, request_ids, "pending-director"),
            source_movie_plan_id=original.plan_id,
            source_revision_request_ids=request_ids,
            revised_movie_plan=None,
            candidate_type="external_director_candidate",
            created_by="pending_director",
            status="pending_director",
            rationale="等待真实 DirectorAgent 生成候选 MoviePlan。",
            metadata={"fixture": "pending_director"},
        )


def _require_plan(value: MoviePlan) -> None:
    if not isinstance(value, MoviePlan):
        raise TypeError("RevisionCandidateFactory 只接受 MoviePlan")


def _request_ids(requests: Iterable[CreativeRevisionRequest]) -> tuple[str, ...]:
    result: list[str] = []
    for request in requests:
        if isinstance(request, CreativeRevisionRequest):
            result.append(request.request_id)
        elif isinstance(request, dict):
            result.append(str(request.get("request_id", "")))
        else:
            raise TypeError("revision requests must contain CreativeRevisionRequest objects")
    return tuple(item for item in result if item)


__all__ = ["RevisionCandidate", "RevisionCandidateFactory"]
