"""Deterministic structural diffing for revision candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Sequence

from .models import MoviePlan, as_plain_data
from .revision_candidate import RevisionCandidate
from .revision_request import CreativeRevisionRequest


_FORBIDDEN_KEYS = {
    "provider", "provider_key", "provider_name", "provider_profile", "api",
    "api_key", "payload", "provider_payload", "request_payload", "video_payload",
    "api_payload", "http_payload", "endpoint", "model", "task", "task_id",
    "video_id", "submit", "poll", "download",
}
_PROMPT_STUFFING_TERMS = (
    "masterpiece", "best quality", "ultra realistic", "ultra-realistic",
    "cinematic masterpiece", "8k", "award winning", "photorealistic masterpiece",
)


@dataclass(frozen=True, slots=True)
class RevisionChange:
    path: str
    change_type: str
    before: object | None = None
    after: object | None = None
    severity: str = "warning"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "change_type": self.change_type,
            "before": _plain(self.before),
            "after": _plain(self.after),
            "severity": self.severity,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RevisionDiff:
    source_movie_plan_id: str
    candidate_id: str
    changes: tuple[RevisionChange, ...] = ()
    preserve_violations: tuple[RevisionChange, ...] = ()
    avoid_violations: tuple[RevisionChange, ...] = ()
    target_responses: tuple[RevisionChange, ...] = ()
    provider_leakage_detected: bool = False
    prompt_stuffing_detected: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    succeeded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_movie_plan_id": self.source_movie_plan_id,
            "candidate_id": self.candidate_id,
            "changes": [item.to_dict() for item in self.changes],
            "preserve_violations": [item.to_dict() for item in self.preserve_violations],
            "avoid_violations": [item.to_dict() for item in self.avoid_violations],
            "target_responses": [item.to_dict() for item in self.target_responses],
            "provider_leakage_detected": bool(self.provider_leakage_detected),
            "prompt_stuffing_detected": bool(self.prompt_stuffing_detected),
            "metrics": {str(key): float(value) for key, value in self.metrics.items()},
            "succeeded": bool(self.succeeded),
        }


class RevisionDiffBuilder:
    """Compare an immutable original plan with a candidate plan."""

    def build_diff(
        self,
        original: MoviePlan,
        candidate: RevisionCandidate | None,
        requests: Sequence[CreativeRevisionRequest | dict[str, Any]] = (),
    ) -> RevisionDiff:
        if not isinstance(original, MoviePlan):
            raise TypeError("RevisionDiffBuilder 只接受 MoviePlan original")
        if candidate is None or candidate.revised_movie_plan is None:
            candidate_id = candidate.candidate_id if candidate is not None else ""
            return RevisionDiff(
                source_movie_plan_id=original.plan_id,
                candidate_id=candidate_id,
                metrics={"pending_director": 1.0},
                succeeded=False,
            )

        revised = candidate.revised_movie_plan
        changes: list[RevisionChange] = []
        if candidate.source_movie_plan_id != original.plan_id:
            changes.append(
                RevisionChange(
                    "candidate.source_movie_plan_id",
                    "modified",
                    original.plan_id,
                    candidate.source_movie_plan_id,
                    "hard",
                    "candidate 必须来自当前 MoviePlan。",
                )
            )

        original_data = as_plain_data(original)
        revised_data = as_plain_data(revised)
        story_fields = (
            "title", "logline", "synopsis", "theme", "ending", "conflict", "stakes", "resolution"
        )
        for field_name in story_fields:
            self._scalar_change(
                changes,
                f"story_plan.{field_name}",
                original_data["story_plan"].get(field_name),
                revised_data["story_plan"].get(field_name),
                reason="StoryPlan narrative field changed.",
            )
        self._sequence_changes(
            changes,
            "story_plan.characters",
            original_data["story_plan"].get("characters", []),
            revised_data["story_plan"].get("characters", []),
            key_name="character_id",
            reason="主要人物集合发生变化。",
        )
        self._sequence_changes(
            changes,
            "story_plan.story_beats",
            original_data["story_plan"].get("story_beats", []),
            revised_data["story_plan"].get("story_beats", []),
            key_name=None,
            reason="StoryPlan story beats 发生变化。",
        )
        director_data = original_data["director_plan"]
        revised_director_data = revised_data["director_plan"]
        for field_name in (
            "pacing_strategy", "suspense_strategy", "audience_knowledge",
            "emotional_intention", "reveal_timing", "withholding_strategy",
            "visual_motif_strategy", "silence_pause_intention", "climax_emphasis",
            "ending_tone",
        ):
            self._scalar_change(
                changes,
                f"director_plan.{field_name}",
                director_data.get(field_name),
                revised_director_data.get(field_name),
                reason="DirectorPlan audience-experience field changed.",
            )

        legacy_fields = (
            "story.title", "story.logline", "story.synopsis", "story.theme", "story.ending",
            "visual_style",
        )
        for path in legacy_fields:
            before = _get_path(original_data, path)
            after = _get_path(revised_data, path)
            self._scalar_change(
                changes,
                path,
                before,
                after,
                reason="Legacy compatibility projection changed.",
            )
        self._sequence_changes(
            changes,
            "character_sheet.characters",
            _get_path(original_data, "character_sheet.characters"),
            _get_path(revised_data, "character_sheet.characters"),
            key_name="character_id",
            reason="Legacy CharacterSheet 人物集合发生变化。",
        )
        self._sequence_changes(
            changes,
            "film_beats",
            original_data.get("film_beats", []),
            revised_data.get("film_beats", []),
            key_name="beat_id",
            reason="Film-level beats 发生变化。",
        )
        if original_data.get("script", {}).get("scenes") != revised_data.get("script", {}).get("scenes"):
            changes.append(
                RevisionChange(
                    "script.scenes",
                    "modified",
                    original_data.get("script", {}).get("scenes"),
                    revised_data.get("script", {}).get("scenes"),
                    "warning",
                    "Legacy Script scenes changed; DirectorAgent must keep compatibility explicit.",
                )
            )

        preserve_violations = self._constraint_changes(changes, requests, "preserve")
        avoid_violations = self._constraint_changes(changes, requests, "avoid")
        provider_leakage = _contains_forbidden(revised_data) or _contains_forbidden(candidate.metadata)
        prompt_stuffing = _contains_prompt_stuffing(revised_data) or _contains_prompt_stuffing(candidate.metadata)
        target_responses = self._target_responses(changes, requests, candidate.metadata)
        metrics = {
            "changed_field_count": float(len(changes)),
            "preserve_violation_count": float(len(preserve_violations)),
            "avoid_violation_count": float(len(avoid_violations)),
            "target_response_count": float(len(target_responses)),
            "provider_leakage_count": float(provider_leakage),
            "prompt_stuffing_count": float(prompt_stuffing),
        }
        return RevisionDiff(
            source_movie_plan_id=original.plan_id,
            candidate_id=candidate.candidate_id,
            changes=tuple(changes),
            preserve_violations=tuple(preserve_violations),
            avoid_violations=tuple(avoid_violations),
            target_responses=tuple(target_responses),
            provider_leakage_detected=provider_leakage,
            prompt_stuffing_detected=prompt_stuffing,
            metrics=metrics,
            succeeded=True,
        )

    @staticmethod
    def _scalar_change(
        changes: list[RevisionChange],
        path: str,
        before: Any,
        after: Any,
        *,
        reason: str,
    ) -> None:
        if before == after:
            return
        severity = "hard" if path.endswith(("conflict", "resolution")) else "warning"
        changes.append(RevisionChange(path, "modified", before, after, severity, reason))

    @staticmethod
    def _sequence_changes(
        changes: list[RevisionChange],
        path: str,
        before: Any,
        after: Any,
        *,
        key_name: str | None,
        reason: str,
    ) -> None:
        before_items = list(before or [])
        after_items = list(after or [])
        if before_items == after_items:
            return
        if key_name:
            before_map = {_item_key(item, key_name): item for item in before_items}
            after_map = {_item_key(item, key_name): item for item in after_items}
            for key in before_map.keys() - after_map.keys():
                changes.append(
                    RevisionChange(
                        f"{path}[{key}]",
                        "removed",
                        before_map[key],
                        None,
                        "hard" if "character" in path else "warning",
                        reason,
                    )
                )
            for key in after_map.keys() - before_map.keys():
                changes.append(
                    RevisionChange(
                        f"{path}[{key}]",
                        "added",
                        None,
                        after_map[key],
                        "warning",
                        reason,
                    )
                )
            for key in before_map.keys() & after_map.keys():
                if before_map[key] != after_map[key]:
                    changes.append(
                        RevisionChange(
                            f"{path}[{key}]",
                            "modified",
                            before_map[key],
                            after_map[key],
                            "warning",
                            reason,
                        )
                    )
            before_order = list(before_map)
            after_order = list(after_map)
            if before_order != after_order and set(before_order) == set(after_order):
                changes.append(
                    RevisionChange(path, "reordered", before_order, after_order, "warning", reason)
                )
            return
        before_keys = {_canonical(item) for item in before_items}
        after_keys = {_canonical(item) for item in after_items}
        for item in before_keys - after_keys:
            changes.append(RevisionChange(f"{path}[item]", "removed", item, None, "warning", reason))
        for item in after_keys - before_keys:
            changes.append(RevisionChange(f"{path}[item]", "added", None, item, "warning", reason))

    def _constraint_changes(
        self,
        changes: Sequence[RevisionChange],
        requests: Sequence[CreativeRevisionRequest | dict[str, Any]],
        field_name: str,
    ) -> list[RevisionChange]:
        violations: list[RevisionChange] = []
        seen: set[tuple[str, str]] = set()
        for request in requests:
            constraints = _request_values(request, field_name)
            for constraint in constraints:
                for change in changes:
                    if not _constraint_matches(constraint, change.path, field_name):
                        continue
                    key = (constraint, change.path)
                    if key in seen:
                        continue
                    seen.add(key)
                    violations.append(
                        RevisionChange(
                            change.path,
                            change.change_type,
                            change.before,
                            change.after,
                            "hard",
                            f"违反 revision request {field_name} 约束：{constraint}",
                        )
                    )
        return violations

    @staticmethod
    def _target_responses(
        changes: Sequence[RevisionChange],
        requests: Sequence[CreativeRevisionRequest | dict[str, Any]],
        metadata: dict[str, object],
    ) -> list[RevisionChange]:
        responses: list[RevisionChange] = []
        addressed = {str(item) for item in (metadata.get("addressed_targets", []) or [])}
        for request in requests:
            target = _request_target(request)
            matched = [change for change in changes if _target_matches(request, change.path)]
            if matched:
                responses.extend(matched)
            elif target in addressed:
                responses.append(
                    RevisionChange(
                        f"metadata.addressed_targets[{target}]",
                        "unchanged",
                        target,
                        target,
                        "warning",
                        "candidate metadata explicitly marks the target as addressed",
                    )
                )
        return _unique_changes(responses)


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _item_key(value: Any, key_name: str) -> str:
    if isinstance(value, dict):
        return str(value.get(key_name, "<missing>"))
    return str(value)


def _canonical(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, default=str)


def _request_values(request: CreativeRevisionRequest | dict[str, Any], field_name: str) -> tuple[str, ...]:
    if isinstance(request, dict):
        raw = request.get(field_name, ())
    else:
        raw = getattr(request, field_name, ())
    return tuple(str(item) for item in (raw or ()))


def _request_target(request: CreativeRevisionRequest | dict[str, Any]) -> str:
    if isinstance(request, dict):
        return str(request.get("target", ""))
    return str(request.target)


def _constraint_matches(constraint: str, path: str, field_name: str) -> bool:
    text = constraint.lower()
    lowered = path.lower()
    if any(token in text for token in ("人物", "character")):
        return "character" in lowered
    if any(token in text for token in ("核心冲突", "冲突", "conflict")):
        return "conflict" in lowered or "stakes" in lowered
    if any(token in text for token in ("stakes", "代价")):
        return "stakes" in lowered or "resolution" in lowered
    if any(token in text for token in ("story beats", "故事节拍", "beat")):
        return "story_beats" in lowered or "film_beats" in lowered
    if any(token in text for token in ("结局", "resolution", "ending")):
        return "resolution" in lowered or lowered.endswith("ending") or "story.ending" in lowered
    if "world" in text or "世界观" in text:
        return any(token in lowered for token in ("story_plan.", "story.", "visual_style"))
    if field_name == "avoid" and any(token in text for token in ("provider", "payload", "prompt")):
        return False
    return False


def _target_matches(request: CreativeRevisionRequest | dict[str, Any], path: str) -> bool:
    target = _request_target(request).lower()
    source_codes = _request_values(request, "source_suggestion_codes")
    text = " ".join((target, *(item.lower() for item in source_codes)))
    lowered = path.lower()
    if any(token in text for token in ("climax", "conflict", "resolution", "cost", "stakes")):
        return any(token in lowered for token in ("conflict", "stakes", "resolution", "ending", "story_beats", "film_beats", "climax", "emotion"))
    if any(token in text for token in ("audience", "reveal", "withholding", "viewer", "setup", "payoff")):
        return any(token in lowered for token in ("audience", "reveal", "withholding", "viewer", "story_beats", "film_beats"))
    if any(token in text for token in ("character", "goal", "arc", "choice")):
        return "character" in lowered or "goal" in lowered or "story_beats" in lowered
    if any(token in text for token in ("plan_layer", "visual_style", "projection", "style")):
        return any(token in lowered for token in ("story_plan", "director_plan", "visual_style", "film_beats", "character_sheet"))
    return False


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                return True
            if _contains_forbidden(child):
                return True
        unsafe_fields = value.get("unsafe_fields", [])
        if any(str(item).lower() in _FORBIDDEN_KEYS for item in (unsafe_fields or [])):
            return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_forbidden(item) for item in value)
    return False


def _contains_prompt_stuffing(value: Any) -> bool:
    if isinstance(value, dict):
        if any(str(item).lower() in _PROMPT_STUFFING_TERMS for item in (value.get("unsafe_terms", []) or [])):
            return True
        return any(_contains_prompt_stuffing(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_prompt_stuffing(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(term in lowered for term in _PROMPT_STUFFING_TERMS)
    return False


def _unique_changes(changes: Iterable[RevisionChange]) -> list[RevisionChange]:
    result: list[RevisionChange] = []
    seen: set[tuple[str, str, str]] = set()
    for change in changes:
        key = (change.path, change.change_type, change.reason)
        if key not in seen:
            seen.add(key)
            result.append(change)
    return result


__all__ = ["RevisionChange", "RevisionDiff", "RevisionDiffBuilder"]
