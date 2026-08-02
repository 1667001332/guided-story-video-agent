"""Fail-closed structural, fingerprint, DAG, and boundary validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .execution_bundle import ExecutionBundle
from .execution_fingerprint import (
    contains_forbidden_provider_data,
    execution_bundle_fingerprint,
    execution_plan_fingerprint,
    execution_unit_fingerprint,
    video_job_fingerprint,
)
from .execution_plan import ExecutionPlan


_DEPENDENCY_TYPES = {
    "serial",
    "reference_frame",
    "audio_dependency",
    "subtitle_dependency",
    "manifest_dependency",
    "barrier",
}


@dataclass(frozen=True, slots=True)
class ExecutionValidationDiagnostic:
    code: str
    message: str
    path: str = ""
    severity: str = "error"
    expected: str = ""
    actual: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True, slots=True)
class ExecutionValidationResult:
    artifact_type: str
    valid: bool
    diagnostics: tuple[ExecutionValidationDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.valid

    @property
    def status(self) -> str:
        if self.valid:
            return "fresh"
        if any(item.code.endswith("unknown_lineage") for item in self.diagnostics):
            return "unknown"
        if any(item.code.startswith("invalid") or item.code.endswith("invalid") for item in self.diagnostics):
            return "invalid"
        return "stale"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "valid": self.valid,
            "status": self.status,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _diag(code: str, message: str, path: str = "", **kwargs: str) -> ExecutionValidationDiagnostic:
    return ExecutionValidationDiagnostic(code, message, path, **kwargs)


def validate_execution_plan(plan: ExecutionPlan | None) -> ExecutionValidationResult:
    diagnostics: list[ExecutionValidationDiagnostic] = []
    if plan is None:
        return ExecutionValidationResult(
            "execution_plan",
            False,
            (_diag("missing_execution_plan", "当前没有 ExecutionPlan；请先执行 /build-execution-plan。"),),
        )
    if not isinstance(plan, ExecutionPlan):
        return ExecutionValidationResult("execution_plan", False, (_diag("invalid_execution_plan", "对象不是 ExecutionPlan。"),))
    forbidden = contains_forbidden_provider_data(plan.to_dict())
    diagnostics.extend(
        _diag("provider_boundary_violation", "ExecutionPlan 包含 Provider/API/runtime 敏感字段。", path)
        for path in forbidden
    )
    if not plan.capability_snapshot.snapshot_id:
        diagnostics.append(_diag("missing_capability_snapshot", "ExecutionPlan 缺少 capability snapshot。", "capability_snapshot"))
    if not plan.execution_units:
        diagnostics.append(_diag("invalid_execution_plan", "ExecutionPlan 至少需要一个 execution unit。", "execution_units"))

    unit_ids = [item.execution_unit_id for item in plan.execution_units]
    if len(unit_ids) != len(set(unit_ids)):
        diagnostics.append(_diag("invalid_duplicate_execution_unit_id", "ExecutionUnit ID 不得重复。", "execution_units"))
    assignment_ids = [item.assignment_id for item in plan.provider_assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        diagnostics.append(_diag("invalid_duplicate_provider_assignment_id", "ProviderAssignment ID 不得重复。", "provider_assignments"))
    assignment_lookup = set(assignment_ids)
    snapshot_ids = {plan.capability_snapshot.snapshot_id}
    for index, unit in enumerate(plan.execution_units):
        if unit.provider_assignment_id not in assignment_lookup:
            diagnostics.append(
                _diag(
                    "dangling_provider_assignment",
                    "ExecutionUnit 引用了不存在的 ProviderAssignment。",
                    f"execution_units[{index}].provider_assignment_id",
                )
            )
        if unit.execution_unit_fingerprint != execution_unit_fingerprint(unit):
            diagnostics.append(
                _diag(
                    "execution_unit_fingerprint_mismatch",
                    "ExecutionUnit fingerprint 不正确。",
                    f"execution_units[{index}].execution_unit_fingerprint",
                )
            )
    for index, assignment in enumerate(plan.provider_assignments):
        if assignment.capability_snapshot_id not in snapshot_ids:
            diagnostics.append(
                _diag(
                    "missing_capability_snapshot",
                    "ProviderAssignment 引用了不存在的 capability snapshot。",
                    f"provider_assignments[{index}].capability_snapshot_id",
                )
            )
        if assignment.provider_key != plan.capability_snapshot.provider_key:
            diagnostics.append(
                _diag(
                    "capability_snapshot_mismatch",
                    "ProviderAssignment.provider_key 与 capability snapshot 不一致。",
                    f"provider_assignments[{index}].provider_key",
                )
            )

    unit_lookup = set(unit_ids)
    edge_pairs: set[tuple[str, str, str]] = set()
    incoming_sources: dict[str, set[str]] = {unit_id: set() for unit_id in unit_ids}
    for index, edge in enumerate(plan.dependency_graph):
        path = f"dependency_graph[{index}]"
        if edge.from_unit_id == edge.to_unit_id:
            diagnostics.append(_diag("invalid_self_dependency", "DAG 不允许自引用。", path))
        if edge.from_unit_id not in unit_lookup or edge.to_unit_id not in unit_lookup:
            diagnostics.append(_diag("invalid_missing_dependency_node", "DependencyEdge 引用了不存在的 ExecutionUnit。", path))
        if edge.dependency_type not in _DEPENDENCY_TYPES:
            diagnostics.append(_diag("invalid_dependency_type", "不支持的 dependency_type。", f"{path}.dependency_type"))
        edge_pairs.add((edge.from_unit_id, edge.to_unit_id, edge.dependency_type))
        incoming_sources.setdefault(edge.to_unit_id, set()).add(edge.from_unit_id)

    for index, unit in enumerate(plan.execution_units):
        expected = incoming_sources.get(unit.execution_unit_id, set())
        actual = set(unit.depends_on)
        if expected != actual:
            diagnostics.append(
                _diag(
                    "dependency_graph_mismatch",
                    "ExecutionUnit.depends_on 与 dependency_graph 不一致。",
                    f"execution_units[{index}].depends_on",
                    expected=",".join(sorted(expected)),
                    actual=",".join(sorted(actual)),
                )
            )
        for reference_index, reference in enumerate(unit.reference_inputs):
            if reference.source_unit_id and (
                reference.source_unit_id,
                unit.execution_unit_id,
                "reference_frame",
            ) not in edge_pairs:
                diagnostics.append(
                    _diag(
                        "reference_dependency_missing",
                        "ReferenceInput 必须对应 reference_frame DependencyEdge。",
                        f"execution_units[{index}].reference_inputs[{reference_index}]",
                    )
                )

    if _has_cycle(unit_lookup, plan.dependency_graph):
        diagnostics.append(_diag("invalid_dependency_cycle", "dependency_graph 必须是 DAG。", "dependency_graph"))
    if (
        not plan.source_movie_plan_id
        or plan.source_movie_plan_version < 1
        or not plan.source_movie_plan_fingerprint
        or not plan.source_movie_plan_lineage_token
    ):
        diagnostics.append(_diag("execution_plan_unknown_lineage", "ExecutionPlan 缺少 MoviePlan provenance。", "source_movie_plan"))
    if not plan.source_film_ir_id or not plan.source_film_ir_fingerprint:
        diagnostics.append(_diag("execution_plan_unknown_lineage", "ExecutionPlan 缺少 FilmIR provenance。", "source_film_ir"))
    if not plan.source_movie_ir_id or not plan.source_movie_ir_fingerprint:
        diagnostics.append(_diag("execution_plan_unknown_lineage", "ExecutionPlan 缺少 MovieIR provenance。", "source_movie_ir"))
    expected_fingerprint = execution_plan_fingerprint(plan)
    if plan.execution_plan_fingerprint != expected_fingerprint:
        diagnostics.append(
            _diag(
                "execution_plan_fingerprint_mismatch",
                "ExecutionPlan fingerprint 不正确。",
                "execution_plan_fingerprint",
                expected=expected_fingerprint,
                actual=plan.execution_plan_fingerprint,
            )
        )
    return ExecutionValidationResult("execution_plan", not diagnostics, tuple(diagnostics))


def validate_execution_bundle(bundle: ExecutionBundle | None) -> ExecutionValidationResult:
    if bundle is None:
        return ExecutionValidationResult(
            "execution_bundle",
            False,
            (_diag("missing_execution_bundle", "当前没有 ExecutionBundle；请先执行 /build-execution-plan。"),),
        )
    if not isinstance(bundle, ExecutionBundle):
        return ExecutionValidationResult("execution_bundle", False, (_diag("invalid_execution_bundle", "对象不是 ExecutionBundle。"),))
    diagnostics = list(validate_execution_plan(bundle.execution_plan).diagnostics)
    forbidden = contains_forbidden_provider_data(bundle.to_dict())
    diagnostics.extend(
        _diag("provider_boundary_violation", "ExecutionBundle 包含 Provider/API/runtime 敏感字段。", path)
        for path in forbidden
    )
    jobs = bundle.video_jobs
    job_ids = [job.job_id for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        diagnostics.append(_diag("duplicate_video_job_id", "VideoJob ID 不得重复。", "video_jobs"))
    job_map = {job.job_id: job for job in jobs}
    referenced: set[str] = set()
    for index, unit in enumerate(bundle.execution_plan.execution_units):
        referenced.add(unit.video_job_id)
        job = job_map.get(unit.video_job_id)
        if job is None:
            diagnostics.append(_diag("missing_video_job", "ExecutionUnit 引用了不存在的 VideoJob。", f"execution_units[{index}].video_job_id"))
            continue
        if unit.video_job_fingerprint != job.video_job_fingerprint:
            diagnostics.append(_diag("video_job_fingerprint_mismatch", "VideoJob fingerprint 与 ExecutionUnit 声明不一致。", f"execution_units[{index}].video_job_fingerprint"))
        if job.video_job_fingerprint != video_job_fingerprint(job):
            diagnostics.append(_diag("video_job_fingerprint_invalid", "VideoJob fingerprint 不正确。", f"video_jobs[{job.job_id}].video_job_fingerprint"))
        plan = bundle.execution_plan
        if job.source_movie_ir_id and job.source_movie_ir_id != plan.source_movie_ir_id:
            diagnostics.append(_diag("video_job_source_mismatch", "VideoJob 来源 MovieIR 与 ExecutionPlan 不一致。", f"video_jobs[{job.job_id}].source_movie_ir_id"))
    unused = sorted(set(job_ids) - referenced)
    if unused:
        diagnostics.append(_diag("unused_video_job", "ExecutionBundle 不得包含未被 ExecutionUnit 引用的 VideoJob。", "video_jobs", actual=",".join(unused)))
    expected_bundle = execution_bundle_fingerprint(bundle)
    if bundle.bundle_fingerprint != expected_bundle:
        diagnostics.append(_diag("bundle_fingerprint_mismatch", "ExecutionBundle fingerprint 不正确。", "bundle_fingerprint", expected=expected_bundle, actual=bundle.bundle_fingerprint))
    return ExecutionValidationResult("execution_bundle", not diagnostics, tuple(diagnostics))


def _has_cycle(unit_ids: set[str], edges: tuple[Any, ...]) -> bool:
    adjacency: dict[str, set[str]] = {item: set() for item in unit_ids}
    indegree: dict[str, int] = {item: 0 for item in unit_ids}
    for edge in edges:
        if edge.from_unit_id not in unit_ids or edge.to_unit_id not in unit_ids:
            continue
        if edge.to_unit_id not in adjacency[edge.from_unit_id]:
            adjacency[edge.from_unit_id].add(edge.to_unit_id)
            indegree[edge.to_unit_id] += 1
    queue = [item for item, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        current = queue.pop(0)
        visited += 1
        for child in adjacency[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited != len(unit_ids)


__all__ = [
    "ExecutionValidationDiagnostic",
    "ExecutionValidationResult",
    "validate_execution_bundle",
    "validate_execution_plan",
]
