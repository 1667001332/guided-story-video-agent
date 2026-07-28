from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from .agent import OpenAIStoryAgent, RuleBasedStoryAgent, StoryAgent
from .rendering import StoryRenderer
from .selfplay import run_selfplay
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider, sanitize_remote_url


@dataclass(frozen=True)
class BatchCase:
    case_id: str
    direction: str
    target_seconds: int | None = None


AgentFactory = Callable[[], StoryAgent]
RendererFactory = Callable[[], Any]
ProgressCallback = Callable[[str], None]
RUN_IDENTITY_SCHEMA_VERSION = 2
RENDER_SUCCESS_STATUSES = {"succeeded", "succeeded_with_warnings"}


def default_cases_source() -> Traversable:
    return resources.files("guided_story_agent").joinpath(
        "resources",
        "batch_cases.jsonl",
    )


def _pipeline_fingerprint() -> dict[str, str]:
    """Identify the installed code, prompts, and rules that affect a batch run."""
    digest = hashlib.sha256()
    package_root = resources.files("guided_story_agent")

    def visit(node: Traversable, relative: str = "") -> None:
        children = sorted(node.iterdir(), key=lambda item: item.name)
        for child in children:
            child_relative = f"{relative}/{child.name}".lstrip("/")
            if child.is_dir():
                if child.name != "__pycache__":
                    visit(child, child_relative)
                continue
            if Path(child.name).suffix.lower() not in {".py", ".md", ".jsonl"}:
                continue
            digest.update(child_relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(child.read_bytes())
            digest.update(b"\0")

    visit(package_root)
    try:
        package_version = importlib.metadata.version("guided-story-video-agent")
    except importlib.metadata.PackageNotFoundError:
        package_version = "uninstalled"
    return {
        "package_version": package_version,
        "content_sha256": digest.hexdigest(),
    }


def _safe_endpoint(value: Any) -> str:
    return sanitize_remote_url(str(value or ""))


def load_cases(path: str | Path | Traversable) -> list[BatchCase]:
    source = Path(path).expanduser().resolve() if isinstance(path, (str, Path)) else path
    if not source.is_file():
        raise FileNotFoundError(f"测试用例文件不存在：{source}")

    suffix = Path(source.name).suffix.lower()
    if suffix == ".jsonl":
        raw_items = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    elif suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        raw_items = raw.get("cases", []) if isinstance(raw, dict) else raw
    elif suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_items = list(csv.DictReader(handle))
    elif suffix == ".txt":
        raw_items = [
            {"direction": line.strip()}
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        raise ValueError("测试用例仅支持 .jsonl、.json、.csv 或 .txt。")

    if not isinstance(raw_items, list):
        raise ValueError("JSON 测试用例必须是数组，或包含 cases 数组。")

    cases: list[BatchCase] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items, 1):
        if isinstance(raw_item, str):
            raw_item = {"direction": raw_item}
        if not isinstance(raw_item, dict):
            raise ValueError(f"第 {index} 条测试用例必须是对象或字符串。")

        direction = str(raw_item.get("direction", "")).strip()
        if not direction:
            raise ValueError(f"第 {index} 条测试用例缺少 direction。")
        case_id = str(raw_item.get("id") or raw_item.get("case_id") or f"case-{index:03d}")
        case_id = case_id.strip()
        if not case_id:
            raise ValueError(f"第 {index} 条测试用例的 id 为空。")
        if case_id in seen_ids:
            raise ValueError(f"测试用例 id 重复：{case_id}")
        seen_ids.add(case_id)

        raw_seconds = raw_item.get("target_seconds")
        target_seconds = None
        if raw_seconds not in (None, ""):
            target_seconds = int(raw_seconds)
            if not 15 <= target_seconds <= 300:
                raise ValueError(f"测试用例 {case_id} 的 target_seconds 必须在 15–300 之间。")
        cases.append(BatchCase(case_id, direction, target_seconds))

    if not cases:
        raise ValueError("测试用例文件中没有可执行用例。")
    return cases


def run_batch(
    *,
    cases: Iterable[BatchCase],
    output_dir: str | Path,
    agent_factory: AgentFactory,
    repeat: int = 1,
    target_seconds_override: int | None = None,
    render: bool = False,
    renderer_factory: RendererFactory | None = None,
    require_live_text: bool = True,
    llm_judge: bool = False,
    resume: bool = False,
    retries: int = 0,
    delay_seconds: float = 0.0,
    progress_callback: ProgressCallback | None = print,
) -> dict[str, Any]:
    if repeat < 1:
        raise ValueError("repeat 必须至少为 1。")
    if retries < 0:
        raise ValueError("retries 不能为负数。")
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds 必须是有限的非负数。")
    if target_seconds_override is not None and not 15 <= target_seconds_override <= 300:
        raise ValueError("target_seconds_override 必须在 15–300 之间。")

    case_list = list(cases)
    if not case_list:
        raise ValueError("没有可执行的测试用例。")

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    skipped = 0
    resume_rejections: dict[str, int] = {}
    total_runs = len(case_list) * repeat
    run_number = 0
    planned_run_ids = _planned_run_ids(case_list, repeat)
    pipeline_fingerprint = _pipeline_fingerprint()

    for case_index, case in enumerate(case_list, 1):
        for repetition in range(1, repeat + 1):
            run_number += 1
            run_id = planned_run_ids[(case_index, repetition)]
            run_dir = target / "runs" / run_id
            result_path = run_dir / "result.json"
            selected_seconds = (
                target_seconds_override
                if target_seconds_override is not None
                else case.target_seconds
            )
            prepared_agent: StoryAgent | None = None
            prepared_renderer: Any | None = None
            preparation_error: Exception | None = None
            try:
                prepared_agent = agent_factory()
                prepared_renderer = _make_renderer(
                    render=render,
                    renderer_factory=renderer_factory,
                )
            except Exception as exc:
                preparation_error = exc

            run_identity = _build_run_identity(
                case=case,
                case_index=case_index,
                repetition=repetition,
                repeat=repeat,
                target_seconds=selected_seconds,
                render=render,
                require_live_text=require_live_text,
                llm_judge=llm_judge,
                agent=prepared_agent,
                renderer=prepared_renderer,
                preparation_error=preparation_error,
                pipeline_fingerprint=pipeline_fingerprint,
            )
            identity_hash = _identity_hash(run_identity)
            run_dir.mkdir(parents=True, exist_ok=True)
            identity_record = {
                "run_identity": run_identity,
                "run_identity_hash": identity_hash,
            }
            identity_path = run_dir / "run_identity.json"
            previous_identity_record = _read_json(identity_path) if resume else None
            _write_json(identity_path, identity_record)
            existing = _read_json(result_path) if resume else None
            if existing is None and previous_identity_record is not None:
                existing = _recover_interrupted_result(
                    run_dir,
                    previous_identity_record,
                )
            active_session: GuidedStorySession | None = None
            active_bench: dict[str, Any] | None = None
            artifact_dir: Path | None = None
            render_output_dir: Path | None = None
            base_attempt = 0
            cumulative_elapsed = 0.0
            resumed_incomplete = False
            if existing is not None:
                can_skip, resume_reason = _can_resume_success(
                    existing,
                    run_identity=run_identity,
                    identity_hash=identity_hash,
                    render=render,
                    require_live_text=require_live_text,
                )
                if can_skip:
                    resumed_result = dict(existing)
                    resumed_result["resumed"] = True
                    resumed_result["resume_identity_verified"] = True
                    results.append(resumed_result)
                    skipped += 1
                    _progress(
                        progress_callback,
                        f"[{run_number}/{total_runs}] 跳过身份一致的成功用例 {case.case_id}",
                    )
                    continue
                recovered, recovery_reason = _load_incomplete_render_state(
                    existing,
                    run_dir=run_dir,
                    run_identity=run_identity,
                    identity_hash=identity_hash,
                    render=render,
                    agent=prepared_agent,
                )
                if recovered is not None:
                    (
                        active_session,
                        active_bench,
                        artifact_dir,
                        render_output_dir,
                        base_attempt,
                        cumulative_elapsed,
                    ) = recovered
                    resumed_incomplete = True
                    _progress(
                        progress_callback,
                        f"[{run_number}/{total_runs}] 继续上次未完成的视频任务 {case.case_id}",
                    )
                else:
                    rejection = recovery_reason or resume_reason
                    resume_rejections[rejection] = resume_rejections.get(rejection, 0) + 1
                    _progress(
                        progress_callback,
                        f"[{run_number}/{total_runs}] 不复用旧结果 {case.case_id}：{rejection}",
                    )

            result: dict[str, Any] | None = None
            for local_attempt in range(1, retries + 2):
                attempt = base_attempt + local_attempt
                attempt_dir = run_dir / f"attempt-{attempt}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                _progress(
                    progress_callback,
                    f"[{run_number}/{total_runs}] {case.case_id}，第 {attempt} 次执行",
                )
                started = time.perf_counter()
                phase = "prepare"
                try:
                    if active_session is None:
                        if local_attempt == 1:
                            if preparation_error is not None:
                                raise preparation_error
                            if prepared_agent is None:
                                raise RuntimeError("文本 Agent 初始化失败。")
                            agent = prepared_agent
                        else:
                            agent = agent_factory()
                        phase = "story"
                        output = run_selfplay(
                            agent=agent,
                            direction=case.direction,
                            target_seconds=selected_seconds,
                            output_dir=attempt_dir,
                            render=False,
                            renderer=None,
                            require_live_text=require_live_text,
                            llm_judge=llm_judge,
                        )
                        active_session = output["session"]
                        active_bench = dict(output["bench"])
                        artifact_dir = attempt_dir

                    if artifact_dir is None or active_bench is None:
                        raise RuntimeError("批测产物状态初始化失败。")

                    status = "succeeded"
                    error = ""
                    warning = ""
                    if render:
                        phase = "render"
                        renderer = (
                            prepared_renderer
                            if local_attempt == 1 and prepared_renderer is not None
                            else _make_renderer(
                                render=True,
                                renderer_factory=renderer_factory,
                            )
                        )
                        render_output_dir = render_output_dir or artifact_dir / "video"
                        manifest = active_session.render_confirmed_plan(
                            renderer,
                            render_output_dir,
                        )
                        render_status = str(manifest.status or "unknown")
                        active_bench["render_status"] = render_status
                        active_bench["failed_shots"] = list(manifest.failed_shots)
                        active_bench["reused_shots"] = list(manifest.reused_shots)
                        _write_json(artifact_dir / "bench.json", active_bench)
                        active_session.save(artifact_dir / "session.json")
                        if render_status not in RENDER_SUCCESS_STATUSES:
                            status = "failed"
                            error = str(manifest.error).strip() or f"视频生成状态为 {render_status}"
                        elif render_status == "succeeded_with_warnings":
                            warning = (
                                str(manifest.error).strip()
                                or "视频生成完成，但 Provider 返回警告。"
                            )

                    attempt_elapsed = time.perf_counter() - started
                    cumulative_elapsed += attempt_elapsed
                    result = {
                        "run_id": run_id,
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "direction": case.direction,
                        "target_seconds_requested": selected_seconds,
                        "status": status,
                        "attempts": attempt,
                        "elapsed_seconds": round(cumulative_elapsed, 3),
                        "attempt_elapsed_seconds": round(attempt_elapsed, 3),
                        "output_dir": str(artifact_dir),
                        "attempt_dir": str(attempt_dir),
                        "render_output_dir": (str(render_output_dir) if render_output_dir else ""),
                        "bench": dict(active_bench),
                        "error_type": "RenderFailed" if error else "",
                        "error": error,
                        "warning": warning,
                        "phase": phase,
                        "run_identity": run_identity,
                        "run_identity_hash": identity_hash,
                        "resumed": False,
                        "resumed_incomplete": resumed_incomplete,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception as exc:
                    attempt_elapsed = time.perf_counter() - started
                    cumulative_elapsed += attempt_elapsed
                    output_dir = artifact_dir or attempt_dir
                    result = {
                        "run_id": run_id,
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "direction": case.direction,
                        "target_seconds_requested": selected_seconds,
                        "status": "failed",
                        "attempts": attempt,
                        "elapsed_seconds": round(cumulative_elapsed, 3),
                        "attempt_elapsed_seconds": round(attempt_elapsed, 3),
                        "output_dir": str(output_dir),
                        "attempt_dir": str(attempt_dir),
                        "render_output_dir": (str(render_output_dir) if render_output_dir else ""),
                        "bench": dict(active_bench or {}),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "warning": "",
                        "traceback": traceback.format_exc(),
                        "phase": phase,
                        "run_identity": run_identity,
                        "run_identity_hash": identity_hash,
                        "resumed": False,
                        "resumed_incomplete": resumed_incomplete,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }

                run_dir.mkdir(parents=True, exist_ok=True)
                _write_json(attempt_dir / "attempt.json", result)
                _write_json(result_path, result)
                if result["status"] == "succeeded" or local_attempt > retries:
                    break
                _progress(
                    progress_callback,
                    f"  失败：{result['error']}；准备重试",
                )
                if delay_seconds:
                    time.sleep(delay_seconds)

            assert result is not None
            results.append(result)
            if result["status"] == "failed":
                _progress(progress_callback, f"  最终失败：{result['error']}")
            else:
                _progress(
                    progress_callback,
                    f"  成功，耗时 {result['elapsed_seconds']} 秒",
                )
            if delay_seconds and run_number < total_runs:
                time.sleep(delay_seconds)

    finished_at = datetime.now(timezone.utc)
    summary = _build_summary(
        results,
        started_at=started_at,
        finished_at=finished_at,
        skipped=skipped,
        render=render,
        require_live_text=require_live_text,
        resume_rejections=resume_rejections,
    )
    _write_json(target / "summary.json", summary)
    _write_results_jsonl(target / "results.jsonl", results)
    _write_results_csv(target / "results.csv", results)
    return {"output_dir": str(target), "summary": summary, "results": results}


def _make_renderer(
    *,
    render: bool,
    renderer_factory: RendererFactory | None,
) -> Any | None:
    if not render:
        return None
    if renderer_factory is not None:
        return renderer_factory()
    return StoryRenderer(AgnesVideoProvider.from_env())


def _build_run_identity(
    *,
    case: BatchCase,
    case_index: int,
    repetition: int,
    repeat: int,
    target_seconds: int | None,
    render: bool,
    require_live_text: bool,
    llm_judge: bool,
    agent: StoryAgent | None,
    renderer: Any | None,
    preparation_error: Exception | None,
    pipeline_fingerprint: dict[str, str],
) -> dict[str, Any]:
    text_client = getattr(agent, "client", None) if agent is not None else None
    text_base_url = getattr(text_client, "base_url", "") if text_client is not None else ""
    video_provider = getattr(renderer, "provider", None) if renderer is not None else None
    return {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "pipeline": dict(pipeline_fingerprint),
        "case": {
            "case_id": case.case_id,
            "case_index": case_index,
            "direction": case.direction,
            "target_seconds": target_seconds,
            "repetition": repetition,
            "repeat": repeat,
        },
        "text": {
            "agent_class": type(agent).__name__ if agent is not None else "unavailable",
            "provider": str(
                getattr(agent, "provider_name", type(agent).__name__)
                if agent is not None
                else "unavailable"
            ),
            "model": str(
                getattr(agent, "model", "rule-based") if agent is not None else "unavailable"
            ),
            "config_source": str(
                getattr(agent, "config_source", "constructor")
                if agent is not None
                else "unavailable"
            ),
            "base_url": _safe_endpoint(text_base_url),
            "json_mode": str(getattr(agent, "json_mode", "")) if agent is not None else "",
            "live_client_available": text_client is not None,
            "require_live_text": require_live_text,
            "llm_judge": llm_judge,
        },
        "video": {
            "enabled": render,
            "renderer_class": (type(renderer).__name__ if renderer is not None else ""),
            "provider": str(
                getattr(video_provider, "provider_name", type(video_provider).__name__)
                if video_provider is not None
                else ""
            ),
            "model": str(
                getattr(video_provider, "endpoint", getattr(video_provider, "model", ""))
                if video_provider is not None
                else ""
            ),
            "api_root": _safe_endpoint(getattr(video_provider, "api_root", "")),
            "config_source": str(getattr(video_provider, "config_source", "") or ""),
        },
        "preparation_error": (
            {
                "type": type(preparation_error).__name__,
                "message": str(preparation_error),
            }
            if preparation_error is not None
            else None
        ),
    }


def _identity_hash(run_identity: dict[str, Any]) -> str:
    canonical = json.dumps(
        run_identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _recover_interrupted_result(
    run_dir: Path,
    identity_record: dict[str, Any],
) -> dict[str, Any] | None:
    """Recover enough state to resume when the process stopped before result.json."""
    run_identity = identity_record.get("run_identity")
    identity_hash = str(identity_record.get("run_identity_hash") or "")
    if not isinstance(run_identity, dict) or not identity_hash:
        return None

    attempts: list[tuple[int, Path]] = []
    try:
        children = list(run_dir.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir() or not child.name.startswith("attempt-"):
            continue
        try:
            number = int(child.name.removeprefix("attempt-"))
        except ValueError:
            continue
        if number > 0 and (child / "session.json").is_file():
            attempts.append((number, child))
    if not attempts:
        return None

    attempt, artifact_dir = max(attempts, key=lambda item: item[0])
    bench = _read_json(artifact_dir / "bench.json") or {}
    session_data = _read_json(artifact_dir / "session.json") or {}
    manifest = session_data.get("render_manifest")
    render_status = (
        str(manifest.get("status") or "")
        if isinstance(manifest, dict)
        else str(bench.get("render_status") or "")
    )
    if render_status:
        bench["render_status"] = render_status
    succeeded = render_status in RENDER_SUCCESS_STATUSES
    warning = (
        str(manifest.get("error") or "")
        if succeeded and render_status == "succeeded_with_warnings" and isinstance(manifest, dict)
        else ""
    )
    case = run_identity.get("case", {})
    return {
        "run_id": run_dir.name,
        "case_id": str(case.get("case_id") or ""),
        "repetition": int(case.get("repetition") or 1),
        "direction": str(case.get("direction") or ""),
        "target_seconds_requested": case.get("target_seconds"),
        "status": "succeeded" if succeeded else "failed",
        "attempts": attempt,
        "elapsed_seconds": 0.0,
        "attempt_elapsed_seconds": 0.0,
        "output_dir": str(artifact_dir),
        "attempt_dir": str(artifact_dir),
        "render_output_dir": str(artifact_dir / "video"),
        "bench": bench,
        "error_type": "" if succeeded else "InterruptedRun",
        "error": "" if succeeded else "上次执行在写入完整结果前中断。",
        "warning": warning,
        "phase": "render",
        "run_identity": run_identity,
        "run_identity_hash": identity_hash,
        "resumed": False,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_incomplete_render_state(
    existing: dict[str, Any],
    *,
    run_dir: Path,
    run_identity: dict[str, Any],
    identity_hash: str,
    render: bool,
    agent: StoryAgent | None,
) -> tuple[
    tuple[
        GuidedStorySession,
        dict[str, Any],
        Path,
        Path,
        int,
        float,
    ]
    | None,
    str,
]:
    if not render or existing.get("phase") != "render" or existing.get("status") == "succeeded":
        return None, ""
    if (
        existing.get("run_identity") != run_identity
        or str(existing.get("run_identity_hash") or "") != identity_hash
    ):
        return None, "identity_mismatch"
    if agent is None:
        return None, "resume_agent_unavailable"

    run_root = run_dir.resolve()

    def checked_path(raw: Any, *, fallback: Path) -> Path:
        candidate = Path(str(raw or fallback)).expanduser().resolve()
        try:
            candidate.relative_to(run_root)
        except ValueError as exc:
            raise ValueError("旧结果路径不在当前运行目录内。") from exc
        return candidate

    try:
        artifact_dir = checked_path(
            existing.get("output_dir"),
            fallback=run_dir / "attempt-1",
        )
        session_path = artifact_dir / "session.json"
        if not session_path.is_file():
            return None, "incomplete_session_missing"
        session = GuidedStorySession.load(session_path, agent=agent)
        if session.stage.value != "render_ready":
            return None, "incomplete_session_not_renderable"
        render_output_dir = checked_path(
            existing.get("render_output_dir"),
            fallback=artifact_dir / "video",
        )
        bench = existing.get("bench")
        if not isinstance(bench, dict):
            bench = _read_json(artifact_dir / "bench.json")
        if not isinstance(bench, dict):
            return None, "incomplete_bench_missing"
        raw_attempt = existing.get("attempts", 0)
        if isinstance(raw_attempt, bool) or not isinstance(raw_attempt, int) or raw_attempt < 1:
            return None, "incomplete_attempt_invalid"
        raw_elapsed = existing.get("elapsed_seconds", 0.0)
        elapsed = float(raw_elapsed)
        if not math.isfinite(elapsed) or elapsed < 0:
            elapsed = 0.0
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, "incomplete_session_invalid"

    return (
        (
            session,
            dict(bench),
            artifact_dir,
            render_output_dir,
            raw_attempt,
            elapsed,
        ),
        "",
    )


def _can_resume_success(
    existing: dict[str, Any],
    *,
    run_identity: dict[str, Any],
    identity_hash: str,
    render: bool,
    require_live_text: bool,
) -> tuple[bool, str]:
    existing_identity = existing.get("run_identity")
    existing_hash = str(existing.get("run_identity_hash") or "")
    if not isinstance(existing_identity, dict) or not existing_hash:
        return False, "legacy_missing_identity"
    if existing_hash != identity_hash or existing_identity != run_identity:
        return False, "identity_mismatch"
    if existing.get("status") != "succeeded":
        return False, "previous_not_succeeded"
    output_dir = str(existing.get("output_dir") or "")
    if not output_dir or not Path(output_dir).is_dir():
        return False, "missing_output_dir"

    bench = existing.get("bench")
    if not isinstance(bench, dict):
        return False, "missing_bench"
    if require_live_text:
        fallback_count = bench.get("text_fallback_count")
        if (
            bench.get("text_api_mode") != "live-required"
            or not isinstance(fallback_count, (int, float))
            or isinstance(fallback_count, bool)
            or fallback_count != 0
        ):
            return False, "strict_text_not_verified"
    if render and bench.get("render_status") not in RENDER_SUCCESS_STATUSES:
        return False, "render_not_verified"
    return True, "identity_match"


def _build_summary(
    results: list[dict[str, Any]],
    *,
    started_at: datetime,
    finished_at: datetime,
    skipped: int,
    render: bool,
    require_live_text: bool,
    resume_rejections: dict[str, int],
) -> dict[str, Any]:
    succeeded = [item for item in results if item["status"] == "succeeded"]
    failed = [item for item in results if item["status"] == "failed"]
    warned = [item for item in succeeded if str(item.get("warning") or "").strip()]
    executed = [item for item in results if not item.get("resumed")]
    executed_succeeded = [item for item in executed if item["status"] == "succeeded"]
    elapsed = [float(item.get("elapsed_seconds", 0.0)) for item in executed]
    benches = [item.get("bench", {}) for item in succeeded]

    def average_metric(name: str) -> float | None:
        values = [
            float(bench[name])
            for bench in benches
            if isinstance(bench.get(name), (int, float)) and not isinstance(bench.get(name), bool)
        ]
        return round(statistics.mean(values), 3) if values else None

    fallback_runs = sum(
        int(item.get("bench", {}).get("text_fallback_count", 0)) > 0 for item in results
    )
    total = len(results)
    return {
        "schema_version": 2,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_time_seconds": round((finished_at - started_at).total_seconds(), 3),
        "total_runs": total,
        "executed_runs": len(executed),
        "resumed_runs": skipped,
        "succeeded": len(succeeded),
        "executed_succeeded": len(executed_succeeded),
        "failed": len(failed),
        "succeeded_with_warnings": len(warned),
        "skipped_by_resume": skipped,
        "success_rate": round(len(succeeded) / max(1, total), 4),
        "executed_success_rate": round(
            len(executed_succeeded) / max(1, len(executed)),
            4,
        ),
        "resume_rejections": dict(sorted(resume_rejections.items())),
        "fallback_runs": fallback_runs,
        "render_enabled": render,
        "live_text_required": require_live_text,
        "average_run_seconds": round(statistics.mean(elapsed), 3) if elapsed else 0.0,
        "p50_run_seconds": _percentile(elapsed, 0.50),
        "p95_run_seconds": _percentile(elapsed, 0.95),
        "average_metrics": {
            "idea_diversity": average_metric("idea_diversity"),
            "duplicate_rate": average_metric("duplicate_rate"),
            "selection_retention": average_metric("selection_retention"),
            "ai_fill_transparency": average_metric("ai_fill_transparency"),
            "visual_anchor_coverage": average_metric("visual_anchor_coverage"),
            "shot_diversity": average_metric("shot_diversity"),
            "story_conflict_coverage": average_metric("story_conflict_coverage"),
            "story_ending_coverage": average_metric("story_ending_coverage"),
            "scene_state_bridge_coverage": average_metric(
                "scene_state_bridge_coverage"
            ),
            "storyboard_action_uniqueness": average_metric(
                "storyboard_action_uniqueness"
            ),
            "storyboard_transition_explicitness": average_metric(
                "storyboard_transition_explicitness"
            ),
            "storyboard_atomic_action_rate": average_metric(
                "storyboard_atomic_action_rate"
            ),
        },
        "duration_tolerance_pass_rate": round(
            sum(bool(bench.get("duration_within_tolerance")) for bench in benches)
            / max(1, len(benches)),
            4,
        ),
        "failure_types": _count_values(
            str(item.get("error_type") or "UnknownError") for item in failed
        ),
        "failed_run_ids": [str(item["run_id"]) for item in failed],
        "warning_run_ids": [str(item["run_id"]) for item in warned],
    }


def _planned_run_ids(
    cases: list[BatchCase],
    repeat: int,
) -> dict[tuple[int, int], str]:
    planned: dict[tuple[int, int], str] = {}
    used: set[str] = set()
    for case_index, case in enumerate(cases, 1):
        for repetition in range(1, repeat + 1):
            candidate = _run_id(
                case_index,
                case.case_id,
                repetition,
                repeat,
            )
            collision_key = candidate.casefold()
            if collision_key in used:
                collision_suffix = hashlib.sha256(
                    (f"{case_index}\0{case.case_id}\0{repetition}\0{repeat}").encode("utf-8")
                ).hexdigest()[:16]
                candidate = f"{candidate}-{collision_suffix}"
                collision_key = candidate.casefold()
            if collision_key in used:
                raise ValueError(f"无法为测试用例 {case.case_id} 生成唯一运行目录。")
            used.add(collision_key)
            planned[(case_index, repetition)] = candidate
    return planned


def _run_id(case_index: int, case_id: str, repetition: int, repeat: int) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-" for char in case_id
    ).strip("-_.")
    safe = safe[:48] or f"case-{case_index:03d}"
    case_digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:10]
    suffix = f"-r{repetition:02d}" if repeat > 1 else ""
    return f"{case_index:03d}-{safe}-{case_digest}{suffix}"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 3)


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_results_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "case_id",
        "repetition",
        "status",
        "resumed",
        "attempts",
        "elapsed_seconds",
        "target_seconds_requested",
        "text_provider",
        "text_model",
        "text_fallback_count",
        "idea_diversity",
        "duplicate_rate",
        "duration_within_tolerance",
        "render_status",
        "error_type",
        "error",
        "warning",
        "output_dir",
        "attempt_dir",
        "render_output_dir",
        "run_identity_hash",
        "direction",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            bench = item.get("bench", {})
            row = {
                "run_id": item.get("run_id", ""),
                "case_id": item.get("case_id", ""),
                "repetition": item.get("repetition", ""),
                "status": item.get("status", ""),
                "resumed": item.get("resumed", False),
                "attempts": item.get("attempts", ""),
                "elapsed_seconds": item.get("elapsed_seconds", ""),
                "target_seconds_requested": item.get("target_seconds_requested", ""),
                "text_provider": bench.get("text_provider", ""),
                "text_model": bench.get("text_model", ""),
                "text_fallback_count": bench.get("text_fallback_count", ""),
                "idea_diversity": bench.get("idea_diversity", ""),
                "duplicate_rate": bench.get("duplicate_rate", ""),
                "duration_within_tolerance": bench.get("duration_within_tolerance", ""),
                "render_status": bench.get("render_status", ""),
                "error_type": item.get("error_type", ""),
                "error": item.get("error", ""),
                "warning": item.get("warning", ""),
                "output_dir": item.get("output_dir", ""),
                "attempt_dir": item.get("attempt_dir", ""),
                "render_output_dir": item.get("render_output_dir", ""),
                "run_identity_hash": item.get("run_identity_hash", ""),
                "direction": item.get("direction", ""),
            }
            writer.writerow({field: _csv_safe_value(row.get(field, "")) for field in fields})


def _csv_safe_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    visible = value.lstrip()
    if visible and visible[0] in {"=", "+", "-", "@"}:
        return "'" + value
    if value.startswith(("\t", "\r")):
        return "'" + value
    return value


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量测试故事、剧本、分镜和可选视频链路")
    parser.add_argument(
        "--input",
        default="",
        help="测试题目文件，支持 JSONL、JSON、CSV 或 TXT；省略时使用包内示例",
    )
    parser.add_argument("--output", default="", help="结果目录")
    parser.add_argument("--repeat", type=int, default=1, help="每条题目重复次数")
    parser.add_argument("--max-cases", type=int, default=0, help="只运行前 N 条，0 表示全部")
    parser.add_argument(
        "--target-seconds",
        type=int,
        default=None,
        help="覆盖全部用例的成片秒数（15–300）",
    )
    parser.add_argument("--resume", action="store_true", help="跳过结果目录中已经成功的用例")
    parser.add_argument("--retries", type=int, default=0, help="失败后额外重试次数")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="用例或重试之间的等待时间，用于控制 API 频率",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="允许文本 API 失败时使用离线兜底；真实 API 验收不建议开启",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="完全使用本地规则，不读取或请求真实文本 API",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="每次额外调用文本模型评价故事、剧本和分镜质量",
    )
    parser.add_argument("--render", action="store_true", help="允许批量调用付费视频 API")
    parser.add_argument(
        "--confirm-paid-video",
        default="",
        help="与 --render 同用时必须精确填写 RENDER",
    )
    args = parser.parse_args()

    if args.render and args.confirm_paid_video != "RENDER":
        parser.error("批量视频会产生费用；请同时添加 --confirm-paid-video RENDER。")
    if not args.render and args.confirm_paid_video:
        parser.error("--confirm-paid-video 只能与 --render 同时使用。")
    if args.max_cases < 0:
        parser.error("--max-cases 不能为负数。")
    if args.repeat < 1:
        parser.error("--repeat 必须至少为 1。")
    if args.retries < 0:
        parser.error("--retries 不能为负数。")
    if not math.isfinite(args.delay_seconds) or args.delay_seconds < 0:
        parser.error("--delay-seconds 必须是有限的非负数。")
    if args.target_seconds is not None and not 15 <= args.target_seconds <= 300:
        parser.error("--target-seconds 必须在 15–300 之间。")
    if args.offline and args.allow_fallback:
        parser.error("--offline 与 --allow-fallback 不能同时使用。")
    if args.offline and args.llm_judge:
        parser.error("--offline 不能启用 --llm-judge。")

    try:
        cases = load_cases(args.input or default_cases_source())
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    if args.max_cases:
        cases = cases[: args.max_cases]

    require_live_text = not args.allow_fallback and not args.offline
    agent_factory: AgentFactory
    if args.offline:
        agent_factory = RuleBasedStoryAgent
    else:
        def configured_agent_factory() -> OpenAIStoryAgent:
            return OpenAIStoryAgent.from_env(
                allow_artifact_fallback=args.allow_fallback
            )

        agent_factory = configured_agent_factory
        probe = OpenAIStoryAgent.from_env(
            allow_artifact_fallback=args.allow_fallback
        )
        if require_live_text and probe.client is None:
            parser.error(
                "真实文本 API 未配置成功："
                + str(getattr(probe, "configuration_error", "") or "请检查 .env")
            )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"outputs/batch/{stamp}"
    try:
        result = run_batch(
            cases=cases,
            output_dir=output,
            agent_factory=agent_factory,
            repeat=args.repeat,
            target_seconds_override=args.target_seconds,
            render=args.render,
            renderer_factory=(
                lambda: StoryRenderer(AgnesVideoProvider.from_env()) if args.render else None
            ),
            require_live_text=require_live_text,
            llm_judge=args.llm_judge,
            resume=args.resume,
            retries=args.retries,
            delay_seconds=args.delay_seconds,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"完整结果：{result['output_dir']}")
    if result["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
