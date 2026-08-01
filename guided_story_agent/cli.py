from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import OpenAIStoryAgent, RuleBasedStoryAgent, StoryAgent
from .models import CreativeBrief, Stage
from .rendering import StoryRenderer, VideoJobRenderer
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider
from .v2.director import DirectorAgent as V2DirectorAgent
from .v2.execution import CompilationOptions, ProviderCapabilities as V2ProviderCapabilities
from .v2.revision_apply import ApplyRevisionCommand, RollbackRevisionCommand
from .v2.lineage import SourceLineageGuard
from .v2.openai_director import (
    DirectorGenerationError,
    OpenAIDirectorAgent,
    RuleBasedDirectorAgent,
)


HELP = """可用命令：
/pick 1 3       保留第1和第3张卡     /more 2      生成8个相似方向
/refresh        换一批8张卡          /mix        混合已选卡
/expand         展开四类故事零件      /choose ending 3  选择结局3
/auto           让AI替你选择          /story      生成完整故事
/revise-story   修改故事               /script     确认故事并生成剧本
/revise-script  修改剧本               /back       回到灵感区
/video          接受剧本并生成完整视频任务（/storyboard 为兼容别名）
/render          付费视频入口
/quit           保存并退出
"""

V2_HELP = """V2 可用命令：
/confirm-plan   确认 DirectorAgent 生成的 MoviePlan
/plan           重新显示当前 MoviePlan
/build-film-ir  将 confirmed MoviePlan 构建为 FilmIR（电影语言层）
/analysis       分析 StoryPlan / DirectorPlan / FilmIR 的创意结构
/build-ir       将 FilmIR 降级为 Provider-neutral MovieIR
/optimize       创建 Creative Optimizer 建议与不可执行候选
/revision       查看 Director-facing Revision Request（离线，不调用 LLM）
/revise         调用 Revision Adapter，生成 candidate 并运行 validate / diff / guard
/revision-guard 运行 Candidate Diff / Guard（无候选时保持 pending_director）
/revision-apply  显式应用已接受 candidate（需要输入 APPLY <candidate_id>）
/revision-rollback 显式回滚到指定 MoviePlan 版本（需要输入 ROLLBACK <movie_plan_id>）
/compile        将已构建的 MovieIR 编译为 V2 VideoJob
/diagnostics    查看 Validator / Pass / Optimizer / Revision 诊断
/render         当前阶段不执行 Provider
/quit           保存并退出
当前阶段只构建 FilmIR/MovieIR/VideoJob，不调用真实视频 Provider。"""


class LiveTextRequiredError(RuntimeError):
    """Raised when strict CLI mode observes any local text fallback."""


def run_interactive(
    *,
    agent: StoryAgent | None = None,
    target_seconds: int | None = None,
    output_dir: str | Path = "outputs/manual_cli",
    allow_render: bool = False,
    require_live_text: bool = False,
    v2: bool = False,
    director_agent: V2DirectorAgent | None = None,
    capabilities: V2ProviderCapabilities | None = None,
    compilation_options: CompilationOptions | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    renderer_factory: Callable[[], Any] | None = None,
) -> GuidedStorySession:
    if v2:
        return run_v2_interactive(
            agent=agent,
            director_agent=director_agent,
            target_seconds=target_seconds,
            output_dir=output_dir,
            require_live_text=require_live_text,
            capabilities=capabilities,
            compilation_options=compilation_options,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    active_agent = agent or RuleBasedStoryAgent()
    fallback_before = int(getattr(active_agent, "fallback_count", 0))
    if require_live_text and getattr(active_agent, "client", None) is None:
        raise LiveTextRequiredError("--require-live-text 已启用，但没有可用的真实文本 API 配置。")
    session = GuidedStorySession(
        CreativeBrief(
            target_seconds=target_seconds,
            duration_mode="custom" if target_seconds is not None else "auto",
        ),
        active_agent,
    )
    target = Path(output_dir).expanduser().resolve()
    output_fn("一句话剧本创意花园：你只需要先说一个方向。")
    direction = input_fn("方向：").strip()
    session.start_ideation(direction)
    _enforce_live_text(active_agent, fallback_before, require_live_text)
    _print_cards(session, output_fn)
    output_fn(HELP)

    while True:
        raw = input_fn("操作：").strip()
        if not raw:
            continue
        try:
            if raw == "/quit":
                break
            if raw == "/help":
                output_fn(HELP)
            elif raw.startswith("/pick") or raw.startswith("/use"):
                command = "/pick" if raw.startswith("/pick") else "/use"
                indexes = [int(item) for item in raw.removeprefix(command).split()]
                if not indexes:
                    raise ValueError("请至少提供一个创意卡序号。")
                cards = session.current_batch.cards
                session.select_ideas(
                    [_one_based_item(cards, index, "创意卡").idea_id for index in indexes]
                )
                output_fn(_selection_text(session))
            elif raw.startswith("/more"):
                index = int(raw.removeprefix("/more").strip())
                card = _one_based_item(session.current_batch.cards, index, "创意卡")
                session.more_like(card.idea_id)
                _print_cards(session, output_fn)
            elif raw in ("/refresh", "/suggest"):
                session.refresh_ideas()
                _print_cards(session, output_fn)
            elif raw == "/mix":
                session.mix_selected()
                _print_cards(session, output_fn)
            elif raw == "/expand":
                palette = session.expand_selected()
                for kind, options in palette.options.items():
                    output_fn(f"[{kind}]")
                    for index, option in enumerate(options, 1):
                        output_fn(f"  {index}. {option.title}｜{option.content}")
            elif raw.startswith("/choose"):
                _, kind, index_text = raw.split(maxsplit=2)
                if session.element_palette is None:
                    raise RuntimeError("请先输入 /expand 展开故事零件。")
                options = session.element_palette.options.get(kind)
                if options is None:
                    raise ValueError(f"未知故事零件类型：{kind}")
                option = _one_based_item(
                    options,
                    int(index_text),
                    f"{kind} 选项",
                )
                session.choose_element(kind, option.option_id)
                output_fn(f"已保留 {kind}：{option.title}")
            elif raw == "/auto":
                session.auto_choose()
                output_fn(_selection_text(session))
            elif raw in ("/story", "/draft", "/outline"):
                story = session.generate_story()
                _print_story(story, output_fn)
            elif raw in ("/revise-story", "/revise"):
                feedback = input_fn("一句话修改故事：").strip()
                story = session.revise_story(feedback)
                _print_story(story, output_fn)
            elif raw == "/script":
                session.confirm_story()
                script = session.generate_script()
                _print_script(script, output_fn)
            elif raw == "/revise-script":
                feedback = input_fn("一句话修改剧本：").strip()
                script = session.revise_script(feedback)
                _print_script(script, output_fn)
            elif raw == "/back":
                session.back_to_ideation()
                _print_cards(session, output_fn)
            elif raw in ("/video", "/storyboard"):
                session.confirm_script()
                job = session.build_video_job()
                output_fn(
                    f"已生成完整视频任务，总时长 {job.target_seconds} 秒；"
                    "Provider 将自行决定是否需要内部拆分。"
                )
            elif raw == "/confirm":
                if session.stage == Stage.STORY_REVIEW:
                    session.confirm_story()
                    output_fn("故事已确认；输入 /script 生成剧本。")
                elif session.stage == Stage.SCRIPT_REVIEW:
                    session.confirm_script()
                    output_fn("剧本已确认；输入 /video 生成完整视频任务。")
                elif session.stage == Stage.STORYBOARD_REVIEW:
                    session.confirm_storyboard()
                    output_fn("旧版分镜已确认。")
                else:
                    raise RuntimeError("当前没有需要确认的内容。")
            elif raw == "/render":
                if not allow_render:
                    raise RuntimeError("本次CLI未使用 --render 启动，付费调用保持关闭。")
                if session.stage != Stage.RENDER_READY:
                    raise RuntimeError("必须先生成并确认完整视频任务。")
                confirmation = input_fn("会调用付费视频API。输入 RENDER 二次确认：").strip()
                if confirmation != "RENDER":
                    output_fn("已取消，没有调用视频API。")
                    continue
                renderer = (
                    renderer_factory()
                    if renderer_factory
                    else StoryRenderer(AgnesVideoProvider.from_env())
                )
                if session.video_job is not None:
                    direct_renderer = (
                        renderer
                        if isinstance(renderer, VideoJobRenderer)
                        else VideoJobRenderer(renderer.provider)
                        if hasattr(renderer, "provider")
                        else VideoJobRenderer(renderer)
                    )
                    manifest = session.render_confirmed_video(
                        direct_renderer,
                        target / "video",
                    )
                else:
                    # Legacy storyboard sessions saved by older releases stay
                    # renderable and are intentionally not migrated in place.
                    manifest = session.render_confirmed_plan(renderer, target / "video")
                output_fn(f"视频状态：{manifest.status}｜{manifest.final_video_path}")
            elif raw.startswith("/"):
                output_fn("未知命令，输入 /help 查看帮助。")
            else:
                result = session.chat_ideation(raw)
                output_fn(result.message)
                _print_cards(session, output_fn)

            _enforce_live_text(active_agent, fallback_before, require_live_text)
        except LiveTextRequiredError:
            raise
        except (RuntimeError, ValueError, IndexError) as exc:
            output_fn(f"未执行：{exc}")

    target.mkdir(parents=True, exist_ok=True)
    session.save(target / "session.json")
    output_fn(f"会话已保存：{target / 'session.json'}")
    return session


def run_v2_interactive(
    *,
    agent: StoryAgent | None = None,
    director_agent: V2DirectorAgent | None = None,
    target_seconds: int | None = None,
    output_dir: str | Path = "outputs/manual_cli_v2",
    require_live_text: bool = False,
    capabilities: V2ProviderCapabilities | None = None,
    compilation_options: CompilationOptions | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> GuidedStorySession:
    if director_agent is None:
        raise RuntimeError("V2 模式缺少 DirectorAgent。")
    if require_live_text and getattr(director_agent, "client", None) is None:
        raise LiveTextRequiredError(
            "--require-live-text 已启用，但 V2 DirectorAgent 没有真实文本客户端。"
        )
    target = Path(output_dir).expanduser().resolve()
    session = GuidedStorySession(
        CreativeBrief(
            target_seconds=target_seconds,
            duration_mode="custom" if target_seconds is not None else "auto",
        ),
        agent or RuleBasedStoryAgent(),
        director_agent=director_agent,
        v2_enabled=True,
    )
    compile_capabilities = capabilities or V2ProviderCapabilities(
        provider_key="offline-v2",
        provider_profile="offline",
        supports_long_video=True,
        supports_multi_scene_prompt=True,
        supports_audio=True,
    )
    output_fn("V2 电影导演模式：DirectorAgent 将一次性生成完整 MoviePlan。")
    direction = input_fn("创作方向：").strip()
    session.generate_movie_plan(direction)
    _print_v2_plan(session, output_fn)
    output_fn(V2_HELP)
    while True:
        raw = input_fn("V2 操作：").strip()
        if not raw:
            continue
        try:
            if raw == "/quit":
                break
            if raw == "/help":
                output_fn(V2_HELP)
            elif raw == "/plan":
                _print_v2_plan(session, output_fn)
            elif raw in {"/confirm", "/confirm-plan"}:
                session.confirm_movie_plan()
                output_fn("MoviePlan 已确认。")
            elif raw == "/build-film-ir":
                film_ir = session.build_film_ir_from_confirmed_movie_plan()
                if film_ir is None:
                    output_fn(
                        "FilmIR 构建被拒绝："
                        + "；".join(
                            f"{item.get('code')}: {item.get('message')}"
                            for item in (
                                session.film_ir_build_diagnostics
                                + session.film_ir_validation_issues
                                + session.film_ir_pass_diagnostics
                            )
                        )
                    )
                else:
                    output_fn(
                        f"FilmIR 已生成：{film_ir.ir_id}；"
                        f"beats={len(film_ir.beats)}。"
                    )
            elif raw == "/analysis":
                try:
                    results = session.run_creative_analysis()
                except RuntimeError as exc:
                    output_fn(f"Creative Analysis 未执行：{exc}")
                else:
                    _print_v2_analysis_summary(session, output_fn, results)
            elif raw == "/optimize":
                try:
                    optimization = session.run_creative_optimization()
                except RuntimeError as exc:
                    output_fn(f"Creative Optimizer 未执行：{exc}")
                else:
                    output_fn(
                        "Creative Optimizer 已完成："
                        f"suggestions={len(session.creative_optimizer_suggestions)}，"
                        f"candidates={len(session.creative_optimizer_candidates)}，"
                        f"revision_requests={len(session.creative_revision_requests)}，"
                        f"stop_reason={optimization.get('revision_stop_reason')}"
                    )
            elif raw == "/revision":
                _print_v2_revision_requests(session, output_fn)
            elif raw == "/revise":
                try:
                    result = session.run_director_revision_guarded()
                except RuntimeError as exc:
                    output_fn(f"Director Revision 未执行：{exc}")
                else:
                    decision = result.get("decision") or {}
                    adapter_result = result.get("adapter_result") or {}
                    output_fn(
                        "Director Revision："
                        f"status={adapter_result.get('status', 'none')}，"
                        f"decision={decision.get('decision', 'none')}，"
                        f"stop_reason={result.get('stop_reason') or 'none'}"
                    )
            elif raw == "/revision-guard":
                try:
                    decision = session.run_revision_guard()
                except RuntimeError as exc:
                    output_fn(f"RevisionGuard 未执行：{exc}")
                else:
                    output_fn(
                        "RevisionGuard："
                        f"decision={decision.get('decision')}，"
                        f"reason={decision.get('reason')}"
                    )
            elif raw == "/revision-apply":
                accepted_ids = {
                    str(item.get("accepted_candidate_id"))
                    for item in session.revision_decisions
                    if item.get("decision") in {"accept", "accept_with_warning"}
                    and item.get("accepted_candidate_id")
                }
                if not accepted_ids:
                    output_fn(
                        "未执行 Apply：当前没有 accepted candidate；"
                        "当前阶段不支持自动 apply。"
                    )
                else:
                    confirmation = input_fn("请输入 APPLY <candidate_id> 以确认应用，或直接回车取消：").strip()
                    parts = confirmation.split(maxsplit=1)
                    if len(parts) != 2 or parts[0].upper() != "APPLY":
                        output_fn("未执行 Apply：需要明确输入 APPLY <candidate_id>。")
                    else:
                        current = session.confirmed_movie_plan or session.movie_plan
                        if current is None:
                            output_fn("未执行 Apply：当前没有 MoviePlan。")
                        else:
                            try:
                                result = session.apply_revision(
                                    ApplyRevisionCommand(
                                        command_id=f"cli-apply-{uuid4().hex[:12]}",
                                        candidate_id=parts[1].strip(),
                                        source_movie_plan_id=current.plan_id,
                                        apply_reason="CLI 用户显式确认 Director revision apply",
                                        confirmed_by="cli_user",
                                    )
                                )
                            except (RuntimeError, ValueError, TypeError) as exc:
                                output_fn(f"Apply 未执行：{exc}")
                            else:
                                output_fn(
                                    "RevisionApply："
                                    f"applied={result.applied}，"
                                    f"stop_reason={result.stop_reason or 'none'}，"
                                    f"invalidated={len(result.invalidated_artifacts)}"
                                )
            elif raw == "/revision-rollback":
                if not session.movie_plan_version_history:
                    output_fn("未执行 Rollback：当前没有可回滚的 MoviePlan 版本。")
                else:
                    confirmation = input_fn(
                        "请输入 ROLLBACK <movie_plan_id> 以确认回滚，或直接回车取消："
                    ).strip()
                    parts = confirmation.split(maxsplit=1)
                    if len(parts) != 2 or parts[0].upper() != "ROLLBACK":
                        output_fn("未执行 Rollback：需要明确输入 ROLLBACK <movie_plan_id>。")
                    else:
                        try:
                            result = session.rollback_revision(
                                RollbackRevisionCommand(
                                    command_id=f"cli-rollback-{uuid4().hex[:12]}",
                                    rollback_to_movie_plan_id=parts[1].strip(),
                                    rollback_reason="CLI 用户显式确认 MoviePlan rollback",
                                    confirmed_by="cli_user",
                                )
                            )
                        except (RuntimeError, ValueError, TypeError) as exc:
                            output_fn(f"Rollback 未执行：{exc}")
                        else:
                            output_fn(
                                "RevisionRollback："
                                f"rolled_back={result.rolled_back}，"
                                f"stop_reason={result.stop_reason or 'none'}，"
                                f"invalidated={len(result.invalidated_artifacts)}"
                            )
            elif raw == "/build-ir":
                try:
                    movie_ir = session.build_movie_ir_from_film_ir()
                except RuntimeError as exc:
                    output_fn(f"MovieIR 构建未执行：{exc}")
                    continue
                if movie_ir is None:
                    output_fn(
                        "MovieIR 构建被拒绝："
                        + "；".join(
                            f"{item.get('code')}: {item.get('message')}"
                            for item in (
                                session.movie_ir_build_diagnostics
                                + session.movie_ir_validation_issues
                                + session.movie_ir_pass_diagnostics
                            )
                        )
                    )
                else:
                    output_fn(
                        f"MovieIR 已生成：{movie_ir.ir_id}；"
                        f"shots={len(movie_ir.shots)}。"
                    )
            elif raw == "/compile":
                try:
                    result = session.compile_confirmed_movie_plan(
                        capabilities=compile_capabilities,
                        options=compilation_options,
                    )
                except RuntimeError as exc:
                    output_fn(f"编译未执行：{exc}")
                else:
                    if result.success and result.video_job is not None:
                        output_fn(
                            f"V2 VideoJob 已生成：{result.video_job.job_id}；"
                            f"时长 {result.video_job.duration_seconds:g} 秒。"
                        )
                    else:
                        output_fn(
                            "编译被拒绝："
                            + "；".join(
                                f"{item.code}: {item.message}" for item in result.errors
                            )
                        )
            elif raw == "/diagnostics":
                _print_v2_diagnostics(session, output_fn)
            elif raw == "/render":
                try:
                    session._require_video_job_lineage()
                except RuntimeError as exc:
                    output_fn(f"V2 render 被拒绝：{exc}")
                else:
                    output_fn(
                        "V2 render is not connected in Phase 3A (offline through Phase 4C). FilmIR、MovieIR 和 VideoJob 只做规划与编译；"
                        "Provider execution belongs to the later Provider Runtime phase."
                    )
            else:
                output_fn("V2 当前支持 /confirm-plan、/plan、/build-film-ir、/analysis、/optimize、/revision、/revise、/revision-guard、/revision-apply、/revision-rollback、/build-ir、/compile、/diagnostics、/render、/help、/quit。")
        except (RuntimeError, ValueError, IndexError) as exc:
            output_fn(f"未执行：{exc}")
    target.mkdir(parents=True, exist_ok=True)
    session.save(target / "session.json")
    output_fn(f"V2 会话已保存：{target / 'session.json'}")
    return session


def _print_v2_diagnostics(
    session: GuidedStorySession,
    output_fn: Callable[[str], None],
) -> None:
    session.refresh_source_lineage_diagnostics()
    groups = (
        ("film_ir_validation_issues", session.film_ir_validation_issues),
        ("film_ir_pass_diagnostics", session.film_ir_pass_diagnostics),
        ("creative_pass_diagnostics", session.creative_pass_diagnostics),
        ("creative_analysis_diagnostics", session.creative_analysis_diagnostics),
        ("creative_optimizer_diagnostics", session.creative_optimizer_diagnostics),
        ("film_ir_optimizer_diagnostics", session.film_ir_optimizer_diagnostics),
        ("movie_ir_validation_issues", session.movie_ir_validation_issues),
        ("movie_ir_pass_diagnostics", session.movie_ir_pass_diagnostics),
        ("movie_ir_optimizer_diagnostics", session.movie_ir_optimizer_diagnostics),
        ("director_revision_history", session.director_revision_history),
        ("creative_revision_requests", session.creative_revision_requests),
        ("creative_revision_request_history", session.creative_revision_request_history),
        ("revision_guard_diagnostics", session.revision_guard_diagnostics),
        ("revision_decisions", session.revision_decisions),
        ("director_revision_adapter_results", session.director_revision_adapter_results),
        ("director_revision_contexts", session.director_revision_contexts),
        ("guarded_revision_results", session.guarded_revision_results),
        ("movie_plan_version_history", session.movie_plan_version_history),
        ("revision_apply_history", session.revision_apply_history),
        ("revision_rollback_history", session.revision_rollback_history),
        ("revision_apply_results", session.revision_apply_results),
        ("revision_rollback_results", session.revision_rollback_results),
        ("stale_artifacts", session.stale_artifacts),
        ("source_lineage_diagnostics", session.source_lineage_diagnostics),
        ("stale_lineage_diagnostics", session.stale_lineage_diagnostics),
    )
    for name, items in groups:
        output_fn(f"{name}: {len(items)}")
        for item in items:
            label = item.get("code") or item.get("request_id") or item.get("target") or item.get("reason") or "record"
            message = item.get("message") or item.get("instruction") or item.get("status", "")
            output_fn(
                f"  {label}: {message}"
            )
    lineage = SourceLineageGuard()
    plan_id, story_id, director_id = session._lineage_ids()
    lineage_results = (
        (
            "FilmIR",
            lineage.check_film_ir(
                session.film_ir,
                current_movie_plan_id=plan_id or "",
                current_story_plan_id=story_id or "",
                current_director_plan_id=director_id or "",
                current_film_ir_id=session.current_film_ir_id or "",
            ),
            "/build-film-ir",
        ),
        (
            "MovieIR",
            lineage.check_movie_ir(
                session.movie_ir,
                current_movie_plan_id=plan_id or "",
                current_film_ir_id=session.current_film_ir_id or "",
                current_movie_ir_id=session.current_movie_ir_id or "",
            ),
            "/build-ir",
        ),
        (
            "VideoJob",
            lineage.check_video_job(
                session.v2_video_job,
                current_movie_plan_id=plan_id or "",
                current_film_ir_id=session.current_film_ir_id or "",
                current_movie_ir_id=session.current_movie_ir_id or "",
                current_video_job_id=session.current_video_job_id or "",
            ),
            "/compile",
        ),
    )
    for label, result, action in lineage_results:
        if result.lineage is None or not getattr(session, {
            "FilmIR": "film_ir",
            "MovieIR": "movie_ir",
            "VideoJob": "v2_video_job",
        }[label], None):
            status = "missing"
        else:
            status = result.status
        output_fn(f"{label}: {status}")
        if result.lineage is not None:
            source_fields = {
                key: value
                for key, value in result.lineage.to_dict().items()
                if key.startswith("source_") and value
            }
            if source_fields:
                output_fn(f"  source_lineage: {source_fields}")
        for item in result.diagnostics:
            output_fn(
                f"  {item.message}"
                + (f"；Action: run {item.action}" if item.action else f"；Action: run {action}")
            )
    output_fn(
        "director_revision_stop_reason: "
        f"{session.director_revision_stop_reason or 'none'}"
    )
    output_fn(f"creative_analysis_results: {len(session.creative_analysis_results)}")
    output_fn(f"creative_analysis_artifacts: {len(session.creative_analysis_artifacts)}")
    output_fn(f"creative_analysis_metrics: {session.creative_analysis_metrics}")
    output_fn(f"creative_optimizer_suggestions: {len(session.creative_optimizer_suggestions)}")
    output_fn(f"creative_optimizer_candidates: {len(session.creative_optimizer_candidates)}")
    output_fn(
        "creative_revision_stop_reason: "
        f"{session.creative_revision_stop_reason or 'none'}"
    )
    output_fn(f"revision_candidates: {len(session.revision_candidates)}")
    output_fn(f"revision_diffs: {len(session.revision_diffs)}")
    output_fn(f"revision_decisions: {len(session.revision_decisions)}")
    output_fn(
        f"director_revision_adapter_results: {len(session.director_revision_adapter_results)}"
    )
    output_fn(f"director_revision_contexts: {len(session.director_revision_contexts)}")
    output_fn(f"guarded_revision_results: {len(session.guarded_revision_results)}")
    output_fn(f"current_movie_plan_id: {session.current_movie_plan_id or 'none'}")
    output_fn(f"previous_movie_plan_id: {session.previous_movie_plan_id or 'none'}")
    output_fn(f"current_film_ir_id: {session.current_film_ir_id or 'none'}")
    output_fn(f"current_movie_ir_id: {session.current_movie_ir_id or 'none'}")
    output_fn(f"current_video_job_id: {session.current_video_job_id or 'none'}")
    output_fn(f"movie_plan_version_history: {len(session.movie_plan_version_history)}")
    output_fn(f"revision_apply_history: {len(session.revision_apply_history)}")
    output_fn(f"revision_rollback_history: {len(session.revision_rollback_history)}")
    output_fn(f"revision_apply_results: {len(session.revision_apply_results)}")
    output_fn(f"revision_rollback_results: {len(session.revision_rollback_results)}")
    output_fn(f"stale_artifacts: {len(session.stale_artifacts)}")
    output_fn(
        "director_revision_attempt_count: "
        f"{session.director_revision_attempt_count}"
    )
    output_fn(
        "director_revision_last_stop_reason: "
        f"{session.director_revision_last_stop_reason or 'none'}"
    )
    output_fn(f"revision_active_candidate_id: {session.revision_active_candidate_id or 'none'}")
    output_fn(f"revision_accepted_movie_plan_id: {session.revision_accepted_movie_plan_id or 'none'}")
    output_fn(f"revision_rollback_movie_plan_id: {session.revision_rollback_movie_plan_id or 'none'}")


def _print_v2_revision_requests(
    session: GuidedStorySession,
    output_fn: Callable[[str], None],
) -> None:
    output_fn(
        "Director Revision Requests："
        f"{len(session.creative_revision_requests)}，"
        f"stop_reason={session.creative_revision_stop_reason or 'none'}"
    )
    for request in session.creative_revision_requests:
        output_fn(
            f"  [{request.get('severity', 'warning')}] "
            f"{request.get('request_id', 'request')} -> {request.get('target', '')}"
        )
        output_fn(f"    {request.get('instruction', '')}")
        output_fn(f"    preserve={', '.join(request.get('preserve', []))}")
        output_fn(f"    avoid={', '.join(request.get('avoid', []))}")
    output_fn(
        "Revision candidates："
        f"{len(session.revision_candidates)}，"
        f"active={session.revision_active_candidate_id or 'none'}"
    )
    for candidate in session.revision_candidates:
        output_fn(
            f"  candidate={candidate.get('candidate_id', '')} "
            f"type={candidate.get('candidate_type', '')} "
            f"status={candidate.get('status', '')}"
        )
    if session.revision_decisions:
        latest = session.revision_decisions[-1]
        output_fn(
            "Revision decision："
            f"{latest.get('decision', 'none')}｜{latest.get('reason', '')}"
        )


def _print_v2_analysis_summary(
    session: GuidedStorySession,
    output_fn: Callable[[str], None],
    results: tuple[dict[str, Any], ...],
) -> None:
    output_fn(f"Creative Analysis 已完成：{len(results)} 个分析")
    for result in results:
        output_fn(
            f"  {result.get('analysis_type', 'analysis')}: "
            f"diagnostics={len(result.get('diagnostics', []))}, "
            f"artifacts={len(result.get('artifacts', []))}, "
            f"succeeded={result.get('succeeded', False)}"
        )
    output_fn(
        f"metrics={len(session.creative_analysis_metrics)}，"
        f"analysis_diagnostics={len(session.creative_analysis_diagnostics)}"
    )


def _one_based_item(items, index: int, label: str):
    if isinstance(index, bool) or not 1 <= index <= len(items):
        raise ValueError(f"{label}序号必须在 1 到 {len(items)} 之间。")
    return items[index - 1]


def _enforce_live_text(agent: StoryAgent, baseline: int, required: bool) -> None:
    if not required:
        return
    if int(getattr(agent, "fallback_count", 0)) <= baseline:
        return
    reason = str(getattr(agent, "last_fallback_reason", "未知原因"))
    kind = str(getattr(agent, "last_fallback_kind", "whole") or "whole")
    raise LiveTextRequiredError(f"真实文本链路发生{kind}本地降级：{reason}")


def _print_cards(session: GuidedStorySession, output_fn: Callable[[str], None]) -> None:
    output_fn("\n8张创意卡：")
    for index, card in enumerate(session.current_batch.cards, 1):
        output_fn(f"{index}. 《{card.title}》｜{card.logline}｜{card.tone}")


def _selection_text(session: GuidedStorySession) -> str:
    return "已保留：" + ("、".join(card.title for card in session.selected_cards) or "暂无")


def _print_story(story, output_fn: Callable[[str], None]) -> None:
    output_fn(f"\n《{story.title}》故事第{story.version}版")
    output_fn(story.story_text)
    output_fn("AI补全：" + ("、".join(story.ai_filled_fields) or "无"))


def _print_script(script, output_fn: Callable[[str], None]) -> None:
    output_fn(f"\n《{script.title}》剧本｜{script.total_duration}秒")
    for scene in script.scenes:
        output_fn(
            f"场景{scene.scene_id} {scene.duration}秒｜{scene.title}｜"
            f"{scene.visible_action or scene.action}"
        )


def _print_v2_plan(session: GuidedStorySession, output_fn: Callable[[str], None]) -> None:
    plan = session.movie_plan
    if plan is None:
        output_fn("当前没有 MoviePlan。")
        return
    output_fn(
        f"\nMoviePlan {plan.plan_id} rev={plan.revision} "
        f"状态={'confirmed' if plan.confirmed else 'review'}"
    )
    output_fn(f"故事：{plan.story.title}｜{plan.story.logline}")
    output_fn(f"视觉风格：{plan.visual_style}")
    if plan.story_plan is not None:
        output_fn(
            f"故事层：事件 {len(plan.story_plan.events)} 个，"
            f"故事节拍 {len(plan.story_plan.story_beats)} 个"
        )
    if plan.director_plan is not None:
        output_fn(
            f"导演层：节奏策略 {plan.director_plan.pacing_strategy or '未声明'}，"
            f"高潮强调 {plan.director_plan.climax_emphasis or '未声明'}"
        )
    output_fn(
        f"场景数：{len(plan.script.scenes)}；"
        f"目标时长：{plan.timing_plan.target_duration_seconds:g} 秒"
    )
    for scene, timing in zip(plan.script.scenes, plan.timing_plan.entries):
        output_fn(
            f"  {scene.scene_id} {timing.duration_seconds:g}s｜"
            f"{scene.goal}｜{scene.emotion}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="一句话剧本创意花园CLI")
    parser.add_argument(
        "--target-seconds",
        type=int,
        default=None,
        help="自定义成片秒数（15–300）；省略时根据完整故事自动估算",
    )
    parser.add_argument("--output", default="")
    text_mode = parser.add_mutually_exclusive_group()
    text_mode.add_argument(
        "--offline",
        action="store_true",
        help="强制使用本地规则代理，不读取文本模型配置，也不调用文本 API",
    )
    text_mode.add_argument("--require-live-text", action="store_true")
    parser.add_argument("--render", action="store_true", help="允许二次确认后的付费视频生成")
    parser.add_argument("--v2", action="store_true", help="启用 DirectorAgent → MoviePlan V2 主链")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"outputs/manual_cli/{stamp}"
    try:
        if args.v2:
            run_interactive(
                agent=RuleBasedStoryAgent(),
                director_agent=(
                    RuleBasedDirectorAgent()
                    if args.offline
                    else OpenAIDirectorAgent.from_env()
                ),
                target_seconds=args.target_seconds,
                output_dir=output,
                require_live_text=args.require_live_text,
                v2=True,
            )
        else:
            run_interactive(
                agent=RuleBasedStoryAgent() if args.offline else OpenAIStoryAgent.from_env(),
                target_seconds=args.target_seconds,
                output_dir=output,
                allow_render=args.render,
                require_live_text=args.require_live_text,
            )
    except LiveTextRequiredError as exc:
        parser.exit(2, f"{exc}\n")
    except DirectorGenerationError as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
