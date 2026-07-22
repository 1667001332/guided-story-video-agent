from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import OpenAIStoryAgent, RuleBasedStoryAgent, StoryAgent
from .models import CreativeBrief, Stage, to_plain_data
from .rendering import StoryRenderer
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider


HELP = """可用命令：
/suggest  获取三个非强制方向      /use 2  采用第二条建议
/show     查看故事圣经            /outline 生成大纲候选
/edit     编辑事实或当前产物      /revise  按反馈局部重写
/review   运行质量检查            /undo    撤销当前产物
/redo     重做当前产物            /confirm 确认并推进阶段
/render   明确进入付费视频生成    /quit    保存并退出
"""


def run_interactive(
    *,
    agent: StoryAgent | None = None,
    target_seconds: int = 45,
    output_dir: str | Path = "outputs/manual_cli",
    allow_render: bool = False,
    require_live_text: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    renderer_factory: Callable[[], Any] | None = None,
) -> GuidedStorySession:
    active_agent = agent or RuleBasedStoryAgent()
    fallback_before = int(getattr(active_agent, "fallback_count", 0))
    if require_live_text and getattr(active_agent, "client", None) is None:
        raise RuntimeError("--require-live-text 已启用，但没有可用的真实文本 API 配置。")
    session = GuidedStorySession(CreativeBrief(target_seconds=target_seconds), active_agent)
    target = Path(output_dir).expanduser().resolve()
    output_fn("引导式剧本共创已开始。AI 每次只追问一个关键缺口。")
    output_fn(HELP)
    output_fn(f"AI：{session.current_question}")

    while True:
        raw = input_fn("你：").strip()
        if not raw:
            continue
        try:
            if raw == "/quit":
                break
            if raw == "/help":
                output_fn(HELP)
            elif raw == "/suggest":
                suggestions = session.request_suggestions()
                for index, item in enumerate(suggestions, 1):
                    output_fn(f"{index}. {item.content}")
            elif raw.startswith("/use"):
                index = int(raw.removeprefix("/use").strip()) - 1
                suggestions = session.request_suggestions()
                if index not in range(len(suggestions)):
                    raise ValueError("建议编号必须是 1、2 或 3。")
                _print_turn(session.apply_suggestion(suggestions[index].suggestion_id), output_fn)
            elif raw == "/show":
                output_fn(json.dumps(to_plain_data(session.story_bible), ensure_ascii=False, indent=2))
            elif raw == "/outline":
                session.build_outline()
                output_fn(_artifact_text(session))
            elif raw == "/edit":
                _edit_current(session, input_fn, output_fn)
            elif raw == "/revise":
                feedback = input_fn("修改意见：").strip()
                _revise_current(session, feedback)
                output_fn(_artifact_text(session))
            elif raw == "/review":
                review = session.review_current_artifact()
                output_fn(json.dumps(to_plain_data(review), ensure_ascii=False, indent=2))
            elif raw == "/undo":
                session.undo_artifact()
                output_fn("已撤销到上一版本。")
            elif raw == "/redo":
                session.redo_artifact()
                output_fn("已恢复下一版本。")
            elif raw == "/confirm":
                _confirm_and_advance(session, output_fn)
            elif raw == "/render":
                if not allow_render:
                    raise RuntimeError("本次 CLI 未使用 --render 启动，付费调用保持关闭。")
                if session.stage != Stage.RENDER_READY:
                    raise RuntimeError("必须先确认分镜。")
                confirmation = input_fn("会调用付费视频 API。输入 RENDER 二次确认：").strip()
                if confirmation != "RENDER":
                    output_fn("已取消，没有调用视频 API。")
                    continue
                renderer = (
                    renderer_factory()
                    if renderer_factory
                    else StoryRenderer(AgnesVideoProvider.from_env())
                )
                manifest = session.render_confirmed_plan(renderer, target / "video")
                output_fn(f"视频状态：{manifest.status}｜{manifest.final_video_path}")
            elif raw.startswith("/"):
                output_fn("未知命令，输入 /help 查看帮助。")
            elif session.stage == Stage.COLLECTING:
                _print_turn(session.submit_user_turn(raw), output_fn)
            elif session.stage == Stage.DETAILING:
                _print_turn(session.answer_detail_question(raw), output_fn)
            else:
                output_fn("当前在产物审阅阶段，请使用 /edit、/revise、/review 或 /confirm。")

            if require_live_text and int(getattr(active_agent, "fallback_count", 0)) > fallback_before:
                reason = str(getattr(active_agent, "last_fallback_reason", "未知原因"))
                raise RuntimeError(f"真实文本链路发生本地降级：{reason}")
        except (RuntimeError, ValueError, IndexError, json.JSONDecodeError) as exc:
            output_fn(f"未执行：{exc}")

    target.mkdir(parents=True, exist_ok=True)
    session.save(target / "session.json")
    output_fn(f"会话已保存：{target / 'session.json'}")
    return session


def _print_turn(result: Any, output_fn: Callable[[str], None]) -> None:
    output_fn(f"AI：{result.assistant_message}")
    if result.extracted_facts:
        summary = "；".join(f"{item.field}={item.value}" for item in result.extracted_facts)
        output_fn(f"已理解：{summary}")
    output_fn(f"完成度：{round(result.readiness_score * 100)}%")
    if result.next_question:
        output_fn(f"AI：{result.next_question}")


def _edit_current(
    session: GuidedStorySession,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> None:
    if session.stage in (Stage.COLLECTING, Stage.DETAILING):
        raw = input_fn("输入 字段=新内容：")
        field, value = raw.split("=", 1)
        session.update_story_bible({field.strip(): value.strip()})
    elif session.stage == Stage.OUTLINE_REVIEW:
        patch = json.loads(input_fn('输入大纲 JSON patch，如 {"title":"新标题"}：'))
        session.update_outline(patch)
    elif session.stage == Stage.SCRIPT_REVIEW:
        scene_id = int(input_fn("场景编号："))
        field, value = input_fn("输入 字段=新内容：").split("=", 1)
        session.update_script_scene(scene_id, {field.strip(): value.strip()})
    elif session.stage in (Stage.STORYBOARD_REVIEW, Stage.RENDER_READY):
        shot_id = int(input_fn("镜头编号："))
        field, value = input_fn("输入 字段=新内容：").split("=", 1)
        session.update_storyboard_shot(shot_id, {field.strip(): value.strip()})
    else:
        raise RuntimeError("当前没有可编辑内容。")
    output_fn("修改已保存为新版本。")


def _revise_current(session: GuidedStorySession, feedback: str) -> None:
    if session.stage == Stage.OUTLINE_REVIEW:
        session.revise_outline(feedback)
    elif session.stage == Stage.SCRIPT_REVIEW:
        session.revise_script(feedback)
    elif session.stage in (Stage.STORYBOARD_REVIEW, Stage.RENDER_READY):
        session.revise_storyboard(feedback)
    else:
        raise RuntimeError("当前阶段没有可局部重写的产物。")


def _confirm_and_advance(
    session: GuidedStorySession, output_fn: Callable[[str], None]
) -> None:
    if session.stage == Stage.OUTLINE_REVIEW:
        session.confirm_outline()
        output_fn(f"大纲已确认。AI：{session.current_question}")
    elif session.stage == Stage.DETAILING:
        session.build_script()
        output_fn("定时剧本候选已生成，请先审阅再确认。")
    elif session.stage == Stage.SCRIPT_REVIEW and session.script and not session.script.confirmed:
        session.confirm_script()
        output_fn("剧本已确认；再次输入 /confirm 生成分镜候选。")
    elif session.stage == Stage.SCRIPT_REVIEW:
        session.build_storyboard()
        output_fn("分镜候选已生成，请逐镜头审阅。")
    elif session.stage == Stage.STORYBOARD_REVIEW:
        session.confirm_storyboard()
        output_fn("分镜已确认。付费视频仍需 /render 和 RENDER 二次确认。")
    else:
        raise RuntimeError("当前阶段没有待确认产物。")


def _artifact_text(session: GuidedStorySession) -> str:
    artifact = session.storyboard or session.script or session.outline
    return json.dumps(to_plain_data(artifact), ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="人工参与的引导式剧本共创 CLI")
    parser.add_argument("--target-seconds", type=int, default=45, choices=range(30, 61))
    parser.add_argument("--output", default="")
    parser.add_argument("--require-live-text", action="store_true")
    parser.add_argument("--render", action="store_true", help="允许进入二次确认后的付费视频生成")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"outputs/manual_cli/{stamp}"
    run_interactive(
        agent=OpenAIStoryAgent.from_env(),
        target_seconds=args.target_seconds,
        output_dir=output,
        allow_render=args.render,
        require_live_text=args.require_live_text,
    )


if __name__ == "__main__":
    main()
