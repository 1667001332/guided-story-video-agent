from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import OpenAIStoryAgent, RuleBasedStoryAgent, StoryAgent
from .models import CreativeBrief, Stage
from .rendering import StoryRenderer
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider


HELP = """可用命令：
/pick 1 3       保留第1和第3张卡     /more 2      生成8个相似方向
/refresh        换一批8张卡          /mix        混合已选卡
/expand         展开四类故事零件      /choose ending 3  选择结局3
/auto           让AI替你选择          /draft      随时生成剧本草稿
/revise         用一句反馈改写         /back       回到灵感区
/storyboard     接受草稿并生成分镜     /render     付费视频入口
/quit           保存并退出
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
        raise RuntimeError("--require-live-text 已启用，但没有可用的真实文本API配置。")
    session = GuidedStorySession(CreativeBrief(target_seconds=target_seconds), active_agent)
    target = Path(output_dir).expanduser().resolve()
    output_fn("一句话剧本创意花园：你只需要先说一个方向。")
    direction = input_fn("方向：").strip()
    session.start_ideation(direction)
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
                cards = session.current_batch.cards
                session.select_ideas([cards[index - 1].idea_id for index in indexes])
                output_fn(_selection_text(session))
            elif raw.startswith("/more"):
                index = int(raw.removeprefix("/more").strip())
                session.more_like(session.current_batch.cards[index - 1].idea_id)
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
                option = session.element_palette.options[kind][int(index_text) - 1]
                session.choose_element(kind, option.option_id)
                output_fn(f"已保留 {kind}：{option.title}")
            elif raw == "/auto":
                session.auto_choose()
                output_fn(_selection_text(session))
            elif raw in ("/draft", "/outline"):
                draft = session.generate_draft()
                _print_draft(draft, output_fn)
            elif raw == "/revise":
                feedback = input_fn("一句话修改：").strip()
                draft = session.revise_draft(feedback)
                _print_draft(draft, output_fn)
            elif raw == "/back":
                session.back_to_ideation()
                _print_cards(session, output_fn)
            elif raw == "/storyboard":
                session.confirm_draft()
                plan = session.build_storyboard()
                output_fn(f"已生成 {len(plan.shots)} 个镜头，总时长 {plan.total_duration} 秒。")
            elif raw == "/confirm":
                if session.stage == Stage.DRAFT_REVIEW:
                    session.confirm_draft()
                    output_fn("草稿已确认；输入 /storyboard 生成分镜。")
                elif session.stage == Stage.STORYBOARD_REVIEW:
                    session.confirm_storyboard()
                    output_fn("分镜已确认。")
                else:
                    raise RuntimeError("当前没有需要确认的内容。")
            elif raw == "/render":
                if not allow_render:
                    raise RuntimeError("本次CLI未使用 --render 启动，付费调用保持关闭。")
                if session.stage == Stage.STORYBOARD_REVIEW:
                    session.confirm_storyboard()
                if session.stage != Stage.RENDER_READY:
                    raise RuntimeError("必须先生成并确认分镜。")
                confirmation = input_fn("会调用付费视频API。输入 RENDER 二次确认：").strip()
                if confirmation != "RENDER":
                    output_fn("已取消，没有调用视频API。")
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
            else:
                result = session.chat_ideation(raw)
                output_fn(result.message)
                _print_cards(session, output_fn)

            if (
                require_live_text
                and int(getattr(active_agent, "fallback_count", 0)) > fallback_before
            ):
                reason = str(getattr(active_agent, "last_fallback_reason", "未知原因"))
                raise RuntimeError(f"真实文本链路发生本地降级：{reason}")
        except (RuntimeError, ValueError, IndexError) as exc:
            output_fn(f"未执行：{exc}")

    target.mkdir(parents=True, exist_ok=True)
    session.save(target / "session.json")
    output_fn(f"会话已保存：{target / 'session.json'}")
    return session


def _print_cards(session: GuidedStorySession, output_fn: Callable[[str], None]) -> None:
    output_fn("\n8张创意卡：")
    for index, card in enumerate(session.current_batch.cards, 1):
        output_fn(f"{index}. 《{card.title}》｜{card.logline}｜{card.tone}")


def _selection_text(session: GuidedStorySession) -> str:
    return "已保留：" + ("、".join(card.title for card in session.selected_cards) or "暂无")


def _print_draft(draft, output_fn: Callable[[str], None]) -> None:
    output_fn(f"\n《{draft.script.title}》第{draft.version}版")
    output_fn("AI补全：" + ("、".join(draft.ai_filled_fields) or "无"))
    for scene in draft.script.scenes:
        output_fn(
            f"场景{scene.scene_id} {scene.duration}秒｜{scene.title}｜"
            f"{scene.visible_action or scene.action}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="一句话剧本创意花园CLI")
    parser.add_argument("--target-seconds", type=int, default=45, choices=range(30, 61))
    parser.add_argument("--output", default="")
    parser.add_argument("--require-live-text", action="store_true")
    parser.add_argument("--render", action="store_true", help="允许二次确认后的付费视频生成")
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
