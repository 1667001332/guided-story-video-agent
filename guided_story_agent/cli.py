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
/auto           让AI替你选择          /story      生成完整故事
/revise-story   修改故事               /script     确认故事并生成剧本
/revise-script  修改剧本               /back       回到灵感区
/storyboard     接受剧本并生成分镜     /render     付费视频入口
/quit           保存并退出
"""


class LiveTextRequiredError(RuntimeError):
    """Raised when strict CLI mode observes any local text fallback."""


def run_interactive(
    *,
    agent: StoryAgent | None = None,
    target_seconds: int | None = None,
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
            elif raw == "/storyboard":
                session.confirm_script()
                plan = session.build_storyboard()
                output_fn(f"已生成 {len(plan.shots)} 个镜头，总时长 {plan.total_duration} 秒。")
            elif raw == "/confirm":
                if session.stage == Stage.STORY_REVIEW:
                    session.confirm_story()
                    output_fn("故事已确认；输入 /script 生成剧本。")
                elif session.stage == Stage.SCRIPT_REVIEW:
                    session.confirm_script()
                    output_fn("剧本已确认；输入 /storyboard 生成分镜。")
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

            _enforce_live_text(active_agent, fallback_before, require_live_text)
        except LiveTextRequiredError:
            raise
        except (RuntimeError, ValueError, IndexError) as exc:
            output_fn(f"未执行：{exc}")

    target.mkdir(parents=True, exist_ok=True)
    session.save(target / "session.json")
    output_fn(f"会话已保存：{target / 'session.json'}")
    return session


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
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"outputs/manual_cli/{stamp}"
    try:
        run_interactive(
            agent=RuleBasedStoryAgent() if args.offline else OpenAIStoryAgent.from_env(),
            target_seconds=args.target_seconds,
            output_dir=output,
            allow_render=args.render,
            require_live_text=args.require_live_text,
        )
    except LiveTextRequiredError as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
