from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from .agent import OpenAIStoryAgent, RuleBasedStoryAgent, StoryAgent
from .models import CreativeBrief, ElementPalette, Stage
from .rendering import StoryRenderer
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider


def card_grid_payload(session: GuidedStorySession | None) -> dict[str, Any]:
    if session is None or session.current_batch is None:
        return {"cards": [], "selected_ids": [], "max_select": 3}
    return {
        "cards": [
            {
                "id": card.idea_id,
                "title": card.title,
                "logline": card.logline,
                "hook": card.hook,
                "tone": card.tone,
            }
            for card in session.current_batch.cards
        ],
        "selected_ids": list(session.selected_idea_ids),
        "max_select": 3,
    }


def _card_grid_update(session: GuidedStorySession | None):
    payload = card_grid_payload(session)
    choices = [
        (
            f"{index}. {card['title']}｜{card['tone']}\n{card['logline']}\n钩子：{card['hook']}",
            card["id"],
        )
        for index, card in enumerate(payload["cards"], start=1)
    ]
    return _gr_update(choices=choices, value=payload["selected_ids"])


def start_garden_view(
    direction: str,
    target_seconds: int | float = 45,
    agent: StoryAgent | None = None,
) -> tuple[GuidedStorySession | None, dict[str, Any], str, list[dict[str, str]], str]:
    try:
        session = GuidedStorySession(
            brief=CreativeBrief(target_seconds=int(target_seconds)),
            agent=agent or RuleBasedStoryAgent(),
        )
        session.start_ideation(direction)
    except Exception as exc:
        return None, _card_grid_update(None), "尚未选择", [], str(exc)
    chat = [
        {"role": "user", "content": session.direction},
        {
            "role": "assistant",
            "content": "我给你铺开了8个完全不同的方向。你可以只点卡片，不必继续写故事。",
        },
    ]
    return (
        session,
        _card_grid_update(session),
        _selection_text(session),
        chat,
        _status_text(session),
    )


def select_cards_view(
    session: GuidedStorySession | None, payload: list[str] | dict[str, Any] | None
) -> tuple[GuidedStorySession | None, dict[str, Any], str, str]:
    if session is None:
        return session, _card_grid_update(None), "尚未选择", "请先给出一句方向。"
    try:
        selected = (
            list((payload or {}).get("selected_ids", []))
            if isinstance(payload, dict)
            else list(payload or [])
        )
        session.select_ideas(selected)
        return session, _card_grid_update(session), _selection_text(session), _status_text(session)
    except Exception as exc:
        return session, _card_grid_update(session), _selection_text(session), str(exc)


def refresh_ideas_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict[str, Any], str, str]:
    if session is None:
        return session, _card_grid_update(None), "尚未选择", "请先开始创作。"
    try:
        session.refresh_ideas()
        return session, _card_grid_update(session), _selection_text(session), "已经换成8个新方向。"
    except Exception as exc:
        return session, _card_grid_update(session), _selection_text(session), str(exc)


def more_like_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict[str, Any], str, str]:
    if session is None:
        return session, _card_grid_update(None), "尚未选择", "请先开始创作。"
    if len(session.selected_idea_ids) != 1:
        return (
            session,
            _card_grid_update(session),
            _selection_text(session),
            "“更多像这个”需要只选择1张卡。",
        )
    try:
        session.more_like(session.selected_idea_ids[0])
        return (
            session,
            _card_grid_update(session),
            _selection_text(session),
            "保留核心吸引力，生成了8个新变体。",
        )
    except Exception as exc:
        return session, _card_grid_update(session), _selection_text(session), str(exc)


def mix_selected_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict[str, Any], str, str]:
    if session is None:
        return session, _card_grid_update(None), "尚未选择", "请先开始创作。"
    try:
        session.mix_selected()
        return (
            session,
            _card_grid_update(session),
            _selection_text(session),
            "已生成8种融合方式，冲突由AI在方案中处理。",
        )
    except Exception as exc:
        return session, _card_grid_update(session), _selection_text(session), str(exc)


def auto_choose_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict[str, Any], str, str]:
    if session is None:
        return session, _card_grid_update(None), "尚未选择", "请先开始创作。"
    try:
        session.auto_choose()
        return (
            session,
            _card_grid_update(session),
            _selection_text(session),
            "我替你保留了一张最容易发展成短片的卡。",
        )
    except Exception as exc:
        return session, _card_grid_update(session), _selection_text(session), str(exc)


def chat_ideation_view(
    session: GuidedStorySession | None,
    message: str,
    history: list[dict[str, str]] | None,
) -> tuple[GuidedStorySession | None, dict[str, Any], list[dict[str, str]], str, str]:
    chat = list(history or [])
    if session is None:
        return session, _card_grid_update(None), chat, "", "请先给出初始方向。"
    cleaned = " ".join(message.split())
    if cleaned:
        chat.append({"role": "user", "content": cleaned})
    try:
        result = session.chat_ideation(cleaned)
        chat.append({"role": "assistant", "content": result.message})
        return session, _card_grid_update(session), chat, "", _status_text(session)
    except Exception as exc:
        return session, _card_grid_update(session), chat, "", str(exc)


def expand_elements_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, Any, Any, Any, Any, str]:
    if session is None:
        return session, None, None, None, None, "请先开始创作。"
    try:
        palette = session.expand_selected()
        values = [
            _element_update(palette, kind)
            for kind in ("character", "conflict", "turning_point", "ending")
        ]
        return session, *values, "故事零件是可选的；不选也能直接生成。"
    except Exception as exc:
        return session, None, None, None, None, str(exc)


def choose_elements_view(
    session: GuidedStorySession | None,
    character: str | None,
    conflict: str | None,
    turning_point: str | None,
    ending: str | None,
) -> tuple[GuidedStorySession | None, str, str]:
    if session is None:
        return session, "尚未选择", "请先开始创作。"
    try:
        for kind, value in {
            "character": character,
            "conflict": conflict,
            "turning_point": turning_point,
            "ending": ending,
        }.items():
            if value:
                session.choose_element(kind, value)
        return session, _selection_text(session), "故事零件已保留；随时可以生成草稿。"
    except Exception as exc:
        return session, _selection_text(session), str(exc)


def generate_draft_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str, str, str]:
    if session is None:
        return session, "", "", "请先给出一句方向。"
    try:
        draft = session.generate_draft()
        return (
            session,
            _draft_markdown(draft),
            _ai_fill_markdown(draft),
            "草稿已生成；AI补全内容已经单独标出。",
        )
    except Exception as exc:
        return session, "", "", str(exc)


def revise_draft_view(
    session: GuidedStorySession | None, feedback: str
) -> tuple[GuidedStorySession | None, str, str, str, str]:
    if session is None:
        return session, "", "", "", "请先生成草稿。"
    try:
        draft = session.revise_draft(feedback)
        return (
            session,
            _draft_markdown(draft),
            _ai_fill_markdown(draft),
            "",
            f"已生成第{draft.version}版。",
        )
    except Exception as exc:
        return session, "", "", feedback, str(exc)


def back_to_ideas_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict[str, Any], str]:
    if session is None:
        return session, _card_grid_update(None), "请先开始创作。"
    session.back_to_ideation()
    return session, _card_grid_update(session), "已回到灵感区，现有草稿和版本没有被覆盖。"


def build_storyboard_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str, Any, str]:
    if session is None:
        return session, "", None, "请先生成草稿。"
    try:
        session.confirm_draft()
        plan = session.build_storyboard()
        choices = [
            (f"镜头 {shot.shot_id}｜{shot.shot_purpose}", str(shot.shot_id)) for shot in plan.shots
        ]
        return (
            session,
            _storyboard_markdown(plan),
            _gr_update(choices=choices, value=choices[0][1]),
            "分镜已生成；确认前不会调用视频API。",
        )
    except Exception as exc:
        return session, "", None, str(exc)


def retake_shot_view(
    session: GuidedStorySession | None, shot_id: str | None, feedback: str
) -> tuple[GuidedStorySession | None, str, str, str]:
    if session is None or not shot_id:
        return session, "", feedback, "请先选择镜头。"
    try:
        shot = next(item for item in session.storyboard.shots if item.shot_id == int(shot_id))
        new_action = f"根据Retake要求“{feedback.strip()}”：{shot.action}"
        session.update_storyboard_shot(
            int(shot_id),
            {"action": new_action, "video_prompt": f"{shot.video_prompt}; {feedback.strip()}"},
        )
        return session, _storyboard_markdown(session.storyboard), "", "该镜头已生成新版本。"
    except Exception as exc:
        return (
            session,
            _storyboard_markdown(session.storyboard) if session.storyboard else "",
            feedback,
            str(exc),
        )


def confirm_storyboard_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str]:
    if session is None:
        return session, "请先生成分镜。"
    try:
        session.confirm_storyboard()
        return session, "分镜已确认。真实视频仍需勾选费用确认。"
    except Exception as exc:
        return session, str(exc)


def render_video_with_progress(
    session: GuidedStorySession | None,
    cost_confirmed: bool = False,
    *,
    provider=None,
    output_dir: str | Path | None = None,
) -> Iterator[tuple[GuidedStorySession | None, str | None, str]]:
    if session is None or session.stage != Stage.RENDER_READY:
        yield session, None, "必须先确认完整分镜，才能生成真实视频。"
        return
    if not cost_confirmed:
        yield session, None, "请先勾选费用确认；当前没有调用视频API。"
        return
    queue: Queue[tuple[float, str]] = Queue()
    result: dict[str, object] = {}

    def progress(stage: str, fraction: float, message: str) -> None:
        queue.put((fraction, message))

    def work() -> None:
        try:
            renderer = StoryRenderer(
                provider or AgnesVideoProvider.from_env(), progress_callback=progress
            )
            target = output_dir or os.getenv("VIDEO_OUTPUT_DIR", "outputs/videos")
            result["manifest"] = session.render_confirmed_plan(renderer, target)
        except Exception as exc:
            result["error"] = exc

    worker = Thread(target=work, daemon=True)
    worker.start()
    fraction, message = 0.0, "正在准备旁白与视频任务"
    yield session, None, _progress_text(fraction, message)
    while worker.is_alive():
        try:
            fraction, message = queue.get(timeout=0.5)
        except Empty:
            pass
        yield session, None, _progress_text(fraction, message)
    worker.join()
    while not queue.empty():
        fraction, message = queue.get_nowait()
    if "error" in result:
        yield session, None, f"生成失败：{result['error']}"
        return
    manifest = result.get("manifest")
    if manifest is None or manifest.status != "succeeded":
        yield session, None, f"生成失败：{getattr(manifest, 'error', '没有生成manifest')}"
        return
    yield session, manifest.final_video_path, _progress_text(1.0, "成片已保存到本地")


def _idea_card_grid_class(gr):
    def IdeaCardGrid(**kwargs):
        """Create the native multi-select card grid without generated component files."""
        kwargs.setdefault("choices", [])
        kwargs.setdefault("value", [])
        kwargs.setdefault("elem_classes", ["idea-card-grid"])
        return gr.CheckboxGroup(**kwargs)

    return IdeaCardGrid


def build_app(agent_factory: Callable[[], StoryAgent] | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('请先安装网页依赖：pip install -e ".[web]"') from exc

    IdeaCardGrid = _idea_card_grid_class(gr)

    def start(direction, seconds):
        return start_garden_view(
            direction,
            seconds,
            agent=(agent_factory() if agent_factory else RuleBasedStoryAgent()),
        )

    start.__name__ = "start_garden_view"

    card_css = """
.idea-card-grid .wrap{display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.idea-card-grid label{align-items:flex-start!important;min-height:150px;padding:14px!important;border:1px solid var(--border-color-primary);border-radius:16px;background:var(--block-background-fill);white-space:pre-line;transition:.15s}
.idea-card-grid label:hover{transform:translateY(-2px);border-color:var(--color-accent)}
.idea-card-grid label:has(input:checked){border:2px solid var(--color-accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--color-accent) 18%,transparent)}
@media(max-width:900px){.idea-card-grid .wrap{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""
    with gr.Blocks(title="一句话剧本创意花园", css=card_css) as app:
        gr.Markdown("# 一句话剧本创意花园\n只要说一个方向。AI负责想很多，你只负责挑喜欢的。")
        session_state = gr.State()
        with gr.Row():
            direction = gr.Textbox(
                label="你的大概方向", placeholder="例如：校园里发生一件带点悬疑的事", scale=5
            )
            target_seconds = gr.Slider(30, 60, value=45, step=1, label="成片秒数", scale=2)
            begin = gr.Button("给我8个想法", variant="primary", scale=2)
        status = gr.Textbox(label="当前提示", interactive=False)
        card_grid = IdeaCardGrid(label="创意卡（直接点击，最多选择3张）")
        selection = gr.Markdown("**已保留：** 暂无。最多选择3张。")
        with gr.Row():
            refresh = gr.Button("换一批")
            more_like = gr.Button("更多像这个")
            mix = gr.Button("混合已选")
            auto = gr.Button("AI替我选")
            expand = gr.Button("展开故事零件")
            make_draft = gr.Button("直接生成草稿", variant="primary")

        with gr.Accordion("还想随便聊两句（完全可选）", open=False):
            chat = gr.Chatbot(type="messages", height=240, allow_tags=False)
            chat_input = gr.Textbox(
                placeholder="例如：更搞笑、不要爱情、换成古代", label="调整方向"
            )
            chat_send = gr.Button("按这句话再想8个")

        with gr.Accordion("故事拼图（全部可跳过）", open=False):
            gr.Markdown("每类最多选一个；不选择的部分由AI补全并标注。")
            with gr.Row():
                character = gr.Radio(label="主角")
                conflict = gr.Radio(label="冲突")
            with gr.Row():
                turning = gr.Radio(label="转折")
                ending = gr.Radio(label="结局")
            keep_elements = gr.Button("保留这些零件")

        gr.Markdown("## 剧本草稿")
        ai_fill = gr.Markdown("")
        draft = gr.Markdown("生成后会按场景展示，不需要填写表格。")
        with gr.Row():
            draft_feedback = gr.Textbox(
                label="一句话修改", placeholder="例如：结局更温暖，减少旁白", scale=5
            )
            rewrite = gr.Button("按这句话改写", scale=2)
        with gr.Row():
            back = gr.Button("回到灵感区")
            make_storyboard = gr.Button("接受草稿并生成分镜", variant="primary")

        gr.Markdown("## 分镜时间线")
        storyboard = gr.Markdown("确认剧本后生成。")
        with gr.Row():
            shot_choice = gr.Dropdown(label="选择要Retake的镜头")
            retake_feedback = gr.Textbox(
                label="Retake要求", placeholder="例如：改成近景，动作更克制"
            )
            retake = gr.Button("重做这个镜头方案")
        confirm_storyboard = gr.Button("确认分镜")
        cost_confirmed = gr.Checkbox(label="我确认下一步会调用付费视频API")
        render = gr.Button("生成真实视频", variant="primary")
        video = gr.Video(label="最终视频", interactive=False)

        base_outputs = [session_state, card_grid, selection, chat, status]
        begin.click(start, [direction, target_seconds], base_outputs, api_name="start_ideation")
        direction.submit(start, [direction, target_seconds], base_outputs)
        card_grid.input(
            select_cards_view,
            [session_state, card_grid],
            [session_state, card_grid, selection, status],
            api_name="select_ideas",
        )
        refresh.click(
            refresh_ideas_view,
            [session_state],
            [session_state, card_grid, selection, status],
            api_name="refresh_ideas",
        )
        more_like.click(
            more_like_view,
            [session_state],
            [session_state, card_grid, selection, status],
            api_name="more_like",
        )
        mix.click(
            mix_selected_view,
            [session_state],
            [session_state, card_grid, selection, status],
            api_name="mix_selected",
        )
        auto.click(
            auto_choose_view,
            [session_state],
            [session_state, card_grid, selection, status],
            api_name="auto_choose",
        )
        chat_send.click(
            chat_ideation_view,
            [session_state, chat_input, chat],
            [session_state, card_grid, chat, chat_input, status],
            api_name="chat_ideation",
        )
        chat_input.submit(
            chat_ideation_view,
            [session_state, chat_input, chat],
            [session_state, card_grid, chat, chat_input, status],
        )
        expand.click(
            expand_elements_view,
            [session_state],
            [session_state, character, conflict, turning, ending, status],
            api_name="expand_selected",
        )
        keep_elements.click(
            choose_elements_view,
            [session_state, character, conflict, turning, ending],
            [session_state, selection, status],
            api_name="choose_elements",
        )
        make_draft.click(
            generate_draft_view,
            [session_state],
            [session_state, draft, ai_fill, status],
            api_name="generate_draft",
        )
        rewrite.click(
            revise_draft_view,
            [session_state, draft_feedback],
            [session_state, draft, ai_fill, draft_feedback, status],
            api_name="revise_draft",
        )
        back.click(
            back_to_ideas_view,
            [session_state],
            [session_state, card_grid, status],
            api_name="back_to_ideas",
        )
        make_storyboard.click(
            build_storyboard_view,
            [session_state],
            [session_state, storyboard, shot_choice, status],
            api_name="build_storyboard",
        )
        retake.click(
            retake_shot_view,
            [session_state, shot_choice, retake_feedback],
            [session_state, storyboard, retake_feedback, status],
            api_name="retake_shot",
        )
        confirm_storyboard.click(
            confirm_storyboard_view,
            [session_state],
            [session_state, status],
            api_name="confirm_storyboard",
        )
        render.click(
            render_video_with_progress,
            [session_state, cost_confirmed],
            [session_state, video, status],
            show_progress="hidden",
            api_name="render_video",
        )
    return app


def _selection_text(session: GuidedStorySession) -> str:
    cards = session.selected_cards
    card_text = "、".join(f"《{card.title}》" for card in cards) or "暂无"
    element_text = "、".join(session.selected_elements) or "暂无"
    return f"**已保留创意：** {card_text}  \n**已保留故事零件：** {element_text}"


def _status_text(session: GuidedStorySession) -> str:
    if session.stage == Stage.IDEATING:
        return "不需要补字段：选卡、聊天或直接生成草稿都可以。"
    if session.stage == Stage.DRAFT_REVIEW:
        return f"当前是第{session.draft.version}版草稿，可以改写或回到灵感区。"
    return f"当前阶段：{session.stage.value}"


def _element_update(palette: ElementPalette, kind: str):
    options = palette.options[kind]
    return _gr_update(
        choices=[(f"{item.title}｜{item.content}", item.option_id) for item in options],
        value=None,
    )


def _gr_update(**kwargs):
    try:
        import gradio as gr

        return gr.update(**kwargs)
    except ImportError:
        return kwargs


def _draft_markdown(draft) -> str:
    sections = [f"### 《{draft.script.title}》 · 第{draft.version}版"]
    for scene in draft.script.scenes:
        sections.append(
            f"#### 场景 {scene.scene_id}｜{scene.title} · {scene.duration}秒\n"
            f"**地点：** {scene.location}  \n"
            f"**画面动作：** {scene.visible_action or scene.action}  \n"
            f"**对白：** {scene.dialogue or '—'}  \n"
            f"**旁白：** {scene.narration or '—'}"
        )
    return "\n\n".join(sections)


def _ai_fill_markdown(draft) -> str:
    labels = {
        "protagonist": "主角细节",
        "conflict": "核心冲突",
        "turning_point": "关键转折",
        "ending": "结局",
        "scene_details": "场景细节",
        "props": "关键道具",
        "narration": "旁白",
        "dialogue": "对白",
        "transitions": "转场",
    }
    items = [labels.get(item, item) for item in draft.ai_filled_fields]
    return "**AI补全：** " + ("、".join(items) if items else "无；全部关键内容来自你的选择。")


def _storyboard_markdown(plan) -> str:
    if plan is None:
        return ""
    sections = [f"### 《{plan.title}》 · {plan.total_duration}秒"]
    for shot in plan.shots:
        sections.append(
            f"**镜头 {shot.shot_id}｜{shot.duration}秒｜{shot.camera}**  \n"
            f"{shot.action}  \n"
            f"连续性：{'；'.join(shot.continuity_notes) or '保持主体与空间一致'}"
        )
    return "\n\n".join(sections)


def _progress_text(fraction: float, message: str) -> str:
    return f"视频进度 {round(min(1.0, max(0.0, fraction)) * 100)}%｜{message}"


def main() -> None:
    build_app(agent_factory=OpenAIStoryAgent.from_env).launch()


if __name__ == "__main__":
    main()
