from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from .agent import ALL_FACT_FIELDS, OpenAIStoryAgent, RuleBasedStoryAgent, StoryAgent
from .models import CreativeBrief, Stage, to_plain_data
from .rendering import StoryRenderer
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider


FACT_LABELS = {
    "premise": "核心概念",
    "genre": "类型",
    "tone": "基调",
    "theme": "主题",
    "audience": "目标观众",
    "opening": "开头",
    "protagonist": "主角",
    "protagonist_goal": "人物目标",
    "motivation": "动机",
    "conflict": "冲突",
    "stakes": "风险与代价",
    "development": "发展",
    "turning_point": "转折",
    "ending": "结局",
    "character_visuals": "人物视觉卡",
    "scene_details": "场景卡",
    "props": "道具",
    "narration_style": "旁白",
    "dialogue_style": "对白风格",
    "camera_style": "摄影风格",
    "visual_anchors": "视觉锚点",
    "transitions": "镜头承接",
}


def initialize_view(
    target_seconds: int | float = 45,
    agent: StoryAgent | None = None,
) -> tuple[GuidedStorySession, list[dict[str, str]], None, None, None, str]:
    session = GuidedStorySession(
        CreativeBrief(target_seconds=int(target_seconds)),
        agent=agent or RuleBasedStoryAgent(),
    )
    chat = [{"role": "assistant", "content": session.current_question}]
    return session, chat, None, None, None, "新创作已开始，请先给出故事开头。"


def submit_message(
    session: GuidedStorySession | None,
    message: str,
    history: list[dict[str, str]] | None,
) -> tuple[GuidedStorySession, list[dict[str, str]], str, str]:
    if session is None:
        session, history, _, _, _, _ = initialize_view(45)
    chat = list(history or [])
    cleaned = " ".join(message.split())
    if cleaned:
        chat.append({"role": "user", "content": cleaned})
    try:
        if session.stage == Stage.COLLECTING:
            result = session.submit_user_turn(message)
        elif session.stage == Stage.DETAILING:
            result = session.answer_detail_question(message)
        else:
            return session, chat, "", f"当前阶段是 {session.stage.value}，请在右侧工作区继续。"
    except Exception as exc:
        return session, chat, "", str(exc)
    assistant = result.assistant_message
    if result.extracted_facts:
        understood = "；".join(
            f"{FACT_LABELS.get(item.field, item.field)}：{item.value}"
            for item in result.extracted_facts
        )
        assistant += f"\n\n我理解了：{understood}"
    if result.conflicts:
        assistant += "\n\n需要你处理冲突：" + "；".join(item.reason for item in result.conflicts)
    if result.next_question:
        assistant += f"\n\n为什么现在问：这是当前对故事完整度影响最大的缺口。\n\n{result.next_question}"
    chat.append({"role": "assistant", "content": assistant})
    return session, chat, "", _status_text(session)


def apply_suggestion_view(
    session: GuidedStorySession | None,
    suggestion_id: str | None,
    history: list[dict[str, str]] | None,
) -> tuple[GuidedStorySession | None, list[dict[str, str]], str]:
    chat = list(history or [])
    if session is None or not suggestion_id:
        return session, chat, "请先选择一个方向。"
    try:
        selected = next(
            item for item in session.request_suggestions() if item.suggestion_id == suggestion_id
        )
        chat.append({"role": "user", "content": f"采用方向：{selected.content}"})
        result = session.apply_suggestion(suggestion_id)
        chat.append(
            {
                "role": "assistant",
                "content": f"{result.assistant_message}\n\n{result.next_question}",
            }
        )
        return session, chat, _status_text(session)
    except Exception as exc:
        return session, chat, str(exc)


def dismiss_suggestions_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str]:
    if session is not None:
        session.pending_suggestions = []
    return session, "已忽略这些方向，故事内容没有改变。"


def build_outline_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict | None, str]:
    if session is None:
        return session, None, "请先开始创作。"
    try:
        outline = session.build_outline()
    except Exception as exc:
        return session, None, str(exc)
    return session, to_plain_data(outline), "大纲候选已生成；尚未确认，可以编辑、撤销或重写。"


def confirm_outline_view(
    session: GuidedStorySession | None,
    history: list[dict[str, str]] | None,
) -> tuple[GuidedStorySession | None, list[dict[str, str]], str]:
    chat = list(history or [])
    if session is None:
        return session, chat, "请先开始创作。"
    try:
        session.confirm_outline()
    except Exception as exc:
        return session, chat, str(exc)
    chat.append({"role": "assistant", "content": f"大纲已确认。接下来只补制作细节。\n\n{session.current_question}"})
    return session, chat, "大纲已确认；后续修改会产生新版本，不会静默覆盖。"


def build_script_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict | None, str]:
    if session is None:
        return session, None, "请先开始创作。"
    try:
        script = session.build_script()
    except Exception as exc:
        return session, None, str(exc)
    return session, to_plain_data(script), "定时剧本候选已生成，请检查可拍摄性和时长。"


def confirm_script_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str]:
    if session is None:
        return session, "请先开始创作。"
    try:
        session.confirm_script()
    except Exception as exc:
        return session, str(exc)
    return session, "剧本已确认，可以生成分镜。"


def build_storyboard_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict | None, str]:
    if session is None:
        return session, None, "请先开始创作。"
    try:
        storyboard = session.build_storyboard()
    except Exception as exc:
        return session, None, str(exc)
    return session, to_plain_data(storyboard), "分镜候选已生成；确认前绝不会调用视频 API。"


def confirm_storyboard_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str]:
    if session is None:
        return session, "请先开始创作。"
    try:
        session.confirm_storyboard()
    except Exception as exc:
        return session, str(exc)
    return session, "分镜已确认。付费按钮仍需勾选费用确认。"


def save_story_bible_view(
    session: GuidedStorySession | None, rows: list[list[Any]] | None
) -> tuple[GuidedStorySession | None, str]:
    if session is None:
        return session, "请先开始创作。"
    reverse = {label: field for field, label in FACT_LABELS.items()}
    patch = {
        reverse.get(str(row[0]), str(row[0])): str(row[1] or "")
        for row in _table_rows(rows)
        if len(row) >= 2 and reverse.get(str(row[0]), str(row[0])) in ALL_FACT_FIELDS
    }
    try:
        session.update_story_bible(patch)
        return session, "故事圣经修改已保存为新版本。"
    except Exception as exc:
        return session, str(exc)


def save_outline_rows(
    session: GuidedStorySession | None, rows: list[list[Any]] | None
) -> tuple[GuidedStorySession | None, str]:
    if session is None or session.outline is None:
        return session, "尚未生成大纲。"
    try:
        beats = []
        for row in _table_rows(rows):
            beats.append(
                {
                    "beat_id": int(row[0]), "purpose": str(row[1]), "event": str(row[2]),
                    "causal_link": str(row[3]), "emotional_change": str(row[4]),
                    "duration": int(row[5]), "source_turn_ids": session.outline.source_turn_ids,
                }
            )
        session.update_outline({"beats": beats})
        return session, "大纲卡片已保存为新版本。"
    except Exception as exc:
        return session, str(exc)


def save_script_rows(
    session: GuidedStorySession | None, rows: list[list[Any]] | None
) -> tuple[GuidedStorySession | None, str]:
    if session is None or session.script is None:
        return session, "尚未生成剧本。"
    try:
        for row in _table_rows(rows):
            session.update_script_scene(
                int(row[0]),
                {
                    "title": str(row[1]), "location": str(row[2]),
                    "visible_action": str(row[3]), "dialogue": str(row[4]),
                    "narration": str(row[5]), "duration": int(row[6]),
                    "start_state": str(row[7]), "end_state": str(row[8]),
                },
            )
        return session, "剧本场景已保存为新版本。"
    except Exception as exc:
        return session, str(exc)


def save_storyboard_rows(
    session: GuidedStorySession | None, rows: list[list[Any]] | None
) -> tuple[GuidedStorySession | None, str]:
    if session is None or session.storyboard is None:
        return session, "尚未生成分镜。"
    try:
        for row in _table_rows(rows):
            session.update_storyboard_shot(
                int(row[0]),
                {
                    "duration": int(row[1]), "shot_purpose": str(row[2]),
                    "action": str(row[3]), "camera": str(row[4]),
                    "camera_movement": str(row[5]), "start_frame": str(row[6]),
                    "end_frame": str(row[7]), "video_prompt": str(row[8]),
                    "negative_prompt": str(row[9]),
                },
            )
        return session, "分镜修改已保存为新版本（Retake 候选）。"
    except Exception as exc:
        return session, str(exc)


def revise_current_view(
    session: GuidedStorySession | None, feedback: str
) -> tuple[GuidedStorySession | None, str]:
    if session is None:
        return session, "请先开始创作。"
    try:
        if session.stage == Stage.OUTLINE_REVIEW:
            session.revise_outline(feedback)
        elif session.stage == Stage.SCRIPT_REVIEW:
            session.revise_script(feedback)
        elif session.stage in (Stage.STORYBOARD_REVIEW, Stage.RENDER_READY):
            session.revise_storyboard(feedback)
        else:
            raise RuntimeError("当前阶段没有可重写的产物。")
        return session, "已根据反馈生成新版本，原版本仍在历史中。"
    except Exception as exc:
        return session, str(exc)


def history_action_view(
    session: GuidedStorySession | None, action: str
) -> tuple[GuidedStorySession | None, str]:
    if session is None:
        return session, "请先开始创作。"
    try:
        session.undo_artifact() if action == "undo" else session.redo_artifact()
        return session, "已撤销。" if action == "undo" else "已重做。"
    except Exception as exc:
        return session, str(exc)


def review_current_view(session: GuidedStorySession | None) -> str:
    if session is None:
        return "请先开始创作。"
    try:
        review = session.review_current_artifact()
    except Exception as exc:
        return str(exc)
    hard = "\n".join(f"- 必须修复：{item}" for item in review.hard_errors) or "- 没有硬错误"
    warnings = "\n".join(f"- 可选建议：{item}" for item in review.warnings) or "- 没有额外建议"
    return f"### 质量检查\n{hard}\n{warnings}"


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
        yield session, None, "请先勾选费用确认；当前没有调用视频 API。"
        return
    queue: Queue[tuple[float, str]] = Queue()
    result: dict[str, object] = {}

    def progress(stage: str, fraction: float, message: str) -> None:
        queue.put((fraction, message))

    def work() -> None:
        try:
            renderer = StoryRenderer(provider or AgnesVideoProvider.from_env(), progress_callback=progress)
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
        yield session, None, f"生成失败：{getattr(manifest, 'error', '没有生成 manifest')}"
        return
    yield session, manifest.final_video_path, _progress_text(1.0, "成片已保存到本地")


def _progress_text(fraction: float, message: str) -> str:
    return f"视频进度 {round(min(1.0, max(0.0, fraction)) * 100)}%｜{message}"


def _table_rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.get("data", []))
    if hasattr(value, "values"):
        return value.values.tolist()
    return list(value)


def _status_text(session: GuidedStorySession) -> str:
    missing = (
        session.facts.missing_outline_fields()
        if session.stage == Stage.COLLECTING
        else session.facts.missing_detail_fields()
    )
    return (
        f"阶段：{session.stage.value}｜完成度：{round(session.readiness_score * 100)}%｜"
        f"有效参与：{session.valid_turns}/5｜缺口：{'、'.join(missing) or '无'}"
    )


def _bible_rows(session: GuidedStorySession | None) -> list[list[str]]:
    if session is None:
        return []
    return [[FACT_LABELS[field], getattr(session.facts, field)] for field in ALL_FACT_FIELDS]


def _outline_rows(session: GuidedStorySession | None) -> list[list[Any]]:
    return [] if session is None or session.outline is None else [
        [beat.beat_id, beat.purpose, beat.event, beat.causal_link, beat.emotional_change, beat.duration]
        for beat in session.outline.beats
    ]


def _script_rows(session: GuidedStorySession | None) -> list[list[Any]]:
    return [] if session is None or session.script is None else [
        [scene.scene_id, scene.title, scene.location, scene.visible_action, scene.dialogue,
         scene.narration, scene.duration, scene.start_state, scene.end_state]
        for scene in session.script.scenes
    ]


def _storyboard_rows(session: GuidedStorySession | None) -> list[list[Any]]:
    return [] if session is None or session.storyboard is None else [
        [shot.shot_id, shot.duration, shot.shot_purpose, shot.action, shot.camera,
         shot.camera_movement, shot.start_frame, shot.end_frame, shot.video_prompt,
         shot.negative_prompt]
        for shot in session.storyboard.shots
    ]


def _workspace_values(session: GuidedStorySession | None) -> tuple[Any, Any, Any, Any, Any]:
    try:
        import gradio as gr
        suggestions = [] if session is None else [
            (f"{item.label}｜{item.content}", item.suggestion_id)
            for item in session.request_suggestions()
        ]
        suggestion_update = gr.update(choices=suggestions, value=None)
    except ImportError:
        suggestion_update = None
    return _bible_rows(session), _outline_rows(session), _script_rows(session), _storyboard_rows(session), suggestion_update


def build_app(agent_factory: Callable[[], StoryAgent] | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('请先安装网页依赖：pip install -e ".[web]"') from exc

    def start(seconds: int | float):
        base = initialize_view(seconds, agent_factory() if agent_factory else RuleBasedStoryAgent())
        bible, outline_rows, script_rows, shot_rows, suggestions = _workspace_values(base[0])
        return (*base, bible, outline_rows, script_rows, shot_rows, suggestions)

    start.__name__ = "initialize_view"

    def send_workspace(session, message, chat):
        updated = submit_message(session, message, chat)
        return (*updated, *_workspace_values(updated[0]))

    send_workspace.__name__ = "submit_message"

    def use_workspace(session, suggestion, chat):
        updated = apply_suggestion_view(session, suggestion, chat)
        bible, outline_rows, script_rows, shot_rows, suggestions = _workspace_values(updated[0])
        return updated[0], updated[1], updated[2], bible, outline_rows, script_rows, shot_rows, suggestions

    use_workspace.__name__ = "apply_suggestion_view"

    def refresh_workspace(session):
        return (*_workspace_values(session)[:4], _status_text(session) if session else "请先开始。")

    def make_outline_candidate(session):
        updated, _, message_text = build_outline_view(session)
        return updated, message_text

    def make_script_candidate(session):
        updated, _, message_text = build_script_view(session)
        return updated, message_text

    def make_storyboard_candidate(session):
        updated, _, message_text = build_storyboard_view(session)
        return updated, message_text

    def undo_current(session):
        return history_action_view(session, "undo")

    def redo_current(session):
        return history_action_view(session, "redo")

    with gr.Blocks(title="引导式剧本到视频 Agent") as app:
        gr.Markdown("# 引导式剧本共创工作台\n你掌控故事，AI 只诊断当前最重要的缺口并提供可拒绝的方向。")
        session_state = gr.State()
        with gr.Row():
            target_seconds = gr.Slider(30, 60, value=45, step=1, label="目标成片时长（秒）")
            restart = gr.Button("开始 / 重新开始")
        status = gr.Textbox(label="当前流程", interactive=False)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(type="messages", label="共创对话", height=560, allow_tags=False)
                suggestion = gr.Radio(label="三个非强制方向", choices=[])
                with gr.Row():
                    use_suggestion = gr.Button("采用所选方向")
                    ignore_suggestion = gr.Button("忽略建议")
                message = gr.Textbox(label="你的方向", placeholder="自由回答，也可以完全不采用建议")
                send = gr.Button("提交", variant="primary")
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("故事地图"):
                        bible = gr.Dataframe(headers=["项目", "已确认内容"], datatype=["str", "str"], interactive=True, wrap=True)
                        save_bible = gr.Button("保存故事圣经修改")
                    with gr.Tab("五节点大纲"):
                        outline_table = gr.Dataframe(headers=["ID", "叙事目的", "具体事件", "因果承接", "情绪变化", "秒"], interactive=True, wrap=True)
                        with gr.Row():
                            make_outline = gr.Button("完成并生成大纲")
                            save_outline = gr.Button("保存大纲修改")
                            confirm_outline = gr.Button("确认大纲")
                    with gr.Tab("定时剧本"):
                        script_table = gr.Dataframe(headers=["ID", "场景", "地点", "可见动作", "对白", "旁白", "秒", "起始状态", "结束状态"], interactive=True, wrap=True)
                        with gr.Row():
                            make_script = gr.Button("生成剧本")
                            save_script = gr.Button("保存剧本修改")
                            confirm_script = gr.Button("确认剧本")
                    with gr.Tab("分镜 Retake"):
                        storyboard_table = gr.Dataframe(headers=["ID", "秒", "目的", "动作", "景别机位", "运动", "起始画面", "结束画面", "正向提示词", "负向提示词"], interactive=True, wrap=True)
                        with gr.Row():
                            make_storyboard = gr.Button("生成分镜")
                            save_storyboard = gr.Button("保存分镜修改")
                            confirm_storyboard = gr.Button("确认分镜")
                feedback = gr.Textbox(label="局部重写意见")
                with gr.Row():
                    revise = gr.Button("按反馈生成新版本")
                    undo = gr.Button("撤销")
                    redo = gr.Button("重做")
                    review = gr.Button("质量检查")
                quality = gr.Markdown("质量检查会区分必须修复的问题和可忽略建议。")
                cost_confirmed = gr.Checkbox(label="我确认下面操作会调用付费视频 API")
                render = gr.Button("生成真实视频", variant="primary")
                video = gr.Video(label="最终视频", interactive=False)

        common_outputs = [session_state, chatbot, message, status, bible, outline_table, script_table, storyboard_table, suggestion]
        restart_outputs = [session_state, chatbot, gr.State(), gr.State(), gr.State(), status, bible, outline_table, script_table, storyboard_table, suggestion]
        restart.click(start, [target_seconds], restart_outputs)
        send.click(send_workspace, [session_state, message, chatbot], common_outputs)
        message.submit(send_workspace, [session_state, message, chatbot], common_outputs)
        use_suggestion.click(use_workspace, [session_state, suggestion, chatbot], [session_state, chatbot, status, bible, outline_table, script_table, storyboard_table, suggestion])
        ignore_suggestion.click(dismiss_suggestions_view, [session_state], [session_state, status])
        save_bible.click(save_story_bible_view, [session_state, bible], [session_state, status])
        save_outline.click(save_outline_rows, [session_state, outline_table], [session_state, status])
        save_script.click(save_script_rows, [session_state, script_table], [session_state, status])
        save_storyboard.click(save_storyboard_rows, [session_state, storyboard_table], [session_state, status])

        def bind_and_refresh(button, fn, inputs):
            return button.click(fn, inputs, [session_state, status]).then(
                refresh_workspace, [session_state], [bible, outline_table, script_table, storyboard_table, status]
            )

        bind_and_refresh(make_outline, make_outline_candidate, [session_state])
        bind_and_refresh(make_script, make_script_candidate, [session_state])
        bind_and_refresh(make_storyboard, make_storyboard_candidate, [session_state])
        confirm_outline.click(confirm_outline_view, [session_state, chatbot], [session_state, chatbot, status])
        confirm_script.click(confirm_script_view, [session_state], [session_state, status])
        confirm_storyboard.click(confirm_storyboard_view, [session_state], [session_state, status])
        revise.click(revise_current_view, [session_state, feedback], [session_state, status]).then(refresh_workspace, [session_state], [bible, outline_table, script_table, storyboard_table, status])
        undo.click(undo_current, [session_state], [session_state, status]).then(refresh_workspace, [session_state], [bible, outline_table, script_table, storyboard_table, status])
        redo.click(redo_current, [session_state], [session_state, status]).then(refresh_workspace, [session_state], [bible, outline_table, script_table, storyboard_table, status])
        review.click(review_current_view, [session_state], [quality])
        render.click(render_video_with_progress, [session_state, cost_confirmed], [session_state, video, status], show_progress="hidden")
        app.load(start, [target_seconds], restart_outputs)
    return app


def main() -> None:
    build_app(agent_factory=OpenAIStoryAgent.from_env).launch()


if __name__ == "__main__":
    main()
