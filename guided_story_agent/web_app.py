from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from .agent import OpenAIStoryAgent
from .models import CreativeBrief, Stage, to_plain_data
from .rendering import StoryRenderer
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider


def initialize_view(
    target_seconds: int | float = 45,
) -> tuple[GuidedStorySession, list[dict[str, str]], None, None, None, str]:
    session = GuidedStorySession(
        CreativeBrief(target_seconds=int(target_seconds)),
        agent=OpenAIStoryAgent.from_env(),
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
            return session, chat, "", f"当前阶段是 {session.stage.value}，请使用下方确认按钮继续。"
    except Exception as exc:
        return session, chat, "", str(exc)
    assistant = result.assistant_message
    if result.next_question:
        assistant += f"\n\n{result.next_question}"
    if result.suggestions and result.next_question:
        assistant += "\n\n可参考：" + " / ".join(result.suggestions)
    chat.append({"role": "assistant", "content": assistant})
    status = (
        f"当前阶段：{session.stage.value}｜有效剧情轮次：{session.valid_turns}｜"
        f"仍缺少：{'、'.join(result.missing_fields) or '无'}"
    )
    return session, chat, "", status


def build_outline_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict | None, str]:
    if session is None:
        return session, None, "请先开始创作。"
    try:
        outline = session.build_outline()
    except Exception as exc:
        return session, None, str(exc)
    return session, to_plain_data(outline), "大纲已生成，请检查后确认。"


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
    chat.append({"role": "assistant", "content": f"大纲已确认。接下来完善制作细节。\n\n{session.current_question}"})
    return session, chat, "大纲已锁定；后续回答不会直接改写已确认的大纲。"


def build_script_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict | None, str]:
    if session is None:
        return session, None, "请先开始创作。"
    try:
        script = session.build_script()
    except Exception as exc:
        return session, None, str(exc)
    return session, to_plain_data(script), "剧本已按目标时长生成，请确认。"


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
    return session, to_plain_data(storyboard), "分镜已生成；确认前不会调用视频 API。"


def confirm_storyboard_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str]:
    if session is None:
        return session, "请先开始创作。"
    try:
        session.confirm_storyboard()
    except Exception as exc:
        return session, str(exc)
    return session, "分镜已确认。只有点击“生成真实视频”才会产生付费请求。"


def render_video_with_progress(
    session: GuidedStorySession | None,
    *,
    provider=None,
    output_dir: str | Path | None = None,
) -> Iterator[tuple[GuidedStorySession | None, str | None, str]]:
    if session is None or session.stage != Stage.RENDER_READY:
        yield session, None, "必须先确认完整分镜，才能生成真实视频。"
        return
    queue: Queue[tuple[float, str]] = Queue()
    result: dict[str, object] = {}

    def progress(stage: str, fraction: float, message: str) -> None:
        queue.put((fraction, message))

    def work() -> None:
        try:
            renderer = StoryRenderer(
                provider or AgnesVideoProvider.from_env(),
                progress_callback=progress,
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
        detail = getattr(manifest, "error", "没有生成 manifest")
        yield session, None, f"生成失败：{detail}"
        return
    yield session, manifest.final_video_path, _progress_text(1.0, "成片已保存到本地")


def _progress_text(fraction: float, message: str) -> str:
    return f"视频进度 {round(min(1.0, max(0.0, fraction)) * 100)}%｜{message}"


def build_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('请先安装网页依赖：pip install -e ".[web]"') from exc

    with gr.Blocks(title="引导式剧本到视频 Agent") as app:
        gr.Markdown(
            "# 引导式剧本到视频 Agent\n"
            "从一句开头开始，由 AI 动态追问；只有故事、剧本和分镜依次确认后，才允许生成视频。"
        )
        session_state = gr.State()
        with gr.Row():
            target_seconds = gr.Slider(30, 60, value=45, step=1, label="目标成片时长（秒）")
            restart = gr.Button("开始 / 重新开始")
        chatbot = gr.Chatbot(
            type="messages", label="创作对话", height=430, allow_tags=False
        )
        message = gr.Textbox(
            label="你的回答",
            placeholder="先写一句故事开头，之后根据 AI 的问题继续补充",
        )
        send = gr.Button("提交这条方向", variant="primary")
        status = gr.Textbox(label="系统状态", interactive=False)

        with gr.Row():
            make_outline = gr.Button("完成并生成大纲")
            confirm_outline = gr.Button("确认大纲")
        outline = gr.JSON(label="故事大纲", show_indices=False)
        with gr.Row():
            make_script = gr.Button("生成定时剧本")
            confirm_script = gr.Button("确认剧本")
        script = gr.JSON(label="定时剧本", show_indices=False)
        with gr.Row():
            make_storyboard = gr.Button("生成分镜")
            confirm_storyboard = gr.Button("确认分镜")
        storyboard = gr.JSON(label="最终分镜", show_indices=False)
        render = gr.Button("生成真实视频（显式调用 API）", variant="primary")
        video = gr.Video(label="最终视频", interactive=False)

        restart.click(
            initialize_view,
            inputs=[target_seconds],
            outputs=[session_state, chatbot, outline, script, storyboard, status],
        )
        send.click(
            submit_message,
            inputs=[session_state, message, chatbot],
            outputs=[session_state, chatbot, message, status],
        )
        message.submit(
            submit_message,
            inputs=[session_state, message, chatbot],
            outputs=[session_state, chatbot, message, status],
        )
        make_outline.click(
            build_outline_view,
            inputs=[session_state],
            outputs=[session_state, outline, status],
        )
        confirm_outline.click(
            confirm_outline_view,
            inputs=[session_state, chatbot],
            outputs=[session_state, chatbot, status],
        )
        make_script.click(
            build_script_view,
            inputs=[session_state],
            outputs=[session_state, script, status],
        )
        confirm_script.click(
            confirm_script_view,
            inputs=[session_state],
            outputs=[session_state, status],
        )
        make_storyboard.click(
            build_storyboard_view,
            inputs=[session_state],
            outputs=[session_state, storyboard, status],
        )
        confirm_storyboard.click(
            confirm_storyboard_view,
            inputs=[session_state],
            outputs=[session_state, status],
        )
        render.click(
            render_video_with_progress,
            inputs=[session_state],
            outputs=[session_state, video, status],
            show_progress="hidden",
        )
        app.load(
            initialize_view,
            inputs=[target_seconds],
            outputs=[session_state, chatbot, outline, script, storyboard, status],
        )
    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
