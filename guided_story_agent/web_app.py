from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Callable, Iterator
from functools import wraps
from hashlib import sha256
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any
from uuid import uuid4

from .agent import OpenAIStoryAgent, RuleBasedStoryAgent, StoryAgent
from .models import CreativeBrief, ElementPalette, Stage
from .rendering import StoryRenderer, VideoJobRenderer  # noqa: F401 - legacy patch/API compatibility
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider


VISUAL_USAGE_CHOICES = [
    ("人物身份参考", "identity_reference"),
    ("地点参考", "location_reference"),
    ("道具参考", "prop_reference"),
    ("场景气氛参考", "scene_reference"),
    ("视频首帧", "start_frame"),
]


def _web_session_path(path: str | Path | None = None) -> Path:
    return (
        Path(path or os.getenv("WEB_SESSION_PATH", "outputs/web/latest_session.json"))
        .expanduser()
        .resolve()
    )


def _autosave_web_session(
    session: GuidedStorySession | None,
    path: str | Path | None = None,
) -> None:
    if session is None or session.render_in_progress:
        return
    try:
        session.save(_web_session_path(path))
    except Exception as exc:
        warnings.warn(f"网页会话自动保存失败：{exc}", RuntimeWarning, stacklevel=2)


def _autosaving_view(callback):
    @wraps(callback)
    def wrapped(*args, **kwargs):
        result = callback(*args, **kwargs)
        if isinstance(result, tuple) and result:
            _autosave_web_session(result[0])
        return result

    return wrapped


def _autosaving_progress(callback):
    @wraps(callback)
    def wrapped(*args, **kwargs):
        for result in callback(*args, **kwargs):
            if isinstance(result, tuple) and result:
                _autosave_web_session(result[0])
            yield result

    return wrapped


def restore_saved_session_view(
    *,
    agent: StoryAgent | None = None,
    path: str | Path | None = None,
) -> tuple[Any, ...]:
    source = _web_session_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"还没有可恢复的网页会话：{source}")
    session = GuidedStorySession.load(source, agent=agent or RuleBasedStoryAgent())
    plan = session.storyboard
    palette_updates = [
        _restored_element_update(session, kind)
        for kind in ("character", "conflict", "turning_point", "ending")
    ]
    final_video = (
        session.render_manifest.final_video_path
        if session.render_manifest
        and session.render_manifest.final_video_path
        and Path(session.render_manifest.final_video_path).is_file()
        else None
    )
    target_seconds = (
        session.brief.target_seconds
        or session.brief.resolved_target_seconds
        or session.effective_target_seconds
    )
    return (
        session,
        session.direction,
        session.brief.duration_mode,
        _gr_update(
            value=target_seconds,
            visible=session.brief.duration_mode == "custom",
        ),
        _card_grid_update(session),
        _selection_text(session),
        list(session.chat_history),
        *palette_updates,
        _story_markdown(session.story) if session.story else "",
        _ai_fill_markdown(session.story) if session.story else "",
        _script_markdown(session.script, session) if session.script else "",
        _storyboard_markdown(plan),
        _shot_choices_update(plan),
        _visual_inputs_markdown(plan),
        _visual_binding_choices_update(plan),
        _visual_reference_choices_update(plan),
        final_video,
        f"已从 {source} 恢复到“{_stage_tab(session.stage)}”阶段。",
        _gr_update(value=False),
        _gr_update(selected=_stage_tab(session.stage)),
        _uncertain_shot_choices_update(plan),
        "",
    )


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
    duration_mode: str = "custom",
) -> tuple[GuidedStorySession | None, dict[str, Any], str, list[dict[str, str]], str]:
    try:
        mode = str(duration_mode).strip().lower()
        custom_seconds = int(target_seconds) if mode == "custom" else None
        session = GuidedStorySession(
            brief=CreativeBrief(
                target_seconds=custom_seconds,
                duration_mode=mode,
            ),
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
        return (
            session,
            _card_grid_update(session),
            _selection_text(session),
            _text_status(session, "已经换成8个新方向。"),
        )
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
            _text_status(session, "保留核心吸引力，生成了8个新变体。"),
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
            _text_status(session, "已生成8种融合方式，冲突由文本模型在方案中处理。"),
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
        return (
            session,
            _card_grid_update(session),
            chat,
            "",
            _text_status(session, "已按你的补充换出8个新方向。"),
        )
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
        return session, *values, _text_status(session, "故事零件已展开；不选也能直接生成完整故事。")
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
        return session, _selection_text(session), "故事零件已保留；随时可以生成完整故事。"
    except Exception as exc:
        return session, _selection_text(session), str(exc)


def generate_story_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str, str, str]:
    if session is None:
        return session, "", "", "请先给出一句方向。"
    try:
        story = session.generate_story()
        return (
            session,
            _story_markdown(story),
            _ai_fill_markdown(story),
            _text_status(session, "完整故事已生成；确认后才能改编剧本。"),
        )
    except Exception as exc:
        story = session.story if session else None
        return (
            session,
            _story_markdown(story) if story else "",
            _ai_fill_markdown(story) if story else "",
            str(exc),
        )


def revise_story_view(
    session: GuidedStorySession | None, feedback: str
) -> tuple[GuidedStorySession | None, str, str, str, str]:
    if session is None:
        return session, "", "", "", "请先生成故事。"
    try:
        story = session.revise_story(feedback)
        return (
            session,
            _story_markdown(story),
            _ai_fill_markdown(story),
            "",
            _text_status(session, f"已生成故事第{story.version}版。"),
        )
    except Exception as exc:
        current = session.story if session else None
        return (
            session,
            _story_markdown(current) if current else "",
            _ai_fill_markdown(current) if current else "",
            feedback,
            str(exc),
        )


def generate_script_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str, str]:
    if session is None:
        return session, "", "请先生成故事。"
    try:
        session.confirm_story()
        script = session.generate_script()
        return (
            session,
            _script_markdown(script, session),
            _text_status(
                session,
                f"故事已确认，剧本已按{script.target_seconds}秒生成。",
            ),
        )
    except Exception as exc:
        current = session.script if session else None
        return session, _script_markdown(current, session) if current else "", str(exc)


def revise_script_view(
    session: GuidedStorySession | None, feedback: str
) -> tuple[GuidedStorySession | None, str, str, str]:
    if session is None:
        return session, "", feedback, "请先生成剧本。"
    try:
        script = session.revise_script(feedback)
        return (
            session,
            _script_markdown(script, session),
            "",
            _text_status(session, "剧本已按反馈改写。"),
        )
    except Exception as exc:
        current = session.script if session else None
        return session, _script_markdown(current, session) if current else "", feedback, str(exc)


def back_to_ideas_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, dict[str, Any], str]:
    if session is None:
        return session, _card_grid_update(None), "请先开始创作。"
    session.back_to_ideation()
    return session, _card_grid_update(session), "已回到灵感区，现有故事和版本没有被覆盖。"


def build_storyboard_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str, Any, str]:
    if session is None:
        return session, "", None, "请先生成剧本。"
    try:
        session.confirm_script()
        job = session.build_video_job()
        return (
            session,
            _video_job_markdown(job),
            _shot_choices_update(None),
            "完整视频任务已生成；Provider 将自行决定是否需要内部拆分。",
        )
    except Exception as exc:
        current = session.storyboard if session else None
        return (
            session,
            _storyboard_markdown(current),
            _shot_choices_update(current),
            str(exc),
        )


def add_visual_reference_view(
    session: GuidedStorySession | None,
    uploaded_image: Any,
    usage: str,
    binding: str | None,
    content_summary: str = "",
) -> tuple[
    GuidedStorySession | None,
    str,
    str,
    Any,
    Any,
    Any,
    str,
    str,
    Any,
]:
    if session is None or session.storyboard is None:
        return (
            session,
            "",
            "",
            _visual_binding_choices_update(None),
            _visual_reference_choices_update(None),
            uploaded_image,
            content_summary,
            "当前直连模式不需要分镜参考图。",
            _gr_update(value=False),
        )
    try:
        source = _uploaded_image_path(uploaded_image)
        if not source:
            raise ValueError("请先上传一张参考图。")
        binding_kind, binding_id = _parse_visual_binding(binding)
        if binding_kind == "asset" and usage == "start_frame":
            raise ValueError("start_frame 只能绑定到具体镜头，不能绑定到通用资产。")
        persisted = _persist_visual_upload(source)
        reference = session.add_visual_reference(
            path=persisted,
            usage=usage,
            binding_kind=binding_kind,
            binding_id=binding_id,
            content_summary=content_summary,
        )
        return (
            session,
            _storyboard_markdown(session.storyboard),
            _visual_inputs_markdown(session.storyboard),
            _visual_binding_choices_update(session.storyboard),
            _visual_reference_choices_update(session.storyboard),
            None,
            "",
            (
                f"参考图 {reference.reference_id} 已绑定但尚未冻结。"
                "请点击“确认这些参考图”，然后重新确认整套分镜。"
            ),
            _gr_update(value=False),
        )
    except Exception as exc:
        return (
            session,
            _storyboard_markdown(session.storyboard),
            _visual_inputs_markdown(session.storyboard),
            _visual_binding_choices_update(session.storyboard),
            _visual_reference_choices_update(session.storyboard),
            uploaded_image,
            content_summary,
            str(exc),
            _gr_update(value=False),
        )


def remove_visual_reference_view(
    session: GuidedStorySession | None,
    reference_id: str | None,
) -> tuple[
    GuidedStorySession | None,
    str,
    str,
    Any,
    Any,
    str,
    Any,
]:
    if session is None or session.storyboard is None:
        return (
            session,
            "",
            "",
            _visual_binding_choices_update(None),
            _visual_reference_choices_update(None),
            "请先生成分镜。",
            _gr_update(value=False),
        )
    try:
        session.remove_visual_reference(str(reference_id or ""))
        return (
            session,
            _storyboard_markdown(session.storyboard),
            _visual_inputs_markdown(session.storyboard),
            _visual_binding_choices_update(session.storyboard),
            _visual_reference_choices_update(session.storyboard),
            "参考图绑定已删除；原分镜确认已失效，请重新确认。",
            _gr_update(value=False),
        )
    except Exception as exc:
        return (
            session,
            _storyboard_markdown(session.storyboard),
            _visual_inputs_markdown(session.storyboard),
            _visual_binding_choices_update(session.storyboard),
            _visual_reference_choices_update(session.storyboard),
            str(exc),
            _gr_update(value=False),
        )


def confirm_visual_inputs_view(
    session: GuidedStorySession | None,
) -> tuple[
    GuidedStorySession | None,
    str,
    str,
    Any,
    Any,
    str,
    Any,
]:
    if session is None or session.storyboard is None:
        return (
            session,
            "",
            "",
            _visual_binding_choices_update(None),
            _visual_reference_choices_update(None),
            "请先生成分镜。",
            _gr_update(value=False),
        )
    try:
        diagnostics = session.confirm_visual_inputs()
        warning = f"；另有 {len(diagnostics)} 条未绑定资产提示" if diagnostics else ""
        return (
            session,
            _storyboard_markdown(session.storyboard),
            _visual_inputs_markdown(session.storyboard),
            _visual_binding_choices_update(session.storyboard),
            _visual_reference_choices_update(session.storyboard),
            f"参考图内容与用途已冻结{warning}。请重新确认整套分镜。",
            _gr_update(value=False),
        )
    except Exception as exc:
        return (
            session,
            _storyboard_markdown(session.storyboard),
            _visual_inputs_markdown(session.storyboard),
            _visual_binding_choices_update(session.storyboard),
            _visual_reference_choices_update(session.storyboard),
            str(exc),
            _gr_update(value=False),
        )


def retake_shot_view(
    session: GuidedStorySession | None, shot_id: str | None, feedback: str
) -> tuple[GuidedStorySession | None, str, str, str]:
    if session is None or not shot_id:
        return session, "", feedback, "请先选择镜头。"
    try:
        shot = next(item for item in session.storyboard.shots if item.shot_id == int(shot_id))
        pending = next(
            (
                item
                for item in reversed(session.storyboard.artifacts)
                if item.shot_id == shot.shot_id and item.status == "pending" and item.request_id
            ),
            None,
        )
        if pending is not None:
            raise RuntimeError(
                f"镜头 {shot.shot_id} 的远端任务仍在处理中"
                f"（任务 ID：{pending.request_id}），暂不能 Retake，避免重复付费。"
            )
        requirement = feedback.strip()
        if not requirement:
            raise ValueError("请先写一句 Retake 要求。")
        session.update_storyboard_shot(int(shot_id), _retake_patch(shot, requirement))
        return session, _storyboard_markdown(session.storyboard), "", "该镜头已生成新版本。"
    except Exception as exc:
        return (
            session,
            _storyboard_markdown(session.storyboard) if session.storyboard else "",
            feedback,
            str(exc),
        )


def _retake_patch(shot, requirement: str) -> dict[str, str]:
    camera = shot.camera
    composition = shot.composition
    camera_label = ""
    for markers, value, label in (
        (("大特写", "极近特写"), "extreme close-up", "极近特写"),
        (("特写",), "close-up", "特写"),
        (("近景",), "medium close-up", "近景"),
        (("中景",), "medium", "中景"),
        (("全景", "远景", "广角"), "wide", "全景"),
        (("俯拍",), "high-angle", "俯拍"),
        (("仰拍",), "low-angle", "仰拍"),
    ):
        if any(marker in requirement for marker in markers):
            camera = value
            camera_label = label
            composition = f"{label}构图，主体与环境关系清晰"
            break

    camera_movement = shot.camera_movement
    for markers, value in (
        (("快速推进", "快速推近"), "fast dolly in"),
        (("缓慢推进", "慢慢推进", "缓慢推近"), "slow dolly in"),
        (("推进", "推镜头", "推近"), "dolly in"),
        (("快速拉远", "快速后拉"), "fast dolly out"),
        (("拉远", "后拉"), "dolly out"),
        (("横摇", "摇镜"), "pan"),
        (("跟拍", "跟随"), "tracking"),
        (("环绕",), "orbit"),
        (("手持",), "handheld"),
        (("静止", "固定镜头"), "static"),
    ):
        if any(marker in requirement for marker in markers):
            camera_movement = value
            break

    first_frame = shot.start_frame
    end_frame = shot.end_frame
    if camera_label:
        first_frame = f"{camera_label}。{first_frame}"
        end_frame = f"{camera_label}。{end_frame}"
    if "首帧" in requirement or "开场画面" in requirement:
        first_frame = f"Retake 首帧要求：{requirement}。原始连续性：{shot.start_frame}"
    if any(marker in requirement for marker in ("结束帧", "尾帧", "末帧", "结尾画面")):
        end_frame = f"Retake 结束帧要求：{requirement}。原始连续性：{shot.end_frame}"

    return {
        "camera": camera,
        "composition": composition,
        "camera_movement": camera_movement,
        "start_frame": first_frame,
        "end_frame": end_frame,
        "retake_instruction": requirement,
    }


def confirm_storyboard_view(
    session: GuidedStorySession | None,
) -> tuple[GuidedStorySession | None, str]:
    if session is None:
        return session, "请先生成分镜。"
    try:
        if session.video_job is not None:
            return session, "完整视频任务已确认。真实视频仍需勾选费用确认。"
        session.confirm_storyboard()
        return session, "旧版分镜已确认。真实视频仍需勾选费用确认。"
    except Exception as exc:
        return session, str(exc)


def resolve_submission_uncertainty_view(
    session: GuidedStorySession | None,
    shot_id: str | int | None,
    provider_request_id: str,
    *,
    accepted_by_provider: bool,
) -> tuple[GuidedStorySession | None, Any, str, str]:
    if session is None or shot_id in (None, ""):
        return session, _uncertain_shot_choices_update(None), provider_request_id, (
            "请先选择一条提交结果不确定的镜头记录。"
        )
    try:
        if session.video_job is not None:
            artifact = session.resolve_video_submission_uncertainty(
                accepted_by_provider=accepted_by_provider,
                provider_request_id=provider_request_id,
            )
            message = (
                "完整视频任务已登记 Provider 真实任务 ID；再次生成时只会继续查询。"
                if accepted_by_provider
                else "完整视频任务已确认为 Provider 未受理；再次生成时允许重新提交。"
            )
            return session, _uncertain_shot_choices_update(session.render_manifest), "", message
        artifact = session.resolve_submission_uncertainty(
            int(shot_id),
            accepted_by_provider=accepted_by_provider,
            provider_request_id=provider_request_id,
        )
        if accepted_by_provider:
            message = (
                f"镜头 {artifact.shot_id} 已登记 Provider 任务 ID "
                f"{artifact.request_id}；再次生成时只会继续查询。"
            )
        else:
            message = (
                f"镜头 {artifact.shot_id} 已确认为 Provider 未受理；"
                "再次生成时允许重新提交。"
            )
        return session, _uncertain_shot_choices_update(session.storyboard), "", message
    except Exception as exc:
        return (
            session,
            _uncertain_shot_choices_update(
                session.render_manifest if session.video_job is not None else session.storyboard
            ),
            provider_request_id,
            str(exc),
        )


def render_video_with_progress(
    session: GuidedStorySession | None,
    cost_confirmed: bool = False,
    *,
    provider=None,
    output_dir: str | Path | None = None,
) -> Iterator[tuple[GuidedStorySession | None, str | None, str, Any]]:
    reset_confirmation = _gr_update(value=False)
    previous_video = (
        session.render_manifest.final_video_path
        if session and session.render_manifest and session.render_manifest.final_video_path
        else None
    )
    if session is not None and bool(getattr(session, "render_in_progress", False)):
        yield (
            session,
            previous_video,
            "当前会话已有视频任务在运行，请等待完成后再试。",
            reset_confirmation,
        )
        return
    if session is None or session.stage != Stage.RENDER_READY:
        yield (
            session,
            previous_video,
            "必须先确认完整视频任务，才能生成真实视频。",
            reset_confirmation,
        )
        return
    if not cost_confirmed:
        yield (
            session,
            previous_video,
            "请先勾选费用确认；当前没有调用视频API。",
            reset_confirmation,
        )
        return
    queue: Queue[tuple[float, str]] = Queue()
    result: dict[str, object] = {}

    def progress(stage: str, fraction: float, message: str) -> None:
        queue.put((fraction, message))

    def work() -> None:
        try:
            base = (
                Path(output_dir or os.getenv("VIDEO_OUTPUT_DIR", "outputs/videos"))
                .expanduser()
                .resolve()
            )
            target = base / f"render_{uuid4().hex}"
            target.mkdir(parents=True, exist_ok=False)
            result["output_dir"] = target
            session.save(target / "session_before_render.json")
            try:
                if provider is not None and not hasattr(provider, "generate_video"):
                    # Explicit shot providers remain supported for the legacy
                    # Storyboard compatibility path and test doubles.
                    renderer = StoryRenderer(provider, progress_callback=progress)
                    result["manifest"] = session.render_confirmed_plan(renderer, target)
                else:
                    active_provider = provider or AgnesVideoProvider.from_env()
                    renderer = VideoJobRenderer(active_provider, progress_callback=progress)
                    result["manifest"] = session.render_confirmed_video(renderer, target)
            finally:
                session.save(target / "session.json")
        except Exception as exc:
            result["error"] = exc

    worker = Thread(target=work, daemon=True)
    worker.start()
    fraction, message = 0.0, "正在准备旁白与视频任务"
    yield session, previous_video, _progress_text(fraction, message), reset_confirmation
    while worker.is_alive():
        try:
            fraction, message = queue.get(timeout=0.5)
        except Empty:
            pass
        yield session, previous_video, _progress_text(fraction, message), reset_confirmation
    worker.join()
    while not queue.empty():
        fraction, message = queue.get_nowait()
    if "error" in result:
        yield session, previous_video, f"生成失败：{result['error']}", reset_confirmation
        return
    manifest = result.get("manifest")
    if manifest is None:
        yield session, previous_video, "生成失败：没有生成 manifest", reset_confirmation
        return
    if manifest.status == "pending":
        yield (
            session,
            previous_video,
            f"视频任务仍在处理中：{manifest.error}",
            reset_confirmation,
        )
        return
    if manifest.status == "submission_uncertain":
        yield (
            session,
            previous_video,
            f"视频提交结果无法确认：{manifest.error}",
            reset_confirmation,
        )
        return
    if manifest.status not in {"succeeded", "succeeded_with_warnings"}:
        yield (
            session,
            previous_video,
            f"生成失败：{manifest.error}｜{_render_evidence_summary(manifest)}",
            reset_confirmation,
        )
        return
    if manifest.status == "succeeded_with_warnings":
        message = f"成片已保存，但存在警告：{manifest.error}"
    else:
        message = "成片已保存到本地"
    message = f"{message}｜{_render_evidence_summary(manifest)}"
    yield (
        session,
        manifest.final_video_path,
        _progress_text(1.0, message),
        reset_confirmation,
    )


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

    def start(direction, duration_mode, seconds):
        return start_garden_view(
            direction,
            seconds,
            agent=(agent_factory() if agent_factory else RuleBasedStoryAgent()),
            duration_mode=duration_mode,
        )

    start.__name__ = "start_garden_view"

    def restore():
        return restore_saved_session_view(
            agent=(agent_factory() if agent_factory else RuleBasedStoryAgent()),
        )

    restore.__name__ = "restore_saved_session_view"

    def toggle_custom_duration(duration_mode):
        return _gr_update(visible=duration_mode == "custom")

    def story_and_open(session):
        return (*generate_story_view(session), _gr_update(selected="story"))

    story_and_open.__name__ = "generate_story_view"

    def script_and_open(session):
        return (*generate_script_view(session), _gr_update(selected="script"))

    script_and_open.__name__ = "generate_script_view"

    def storyboard_and_open(session):
        result = build_storyboard_view(session)
        plan = result[0].storyboard if result[0] is not None else None
        return (
            *result,
            _visual_inputs_markdown(plan),
            _visual_binding_choices_update(plan),
            _visual_reference_choices_update(plan),
            _gr_update(value=False),
            _gr_update(selected="storyboard"),
        )

    storyboard_and_open.__name__ = "build_storyboard_view"

    def confirm_storyboard_and_open(session):
        active, message = confirm_storyboard_view(session)
        plan = active.storyboard if active is not None else None
        job = active.video_job if active is not None else None
        return (
            active,
            _storyboard_markdown(plan) if plan is not None else _video_job_markdown(job),
            _visual_inputs_markdown(plan),
            _visual_reference_choices_update(plan),
            message,
            _gr_update(value=False),
            _gr_update(selected="video"),
        )

    confirm_storyboard_and_open.__name__ = "confirm_storyboard_view"

    def retake_and_reset(session, shot_id, feedback):
        result = retake_shot_view(session, shot_id, feedback)
        plan = result[0].storyboard if result[0] is not None else None
        return (
            *result,
            _visual_inputs_markdown(plan),
            _visual_binding_choices_update(plan),
            _visual_reference_choices_update(plan),
            _gr_update(value=False),
        )

    retake_and_reset.__name__ = "retake_shot_view"

    def back_and_open(session):
        return (*back_to_ideas_view(session), _gr_update(selected="ideas"))

    back_and_open.__name__ = "back_to_ideas_view"

    def accept_uncertain(session, shot_id, request_id):
        return resolve_submission_uncertainty_view(
            session,
            shot_id,
            request_id,
            accepted_by_provider=True,
        )

    def reject_uncertain(session, shot_id, request_id):
        return resolve_submission_uncertainty_view(
            session,
            shot_id,
            request_id,
            accepted_by_provider=False,
        )

    def uncertain_choices(session):
        return _uncertain_shot_choices_update(
            (
                session.render_manifest
                if session is not None and session.video_job is not None
                else session.storyboard
                if session is not None
                else None
            )
        )

    start_saved = _autosaving_view(start)
    select_cards_saved = _autosaving_view(select_cards_view)
    refresh_saved = _autosaving_view(refresh_ideas_view)
    more_like_saved = _autosaving_view(more_like_view)
    mix_saved = _autosaving_view(mix_selected_view)
    auto_saved = _autosaving_view(auto_choose_view)
    chat_saved = _autosaving_view(chat_ideation_view)
    expand_saved = _autosaving_view(expand_elements_view)
    choose_elements_saved = _autosaving_view(choose_elements_view)
    story_saved = _autosaving_view(story_and_open)
    revise_story_saved = _autosaving_view(revise_story_view)
    back_saved = _autosaving_view(back_and_open)
    script_saved = _autosaving_view(script_and_open)
    revise_script_saved = _autosaving_view(revise_script_view)
    storyboard_saved = _autosaving_view(storyboard_and_open)
    retake_saved = _autosaving_view(retake_and_reset)
    add_visual_saved = _autosaving_view(add_visual_reference_view)
    remove_visual_saved = _autosaving_view(remove_visual_reference_view)
    confirm_visual_saved = _autosaving_view(confirm_visual_inputs_view)
    confirm_storyboard_saved = _autosaving_view(confirm_storyboard_and_open)
    accept_uncertain_saved = _autosaving_view(accept_uncertain)
    reject_uncertain_saved = _autosaving_view(reject_uncertain)
    render_saved = _autosaving_progress(render_video_with_progress)

    studio_css = """
:root{
  --studio-ink:#27221f;
  --studio-muted:#746c65;
  --studio-paper:#fffdf8;
  --studio-canvas:#f3efe8;
  --studio-line:#ded7cd;
  --studio-accent:#d86435;
  --studio-accent-dark:#b94c25;
  --studio-night:#111315;
  --studio-night-panel:#1a1d20;
  --studio-night-line:#30353a;
  --studio-cyan:#69d8d0;
}
body,.gradio-container{
  background:
    radial-gradient(circle at 10% -10%,rgba(216,100,53,.10),transparent 30rem),
    var(--studio-canvas)!important;
  color:var(--studio-ink);
}
.gradio-container{max-width:none!important;padding:0!important}
footer{display:none!important}
#studio-shell{max-width:1480px;margin:0 auto;padding:24px 28px 52px}
#studio-brand{
  margin-bottom:18px;padding:6px 4px;
  display:flex;align-items:flex-end;justify-content:space-between;gap:24px;
}
#studio-brand .brand-lockup{display:flex;align-items:center;gap:14px}
#studio-brand .brand-mark{
  width:43px;height:43px;border-radius:13px;display:grid;place-items:center;
  background:var(--studio-ink);color:#fff;font-size:18px;font-weight:800;
  box-shadow:0 10px 24px rgba(39,34,31,.16);
}
#studio-brand .brand-name{
  font-size:12px;line-height:1.1;letter-spacing:.19em;text-transform:uppercase;
  color:var(--studio-muted);font-weight:750;
}
#studio-brand h1{font-size:25px;line-height:1.15;margin:3px 0 0;letter-spacing:-.03em}
#studio-brand .brand-note{max-width:440px;text-align:right;color:var(--studio-muted);font-size:13px}
#workflow-tabs{
  overflow:hidden;border:1px solid var(--studio-line);border-radius:22px;
  background:var(--studio-paper);box-shadow:0 20px 60px rgba(71,59,49,.09);
}
#workflow-tabs>.tab-wrapper{
  padding:8px 12px!important;gap:7px!important;border-bottom:1px solid var(--studio-line);
  background:rgba(255,253,248,.96);position:sticky;top:0;z-index:20;
}
#workflow-tabs>.tab-wrapper [role="tablist"]{
  display:grid!important;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px!important;
  width:100%!important;
}
#workflow-tabs>.tab-wrapper [role="tab"]{
  width:100%!important;justify-content:center!important;min-height:50px!important;
  border:0!important;border-radius:13px!important;color:#8a8179!important;
  font-size:13px!important;font-weight:720!important;letter-spacing:.02em;
  transition:all .18s ease!important;
}
#workflow-tabs>.tab-wrapper [role="tab"]:hover{
  background:#f4eee5!important;color:var(--studio-ink)!important
}
#workflow-tabs>.tab-wrapper [role="tab"].selected{
  background:var(--studio-ink)!important;color:#fff!important;
  box-shadow:0 7px 18px rgba(39,34,31,.14)!important;
}
.stage-surface{padding:28px 30px 34px!important;background:var(--studio-paper)}
.stage-kicker{
  color:var(--studio-accent);font-weight:780;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;margin-bottom:7px;
}
.stage-heading{font-size:27px;font-weight:760;letter-spacing:-.035em;line-height:1.2;margin:0 0 7px}
.stage-copy{color:var(--studio-muted);font-size:14px;line-height:1.7;margin:0 0 22px;max-width:760px}
.idea-brief,.paper-panel,.side-panel,.control-card{
  border:1px solid var(--studio-line)!important;border-radius:18px!important;
  background:#fff!important;box-shadow:0 10px 30px rgba(63,52,44,.055)!important;
}
.idea-brief{padding:18px!important;margin-bottom:18px!important}
.paper-panel{padding:24px 26px!important;min-height:590px}
.side-panel{padding:20px!important;align-self:flex-start;position:sticky;top:88px}
.control-card{padding:18px!important}
.panel-label{
  margin:0 0 12px;color:var(--studio-muted);font-size:11px;font-weight:780;
  letter-spacing:.13em;text-transform:uppercase;
}
.idea-card-grid{border:0!important;background:transparent!important;padding:0!important}
.idea-card-grid .wrap{
  display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;
}
.idea-card-grid label{
  position:relative;align-items:flex-start!important;min-height:178px;padding:17px!important;
  border:1px solid var(--studio-line)!important;border-radius:16px!important;
  background:#fff!important;white-space:pre-line;line-height:1.55!important;
  box-shadow:0 6px 18px rgba(63,52,44,.045);transition:.18s ease;
}
.idea-card-grid label:hover{
  transform:translateY(-3px);border-color:#c7b7aa!important;
  box-shadow:0 12px 25px rgba(63,52,44,.09);
}
.idea-card-grid label:has(input:checked){
  border:2px solid var(--studio-accent)!important;padding:16px!important;
  background:#fff8f3!important;box-shadow:0 0 0 4px rgba(216,100,53,.11);
}
.idea-card-grid input{accent-color:var(--studio-accent)}
#selection-note{
  margin:14px 0 5px;padding:12px 15px;border-radius:12px;background:#f6f2ec;
  color:#625a54;font-size:13px;
}
.action-row{gap:9px!important;margin-top:13px}
.action-row button{min-height:42px!important}
.primary-action{
  background:var(--studio-accent)!important;border-color:var(--studio-accent)!important;
  color:#fff!important;font-weight:760!important;
  box-shadow:0 8px 18px rgba(216,100,53,.20)!important;
}
.primary-action:hover{
  background:var(--studio-accent-dark)!important;border-color:var(--studio-accent-dark)!important;
  transform:translateY(-1px);
}
.quiet-action{background:#f5f0e9!important;border-color:#e3dbd1!important;color:#544c46!important}
.story-copy h3,.script-copy h3{font-size:28px!important;letter-spacing:-.03em!important}
.story-copy p{font-family:Georgia,"Noto Serif SC","Songti SC",serif;font-size:16px;line-height:2}
.script-copy h4{margin-top:25px!important;padding-top:20px;border-top:1px solid #e5ded4}
.script-copy p{line-height:1.85}
#ai-fill{padding:10px 13px;border-radius:11px;background:#f3efe8;color:#6f665f;font-size:12px}
.empty-state{color:#9a9189;font-style:italic}
#studio-status{
  margin:0!important;padding:12px 17px!important;border:0!important;border-radius:0!important;
  background:#eee8df!important;color:#5c554f!important;font-size:13px!important;
}
#studio-status p{margin:0!important}
.assistant-note{
  padding:15px;border-radius:14px;background:#f5f0e9;color:#645c55;
  font-size:13px;line-height:1.7;margin-bottom:16px;
}
.assistant-note strong{color:var(--studio-ink)}
.compact-accordion{border-color:var(--studio-line)!important;border-radius:14px!important}
.shot-strip{
  padding:13px 15px;border:1px solid var(--studio-line);border-radius:14px;
  color:#756c65;background:#f6f2ec;font-size:12px;letter-spacing:.04em;
}
.storyboard-paper{min-height:620px}
.storyboard-copy{color:var(--studio-ink)}
.storyboard-copy h3{font-size:24px!important;letter-spacing:-.02em!important}
.storyboard-copy strong{color:var(--studio-accent-dark)!important}
.storyboard-copy p{line-height:1.78}
.storyboard-copy hr{border-color:var(--studio-line)}
.storyboard-controls .wrap{gap:12px!important}
.video-paper{min-height:520px}
.video-paper .panel-label{margin-bottom:14px}
#final-video{
  overflow:hidden;border:1px solid var(--studio-line)!important;border-radius:16px!important;
  background:#f3eee7!important;
}
#final-video video{border-radius:15px;background:#171513}
.video-ready-note{
  margin-top:14px;padding:13px 15px;border-radius:12px;background:#f6f2ec;
  color:#756c65;font-size:13px;line-height:1.65;
}
.video-controls .wrap{gap:14px!important}
.video-controls .form{border:0!important}
@media(max-width:1100px){
  .idea-card-grid .wrap{grid-template-columns:repeat(2,minmax(0,1fr))}
  .side-panel{position:static}
}
@media(max-width:720px){
  #studio-shell{padding:12px}
  #studio-brand{align-items:flex-start}
  #studio-brand .brand-note{display:none}
  #workflow-tabs>.tab-wrapper{overflow-x:auto}
  #workflow-tabs>.tab-wrapper [role="tablist"]{
    display:flex!important;justify-content:flex-start!important;width:max-content!important;
  }
  #workflow-tabs>.tab-wrapper [role="tab"]{min-width:105px!important}
  .stage-surface{padding:22px 16px 28px!important}
  .idea-card-grid .wrap{grid-template-columns:1fr}
  .paper-panel{padding:18px;min-height:420px}
}
"""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.gray,
        radius_size=gr.themes.sizes.radius_lg,
        font=["Inter", "Noto Sans SC", "Microsoft YaHei", "ui-sans-serif", "system-ui"],
    )
    with gr.Blocks(title="MOTIF｜一句话影像工坊", css=studio_css, theme=theme) as app:
        session_state = gr.State()
        with gr.Column(elem_id="studio-shell"):
            gr.HTML(
                """
                <header id="studio-brand">
                  <div class="brand-lockup">
                    <div class="brand-mark">M</div>
                    <div>
                      <div class="brand-name">Motif Story Studio</div>
                      <h1>一句话影像工坊</h1>
                    </div>
                  </div>
                  <div class="brand-note">从一个念头，到一篇完整故事，再到可以拍出来的镜头。</div>
                </header>
                """
            )
            with gr.Tabs(selected="ideas", elem_id="workflow-tabs") as workflow:
                with gr.Tab("01  灵感", id="ideas", elem_classes=["stage-surface"]):
                    gr.HTML(
                        """
                        <div class="stage-kicker">Discover the story</div>
                        <div class="stage-heading">先找到真正想讲的故事</div>
                        <p class="stage-copy">写下一句模糊的方向即可。挑选喜欢的创意，
                        或者什么都不选，让系统先写出一版完整故事。</p>
                        """
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7):
                            with gr.Column(elem_classes=["idea-brief"]):
                                direction = gr.Textbox(
                                    label="你的故事从哪里开始？",
                                    placeholder="例如：校园里发生一件带点悬疑的事",
                                    lines=2,
                                )
                                with gr.Row():
                                    duration_mode = gr.Radio(
                                        choices=[
                                            ("自动估算（推荐）", "auto"),
                                            ("自定义", "custom"),
                                        ],
                                        value="auto",
                                        label="成片时长",
                                        scale=3,
                                    )
                                    target_seconds = gr.Number(
                                        value=90,
                                        minimum=15,
                                        maximum=300,
                                        step=5,
                                        precision=0,
                                        label="自定义秒数",
                                        visible=False,
                                        scale=2,
                                    )
                                    begin = gr.Button(
                                        "生成 8 个灵感方向",
                                        variant="primary",
                                        scale=2,
                                        elem_classes=["primary-action"],
                                    )
                                restore_session = gr.Button(
                                    "恢复上次自动保存的会话",
                                    size="sm",
                                    elem_classes=["quiet-action"],
                                )
                            gr.HTML('<div class="panel-label">灵感画廊 · 最多保留 3 个</div>')
                            card_grid = IdeaCardGrid(show_label=False)
                            selection = gr.Markdown(
                                "**已保留：** 暂无。最多选择3张。",
                                elem_id="selection-note",
                            )
                            with gr.Row(elem_classes=["action-row"]):
                                refresh = gr.Button(
                                    "换一批", size="sm", elem_classes=["quiet-action"]
                                )
                                more_like = gr.Button(
                                    "更多像这个", size="sm", elem_classes=["quiet-action"]
                                )
                                mix = gr.Button(
                                    "混合已选", size="sm", elem_classes=["quiet-action"]
                                )
                                auto = gr.Button(
                                    "替我选择", size="sm", elem_classes=["quiet-action"]
                                )
                                expand = gr.Button(
                                    "展开故事素材", size="sm", elem_classes=["quiet-action"]
                                )
                                make_story = gr.Button(
                                    "生成完整故事  →",
                                    variant="primary",
                                    elem_classes=["primary-action"],
                                )
                        with gr.Column(scale=3, elem_classes=["side-panel"]):
                            gr.HTML(
                                """
                                <div class="panel-label">创作助手</div>
                                <div class="assistant-note"><strong>你不需要先写大纲。</strong><br>
                                可以挑选灵感，也可以直接生成故事。人物、冲突和结局都只是可选素材。</div>
                                """
                            )
                            chat = gr.Chatbot(
                                type="messages",
                                height=290,
                                allow_tags=False,
                                show_label=False,
                                placeholder="这里可以继续补充你的偏好",
                            )
                            chat_input = gr.Textbox(
                                placeholder="比如：更幽默，不要爱情线",
                                label="补充一句",
                                lines=2,
                            )
                            chat_send = gr.Button("按这句话重新构思", size="sm")
                            with gr.Accordion(
                                "可选：人物与情节素材",
                                open=False,
                                elem_classes=["compact-accordion"],
                            ):
                                gr.Markdown("不选择的部分会由 AI 自然补全。")
                                character = gr.Radio(label="主角")
                                conflict = gr.Radio(label="冲突")
                                turning = gr.Radio(label="转折")
                                ending = gr.Radio(label="结局")
                                keep_elements = gr.Button("保留这些素材", size="sm")

                with gr.Tab("02  故事", id="story", elem_classes=["stage-surface"]):
                    gr.HTML(
                        """
                        <div class="stage-kicker">Write the whole story</div>
                        <div class="stage-heading">把故事读完，再决定是否拍它</div>
                        <p class="stage-copy">这里是一篇完整故事，而不是固定节点大纲。
                        你可以反复修改，直到人物、冲突和结局都成立。</p>
                        """
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, elem_classes=["paper-panel"]):
                            ai_fill = gr.Markdown("", elem_id="ai-fill")
                            story = gr.Markdown(
                                "*生成后，完整故事会出现在这里。*",
                                elem_classes=["story-copy", "empty-state"],
                            )
                        with gr.Column(scale=3, elem_classes=["side-panel"]):
                            gr.HTML(
                                """
                                <div class="panel-label">导演批注</div>
                                <div class="assistant-note">先关注“这个故事是否值得讲”，
                                暂时不用考虑镜头数量和生成次数。</div>
                                """
                            )
                            story_feedback = gr.Textbox(
                                label="你想怎样修改？",
                                placeholder="例如：让人物关系更复杂，结局更克制",
                                lines=5,
                            )
                            rewrite_story = gr.Button(
                                "改写这一版故事", elem_classes=["quiet-action"]
                            )
                            make_script = gr.Button(
                                "确认故事并生成剧本  →",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )
                            back = gr.Button("返回灵感区", size="sm")

                with gr.Tab("03  剧本", id="script", elem_classes=["stage-surface"]):
                    gr.HTML(
                        """
                        <div class="stage-kicker">Adapt for the screen</div>
                        <div class="stage-heading">让文字变成可以被看见的行动</div>
                        <p class="stage-copy">场景数量由故事内容与目标时长自然决定。
                        剧本只负责把完整故事改造成画面、动作、对白和旁白。</p>
                        """
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, elem_classes=["paper-panel"]):
                            script = gr.Markdown(
                                "*确认故事后，改编剧本会出现在这里。*",
                                elem_classes=["script-copy", "empty-state"],
                            )
                        with gr.Column(scale=3, elem_classes=["side-panel"]):
                            gr.HTML(
                                """
                                <div class="panel-label">剧本工作台</div>
                                <div class="assistant-note">检查动作是否可见、对白是否必要，
                                不必人为凑够固定场景数量。</div>
                                """
                            )
                            script_feedback = gr.Textbox(
                                label="剧本修改意见",
                                placeholder="例如：减少旁白，加强可见动作",
                                lines=5,
                            )
                            rewrite_script = gr.Button("改写剧本", elem_classes=["quiet-action"])
                            make_storyboard = gr.Button(
                                "接受剧本并生成完整视频任务  →",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )

                with gr.Tab("04  视频任务", id="storyboard", elem_classes=["stage-surface"]):
                    gr.HTML(
                        """
                        <div class="stage-kicker">Design the shots</div>
                        <div class="stage-heading">把确认后的剧本交给视频 Provider</div>
                        <p class="stage-copy">这是一个完整视频任务。场景是叙事上下文，
                        不会被本地机械切成相同时长的镜头。</p>
                        <div class="shot-strip">VIDEO JOB　·　FULL SCRIPT　·　PROVIDER CONTROLLED</div>
                        """
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, elem_classes=["paper-panel", "storyboard-paper"]):
                            storyboard = gr.Markdown(
                                "*确认剧本后，完整视频任务会出现在这里。*",
                                elem_classes=["storyboard-copy", "empty-state"],
                            )
                            visual_inputs = gr.Markdown(
                                "*当前直连模式不需要先制作分镜。*",
                                elem_classes=["visual-input-summary"],
                            )
                        with gr.Column(scale=3, elem_classes=["side-panel", "storyboard-controls"]):
                            gr.HTML(
                                """
                                <div class="panel-label">完整视频任务</div>
                                <div class="assistant-note"><strong>先检查剧本。</strong><br>
                                Provider 会根据完整剧本和自身能力决定生成策略；
                                本地不预设镜头数量。</div>
                                """
                            )
                            shot_choice = gr.Dropdown(label="选择镜头")
                            retake_feedback = gr.Textbox(
                                label="Retake 要求",
                                placeholder="例如：改成近景，动作更克制",
                                lines=4,
                            )
                            retake = gr.Button("（兼容入口）修改旧版镜头方案", elem_classes=["quiet-action"])
                            with gr.Accordion("参考图管理", open=True):
                                visual_upload = gr.Image(
                                    label="上传参考图",
                                    type="filepath",
                                    sources=["upload"],
                                    height=180,
                                )
                                visual_usage = gr.Dropdown(
                                    label="图片用途",
                                    choices=VISUAL_USAGE_CHOICES,
                                    value="identity_reference",
                                )
                                visual_binding = gr.Dropdown(
                                    label="绑定到镜头或资产",
                                    choices=[],
                                )
                                visual_summary = gr.Textbox(
                                    label="内容说明（可选）",
                                    placeholder="例如：林夏正面定妆照，黑色短发、灰色风衣",
                                    lines=2,
                                )
                                add_visual = gr.Button(
                                    "上传并绑定",
                                    elem_classes=["quiet-action"],
                                )
                                visual_delete = gr.Dropdown(
                                    label="删除已绑定参考图",
                                    choices=[],
                                )
                                remove_visual = gr.Button(
                                    "删除这条绑定",
                                    elem_classes=["quiet-action"],
                                )
                                confirm_visual = gr.Button(
                                    "确认这些参考图",
                                    elem_classes=["quiet-action"],
                                )
                            confirm_storyboard = gr.Button(
                                "确认完整视频任务  →",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )

                with gr.Tab("05  视频", id="video", elem_classes=["stage-surface"]):
                    gr.HTML(
                        """
                        <div class="stage-kicker">Render the film</div>
                        <div class="stage-heading">最后一步：把完整视频任务变成影片</div>
                        <p class="stage-copy">只有确认完整视频任务并主动勾选费用确认后，
                        才会调用真实视频服务。</p>
                        """
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, elem_classes=["paper-panel", "video-paper"]):
                            gr.HTML('<div class="panel-label">最终成片</div>')
                            video = gr.Video(
                                show_label=False,
                                interactive=False,
                                elem_id="final-video",
                            )
                            gr.HTML(
                                """
                                <div class="video-ready-note">
                                成片生成后会在这里直接播放；完整任务的 Provider 状态和恢复记录
                                都会保存在本次项目中。
                                </div>
                                """
                            )
                        with gr.Column(scale=3, elem_classes=["side-panel", "video-controls"]):
                            gr.HTML(
                                """
                                <div class="panel-label">生成设置</div>
                                <div class="assistant-note"><strong>先确认，再生成。</strong><br>
                                视频会作为一个完整 VideoJob 提交；Provider 若需要分段，
                                由 Provider 在适配器内部处理。</div>
                                """
                            )
                            cost_confirmed = gr.Checkbox(label="我确认下一步会调用付费视频 API")
                            render = gr.Button(
                                "生成真实视频",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )
                            with gr.Accordion("处理提交结果不确定的任务", open=False):
                                gr.Markdown(
                                    "只在 Provider 后台核对后操作。"
                                    "确认已受理必须填写后台真实任务 ID；"
                                    "确认未受理后，系统才允许重新提交。"
                                )
                                uncertain_shot = gr.Dropdown(
                                    label="待核对镜头",
                                    choices=[],
                                )
                                provider_request_id = gr.Textbox(
                                    label="Provider 后台任务 ID",
                                    placeholder="确认已受理时必填",
                                )
                                accepted_uncertain = gr.Button(
                                    "后台确认已受理",
                                    elem_classes=["quiet-action"],
                                )
                                rejected_uncertain = gr.Button(
                                    "后台确认未受理",
                                    elem_classes=["quiet-action"],
                                )
            status = gr.Markdown(
                "写下一句话，我们从寻找故事开始。",
                elem_id="studio-status",
            )

        base_outputs = [session_state, card_grid, selection, chat, status]
        restore_outputs = [
            session_state,
            direction,
            duration_mode,
            target_seconds,
            card_grid,
            selection,
            chat,
            character,
            conflict,
            turning,
            ending,
            story,
            ai_fill,
            script,
            storyboard,
            shot_choice,
            visual_inputs,
            visual_binding,
            visual_delete,
            video,
            status,
            cost_confirmed,
            workflow,
            uncertain_shot,
            provider_request_id,
        ]
        duration_mode.change(
            toggle_custom_duration,
            [duration_mode],
            [target_seconds],
        )
        restore_session.click(
            restore,
            outputs=restore_outputs,
            api_name="restore_session",
        )
        begin.click(
            start_saved,
            [direction, duration_mode, target_seconds],
            base_outputs,
            api_name="start_ideation",
        )
        direction.submit(
            start_saved,
            [direction, duration_mode, target_seconds],
            base_outputs,
        )
        card_grid.input(
            select_cards_saved,
            [session_state, card_grid],
            [session_state, card_grid, selection, status],
            api_name="select_ideas",
        )
        refresh.click(
            refresh_saved,
            [session_state],
            [session_state, card_grid, selection, status],
            api_name="refresh_ideas",
        )
        more_like.click(
            more_like_saved,
            [session_state],
            [session_state, card_grid, selection, status],
            api_name="more_like",
        )
        mix.click(
            mix_saved,
            [session_state],
            [session_state, card_grid, selection, status],
            api_name="mix_selected",
        )
        auto.click(
            auto_saved,
            [session_state],
            [session_state, card_grid, selection, status],
            api_name="auto_choose",
        )
        chat_send.click(
            chat_saved,
            [session_state, chat_input, chat],
            [session_state, card_grid, chat, chat_input, status],
            api_name="chat_ideation",
        )
        chat_input.submit(
            chat_saved,
            [session_state, chat_input, chat],
            [session_state, card_grid, chat, chat_input, status],
        )
        expand.click(
            expand_saved,
            [session_state],
            [session_state, character, conflict, turning, ending, status],
            api_name="expand_selected",
        )
        keep_elements.click(
            choose_elements_saved,
            [session_state, character, conflict, turning, ending],
            [session_state, selection, status],
            api_name="choose_elements",
        )
        make_story.click(
            story_saved,
            [session_state],
            [session_state, story, ai_fill, status, workflow],
            api_name="generate_story",
        )
        rewrite_story.click(
            revise_story_saved,
            [session_state, story_feedback],
            [session_state, story, ai_fill, story_feedback, status],
            api_name="revise_story",
        )
        back.click(
            back_saved,
            [session_state],
            [session_state, card_grid, status, workflow],
            api_name="back_to_ideas",
        )
        make_script.click(
            script_saved,
            [session_state],
            [session_state, script, status, workflow],
            api_name="generate_script",
        )
        rewrite_script.click(
            revise_script_saved,
            [session_state, script_feedback],
            [session_state, script, script_feedback, status],
            api_name="revise_script",
        )
        make_storyboard.click(
            storyboard_saved,
            [session_state],
            [
                session_state,
                storyboard,
                shot_choice,
                status,
                visual_inputs,
                visual_binding,
                visual_delete,
                cost_confirmed,
                workflow,
            ],
            api_name="build_storyboard",
        )
        retake.click(
            retake_saved,
            [session_state, shot_choice, retake_feedback],
            [
                session_state,
                storyboard,
                retake_feedback,
                status,
                visual_inputs,
                visual_binding,
                visual_delete,
                cost_confirmed,
            ],
            api_name="retake_shot",
        )
        add_visual.click(
            add_visual_saved,
            [
                session_state,
                visual_upload,
                visual_usage,
                visual_binding,
                visual_summary,
            ],
            [
                session_state,
                storyboard,
                visual_inputs,
                visual_binding,
                visual_delete,
                visual_upload,
                visual_summary,
                status,
                cost_confirmed,
            ],
            api_name="add_visual_reference",
        )
        remove_visual.click(
            remove_visual_saved,
            [session_state, visual_delete],
            [
                session_state,
                storyboard,
                visual_inputs,
                visual_binding,
                visual_delete,
                status,
                cost_confirmed,
            ],
            api_name="remove_visual_reference",
        )
        confirm_visual.click(
            confirm_visual_saved,
            [session_state],
            [
                session_state,
                storyboard,
                visual_inputs,
                visual_binding,
                visual_delete,
                status,
                cost_confirmed,
            ],
            api_name="confirm_visual_inputs",
        )
        confirm_storyboard.click(
            confirm_storyboard_saved,
            [session_state],
            [
                session_state,
                storyboard,
                visual_inputs,
                visual_delete,
                status,
                cost_confirmed,
                workflow,
            ],
            api_name="confirm_storyboard",
        )
        render_event = render.click(
            render_saved,
            [session_state, cost_confirmed],
            [session_state, video, status, cost_confirmed],
            show_progress="hidden",
            api_name="render_video",
        )
        render_event.then(
            uncertain_choices,
            [session_state],
            [uncertain_shot],
        )
        accepted_uncertain.click(
            accept_uncertain_saved,
            [session_state, uncertain_shot, provider_request_id],
            [session_state, uncertain_shot, provider_request_id, status],
            api_name="accept_uncertain_submission",
        )
        rejected_uncertain.click(
            reject_uncertain_saved,
            [session_state, uncertain_shot, provider_request_id],
            [session_state, uncertain_shot, provider_request_id, status],
            api_name="reject_uncertain_submission",
        )
    return app


def _selection_text(session: GuidedStorySession) -> str:
    cards = session.selected_cards
    card_text = "、".join(f"《{card.title}》" for card in cards) or "暂无"
    element_text = "、".join(session.selected_elements) or "暂无"
    return f"**已保留创意：** {card_text}  \n**已保留故事零件：** {element_text}"


def _provider_label(agent: StoryAgent) -> str:
    return str(getattr(agent, "provider_label", "真实文本模型"))


def _status_text(session: GuidedStorySession) -> str:
    if bool(getattr(session.agent, "last_used_fallback", False)):
        reason = str(getattr(session.agent, "last_fallback_reason", "文本 API 不可用"))
        return (
            "⚠️ 当前展示的是离线兜底创意，不是 LLM 结果。"
            f"原因：{reason}。请检查当前项目的 .env 后重试。"
        )
    if session.stage == Stage.IDEATING:
        if isinstance(session.agent, OpenAIStoryAgent):
            return (
                f"✓ 已使用真实文本模型生成（{_provider_label(session.agent)}）。"
                "选卡、聊天或直接生成完整故事都可以。"
            )
        return "离线演示模式：选卡、聊天或直接生成完整故事都可以。"
    if session.stage == Stage.STORY_REVIEW:
        return f"当前是故事第{session.story.version}版，可以改写或确认后生成剧本。"
    if session.stage == Stage.SCRIPT_REVIEW:
        return "剧本等待确认；场景数量由故事内容自然决定。"
    return f"当前阶段：{session.stage.value}"


def _text_status(session: GuidedStorySession, success_message: str) -> str:
    """Keep the text-provider result visible after every text operation."""
    if bool(getattr(session.agent, "last_used_fallback", False)):
        reason = str(getattr(session.agent, "last_fallback_reason", "文本 API 不可用"))
        return (
            "⚠️ 本次使用的是离线兜底，不是 LLM 结果。"
            f"{success_message}原因：{reason}。请检查当前项目的 .env 后重试。"
        )
    if isinstance(session.agent, OpenAIStoryAgent):
        return f"✓ {success_message}本次使用真实文本模型（{_provider_label(session.agent)}）。"
    return f"离线演示模式：{success_message}"


def _element_update(palette: ElementPalette, kind: str):
    options = palette.options[kind]
    return _gr_update(
        choices=[(f"{item.title}｜{item.content}", item.option_id) for item in options],
        value=None,
    )


def _restored_element_update(session: GuidedStorySession, kind: str):
    palette = session.element_palette
    if palette is None:
        return _gr_update(choices=[], value=None)
    options = palette.options.get(kind, [])
    selected = session.selected_elements.get(kind)
    return _gr_update(
        choices=[(f"{item.title}｜{item.content}", item.option_id) for item in options],
        value=selected if any(item.option_id == selected for item in options) else None,
    )


def _stage_tab(stage: Stage) -> str:
    if stage == Stage.STORY_REVIEW:
        return "story"
    if stage == Stage.SCRIPT_REVIEW:
        return "script"
    if stage == Stage.STORYBOARD_REVIEW:
        return "storyboard"
    if stage in {Stage.RENDER_READY, Stage.COMPLETED}:
        return "video"
    return "ideas"


def _shot_choices_update(plan):
    choices = (
        [(f"镜头 {shot.shot_id}｜{shot.shot_purpose}", str(shot.shot_id)) for shot in plan.shots]
        if plan
        else []
    )
    return _gr_update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )


def _uncertain_shot_choices_update(plan):
    latest: dict[int, Any] = {}
    if plan is not None:
        for artifact in plan.artifacts:
            if artifact.status == "submission_uncertain":
                latest[artifact.shot_id] = artifact
    choices = [
        (
            f"镜头 {shot_id}｜本地操作 ID：{artifact.request_id or '未知'}",
            str(shot_id),
        )
        for shot_id, artifact in sorted(latest.items())
    ]
    return _gr_update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )


def _visual_binding_choices_update(plan):
    choices: list[tuple[str, str]] = []
    if plan is not None:
        choices.extend(
            (
                f"资产｜{asset.name}（{asset.kind}）",
                f"asset|{asset.asset_id}",
            )
            for asset in plan.visual_bible.assets
        )
        choices.extend(
            (
                f"镜头 {shot.shot_id}｜{shot.shot_purpose}",
                f"shot|{shot.shot_id}",
            )
            for shot in plan.shots
        )
    return _gr_update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )


def _visual_reference_choices_update(plan):
    choices: list[tuple[str, str]] = []
    seen: set[str] = set()
    if plan is not None:
        for label, reference in _bound_visual_references(plan):
            if reference.reference_id in seen:
                continue
            seen.add(reference.reference_id)
            status = "已确认" if reference.confirmed else "待确认"
            choices.append(
                (
                    f"{label}｜{reference.usage}｜{status}｜{Path(reference.path).name}",
                    reference.reference_id,
                )
            )
    return _gr_update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )


def _visual_inputs_markdown(plan) -> str:
    if plan is None:
        return "*生成分镜后，可在这里上传并绑定参考图。*"
    rows = []
    for label, reference in _bound_visual_references(plan):
        status = "✓ 已确认" if reference.confirmed else "○ 待确认"
        rows.append(
            f"| {reference.reference_id} | {label} | {reference.usage} | "
            f"{status} | {Path(reference.path).name} |"
        )
    if not rows:
        return (
            "#### 参考图绑定\n\n"
            "尚未上传参考图。人物定妆图、地点图和道具图不会自动变成视频首帧；"
            "`start_frame` 必须明确绑定到具体镜头。"
        )
    return "\n".join(
        [
            "#### 参考图绑定",
            "",
            "| ID | 绑定对象 | 用途 | 状态 | 文件 |",
            "|---|---|---|---|---|",
            *rows,
        ]
    )


def _bound_visual_references(plan):
    for asset in plan.visual_bible.assets:
        for reference in asset.references:
            yield f"资产 {asset.asset_id}", reference
    for shot in plan.shots:
        for reference in shot.confirmed_visual_inputs:
            if reference.binding_kind == "asset":
                continue
            yield f"镜头 {shot.shot_id}", reference


def _uploaded_image_path(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, dict):
        return str(value.get("path") or value.get("name") or "")
    return str(getattr(value, "path", "") or getattr(value, "name", "") or "")


def _parse_visual_binding(value: str | None) -> tuple[str, str]:
    raw = str(value or "").strip()
    if "|" not in raw:
        raise ValueError("请明确选择要绑定的镜头或视觉资产。")
    kind, binding_id = raw.split("|", 1)
    if kind not in {"asset", "shot"} or not binding_id:
        raise ValueError("参考图绑定目标无效。")
    return kind, binding_id


def _persist_visual_upload(source_path: str | Path) -> Path:
    source = Path(source_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError("上传的参考图不存在或为空。")
    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"不支持的图片类型：{suffix or '无扩展名'}")
    digest = sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    # Keep the configured lexical root in persisted references. Canonical
    # paths are still used for the traversal check below, but returning a
    # fully resolved path can expand Windows 8.3 aliases and break containment
    # checks against the caller's original temp directory.
    root = Path(os.getenv("VISUAL_INPUT_DIR", "outputs/visual_inputs")).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{digest.hexdigest()[:24]}{suffix}"
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("视觉输入目录配置无效。") from exc
    if not target.is_file():
        temporary = root / f".{target.name}.{uuid4().hex}.tmp"
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def _gr_update(**kwargs):
    try:
        import gradio as gr

        return gr.update(**kwargs)
    except ImportError:
        return kwargs


def _story_markdown(story) -> str:
    characters = "；".join(f"**{item.name}**：{item.description}" for item in story.characters)
    locations = "；".join(f"**{item.name}**：{item.description}" for item in story.locations)
    return (
        f"### 《{story.title}》 · 故事第{story.version}版\n\n"
        f"*{story.logline}*\n\n"
        f"{story.story_text}\n\n"
        f"**人物：** {characters or '—'}  \n"
        f"**地点：** {locations or '—'}  \n"
        f"**核心冲突：** {story.core_conflict}  \n"
        f"**结局：** {story.ending}"
    )


def _script_markdown(script, session: GuidedStorySession | None = None) -> str:
    sections = [f"### 《{script.title}》 · {script.total_duration}秒"]
    if session is not None:
        event = next(
            (
                item
                for item in reversed(session.text_generation_events)
                if item.get("artifact_type") == "script"
                and item.get("status") in {"succeeded", "fallback", "offline"}
            ),
            None,
        )
        if event:
            provider = f"{event.get('provider', 'offline')} · {event.get('model', 'rule-based')}"
            if event.get("status") in {"fallback", "offline"}:
                sections.append(
                    f"> ⚠️ 来源：离线兜底（{provider}）。这不是远程模型结果。"
                )
            else:
                sections.append(f"> 来源：真实文本模型（{provider}）")
    for scene in script.scenes:
        sections.append(
            f"#### 场景 {scene.scene_id}｜{scene.title} · {scene.duration}秒\n"
            f"**地点：** {scene.location}  \n"
            f"**画面动作：** {scene.visible_action or scene.action}  \n"
            f"**对白：** {scene.dialogue or '—'}  \n"
            f"**旁白：** {scene.narration or '—'}"
        )
    return "\n\n".join(sections)


def _ai_fill_markdown(artifact) -> str:
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
    items = [labels.get(item, item) for item in artifact.ai_filled_fields]
    return "**AI补全：** " + ("、".join(items) if items else "无；全部关键内容来自你的选择。")


def _storyboard_markdown(plan) -> str:
    if plan is None:
        return ""
    bible = plan.visual_bible
    assets = "；".join(
        f"`{item.asset_id}` {item.name}（{item.kind}，"
        f"{len(item.reference_images) + len(item.references)} 张固定参考图）"
        for item in bible.assets
    )
    sections = [
        f"### 《{plan.title}》 · {plan.total_duration}秒 · {len(plan.shots)}个动态镜头",
        "#### 视觉圣经\n"
        f"**视觉风格：** {bible.visual_style}  \n"
        f"**色彩：** {bible.color_palette}  \n"
        f"**光线：** {bible.lighting_rules}  \n"
        f"**摄影原则：** {bible.camera_language}  \n"
        f"**待生成参考资产：** {assets or '暂无'}",
    ]
    for shot in plan.shots:
        sections.append(
            f"#### 镜头 {shot.shot_id}｜{shot.duration}秒｜{shot.shot_kind}｜{shot.camera}\n"
            f"**存在理由：** {shot.shot_purpose}  \n"
            f"**时长分配：** 最终 {shot.duration} 秒；估算 "
            f"{shot.estimated_duration:.1f} 秒；权重 {shot.duration_weight:.2f}  \n"
            f"**时长理由：** {shot.duration_reason or '旧会话未记录'}  \n"
            f"**Seed：** {shot.seed if shot.seed is not None else '旧会话未分配'}  \n"
            f"**首帧：** {shot.first_frame_prompt}  \n"
            f"**动作：** {shot.motion_prompt}  \n"
            f"**结束帧：** {shot.end_frame_prompt}  \n"
            f"**连续性模式：** {shot.continuity_mode}  \n"
            f"**切镜方式：** {shot.transition_type}  \n"
            f"**切镜理由：** {shot.transition_reason or '旧会话未记录'}  \n"
            f"**继承上一镜头末帧：** "
            f"{'是，仅用于连续动作' if shot.inherit_previous_frame else '否，重新建立机位和构图'}  \n"
            f"**引用资产：** {'、'.join(shot.reference_asset_ids) or '暂无'}  \n"
            f"**已解析固定图：** "
            f"{'、'.join(shot.reference_image_paths) or '渲染时从视觉圣经解析'}  \n"
            f"**参考图用途：** {_visual_input_summary(shot)}  \n"
            f"**起始参考：** {_shot_start_reference(shot)}"
        )
    return "\n\n".join(sections)


def _video_job_markdown(job) -> str:
    if job is None:
        return ""
    scene_count = len(job.metadata.get("scenes", [])) if isinstance(job.metadata, dict) else 0
    return (
        f"### 《{job.title}》 · 完整视频任务 · {job.target_seconds}秒\n\n"
        "剧本已作为一个连续的视频生成请求提交给 Provider。"
        "场景只作为叙事上下文，不会在本地被切成等长镜头。\n\n"
        f"**场景上下文：** {scene_count} 个\n\n"
        f"**视觉风格：** {job.visual_style or '由剧本与 Provider 默认能力决定'}\n\n"
        "**视觉圣经：** 已并入完整视频任务的视觉与连续性约束。\n\n"
        "**首帧：** 由完整叙事和已确认视觉输入共同决定。\n\n"
        "**引用资产：** 已确认的视觉输入随任务元数据传递；无则由 Provider 自行生成。\n\n"
        f"**提示词摘要：** {job.prompt[:800]}"
    )


def _shot_start_reference(shot) -> str:
    explicit = next(
        (
            item
            for item in shot.confirmed_visual_inputs
            if item.confirmed and item.usage == "start_frame"
        ),
        None,
    )
    if explicit is not None:
        return f"已确认 start_frame：{explicit.reference_id}"
    if shot.initial_frame_path:
        return shot.initial_frame_path
    if shot.continuity_mode == "same_scene_chain" and shot.previous_shot_id is not None:
        return f"镜头 {shot.previous_shot_id} 的生成末帧"
    if shot.continuity_mode == "same_scene_reference":
        return "同场景切换机位；不继承上一镜头画面，仅共享连续性状态与固定参考图"
    if shot.continuity_mode == "new_scene_reference":
        return "本场景已确认固定参考图；若缺失会明确提示或标记无参考回退"
    return "独立镜头，不继承上一镜头"


def _visual_input_summary(shot) -> str:
    if not shot.confirmed_visual_inputs:
        return "暂无已确认视觉输入"
    return "；".join(
        f"{item.reference_id}={item.usage}"
        for item in shot.confirmed_visual_inputs
    )


def _render_evidence_summary(manifest) -> str:
    parts = [
        f"重新生成 {len(manifest.generated_shots)}",
        f"复用 {len(manifest.reused_shots)}",
        f"依赖失败 {len(manifest.dependency_failed_shots)}",
        f"无参考回退 {len(manifest.unreferenced_fallback_shots)}",
    ]
    if manifest.final_video_path:
        parts.append("成片 1")
    diagnostics = []
    for artifact in manifest.artifacts:
        for message in artifact.continuity_diagnostics:
            if message not in diagnostics:
                diagnostics.append(message)
    if diagnostics:
        parts.append("连续性提示：" + "；".join(diagnostics[:3]))
    return "；".join(parts)


def _progress_text(fraction: float, message: str) -> str:
    return f"视频进度 {round(min(1.0, max(0.0, fraction)) * 100)}%｜{message}"


def main() -> None:
    build_app(agent_factory=OpenAIStoryAgent.from_env).launch()


if __name__ == "__main__":
    main()
