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
        return session, _card_grid_update(session), chat, "", _text_status(
            session, "已按你的补充换出8个新方向。"
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
        return session, "", "", str(exc)


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
        return session, "", "", feedback, str(exc)


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
            _script_markdown(script),
            _text_status(
                session,
                f"故事已确认，剧本已按{script.target_seconds}秒生成。",
            ),
        )
    except Exception as exc:
        return session, "", str(exc)


def revise_script_view(
    session: GuidedStorySession | None, feedback: str
) -> tuple[GuidedStorySession | None, str, str, str]:
    if session is None:
        return session, "", feedback, "请先生成剧本。"
    try:
        script = session.revise_script(feedback)
        return (
            session,
            _script_markdown(script),
            "",
            _text_status(session, "剧本已按反馈改写。"),
        )
    except Exception as exc:
        return session, "", feedback, str(exc)


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
        requirement = feedback.strip()
        if not requirement:
            raise ValueError("请先写一句 Retake 要求。")
        new_action = f"根据 Retake 要求“{requirement}”：{shot.action}"
        new_motion = (
            f"{new_action} 摄影机运动：{shot.camera_movement}。"
            "保持人物身份、动作方向、道具位置与首帧连续。"
        )
        new_prompt = (
            f"Cinematic narrative shot. Purpose: {shot.shot_purpose}. "
            f"FIRST FRAME: {shot.first_frame_prompt} "
            f"MOTION: {new_motion} "
            f"END FRAME: {shot.end_frame_prompt}"
        )
        session.update_storyboard_shot(
            int(shot_id),
            {
                "action": new_action,
                "motion_prompt": new_motion,
                "video_prompt": new_prompt,
            },
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

    def start(direction, duration_mode, seconds):
        return start_garden_view(
            direction,
            seconds,
            agent=(agent_factory() if agent_factory else RuleBasedStoryAgent()),
            duration_mode=duration_mode,
        )

    start.__name__ = "start_garden_view"

    def toggle_custom_duration(duration_mode):
        return _gr_update(visible=duration_mode == "custom")

    def story_and_open(session):
        return (*generate_story_view(session), _gr_update(selected="story"))

    story_and_open.__name__ = "generate_story_view"

    def script_and_open(session):
        return (*generate_script_view(session), _gr_update(selected="script"))

    script_and_open.__name__ = "generate_script_view"

    def storyboard_and_open(session):
        return (*build_storyboard_view(session), _gr_update(selected="storyboard"))

    storyboard_and_open.__name__ = "build_storyboard_view"

    def confirm_storyboard_and_open(session):
        return (*confirm_storyboard_view(session), _gr_update(selected="video"))

    confirm_storyboard_and_open.__name__ = "confirm_storyboard_view"

    def back_and_open(session):
        return (*back_to_ideas_view(session), _gr_update(selected="ideas"))

    back_and_open.__name__ = "back_to_ideas_view"

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
                            rewrite_script = gr.Button(
                                "改写剧本", elem_classes=["quiet-action"]
                            )
                            make_storyboard = gr.Button(
                                "接受剧本并生成分镜  →",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )

                with gr.Tab("04  分镜", id="storyboard", elem_classes=["stage-surface"]):
                    gr.HTML(
                        """
                        <div class="stage-kicker">Design the shots</div>
                        <div class="stage-heading">在生成视频前，先把每个镜头看清楚</div>
                        <p class="stage-copy">检查动作、景别、节奏和连续性。
                        此时仍然不会调用真实视频 API。</p>
                        <div class="shot-strip">SHOT PLAN　·　ACTION　·　CAMERA　·　CONTINUITY</div>
                        """
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(
                            scale=7, elem_classes=["paper-panel", "storyboard-paper"]
                        ):
                            storyboard = gr.Markdown(
                                "*确认剧本后，镜头时间线会出现在这里。*",
                                elem_classes=["storyboard-copy", "empty-state"],
                            )
                        with gr.Column(
                            scale=3, elem_classes=["side-panel", "storyboard-controls"]
                        ):
                            gr.HTML(
                                """
                                <div class="panel-label">镜头修改</div>
                                <div class="assistant-note"><strong>先检查连续性。</strong><br>
                                选择任一镜头修改景别、动作或节奏；确认整套分镜之前，
                                不会调用真实视频服务。</div>
                                """
                            )
                            shot_choice = gr.Dropdown(label="选择镜头")
                            retake_feedback = gr.Textbox(
                                label="Retake 要求",
                                placeholder="例如：改成近景，动作更克制",
                                lines=4,
                            )
                            retake = gr.Button(
                                "重做这个镜头方案", elem_classes=["quiet-action"]
                            )
                            confirm_storyboard = gr.Button(
                                "确认整套分镜  →",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )

                with gr.Tab("05  视频", id="video", elem_classes=["stage-surface"]):
                    gr.HTML(
                        """
                        <div class="stage-kicker">Render the film</div>
                        <div class="stage-heading">最后一步：把确认过的镜头变成影片</div>
                        <p class="stage-copy">只有确认完整分镜并主动勾选费用确认后，
                        才会调用真实视频服务。生成进度会保留，失败镜头可以单独重试。</p>
                        """
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(
                            scale=7, elem_classes=["paper-panel", "video-paper"]
                        ):
                            gr.HTML('<div class="panel-label">最终成片</div>')
                            video = gr.Video(
                                show_label=False,
                                interactive=False,
                                elem_id="final-video",
                            )
                            gr.HTML(
                                """
                                <div class="video-ready-note">
                                成片生成后会在这里直接播放；每个镜头的成功、失败和重试记录
                                都会保存在本次项目中。
                                </div>
                                """
                            )
                        with gr.Column(
                            scale=3, elem_classes=["side-panel", "video-controls"]
                        ):
                            gr.HTML(
                                """
                                <div class="panel-label">生成设置</div>
                                <div class="assistant-note"><strong>先确认，再生成。</strong><br>
                                视频会按照已确认分镜逐镜头制作，成功片段会即时保存，
                                全部完成后自动合成为一条成片。</div>
                                """
                            )
                            cost_confirmed = gr.Checkbox(
                                label="我确认下一步会调用付费视频 API"
                            )
                            render = gr.Button(
                                "生成真实视频",
                                variant="primary",
                                elem_classes=["primary-action"],
                            )
            status = gr.Markdown(
                "写下一句话，我们从寻找故事开始。",
                elem_id="studio-status",
            )

        base_outputs = [session_state, card_grid, selection, chat, status]
        duration_mode.change(
            toggle_custom_duration,
            [duration_mode],
            [target_seconds],
        )
        begin.click(
            start,
            [direction, duration_mode, target_seconds],
            base_outputs,
            api_name="start_ideation",
        )
        direction.submit(
            start,
            [direction, duration_mode, target_seconds],
            base_outputs,
        )
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
        make_story.click(
            story_and_open,
            [session_state],
            [session_state, story, ai_fill, status, workflow],
            api_name="generate_story",
        )
        rewrite_story.click(
            revise_story_view,
            [session_state, story_feedback],
            [session_state, story, ai_fill, story_feedback, status],
            api_name="revise_story",
        )
        back.click(
            back_and_open,
            [session_state],
            [session_state, card_grid, status, workflow],
            api_name="back_to_ideas",
        )
        make_script.click(
            script_and_open,
            [session_state],
            [session_state, script, status, workflow],
            api_name="generate_script",
        )
        rewrite_script.click(
            revise_script_view,
            [session_state, script_feedback],
            [session_state, script, script_feedback, status],
            api_name="revise_script",
        )
        make_storyboard.click(
            storyboard_and_open,
            [session_state],
            [session_state, storyboard, shot_choice, status, workflow],
            api_name="build_storyboard",
        )
        retake.click(
            retake_shot_view,
            [session_state, shot_choice, retake_feedback],
            [session_state, storyboard, retake_feedback, status],
            api_name="retake_shot",
        )
        confirm_storyboard.click(
            confirm_storyboard_and_open,
            [session_state],
            [session_state, status, workflow],
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
        return (
            f"✓ {success_message}本次使用真实文本模型"
            f"（{_provider_label(session.agent)}）。"
        )
    return f"离线演示模式：{success_message}"


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


def _story_markdown(story) -> str:
    characters = "；".join(
        f"**{item.name}**：{item.description}" for item in story.characters
    )
    locations = "；".join(
        f"**{item.name}**：{item.description}" for item in story.locations
    )
    return (
        f"### 《{story.title}》 · 故事第{story.version}版\n\n"
        f"*{story.logline}*\n\n"
        f"{story.story_text}\n\n"
        f"**人物：** {characters or '—'}  \n"
        f"**地点：** {locations or '—'}  \n"
        f"**核心冲突：** {story.core_conflict}  \n"
        f"**结局：** {story.ending}"
    )


def _script_markdown(script) -> str:
    sections = [f"### 《{script.title}》 · {script.total_duration}秒"]
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
        f"`{item.asset_id}` {item.name}（{item.kind}）"
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
            f"**首帧：** {shot.first_frame_prompt}  \n"
            f"**动作：** {shot.motion_prompt}  \n"
            f"**结束帧：** {shot.end_frame_prompt}  \n"
            f"**引用资产：** {'、'.join(shot.reference_asset_ids) or '暂无'}"
        )
    return "\n\n".join(sections)


def _progress_text(fraction: float, message: str) -> str:
    return f"视频进度 {round(min(1.0, max(0.0, fraction)) * 100)}%｜{message}"


def main() -> None:
    build_app(agent_factory=OpenAIStoryAgent.from_env).launch()


if __name__ == "__main__":
    main()
