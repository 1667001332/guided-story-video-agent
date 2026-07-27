from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from guided_story_agent import RuleBasedStoryAgent, Stage
from guided_story_agent.agent import OpenAIStoryAgent
from guided_story_agent.models import RenderManifest, VideoArtifact
from guided_story_agent.selfplay import run_selfplay
from guided_story_agent.web_app import (
    add_visual_reference_view,
    build_app,
    card_grid_payload,
    confirm_visual_inputs_view,
    generate_story_view,
    remove_visual_reference_view,
    retake_shot_view,
    render_video_with_progress,
    refresh_ideas_view,
    revise_story_view,
    start_garden_view,
)


def make_render_ready_session():
    session, *_ = start_garden_view("雨夜车站", 30, RuleBasedStoryAgent())
    session.auto_choose()
    session.generate_story()
    session.confirm_story()
    session.generate_script()
    session.confirm_script()
    session.build_storyboard()
    session.confirm_storyboard()
    return session


class CreativeGardenWebTests(unittest.TestCase):
    def test_start_handler_returns_eight_cards(self) -> None:
        session, update, selection, chat, status = start_garden_view(
            "校园悬疑", 30, RuleBasedStoryAgent()
        )
        self.assertEqual(8, len(card_grid_payload(session)["cards"]))
        self.assertEqual([], session.selected_idea_ids)
        self.assertIn("暂无", selection)
        self.assertEqual(2, len(chat))
        self.assertIn("离线演示模式", status)
        self.assertIsNotNone(update)

    def test_render_progress_rejects_unconfirmed_session(self) -> None:
        session, *_ = start_garden_view("校园悬疑", 30, RuleBasedStoryAgent())
        updates = list(render_video_with_progress(session, False))
        self.assertEqual(1, len(updates))
        self.assertIn("必须先确认", updates[0][2])

    def test_web_explicitly_labels_offline_fallback(self) -> None:
        session, _, _, _, status = start_garden_view(
            "情人节杀人案", 30, OpenAIStoryAgent(None, "offline-test")
        )
        self.assertEqual(8, len(session.current_batch.cards))
        self.assertIn("离线兜底", status)
        self.assertIn("不是 LLM 结果", status)

    def test_followup_text_actions_keep_fallback_visible(self) -> None:
        session, *_ = start_garden_view("校园悬疑", 30, OpenAIStoryAgent(None, "offline-test"))
        session, _, _, refresh_status = refresh_ideas_view(session)
        self.assertIn("离线兜底", refresh_status)
        session, _, _, story_status = generate_story_view(session)
        self.assertIn("离线兜底", story_status)
        self.assertIn("完整故事已生成", story_status)

    def test_app_has_native_multiselect_grid_and_no_dataframe(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = build_app()
        self.addCleanup(app.close)
        card_grids = [
            component
            for component in app.blocks.values()
            if "idea-card-grid" in (getattr(component, "elem_classes", []) or [])
        ]
        self.assertEqual(1, len(card_grids))
        names = {component.__class__.__name__ for component in app.blocks.values()}
        self.assertNotIn("Dataframe", names)
        api_names = {dependency.api_name for dependency in app.fns.values() if dependency.api_name}
        for name in (
            "start_ideation",
            "select_ideas",
            "mix_selected",
            "auto_choose",
            "generate_story",
            "generate_script",
            "back_to_ideas",
            "add_visual_reference",
            "remove_visual_reference",
            "confirm_visual_inputs",
            "render_video",
        ):
            self.assertIn(name, api_names)

    def test_web_visual_bind_confirm_delete_invalidates_storyboard(self) -> None:
        session = make_render_ready_session()
        asset = session.storyboard.visual_bible.assets[0]
        with tempfile.TemporaryDirectory() as temp:
            upload = Path(temp) / "identity.png"
            upload.write_bytes(b"identity-reference")
            visual_dir = Path(temp) / "managed-inputs"
            with patch.dict(
                "os.environ",
                {"VISUAL_INPUT_DIR": str(visual_dir)},
            ):
                added = add_visual_reference_view(
                    session,
                    str(upload),
                    "identity_reference",
                    f"asset|{asset.asset_id}",
                    "主角正面定妆照",
                )

            self.assertIn("尚未冻结", added[7])
            self.assertFalse(session.storyboard.confirmed)
            self.assertEqual(Stage.STORYBOARD_REVIEW, session.stage)
            self.assertEqual(1, len(asset.references))
            reference = asset.references[0]
            self.assertTrue(Path(reference.path).is_relative_to(visual_dir))
            self.assertEqual("asset", reference.binding_kind)
            self.assertFalse(reference.confirmed)

            confirmed = confirm_visual_inputs_view(session)
            self.assertIn("已冻结", confirmed[5])
            self.assertTrue(asset.references[0].confirmed)
            self.assertTrue(
                any(
                    item.reference_id == reference.reference_id and item.confirmed
                    for shot in session.storyboard.shots
                    for item in shot.confirmed_visual_inputs
                )
            )
            session.confirm_storyboard()
            self.assertEqual(Stage.RENDER_READY, session.stage)

            removed = remove_visual_reference_view(session, reference.reference_id)
            self.assertIn("确认已失效", removed[5])
            self.assertFalse(session.storyboard.confirmed)
            self.assertEqual(Stage.STORYBOARD_REVIEW, session.stage)
            self.assertFalse(
                any(
                    item.reference_id == reference.reference_id
                    for shot in session.storyboard.shots
                    for item in shot.confirmed_visual_inputs
                )
            )

    def test_start_frame_cannot_be_bound_to_general_asset(self) -> None:
        session = make_render_ready_session()
        asset = session.storyboard.visual_bible.assets[0]
        with tempfile.TemporaryDirectory() as temp:
            upload = Path(temp) / "start.png"
            upload.write_bytes(b"start-frame")
            with patch.dict(
                "os.environ",
                {"VISUAL_INPUT_DIR": str(Path(temp) / "managed-inputs")},
            ):
                result = add_visual_reference_view(
                    session,
                    str(upload),
                    "start_frame",
                    f"asset|{asset.asset_id}",
                )
        self.assertIn("只能绑定到具体镜头", result[7])
        self.assertEqual([], asset.references)

    def test_process_api_one_sentence_select_story_and_script(self) -> None:
        from gradio.state_holder import SessionState

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = build_app()
        self.addCleanup(app.close)
        state = SessionState(app)
        indexes = {
            dependency.api_name: index
            for index, dependency in app.fns.items()
            if dependency.api_name
        }

        async def process() -> None:
            started = await app.process_api(
                indexes["start_ideation"], ["校园悬疑", "auto", 90], state=state
            )
            grid = started["data"][1]
            self.assertEqual(8, len(grid["choices"]))
            first_id = grid["choices"][0][1]
            selected = await app.process_api(
                indexes["select_ideas"], [None, [first_id]], state=state
            )
            self.assertIn(first_id, selected["data"][1]["value"])
            mixed = await app.process_api(indexes["mix_selected"], [None], state=state)
            self.assertEqual(8, len(mixed["data"][1]["choices"]))
            auto = await app.process_api(indexes["auto_choose"], [None], state=state)
            self.assertEqual(1, len(auto["data"][1]["value"]))
            written = await app.process_api(indexes["generate_story"], [None], state=state)
            self.assertIn("完整故事", written["data"][3])
            state_id = app.fns[indexes["generate_story"]].inputs[0]._id
            active = state[state_id]
            self.assertEqual(Stage.STORY_REVIEW, active.stage)
            scripted = await app.process_api(indexes["generate_script"], [None], state=state)
            self.assertIn("场景 1", scripted["data"][1])
            self.assertEqual(Stage.SCRIPT_REVIEW, active.stage)
            self.assertEqual("auto", active.brief.duration_mode)
            self.assertEqual(
                active.script.target_seconds,
                active.brief.resolved_target_seconds,
            )
            await app.process_api(indexes["back_to_ideas"], [None], state=state)
            self.assertIsNotNone(active.story)
            self.assertIsNotNone(active.script)

        asyncio.run(process())

    def test_process_api_paid_gate_does_not_call_provider(self) -> None:
        from gradio.state_holder import SessionState

        app = build_app()
        self.addCleanup(app.close)
        state = SessionState(app)
        indexes = {
            dependency.api_name: index
            for index, dependency in app.fns.items()
            if dependency.api_name
        }

        async def process() -> None:
            await app.process_api(
                indexes["start_ideation"],
                ["雨夜车站", "custom", 30],
                state=state,
            )
            await app.process_api(indexes["auto_choose"], [None], state=state)
            await app.process_api(indexes["generate_story"], [None], state=state)
            await app.process_api(indexes["generate_script"], [None], state=state)
            planned = await app.process_api(indexes["build_storyboard"], [None], state=state)
            self.assertIn("视觉圣经", planned["data"][1])
            self.assertIn("首帧", planned["data"][1])
            self.assertIn("引用资产", planned["data"][1])
            await app.process_api(indexes["confirm_storyboard"], [None], state=state)
            blocked = await app.process_api(indexes["render_video"], [None, False], state=state)
            self.assertIn("费用确认", blocked["data"][2])

        asyncio.run(process())

    def test_retake_updates_camera_movement_and_frame_fields(self) -> None:
        session = make_render_ready_session()
        session.stage = Stage.STORYBOARD_REVIEW
        session.storyboard.confirmed = False
        shot = session.storyboard.shots[0]
        shot.camera = "wide"
        shot.camera_movement = "static"
        original_first_frame = shot.first_frame_prompt

        _, _, _, status = retake_shot_view(
            session,
            str(shot.shot_id),
            "改成近景，镜头快速推进，首帧突出主角手里的信",
        )
        shot = session.storyboard.shots[0]

        self.assertEqual("medium close-up", shot.camera)
        self.assertEqual("fast dolly in", shot.camera_movement)
        self.assertNotEqual(original_first_frame, shot.first_frame_prompt)
        self.assertIn("首帧要求", shot.first_frame_prompt)
        self.assertIn("CAMERA: medium close-up", shot.video_prompt)
        self.assertIn("新版本", status)

    def test_render_uses_unique_directory_saves_session_and_resets_confirmation(self) -> None:
        rendered_targets: list[Path] = []

        class Renderer:
            def __init__(self, _provider, *, progress_callback=None):
                self.progress_callback = progress_callback

            def render(self, plan, output_dir):
                target = Path(output_dir)
                rendered_targets.append(target)
                final = target / "fake-final.mp4"
                final.write_bytes(b"video")
                if self.progress_callback:
                    self.progress_callback("completed", 1.0, "done")
                return RenderManifest(
                    status="succeeded_with_warnings",
                    output_dir=str(target),
                    final_video_path=str(final),
                    error="旁白不可用，已保留字幕",
                )

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("guided_story_agent.web_app.StoryRenderer", Renderer),
        ):
            first_session = make_render_ready_session()
            list(
                render_video_with_progress(
                    first_session,
                    True,
                    provider=object(),
                    output_dir=temp_dir,
                )
            )
            second_session = make_render_ready_session()
            second = list(
                render_video_with_progress(
                    second_session,
                    True,
                    provider=object(),
                    output_dir=temp_dir,
                )
            )

            self.assertEqual(2, len(rendered_targets))
            self.assertNotEqual(rendered_targets[0], rendered_targets[1])
            for target in rendered_targets:
                self.assertTrue((target / "session_before_render.json").is_file())
                self.assertTrue((target / "session.json").is_file())
            final_update = second[-1]
            self.assertIn("警告", final_update[2])
            self.assertFalse(final_update[3]["value"])

    def test_render_gate_preserves_previous_video_and_consumes_confirmation(self) -> None:
        session = make_render_ready_session()
        session.render_manifest = RenderManifest(
            status="failed",
            output_dir="previous",
            final_video_path="previous.mp4",
        )

        update = list(render_video_with_progress(session, False))[0]

        self.assertEqual("previous.mp4", update[1])
        self.assertIn("费用确认", update[2])
        self.assertFalse(update[3]["value"])

    def test_render_and_retake_are_rejected_while_session_is_rendering(self) -> None:
        session = make_render_ready_session()
        shot = session.storyboard.shots[0]
        original_prompt = shot.video_prompt
        session._render_in_progress = True
        try:
            render_update = list(render_video_with_progress(session, True))[0]
            _, _, _, retake_status = retake_shot_view(
                session,
                str(shot.shot_id),
                "改成特写",
            )
        finally:
            session._render_in_progress = False

        self.assertIn("已有视频任务", render_update[2])
        self.assertIn("正在进行", retake_status)
        self.assertEqual(original_prompt, session.storyboard.shots[0].video_prompt)

    def test_retake_rejects_a_shot_with_a_pending_remote_task(self) -> None:
        session = make_render_ready_session()
        shot = session.storyboard.shots[0]
        original_prompt = shot.video_prompt
        session.storyboard.artifacts.append(
            VideoArtifact(
                "pending",
                shot.shot_id,
                "agnes",
                "agnes-video-v2.0",
                "pending",
                "",
                "",
                shot.duration,
                shot.video_prompt,
                "now",
                request_id="job-pending",
            )
        )

        _, _, feedback, status = retake_shot_view(
            session,
            str(shot.shot_id),
            "改成近景",
        )

        self.assertIn("避免重复付费", status)
        self.assertEqual("改成近景", feedback)
        self.assertEqual(original_prompt, session.storyboard.shots[0].video_prompt)

    def test_failed_story_revision_keeps_previous_story_visible(self) -> None:
        session, *_ = start_garden_view("校园悬疑", 30, RuleBasedStoryAgent())
        session.generate_story()
        title = session.story.title

        with patch.object(session, "revise_story", side_effect=RuntimeError("rewrite failed")):
            _, markdown, ai_fill, feedback, status = revise_story_view(
                session,
                "再克制一些",
            )

        self.assertIn(title, markdown)
        self.assertIn("AI补全", ai_fill)
        self.assertEqual("再克制一些", feedback)
        self.assertIn("rewrite failed", status)

    def test_selfplay_saves_render_artifacts_after_render(self) -> None:
        class Renderer:
            def render(self, plan, output_dir):
                shot = plan.shots[0]
                artifact = VideoArtifact(
                    "saved-artifact",
                    shot.shot_id,
                    "fake",
                    "fake-model",
                    "succeeded",
                    str(Path(output_dir) / "shot.mp4"),
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                )
                plan.artifacts.append(artifact)
                return RenderManifest(
                    status="succeeded",
                    output_dir=str(output_dir),
                    artifacts=[artifact],
                    final_video_path=str(Path(output_dir) / "final.mp4"),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_selfplay(
                agent=RuleBasedStoryAgent(),
                direction="雨夜车站",
                target_seconds=15,
                output_dir=temp_dir,
                render=True,
                renderer=Renderer(),
            )
            saved = json.loads((Path(temp_dir) / "session.json").read_text(encoding="utf-8"))

        self.assertEqual("succeeded", result["bench"]["render_status"])
        self.assertEqual(
            "saved-artifact",
            saved["storyboard"]["artifacts"][0]["artifact_id"],
        )
        self.assertEqual("succeeded", saved["render_manifest"]["status"])

    def test_selfplay_main_requires_paid_video_confirmation(self) -> None:
        from guided_story_agent import selfplay

        with (
            patch.object(sys, "argv", ["guided-story-selfplay", "--render"]),
            patch("guided_story_agent.selfplay.run_selfplay") as run,
            self.assertRaises(SystemExit) as raised,
        ):
            selfplay.main()

        self.assertEqual(2, raised.exception.code)
        run.assert_not_called()

    def test_selfplay_main_offline_never_constructs_remote_agent(self) -> None:
        from guided_story_agent import selfplay

        with (
            patch.object(sys, "argv", ["guided-story-selfplay", "--offline"]),
            patch("guided_story_agent.selfplay.OpenAIStoryAgent.from_env") as remote_agent,
            patch(
                "guided_story_agent.selfplay.run_selfplay",
                return_value={"output_dir": "offline", "bench": {}},
            ) as run,
            patch("builtins.print"),
        ):
            selfplay.main()

        remote_agent.assert_not_called()
        self.assertIsInstance(run.call_args.kwargs["agent"], RuleBasedStoryAgent)

    def test_selfplay_main_rejects_unused_paid_confirmation(self) -> None:
        from guided_story_agent import selfplay

        with (
            patch.object(
                sys,
                "argv",
                ["guided-story-selfplay", "--confirm-paid-video", "RENDER"],
            ),
            patch("guided_story_agent.selfplay.run_selfplay") as run,
            self.assertRaises(SystemExit) as raised,
        ):
            selfplay.main()

        self.assertEqual(2, raised.exception.code)
        run.assert_not_called()

    def test_selfplay_main_exits_nonzero_for_incomplete_video(self) -> None:
        from guided_story_agent import selfplay

        with (
            patch.object(
                sys,
                "argv",
                [
                    "guided-story-selfplay",
                    "--render",
                    "--confirm-paid-video",
                    "RENDER",
                ],
            ),
            patch(
                "guided_story_agent.selfplay.run_selfplay",
                return_value={
                    "output_dir": "pending",
                    "bench": {"render_status": "pending"},
                },
            ),
            patch("builtins.print"),
            self.assertRaises(SystemExit) as raised,
        ):
            selfplay.main()

        self.assertEqual(1, raised.exception.code)


if __name__ == "__main__":
    unittest.main()
