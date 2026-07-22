from __future__ import annotations

import asyncio
import tempfile
import unittest
import warnings
from pathlib import Path

from guided_story_agent import RuleBasedStoryAgent, Stage
from guided_story_agent.selfplay import run_selfplay
from guided_story_agent.web_app import (
    build_app,
    build_outline_view,
    initialize_view,
    render_video_with_progress,
    submit_message,
)


class SelfPlayWebTests(unittest.TestCase):
    def test_selfplay_completes_without_video_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_selfplay(
                agent=RuleBasedStoryAgent(),
                target_seconds=45,
                max_turns=12,
                output_dir=temp_dir,
                render=False,
            )
            bench = result["bench"]
            self.assertFalse(bench["video_requested"])
            self.assertTrue(bench["duration_within_tolerance"])
            self.assertTrue((Path(temp_dir) / "transcript.json").is_file())
            self.assertTrue((Path(temp_dir) / "storyboard.json").is_file())
            self.assertEqual(Stage.RENDER_READY, result["session"].stage)

    def test_web_handlers_are_thin_and_testable(self) -> None:
        session, chat, outline, script, storyboard, status = initialize_view(30)
        self.assertEqual(30, session.brief.target_seconds)
        self.assertIsNone(outline)
        self.assertIn("开头", chat[0]["content"])
        returned, chat, cleared, status = submit_message(session, "雨夜，车站出现一封信。", chat)
        self.assertIs(returned, session)
        self.assertEqual("", cleared)
        self.assertEqual(1, session.valid_turns)
        _, payload, status = build_outline_view(session)
        self.assertIsNone(payload)
        self.assertIn("尚未达到", status)

    def test_render_progress_rejects_unconfirmed_session_without_provider(self) -> None:
        session, *_ = initialize_view(45)
        updates = list(render_video_with_progress(session))
        self.assertEqual(1, len(updates))
        self.assertIn("必须先确认", updates[0][2])

    def test_gradio_app_exposes_core_handlers(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            app = build_app()
        self.addCleanup(app.close)
        dependency_names = {
            getattr(dependency.fn, "__name__", "")
            for dependency in app.fns.values()
            if getattr(dependency, "fn", None)
        }
        self.assertIn("submit_message", dependency_names)
        self.assertIn("render_video_with_progress", dependency_names)

    def test_gradio_process_api_initializes_real_component_outputs(self) -> None:
        from gradio.state_holder import SessionState

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            app = build_app()
        self.addCleanup(app.close)
        state = SessionState(app)

        async def process() -> dict:
            return await app.process_api(0, [30], state=state)

        initialized = asyncio.run(process())
        self.assertEqual(11, len(initialized["data"]))
        self.assertIn("开头", initialized["data"][1][0]["content"])
        self.assertIsNone(initialized["data"][2])
        self.assertIn("新创作已开始", initialized["data"][5])
        self.assertTrue(initialized["data"][6])

    def test_gradio_process_api_covers_suggestion_edit_undo_confirm_and_cost_gate(self) -> None:
        from gradio.state_holder import SessionState

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            app = build_app()
        self.addCleanup(app.close)
        state = SessionState(app)
        indexes = {
            getattr(dependency.fn, "__name__", ""): index
            for index, dependency in app.fns.items()
            if getattr(dependency, "fn", None)
        }

        async def process() -> None:
            initialized = await app.process_api(indexes["initialize_view"], [30], state=state)
            submitted = await app.process_api(
                indexes["submit_message"],
                [None, "雨夜，一封信落在车站。", initialized["data"][1]],
                state=state,
            )
            self.assertEqual(3, len(submitted["data"][8]["choices"]))
            suggestion_id = submitted["data"][8]["choices"][0][1]
            used = await app.process_api(
                indexes["apply_suggestion_view"],
                [None, suggestion_id, submitted["data"][1]],
                state=state,
            )
            bible = used["data"][3]
            theme_row = next(row for row in bible["data"] if row[0] == "主题")
            theme_row[1] = "记忆与承担"
            await app.process_api(
                indexes["save_story_bible_view"], [None, bible], state=state
            )
            state_id = app.fns[indexes["save_story_bible_view"]].inputs[0]._id
            active_session = state[state_id]
            self.assertEqual("记忆与承担", active_session.facts.theme)
            await app.process_api(indexes["undo_current"], [None], state=state)
            self.assertEqual("", active_session.facts.theme)

            with tempfile.TemporaryDirectory() as temp_dir:
                ready = run_selfplay(
                    agent=RuleBasedStoryAgent(), target_seconds=30, max_turns=12,
                    output_dir=temp_dir,
                )["session"]
                ready.update_storyboard_shot(1, {"action": "主角停步回头。"})
                confirm_id = app.fns[indexes["confirm_storyboard_view"]].inputs[0]._id
                state[confirm_id] = ready
                confirmed = await app.process_api(
                    indexes["confirm_storyboard_view"], [None], state=state
                )
                self.assertIn("分镜已确认", confirmed["data"][1])
                blocked = await app.process_api(
                    indexes["render_video_with_progress"], [None, False], state=state
                )
                self.assertIn("费用确认", blocked["data"][2])
                self.assertIsNone(ready.render_manifest)

        asyncio.run(process())


if __name__ == "__main__":
    unittest.main()
