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
        self.assertEqual(6, len(initialized["data"]))
        self.assertIn("开头", initialized["data"][1][0]["content"])
        self.assertIsNone(initialized["data"][2])
        self.assertIn("新创作已开始", initialized["data"][5])


if __name__ == "__main__":
    unittest.main()
