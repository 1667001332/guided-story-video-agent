from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guided_story_agent import OpenAIStoryAgent, RuleBasedStoryAgent, Stage
from guided_story_agent.cli import run_interactive
from guided_story_agent.selfplay import run_selfplay


class CreativeGardenIntegrationTests(unittest.TestCase):
    def test_selfplay_uses_one_text_input_and_no_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_selfplay(
                agent=RuleBasedStoryAgent(),
                target_seconds=45,
                output_dir=temp_dir,
            )
            bench = result["bench"]
            self.assertEqual(8, bench["idea_count"])
            self.assertEqual(1, bench["free_text_required_count"])
            self.assertEqual(0, bench["mandatory_followup_text_count"])
            self.assertLessEqual(bench["clicks_to_draft"], 2)
            self.assertEqual(1.0, bench["selection_retention"])
            self.assertEqual(1.0, bench["ai_fill_transparency"])
            self.assertFalse(bench["video_requested"])
            self.assertEqual(Stage.RENDER_READY, result["session"].stage)
            for name in (
                "ideas.json",
                "selection.json",
                "draft.json",
                "storyboard.json",
                "session.json",
                "bench.json",
            ):
                self.assertTrue((Path(temp_dir) / name).is_file())

    def test_require_live_text_rejects_offline_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "真实文本 API"):
                run_selfplay(
                    agent=RuleBasedStoryAgent(),
                    target_seconds=30,
                    output_dir=temp_dir,
                    require_live_text=True,
                )

    def test_unconfigured_openai_agent_falls_back_without_recursion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_selfplay(
                agent=OpenAIStoryAgent(None, "offline-test"),
                target_seconds=30,
                output_dir=temp_dir,
            )
            self.assertEqual(Stage.RENDER_READY, result["session"].stage)
            self.assertGreater(result["bench"]["text_fallback_count"], 0)

    def test_cli_reaches_draft_with_one_free_text_and_two_commands(self) -> None:
        commands = iter(["校园里的轻喜剧悬疑", "/pick 1", "/draft", "/quit"])
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            session = run_interactive(
                agent=RuleBasedStoryAgent(),
                target_seconds=30,
                output_dir=temp_dir,
                input_fn=lambda _: next(commands),
                output_fn=messages.append,
            )
            self.assertEqual(Stage.DRAFT_REVIEW, session.stage)
            self.assertEqual(1, session.free_text_count)
            self.assertIsNone(session.render_manifest)
            self.assertTrue((Path(temp_dir) / "session.json").is_file())

    def test_cli_render_stays_closed_without_explicit_flag(self) -> None:
        commands = iter(
            ["雨夜车站悬疑", "/auto", "/draft", "/storyboard", "/confirm", "/render", "/quit"]
        )
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            session = run_interactive(
                agent=RuleBasedStoryAgent(),
                target_seconds=30,
                output_dir=temp_dir,
                input_fn=lambda _: next(commands),
                output_fn=messages.append,
            )
            self.assertEqual(Stage.RENDER_READY, session.stage)
            self.assertIsNone(session.render_manifest)
            self.assertTrue(any("付费调用保持关闭" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
