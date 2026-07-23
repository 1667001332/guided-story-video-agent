from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guided_story_agent import (
    CreativeBrief,
    GuidedStorySession,
    OpenAIStoryAgent,
    RuleBasedStoryAgent,
    Stage,
)
from guided_story_agent.cli import run_interactive
from guided_story_agent.selfplay import run_selfplay


class CreativeGardenIntegrationTests(unittest.TestCase):
    def test_from_env_prefers_deepseek_for_text_generation(self) -> None:
        captured: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_module = SimpleNamespace(OpenAI=FakeOpenAI)
        environment = {
            "DEEPSEEK_API_KEY": "deepseek-test-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_TEXT_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_TIMEOUT": "90",
            "AGNES_API_KEY": "agnes-video-key",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.dict(sys.modules, {"openai": fake_module}),
        ):
            agent = OpenAIStoryAgent.from_env()

        self.assertEqual("deepseek-v4-pro", agent.model)
        self.assertEqual("deepseek-test-key", captured["api_key"])
        self.assertEqual("https://api.deepseek.com", captured["base_url"])
        self.assertEqual(90.0, captured["timeout"])

    def test_model_script_accepts_story_driven_scene_count(self) -> None:
        story_text = (
            "雨夜里，邮差收到一封写给明天的信。他循着信上的水迹进入废弃车站，"
            "发现寄信人正是年老后的自己。车站时钟不断倒退，每次回拨都会抹去一段记忆。"
            "邮差必须在保留过去和救下陌生人之间选择。他最终烧掉信件，让时间恢复前进，"
            "也接受自己再也无法证明这场相遇。天亮后，空信封里只剩一枚停止走动的怀表。"
        )

        class Completions:
            def create(self, **request):
                system = request["messages"][0]["content"]
                if "故事创作者" in system:
                    payload = {
                        "story": {
                            "title": "写给明天的信",
                            "logline": "邮差在倒流的车站收到未来自己的来信。",
                            "story_text": story_text,
                            "characters": [
                                {
                                    "name": "邮差",
                                    "description": "害怕失去记忆的年轻邮差",
                                    "visual_identity": "深蓝雨衣和铜怀表",
                                }
                            ],
                            "locations": [
                                {
                                    "name": "废弃车站",
                                    "description": "雨水倒流的旧站台",
                                    "visual_identity": "冷蓝灯光和停摆时钟",
                                }
                            ],
                            "tone": "悬疑克制",
                            "theme": "接受无法保留的一切",
                            "core_conflict": "救人会失去自己的记忆",
                            "ending": "邮差烧掉信件，让时间继续前进",
                            "visual_anchors": ["深蓝雨衣", "铜怀表"],
                        }
                    }
                else:
                    payload = {
                        "script": {
                            "title": "写给明天的信",
                            "scenes": [
                                {
                                    "title": "来信",
                                    "location": "废弃车站",
                                    "characters": ["邮差"],
                                    "visible_action": "邮差接住从空中落下的湿信封。",
                                },
                                {
                                    "title": "选择",
                                    "location": "站台",
                                    "characters": ["邮差"],
                                    "visible_action": "邮差点燃信件，倒退的时钟恢复前进。",
                                },
                            ],
                        }
                    }
                message = SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        session = GuidedStorySession(CreativeBrief(target_seconds=30), RuleBasedStoryAgent())
        session.start_ideation("雨夜车站")
        session.agent = OpenAIStoryAgent(client, "fake-model")
        session.generate_story()
        session.confirm_story()
        script = session.generate_script()
        self.assertEqual(2, len(script.scenes))
        self.assertEqual(30, script.total_duration)

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
            self.assertLessEqual(bench["clicks_to_story"], 2)
            self.assertEqual(1.0, bench["selection_retention"])
            self.assertEqual(1.0, bench["ai_fill_transparency"])
            self.assertFalse(bench["video_requested"])
            self.assertEqual(Stage.RENDER_READY, result["session"].stage)
            for name in (
                "ideas.json",
                "selection.json",
                "story.json",
                "script.json",
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

    def test_cli_reaches_story_with_one_free_text_and_two_commands(self) -> None:
        commands = iter(["校园里的轻喜剧悬疑", "/pick 1", "/story", "/quit"])
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            session = run_interactive(
                agent=RuleBasedStoryAgent(),
                target_seconds=30,
                output_dir=temp_dir,
                input_fn=lambda _: next(commands),
                output_fn=messages.append,
            )
            self.assertEqual(Stage.STORY_REVIEW, session.stage)
            self.assertEqual(1, session.free_text_count)
            self.assertIsNone(session.render_manifest)
            self.assertTrue((Path(temp_dir) / "session.json").is_file())

    def test_cli_render_stays_closed_without_explicit_flag(self) -> None:
        commands = iter(
            [
                "雨夜车站悬疑",
                "/auto",
                "/story",
                "/script",
                "/storyboard",
                "/confirm",
                "/render",
                "/quit",
            ]
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
