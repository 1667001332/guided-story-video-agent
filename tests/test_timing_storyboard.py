from __future__ import annotations

import unittest

from guided_story_agent.models import StoryFacts, StoryScene, StoryScript
from guided_story_agent.storyboard import build_storyboard
from guided_story_agent.timing import allocate_durations


class TimingTests(unittest.TestCase):
    def test_30_45_60_second_plans_are_exact(self) -> None:
        for target in (30, 45, 60):
            values = allocate_durations(target, 5)
            self.assertEqual(target, sum(values))
            self.assertTrue(all(3 <= value <= 15 for value in values))

    def test_impossible_duration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allocate_durations(20, 10)

    def test_storyboard_requires_confirmed_script(self) -> None:
        script = StoryScript("测试", 30, [])
        with self.assertRaises(RuntimeError):
            build_storyboard(script, StoryFacts())

    def test_storyboard_keeps_narration_and_continuity(self) -> None:
        script = StoryScript(
            "测试",
            30,
            [
                StoryScene(i, f"场景{i}", "车站", "雨夜", ["邮差"], f"动作{i}", f"旁白{i}", 6)
                for i in range(1, 6)
            ],
            confirmed=True,
        )
        facts = StoryFacts(
            character_visuals="蓝色制服",
            props="铜怀表",
            transitions="匹配剪辑",
        )
        plan = build_storyboard(script, facts)
        self.assertEqual(30, plan.total_duration)
        self.assertTrue(2 <= len(plan.shots) <= 10)
        self.assertIn("蓝色制服", plan.shots[0].first_frame_prompt)
        self.assertIn("旁白5", plan.narration_text)
        self.assertTrue(plan.visual_bible.assets)
        self.assertTrue(all(shot.first_frame_prompt for shot in plan.shots))
        self.assertTrue(all(shot.motion_prompt for shot in plan.shots))
        self.assertTrue(all(shot.end_frame_prompt for shot in plan.shots))
        self.assertTrue(all("FIRST FRAME:" in shot.video_prompt for shot in plan.shots))

    def test_story_content_drives_shot_count(self) -> None:
        simple = StoryScript(
            "简单动作",
            30,
            [
                StoryScene(
                    1,
                    "等待",
                    "空站台",
                    "清晨",
                    ["旅人"],
                    "旅人安静等待列车",
                    "",
                    30,
                )
            ],
            confirmed=True,
        )
        complex_scene = StoryScene(
            1,
            "交出证据",
            "审讯室",
            "深夜",
            ["警探", "证人"],
            "证人把沾雨的怀表推到警探面前",
            "",
            30,
            dialogue="这不是我的表。",
            props=["铜怀表"],
            visible_action="证人把沾雨的怀表推到警探面前",
            start_state="两人隔桌对坐，怀表藏在证人口袋里",
            end_state="怀表停在警探手边，证人移开视线",
            emotional_change="警探从怀疑转为确认",
        )
        complex_script = StoryScript(
            "复杂动作",
            30,
            [complex_scene],
            confirmed=True,
        )
        simple_plan = build_storyboard(simple, StoryFacts())
        complex_plan = build_storyboard(
            complex_script,
            StoryFacts(character_visuals="警探：灰色风衣；证人：深蓝雨衣"),
        )
        self.assertGreater(len(complex_plan.shots), len(simple_plan.shots))
        self.assertIn("detail", {shot.shot_kind for shot in complex_plan.shots})
        self.assertIn("dialogue", {shot.shot_kind for shot in complex_plan.shots})
        self.assertTrue(
            any(
                "character-" in asset_id
                for shot in complex_plan.shots
                for asset_id in shot.reference_asset_ids
            )
        )


if __name__ == "__main__":
    unittest.main()
