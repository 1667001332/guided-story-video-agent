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
        self.assertEqual(5, len(plan.shots))
        self.assertIn("蓝色制服", plan.shots[0].continuity_notes[0])
        self.assertIn("旁白5", plan.narration_text)


if __name__ == "__main__":
    unittest.main()
