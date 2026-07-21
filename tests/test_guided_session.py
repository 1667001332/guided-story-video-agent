from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guided_story_agent import CreativeBrief, GuidedStorySession, Stage
from guided_story_agent.models import RenderManifest


def complete_story(session: GuidedStorySession) -> None:
    answers = [
        "暴雨夜，邮差在废弃车站收到一封写给明天的信。",
        "邮差必须在午夜前找到收信人，阻止一场事故。",
        "冲突来自封锁车站的管理员，他阻止邮差进入站台。",
        "邮差沿旧时刻表寻找线索，危险不断升级。",
        "转折是收信人其实是过去的自己，最后他拉下制动杆救下乘客。",
    ]
    for answer in answers:
        session.submit_user_turn(answer)


def complete_details(session: GuidedStorySession) -> None:
    answers = [
        "主角穿深蓝旧制服和邮差帽，管理员穿灰色风衣。",
        "故事发生在雨夜老车站、地下站台和停驶列车内。",
        "关键道具是湿信封、铜怀表和红色制动杆。",
        "第一人称低沉旁白，每句话简短克制。",
        "用怀表特写做匹配剪辑，结尾回到空站台。",
    ]
    for answer in answers:
        session.answer_detail_question(answer)


class GuidedStorySessionTests(unittest.TestCase):
    def test_target_duration_must_be_between_30_and_60(self) -> None:
        with self.assertRaises(ValueError):
            GuidedStorySession(CreativeBrief(target_seconds=29))
        with self.assertRaises(ValueError):
            GuidedStorySession(CreativeBrief(target_seconds=61))

    def test_empty_and_duplicate_turns_do_not_count(self) -> None:
        session = GuidedStorySession()
        self.assertFalse(session.submit_user_turn(" ").accepted)
        first = session.submit_user_turn("暴雨夜，车站收到一封信。")
        self.assertTrue(first.accepted)
        self.assertFalse(session.submit_user_turn("暴雨夜，车站收到一封信。").accepted)
        self.assertEqual(1, session.valid_turns)

    def test_fewer_than_five_turns_cannot_build_outline(self) -> None:
        session = GuidedStorySession()
        for text in ("开头", "目标", "冲突", "发展"):
            session.submit_user_turn(text)
        with self.assertRaises(RuntimeError):
            session.build_outline()

    def test_five_turns_without_ending_cannot_build_outline(self) -> None:
        session = GuidedStorySession()
        for text in ("开头", "目标", "冲突", "发展", "只有转折没有收尾"):
            session.submit_user_turn(text)
        self.assertFalse(session.can_build_outline)
        self.assertIn("ending", session.facts.missing_outline_fields())

    def test_five_meaningful_turns_with_ending_can_build_outline(self) -> None:
        session = GuidedStorySession()
        complete_story(session)
        self.assertEqual(5, session.valid_turns)
        self.assertTrue(session.can_build_outline)
        outline = session.build_outline()
        self.assertEqual(Stage.OUTLINE_REVIEW, session.stage)
        self.assertIn("制动杆", outline.ending)
        self.assertEqual([1, 2, 3, 4, 5], outline.source_turn_ids)

    def test_full_confirmation_flow_builds_timed_storyboard(self) -> None:
        session = GuidedStorySession(CreativeBrief(target_seconds=45))
        complete_story(session)
        session.build_outline()
        session.confirm_outline()
        self.assertEqual(Stage.DETAILING, session.stage)
        complete_details(session)
        self.assertTrue(session.can_build_script)
        script = session.build_script()
        self.assertEqual(45, script.total_duration)
        session.confirm_script()
        plan = session.build_storyboard()
        self.assertEqual(45, plan.total_duration)
        self.assertTrue(all(3 <= shot.duration <= 15 for shot in plan.shots))
        session.confirm_storyboard()
        self.assertEqual(Stage.RENDER_READY, session.stage)

    def test_rendering_is_blocked_before_storyboard_confirmation(self) -> None:
        session = GuidedStorySession()

        class SpyRenderer:
            def render(self, plan, output_dir):
                raise AssertionError("renderer must not be called")

        with self.assertRaises(RuntimeError):
            session.render_confirmed_plan(SpyRenderer(), "outputs")

    def test_successful_render_moves_session_to_completed(self) -> None:
        session = GuidedStorySession()
        complete_story(session)
        session.build_outline()
        session.confirm_outline()
        complete_details(session)
        session.build_script()
        session.confirm_script()
        session.build_storyboard()
        session.confirm_storyboard()

        class Renderer:
            def render(self, plan, output_dir):
                return RenderManifest(status="succeeded", output_dir=str(output_dir))

        result = session.render_confirmed_plan(Renderer(), "outputs")
        self.assertEqual("succeeded", result.status)
        self.assertEqual(Stage.COMPLETED, session.stage)

    def test_session_exports_traceable_json(self) -> None:
        session = GuidedStorySession()
        complete_story(session)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = session.save(Path(temp_dir) / "session.json")
            content = path.read_text(encoding="utf-8")
        self.assertIn('"source": "human"', content)
        self.assertIn('"opening"', content)


if __name__ == "__main__":
    unittest.main()
