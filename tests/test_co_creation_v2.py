from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guided_story_agent import (
    GuidedStorySession,
    OpenAIStoryAgent,
    RuleBasedStoryAgent,
    Stage,
)
from guided_story_agent.cli import run_interactive
from guided_story_agent.selfplay import run_selfplay


class ConflictAgent(RuleBasedStoryAgent):
    def coach_turn(self, text, facts, history, *, phase):
        result = super().coach_turn(text, facts, history, phase=phase)
        if facts.ending and "改成" in text:
            result["conflicts"] = [
                {
                    "field": "ending",
                    "existing_value": facts.ending,
                    "proposed_value": text,
                    "reason": "新结局与已经记录的结局不一致",
                }
            ]
        return result


class CoCreationV2Tests(unittest.TestCase):
    def test_one_turn_extracts_multiple_fields_and_asks_highest_gap(self) -> None:
        session = GuidedStorySession(agent=RuleBasedStoryAgent())
        result = session.submit_user_turn(
            "雨夜列车突然启动，邮差必须在午夜前停车；管理员阻止主角，"
            "随后车厢开始消失，最后邮差拉下制动救下乘客。"
        )
        extracted = {item.field for item in result.extracted_facts}
        self.assertTrue({"opening", "protagonist_goal", "conflict", "development", "ending"} <= extracted)
        self.assertEqual([], session.facts.missing_outline_fields())
        self.assertIn("发现", result.next_question)

    def test_suggestion_changes_nothing_until_explicitly_applied(self) -> None:
        session = GuidedStorySession(agent=RuleBasedStoryAgent())
        session.submit_user_turn("雨夜，一封来自明天的信落在废弃车站。")
        before = session.facts.protagonist_goal
        suggestions = session.request_suggestions()
        self.assertEqual(before, session.facts.protagonist_goal)
        result = session.apply_suggestion(suggestions[0].suggestion_id)
        self.assertTrue(result.accepted)
        self.assertNotEqual(before, session.facts.protagonist_goal)

    def test_conflict_is_rejected_until_user_edits_story_bible(self) -> None:
        session = GuidedStorySession(agent=ConflictAgent())
        session.update_story_bible({"ending": "主角烧掉信件。"})
        result = session.submit_user_turn("结局改成主角保留信件。")
        self.assertFalse(result.accepted)
        self.assertEqual(0, session.valid_turns)
        self.assertEqual("主角烧掉信件。", session.facts.ending)
        session.update_story_bible({"ending": "主角保留信件。"})
        self.assertEqual([], session.unresolved_conflicts)

    def test_artifacts_are_editable_versioned_and_undoable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_selfplay(
                agent=RuleBasedStoryAgent(), target_seconds=45, max_turns=12,
                output_dir=temp_dir,
            )
            session = result["session"]
            original = session.storyboard.shots[0].action
            session.update_storyboard_shot(1, {"action": "主角突然停步并回头。"})
            self.assertEqual(Stage.STORYBOARD_REVIEW, session.stage)
            self.assertFalse(session.storyboard.confirmed)
            self.assertNotEqual(original, session.storyboard.shots[0].action)
            session.undo_artifact("storyboard")
            self.assertEqual(original, session.storyboard.shots[0].action)
            session.redo_artifact("storyboard")
            self.assertEqual("主角突然停步并回头。", session.storyboard.shots[0].action)

    def test_dynamic_storyboard_counts_and_quality_metrics(self) -> None:
        expected = {30: 5, 45: 8, 60: 10}
        with tempfile.TemporaryDirectory() as temp_dir:
            for seconds, count in expected.items():
                result = run_selfplay(
                    agent=RuleBasedStoryAgent(), target_seconds=seconds, max_turns=12,
                    output_dir=Path(temp_dir) / str(seconds),
                )
                storyboard = result["session"].storyboard
                self.assertEqual(count, len(storyboard.shots))
                self.assertEqual(seconds, storyboard.total_duration)
                self.assertTrue(all(3 <= shot.duration <= 15 for shot in storyboard.shots))
                bench = result["bench"]
                for key in (
                    "question_repetition_rate", "user_fact_retention",
                    "conflict_resolution_rate", "causal_completeness",
                    "visual_anchor_coverage", "shot_diversity",
                ):
                    self.assertIn(key, bench)

    def test_require_live_text_rejects_offline_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "真实文本 API"):
                run_selfplay(
                    agent=RuleBasedStoryAgent(), target_seconds=30, max_turns=12,
                    output_dir=temp_dir, require_live_text=True,
                )

    def test_unconfigured_online_agent_falls_back_without_recursion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_selfplay(
                agent=OpenAIStoryAgent(None, "offline-test"),
                target_seconds=30,
                max_turns=12,
                output_dir=temp_dir,
            )
            self.assertEqual(Stage.RENDER_READY, result["session"].stage)
            self.assertGreater(result["bench"]["text_fallback_count"], 0)

    def test_schema_v2_save_and_v1_read_only_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_path = Path(temp_dir) / "v1.json"
            old_path.write_text(
                json.dumps(
                    {
                        "stage": "collecting",
                        "brief": {"target_seconds": 30},
                        "facts": {"opening": "旧版开头", "ending": "旧版结局"},
                        "contributions": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            session = GuidedStorySession.load(old_path, agent=RuleBasedStoryAgent())
            self.assertEqual("旧版开头", session.facts.opening)
            new_path = Path(temp_dir) / "v2.json"
            session.save(new_path)
            saved = json.loads(new_path.read_text(encoding="utf-8"))
            self.assertEqual(2, saved["schema_version"])
            self.assertEqual("旧版开头", json.loads(old_path.read_text(encoding="utf-8"))["facts"]["opening"])

    def test_scripted_manual_cli_reaches_render_gate_without_api(self) -> None:
        commands = iter(
            [
                "暴雨夜，邮差在车站收到未来的信。",
                "邮差必须在午夜前找到收信人。",
                "管理员阻止主角进入站台。",
                "随后停驶列车突然重新启动。",
                "最后邮差烧掉信并救下乘客。",
                "/outline", "/confirm",
                "主角穿深蓝邮差制服和旧帽子。",
                "故事发生在冷色雨夜车站和列车内。",
                "关键道具是湿信封、怀表和红色制动杆。",
                "第一人称克制旁白，少量短对白。",
                "用怀表特写承接镜头，结尾呼应空站台。",
                "/confirm", "/confirm", "/confirm", "/confirm",
                "/render", "/quit",
            ]
        )
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            session = run_interactive(
                agent=RuleBasedStoryAgent(), target_seconds=30, output_dir=temp_dir,
                input_fn=lambda _: next(commands), output_fn=messages.append,
            )
            self.assertEqual(Stage.RENDER_READY, session.stage)
            self.assertIsNone(session.render_manifest)
            self.assertTrue((Path(temp_dir) / "session.json").is_file())
            self.assertTrue(any("付费调用保持关闭" in item for item in messages))


if __name__ == "__main__":
    unittest.main()
