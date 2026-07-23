from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guided_story_agent import CreativeBrief, GuidedStorySession, RuleBasedStoryAgent, Stage
from guided_story_agent.models import RenderManifest


def started(seconds: int = 45) -> GuidedStorySession:
    session = GuidedStorySession(CreativeBrief(target_seconds=seconds), RuleBasedStoryAgent())
    session.start_ideation("校园里发生一件带点悬疑的事")
    return session


class RepairableScriptAgent(RuleBasedStoryAgent):
    def generate_script(self, story, target_seconds):
        script = super().generate_script(story, target_seconds)
        for scene in script.scenes:
            scene.duration = 3
        script.scenes[0].action = ""
        script.scenes[0].visible_action = ""
        return script


class GuidedStorySessionV4Tests(unittest.TestCase):
    def test_direction_immediately_creates_eight_distinct_cards(self) -> None:
        session = started()
        self.assertEqual(Stage.IDEATING, session.stage)
        self.assertEqual(8, len(session.current_batch.cards))
        self.assertEqual(8, len({card.fingerprint for card in session.current_batch.cards}))
        self.assertEqual(1, session.free_text_count)

    def test_murder_mystery_fallback_is_specific_not_template_substitution(self) -> None:
        session = GuidedStorySession(agent=RuleBasedStoryAgent())
        session.start_ideation("情人节杀人案")
        cards = session.current_batch.cards
        self.assertEqual(8, len(cards))
        self.assertTrue(all("把“情人节杀人案”变成" not in card.logline for card in cards))
        self.assertEqual(8, len({card.protagonist for card in cards}))
        self.assertIn("死后送达的玫瑰", {card.title for card in cards})
        self.assertTrue(all(len(card.central_conflict) >= 25 for card in cards))

    def test_one_sentence_can_generate_story_then_script_without_selection(self) -> None:
        session = started(30)
        story = session.generate_story()
        self.assertEqual(Stage.STORY_REVIEW, session.stage)
        self.assertGreater(len(story.story_text), 120)
        with self.assertRaisesRegex(RuntimeError, "确认完整故事"):
            session.generate_script()
        session.confirm_story()
        script = session.generate_script()
        self.assertEqual(Stage.SCRIPT_REVIEW, session.stage)
        self.assertEqual(30, script.total_duration)
        self.assertGreater(len(story.ai_filled_fields), 0)
        self.assertTrue(all(field in story.field_sources for field in story.ai_filled_fields))

    def test_duration_and_empty_action_are_repaired_without_questioning_user(self) -> None:
        session = GuidedStorySession(CreativeBrief(target_seconds=45), RepairableScriptAgent())
        session.start_ideation("一个发生在天台的温暖故事")
        session.generate_story()
        session.confirm_story()
        script = session.generate_script()
        self.assertEqual(45, script.total_duration)
        self.assertTrue(script.scenes[0].visible_action)

    def test_selects_one_to_three_and_preserves_all_sources(self) -> None:
        session = started()
        ids = [card.idea_id for card in session.current_batch.cards[:3]]
        session.select_ideas(ids)
        story = session.generate_story()
        self.assertEqual(set(ids), set(story.field_sources["selected_ideas"].source_ids))
        with self.assertRaisesRegex(ValueError, "最多"):
            session.select_ideas([card.idea_id for card in session.current_batch.cards[:4]])

    def test_more_like_and_mix_keep_sources(self) -> None:
        session = started()
        anchor = session.current_batch.cards[1]
        similar = session.more_like(anchor.idea_id)
        self.assertEqual(8, len(similar.cards))
        self.assertTrue(all(anchor.idea_id in card.source_idea_ids for card in similar.cards))
        selected = [card.idea_id for card in similar.cards[:2]]
        session.select_ideas(selected)
        mixed = session.mix_selected()
        self.assertTrue(all(set(card.source_idea_ids) == set(selected) for card in mixed.cards))

    def test_refresh_does_not_repeat_previous_batch(self) -> None:
        session = started()
        seen: set[str] = set()
        for _ in range(5):
            current = {card.fingerprint for card in session.current_batch.cards}
            self.assertTrue(seen.isdisjoint(current))
            seen.update(current)
            session.refresh_ideas()

    def test_elements_are_optional_four_by_four_and_explicit_choice_wins(self) -> None:
        session = started()
        palette = session.expand_selected()
        self.assertEqual(
            {"character", "conflict", "turning_point", "ending"},
            set(palette.options),
        )
        self.assertTrue(all(len(options) == 4 for options in palette.options.values()))
        ending = palette.options["ending"][2]
        session.choose_element("ending", ending.option_id)
        story = session.generate_story()
        self.assertEqual(ending.content, story.ending)
        self.assertEqual("selected_element", story.field_sources["ending"].source_type)

    def test_story_versions_survive_return_to_ideas(self) -> None:
        session = started()
        first = session.generate_story()
        session.back_to_ideation()
        self.assertIs(first, session.story)
        session.refresh_ideas()
        session.auto_choose()
        second = session.generate_story()
        self.assertEqual(2, second.version)
        self.assertEqual(2, len(session.story_history))

    def test_story_revision_cannot_silently_change_selected_card(self) -> None:
        session = started()
        selected = session.current_batch.cards[0]
        session.select_ideas([selected.idea_id])
        session.generate_story()
        revised = session.revise_story("让结局更安静")
        self.assertEqual(selected.ending_direction, revised.ending)
        self.assertEqual([selected.idea_id], revised.field_sources["selected_ideas"].source_ids)

    def test_duration_and_dynamic_storyboards(self) -> None:
        scene_counts = set()
        shot_counts = set()
        for seconds in (30, 45, 60):
            session = started(seconds)
            session.generate_story()
            session.confirm_story()
            session.generate_script()
            scene_counts.add(len(session.script.scenes))
            session.confirm_script()
            storyboard = session.build_storyboard()
            shot_counts.add(len(storyboard.shots))
            self.assertEqual(seconds, storyboard.total_duration)
            self.assertTrue(all(3 <= shot.duration <= 15 for shot in storyboard.shots))
            self.assertTrue(storyboard.visual_bible.assets)
            self.assertTrue(all(shot.first_frame_prompt for shot in storyboard.shots))
            self.assertTrue(all(shot.motion_prompt for shot in storyboard.shots))
            self.assertTrue(all(shot.end_frame_prompt for shot in storyboard.shots))
        self.assertGreater(len(scene_counts), 1)
        self.assertGreater(len(shot_counts), 1)

    def test_render_is_blocked_until_confirmation(self) -> None:
        session = started()

        class SpyRenderer:
            def render(self, plan, output_dir):
                raise AssertionError("renderer must not be called")

        with self.assertRaises(RuntimeError):
            session.render_confirmed_plan(SpyRenderer(), "outputs")

    def test_successful_render_moves_to_completed(self) -> None:
        session = started(30)
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        session.build_storyboard()
        session.confirm_storyboard()

        class Renderer:
            def render(self, plan, output_dir):
                return RenderManifest(status="succeeded", output_dir=str(output_dir))

        session.render_confirmed_plan(Renderer(), "outputs")
        self.assertEqual(Stage.COMPLETED, session.stage)

    def test_schema_v4_roundtrip_and_v2_read_only_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = started()
            session.auto_choose()
            session.generate_story()
            session.confirm_story()
            session.generate_script()
            session.confirm_script()
            storyboard = session.build_storyboard()
            path = session.save(Path(temp_dir) / "v4.json")
            loaded = GuidedStorySession.load(path, agent=RuleBasedStoryAgent())
            self.assertEqual(4, json.loads(path.read_text(encoding="utf-8"))["schema_version"])
            self.assertEqual(session.direction, loaded.direction)
            self.assertEqual(session.selected_idea_ids, loaded.selected_idea_ids)
            self.assertEqual(session.story.story_text, loaded.story.story_text)
            self.assertEqual(len(session.script.scenes), len(loaded.script.scenes))
            self.assertEqual(
                storyboard.visual_bible.assets[0].asset_id,
                loaded.storyboard.visual_bible.assets[0].asset_id,
            )
            self.assertEqual(
                storyboard.shots[0].first_frame_prompt,
                loaded.storyboard.shots[0].first_frame_prompt,
            )

            old_path = Path(temp_dir) / "v2.json"
            old_data = {
                "schema_version": 2,
                "stage": "collecting",
                "brief": {"target_seconds": 30},
                "facts": {"opening": "旧版开头", "ending": "旧版结局"},
            }
            old_path.write_text(json.dumps(old_data, ensure_ascii=False), encoding="utf-8")
            migrated = GuidedStorySession.load(old_path, agent=RuleBasedStoryAgent())
            self.assertEqual("旧版开头", migrated.facts.opening)
            self.assertEqual(old_data, json.loads(old_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
