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


class RepairableDraftAgent(RuleBasedStoryAgent):
    def generate_draft(self, direction, selected_cards, selected_elements, target_seconds):
        draft = super().generate_draft(direction, selected_cards, selected_elements, target_seconds)
        for scene, beat in zip(draft.script.scenes, draft.outline.beats):
            scene.duration = 3
            beat.duration = 3
        draft.script.scenes[0].action = ""
        draft.script.scenes[0].visible_action = ""
        return draft


class GuidedStorySessionV3Tests(unittest.TestCase):
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

    def test_one_sentence_can_generate_complete_draft_without_selection(self) -> None:
        session = started(30)
        draft = session.generate_draft()
        self.assertEqual(Stage.DRAFT_REVIEW, session.stage)
        self.assertEqual(30, draft.script.total_duration)
        self.assertEqual(5, len(draft.script.scenes))
        self.assertGreater(len(draft.ai_filled_fields), 0)
        self.assertTrue(all(field in draft.field_sources for field in draft.ai_filled_fields))

    def test_duration_and_empty_action_are_repaired_without_questioning_user(self) -> None:
        session = GuidedStorySession(CreativeBrief(target_seconds=45), RepairableDraftAgent())
        session.start_ideation("一个发生在天台的温暖故事")
        draft = session.generate_draft()
        self.assertEqual(45, draft.script.total_duration)
        self.assertTrue(draft.script.scenes[0].visible_action)

    def test_selects_one_to_three_and_preserves_all_sources(self) -> None:
        session = started()
        ids = [card.idea_id for card in session.current_batch.cards[:3]]
        session.select_ideas(ids)
        draft = session.generate_draft()
        self.assertEqual(set(ids), set(draft.field_sources["selected_ideas"].source_ids))
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
        draft = session.generate_draft()
        self.assertEqual(ending.content, draft.outline.ending)
        self.assertEqual("selected_element", draft.field_sources["ending"].source_type)

    def test_draft_versions_survive_return_to_ideas(self) -> None:
        session = started()
        first = session.generate_draft()
        session.back_to_ideation()
        self.assertIs(first, session.draft)
        session.refresh_ideas()
        session.auto_choose()
        second = session.generate_draft()
        self.assertEqual(2, second.version)
        self.assertEqual(2, len(session.draft_history))

    def test_revision_cannot_silently_change_selected_card(self) -> None:
        session = started()
        selected = session.current_batch.cards[0]
        session.select_ideas([selected.idea_id])
        session.generate_draft()
        revised = session.revise_draft("让结局更安静")
        self.assertEqual(selected.ending_direction, revised.outline.ending)
        self.assertEqual([selected.idea_id], revised.field_sources["selected_ideas"].source_ids)

    def test_duration_and_dynamic_storyboards(self) -> None:
        for seconds, expected_shots in {30: 5, 45: 8, 60: 10}.items():
            session = started(seconds)
            session.generate_draft()
            session.confirm_draft()
            storyboard = session.build_storyboard()
            self.assertEqual(expected_shots, len(storyboard.shots))
            self.assertEqual(seconds, storyboard.total_duration)
            self.assertTrue(all(3 <= shot.duration <= 15 for shot in storyboard.shots))

    def test_render_is_blocked_until_confirmation(self) -> None:
        session = started()

        class SpyRenderer:
            def render(self, plan, output_dir):
                raise AssertionError("renderer must not be called")

        with self.assertRaises(RuntimeError):
            session.render_confirmed_plan(SpyRenderer(), "outputs")

    def test_successful_render_moves_to_completed(self) -> None:
        session = started(30)
        session.generate_draft()
        session.confirm_draft()
        session.build_storyboard()
        session.confirm_storyboard()

        class Renderer:
            def render(self, plan, output_dir):
                return RenderManifest(status="succeeded", output_dir=str(output_dir))

        session.render_confirmed_plan(Renderer(), "outputs")
        self.assertEqual(Stage.COMPLETED, session.stage)

    def test_schema_v3_roundtrip_and_v2_read_only_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = started()
            session.auto_choose()
            session.generate_draft()
            path = session.save(Path(temp_dir) / "v3.json")
            loaded = GuidedStorySession.load(path, agent=RuleBasedStoryAgent())
            self.assertEqual(3, json.loads(path.read_text(encoding="utf-8"))["schema_version"])
            self.assertEqual(session.direction, loaded.direction)
            self.assertEqual(session.selected_idea_ids, loaded.selected_idea_ids)

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
