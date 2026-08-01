from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guided_story_agent import CreativeBrief, GuidedStorySession, RuleBasedStoryAgent, Stage
from guided_story_agent.models import (
    RenderManifest,
    StoryCharacter,
    StoryDraft,
    StoryScene,
    StoryScript,
)


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
    def test_auto_duration_is_resolved_from_complete_story(self) -> None:
        session = GuidedStorySession(CreativeBrief(), RuleBasedStoryAgent())
        session.start_ideation("一名邮差在雨夜收到未来自己的来信")
        session.generate_story()
        session.confirm_story()
        script = session.generate_script()
        self.assertEqual("auto", session.brief.duration_mode)
        self.assertEqual(script.target_seconds, session.brief.resolved_target_seconds)
        self.assertTrue(15 <= script.target_seconds <= 300)

    def test_custom_duration_accepts_15_to_300_seconds(self) -> None:
        for seconds in (15, 90, 300):
            brief = CreativeBrief(target_seconds=seconds)
            brief.validate()
            self.assertEqual("custom", brief.duration_mode)
        for seconds in (14, 301):
            with self.assertRaisesRegex(ValueError, "15 到 300"):
                GuidedStorySession(CreativeBrief(target_seconds=seconds))

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
        session.back_to_ideation()
        with self.assertRaisesRegex(ValueError, "最多"):
            session.select_ideas([card.idea_id for card in session.current_batch.cards[:4]])

    def test_selected_protagonist_accepts_semantic_realization_without_exact_phrase(self) -> None:
        session = started()
        card = session.current_batch.cards[0]
        session.select_ideas([card.idea_id])
        card.protagonist = "年轻邮差（刚入职）"
        story = StoryDraft(
            title="测试故事",
            logline="一名新邮差在雨夜完成关键投递。",
            story_text=(
                "年轻邮差刚刚入职，在暴雨里接下旧车站的最后一封信。"
                "他沿着站台留下的线索行动，最终完成投递并承担自己的选择。"
            ),
            characters=[
                StoryCharacter(
                    name="林川",
                    description="刚刚入职的年轻邮差，对废弃车站并不熟悉。",
                )
            ],
            core_conflict=card.central_conflict,
            ending=card.ending_direction,
        )

        session._validate_selected_constraints(story, {})

    def test_script_boundary_accepts_semantic_role_without_full_card_phrase(self) -> None:
        session = started(15)
        card = session.current_batch.cards[0]
        session.select_ideas([card.idea_id])
        card.protagonist = "一个循规蹈矩的年轻邮差，深爱着生病在床的妻子。"
        session.story = StoryDraft(
            title="测试故事",
            logline="年轻邮差在雨夜作出选择。",
            story_text="年轻邮差林川为了病中的妻子，在雨夜承担了一次危险投递。",
            characters=[
                StoryCharacter(
                    name="林川",
                    description="循规蹈矩的年轻邮差，深爱病中的妻子。",
                )
            ],
            core_conflict=card.central_conflict,
            ending=card.ending_direction,
        )
        self.assertEqual(["林川"], session._required_story_characters())
        visible_action = (
            f"循规蹈矩的年轻邮差林川深爱病中的妻子，并为她完成投递；"
            f"{card.central_conflict}；"
            f"{card.ending_direction}"
        )
        script = StoryScript(
            title="测试剧本",
            target_seconds=15,
            scenes=[
                StoryScene(
                    scene_id=1,
                    title="最后投递",
                    location="旧车站",
                    time_of_day="雨夜",
                    characters=["年轻邮差林川"],
                    action=visible_action,
                    narration="",
                    visible_action=visible_action,
                    start_state="林川带着信进入旧车站。",
                    end_state=card.ending_direction,
                    duration=15,
                )
            ],
        )

        session._validate_script_story_boundary(script)

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

    def test_storyboard_gate_uses_readable_floor_not_soft_preferred_total(self) -> None:
        session = started(15)
        session.auto_choose()
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        storyboard = session.build_storyboard()

        self.assertGreater(
            sum(shot.estimated_duration for shot in storyboard.shots),
            storyboard.target_seconds,
        )
        initial_review = session.review_current_artifact("storyboard")
        self.assertTrue(initial_review.can_confirm)
        self.assertEqual(1.0, initial_review.scores["timing_budget_fit"])
        self.assertLess(initial_review.scores["timing_preferred_fit"], 1.0)

        underfunded = next(
            shot
            for shot in storyboard.shots
            if shot.minimum_readable_duration > 3
        )
        replacement_duration = underfunded.minimum_readable_duration - 1
        transferred = underfunded.duration - replacement_duration
        donor = next(
            shot
            for shot in storyboard.shots
            if shot.shot_id != underfunded.shot_id
            and shot.duration + transferred <= 15
        )
        underfunded.duration = replacement_duration
        donor.duration += transferred

        blocked_review = session.review_current_artifact("storyboard")
        self.assertFalse(blocked_review.can_confirm)
        self.assertTrue(
            any("低于内容可读下限" in error for error in blocked_review.hard_errors)
        )
        with self.assertRaisesRegex(RuntimeError, "内容可读下限"):
            session.confirm_storyboard()
        self.assertEqual(Stage.STORYBOARD_REVIEW, session.stage)
        self.assertFalse(storyboard.confirmed)

    def test_render_gate_rechecks_current_content_budget_before_provider_call(self) -> None:
        session = started(15)
        session.auto_choose()
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        storyboard = session.build_storyboard()
        session.confirm_storyboard()

        underfunded = next(
            shot
            for shot in storyboard.shots
            if shot.minimum_readable_duration > 3
        )
        replacement_duration = underfunded.minimum_readable_duration - 1
        transferred = underfunded.duration - replacement_duration
        donor = next(
            shot
            for shot in storyboard.shots
            if shot.shot_id != underfunded.shot_id
            and shot.duration + transferred <= 15
        )
        underfunded.duration = replacement_duration
        donor.duration += transferred

        class Renderer:
            called = False

            def render(self, plan, output_dir):
                self.called = True
                raise AssertionError("时长门失败后不应调用视频 Provider")

        renderer = Renderer()
        with self.assertRaisesRegex(RuntimeError, "内容可读下限"):
            session.render_confirmed_plan(renderer, ".")
        self.assertFalse(renderer.called)

    def test_long_custom_duration_is_not_capped_at_twelve_shots(self) -> None:
        session = started(300)
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        storyboard = session.build_storyboard()
        self.assertEqual(300, storyboard.total_duration)
        self.assertGreater(len(storyboard.shots), 12)
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

    def test_schema_v5_roundtrip_and_legacy_read_only_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = started()
            session.auto_choose()
            session.generate_story()
            session.confirm_story()
            session.generate_script()
            session.confirm_script()
            storyboard = session.build_storyboard()
            path = session.save(Path(temp_dir) / "v5.json")
            loaded = GuidedStorySession.load(path, agent=RuleBasedStoryAgent())
            self.assertEqual(5, json.loads(path.read_text(encoding="utf-8"))["schema_version"])
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

            v4_path = Path(temp_dir) / "v4.json"
            v4_data = json.loads(path.read_text(encoding="utf-8"))
            v4_data["schema_version"] = 4
            for shot in v4_data["storyboard"]["shots"]:
                shot.pop("dialogue", None)
                shot.pop("minimum_readable_duration", None)
                shot.pop("source_action", None)
                shot.pop("retake_instruction", None)
                shot.pop("time_of_day", None)
                shot.pop("visual_style", None)
                shot.pop("color_palette", None)
            v4_path.write_text(
                json.dumps(v4_data, ensure_ascii=False),
                encoding="utf-8",
            )
            migrated_v4 = GuidedStorySession.load(
                v4_path,
                agent=RuleBasedStoryAgent(),
            )
            self.assertEqual("", migrated_v4.storyboard.shots[0].dialogue)
            self.assertEqual(
                0,
                migrated_v4.storyboard.shots[0].minimum_readable_duration,
            )
            self.assertEqual(v4_data, json.loads(v4_path.read_text(encoding="utf-8")))

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
