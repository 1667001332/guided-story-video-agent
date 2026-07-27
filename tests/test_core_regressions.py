from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guided_story_agent import CreativeBrief, GuidedStorySession, RuleBasedStoryAgent, Stage
from guided_story_agent.agent import OpenAIStoryAgent
from guided_story_agent.cli import LiveTextRequiredError, main as cli_main, run_interactive
from guided_story_agent.models import (
    RenderManifest,
    StoryScene,
    StoryScript,
    VideoArtifact,
)


def render_ready_session(seconds: int = 30) -> GuidedStorySession:
    session = GuidedStorySession(
        CreativeBrief(target_seconds=seconds),
        RuleBasedStoryAgent(),
    )
    session.start_ideation("雨夜车站悬疑")
    session.generate_story()
    session.confirm_story()
    session.generate_script()
    session.confirm_script()
    session.build_storyboard()
    session.confirm_storyboard()
    return session


class CoreStateRegressionTests(unittest.TestCase):
    def test_storyboard_patch_is_atomic_and_valid_patch_invalidates_render(self) -> None:
        session = render_ready_session()
        original_prompt = session.storyboard.shots[0].video_prompt

        with self.assertRaisesRegex(ValueError, "不支持"):
            session.update_storyboard_shot(
                1,
                {
                    "video_prompt": "未经确认的新提示词",
                    "not_a_field": "boom",
                },
            )

        self.assertEqual(original_prompt, session.storyboard.shots[0].video_prompt)
        self.assertTrue(session.storyboard.confirmed)
        self.assertEqual(Stage.RENDER_READY, session.stage)

        session.update_storyboard_shot(1, {"action": "重新设计的镜头动作"})
        self.assertFalse(session.storyboard.confirmed)
        self.assertEqual(Stage.STORYBOARD_REVIEW, session.stage)
        self.assertIsNone(session.render_manifest)

    def test_script_patch_is_atomic_and_clears_all_downstream_state(self) -> None:
        session = render_ready_session()
        original_action = session.script.scenes[0].visible_action

        with self.assertRaisesRegex(ValueError, "不支持"):
            session.update_script_scene(
                1,
                {"visible_action": "未经确认的新动作", "not_a_field": "boom"},
            )

        self.assertEqual(original_action, session.script.scenes[0].visible_action)
        self.assertTrue(session.script.confirmed)
        self.assertIsNotNone(session.storyboard)

        session.update_script_scene(1, {"visible_action": "主角把信封放到站台长椅上。"})
        self.assertFalse(session.script.confirmed)
        self.assertEqual(Stage.SCRIPT_REVIEW, session.stage)
        self.assertIsNone(session.storyboard)
        self.assertIsNone(session.render_manifest)

    def test_render_lock_blocks_mutation_and_rechecks_plan_signature(self) -> None:
        session = render_ready_session()
        observed: dict[str, object] = {}

        class LockedRenderer:
            def render(self, plan, output_dir):
                observed["in_progress"] = session.render_in_progress
                try:
                    session.update_storyboard_shot(1, {"action": "并发修改"})
                except RuntimeError as exc:
                    observed["error"] = str(exc)
                return RenderManifest(status="succeeded", output_dir=str(output_dir))

        session.render_confirmed_plan(LockedRenderer(), "outputs")
        self.assertTrue(observed["in_progress"])
        self.assertIn("视频生成正在进行", str(observed["error"]))
        self.assertFalse(session.render_in_progress)
        self.assertEqual(Stage.COMPLETED, session.stage)

        changed = render_ready_session()

        class MutatingRenderer:
            def render(self, plan, output_dir):
                plan.shots[0].video_prompt = "绕过 Session 的直接修改"
                return RenderManifest(status="succeeded", output_dir=str(output_dir))

        with self.assertRaisesRegex(RuntimeError, "确认计划发生变化"):
            changed.render_confirmed_plan(MutatingRenderer(), "outputs")
        self.assertFalse(changed.render_in_progress)
        self.assertFalse(changed.storyboard.confirmed)
        self.assertEqual(Stage.STORYBOARD_REVIEW, changed.stage)
        self.assertIsNone(changed.render_manifest)

    def test_renderer_exception_keeps_remote_task_id_and_blocks_mutation(self) -> None:
        session = render_ready_session()

        class Renderer:
            def render(self, plan, output_dir):
                shot = plan.shots[0]
                plan.artifacts.append(
                    VideoArtifact(
                        artifact_id="pending-after-error",
                        shot_id=shot.shot_id,
                        provider="fake",
                        model="fake-model",
                        status="pending",
                        local_path="",
                        remote_url="",
                        duration=shot.duration,
                        prompt=shot.video_prompt,
                        created_at="now",
                        request_id="remote-job-after-error",
                    )
                )
                raise OSError("manifest write failed")

        with self.assertRaisesRegex(OSError, "manifest write failed"):
            session.render_confirmed_plan(Renderer(), "outputs")

        self.assertEqual(
            "remote-job-after-error",
            session.storyboard.artifacts[0].request_id,
        )
        with self.assertRaisesRegex(RuntimeError, "重复付费"):
            session.back_to_ideation()

    def test_upstream_selection_is_locked_until_back_then_cascades(self) -> None:
        session = render_ready_session()
        idea_id = session.current_batch.cards[1].idea_id
        with self.assertRaisesRegex(RuntimeError, "返回灵感区"):
            session.select_ideas([idea_id])
        self.assertIsNotNone(session.storyboard)
        self.assertEqual(Stage.RENDER_READY, session.stage)

        session.back_to_ideation()
        session.select_ideas([idea_id])
        self.assertEqual([idea_id], session.selected_idea_ids)
        self.assertIsNone(session.story)
        self.assertIsNone(session.script)
        self.assertIsNone(session.storyboard)
        self.assertIsNone(session.render_manifest)

    def test_failed_start_and_refresh_leave_session_unchanged(self) -> None:
        class ToggleAgent(RuleBasedStoryAgent):
            fail = False

            def generate_ideas(self, *args, **kwargs):
                if self.fail:
                    raise RuntimeError("offline failure")
                return super().generate_ideas(*args, **kwargs)

        agent = ToggleAgent()
        session = GuidedStorySession(agent=agent)
        session.start_ideation("旧方向")
        session.generate_story()
        before = session.to_dict()

        agent.fail = True
        with self.assertRaisesRegex(RuntimeError, "offline failure"):
            session.start_ideation("新方向")
        self.assertEqual(before, session.to_dict())

        session.back_to_ideation()
        before_refresh = session.to_dict()
        with self.assertRaisesRegex(RuntimeError, "offline failure"):
            session.refresh_ideas()
        self.assertEqual(before_refresh, session.to_dict())

        agent.fail = False
        session.start_ideation("真正的新项目")
        self.assertEqual({}, session.revisions)
        self.assertEqual([], session.draft_history)

    def test_loaded_state_rejects_future_schema_string_bool_and_broken_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session.json"
            data = render_ready_session().to_dict()

            future = dict(data)
            future["schema_version"] = GuidedStorySession.schema_version + 1
            path.write_text(json.dumps(future, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不支持"):
                GuidedStorySession.load(path)

            string_bool = json.loads(json.dumps(data))
            string_bool["storyboard"]["confirmed"] = "false"
            path.write_text(json.dumps(string_bool, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "布尔值"):
                GuidedStorySession.load(path)

            broken = json.loads(json.dumps(data))
            broken["script"]["confirmed"] = False
            path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "确认链"):
                GuidedStorySession.load(path)


class CoreGenerationRegressionTests(unittest.TestCase):
    def test_dense_script_scenes_are_merged_without_losing_actions(self) -> None:
        class DenseScriptAgent(RuleBasedStoryAgent):
            def generate_script(self, story, target_seconds):
                protagonist = story.characters[0].name
                return StoryScript(
                    title=story.title,
                    target_seconds=target_seconds,
                    scenes=[
                        StoryScene(
                            scene_id=index,
                            title=f"场景 {index}",
                            location="同一车站",
                            time_of_day="夜",
                            characters=[protagonist],
                            action=f"动作{index}",
                            visible_action=f"动作{index}",
                            narration="",
                            duration=1,
                        )
                        for index in range(1, 11)
                    ],
                )

        session = GuidedStorySession(
            CreativeBrief(target_seconds=15),
            DenseScriptAgent(),
        )
        session.start_ideation("雨夜车站")
        session.generate_story()
        session.confirm_story()
        script = session.generate_script()

        self.assertEqual(5, len(script.scenes))
        merged_text = "\n".join(scene.visible_action for scene in script.scenes)
        for index in range(1, 11):
            self.assertIn(f"动作{index}", merged_text)
        self.assertTrue(all(scene.duration >= 3 for scene in script.scenes))

        session.confirm_script()
        plan = session.build_storyboard()
        shot_text = "\n".join(shot.action for shot in plan.shots)
        for index in range(1, 11):
            self.assertIn(f"动作{index}", shot_text)

    def test_character_and_turning_point_choices_become_verifiable_constraints(self) -> None:
        session = GuidedStorySession(agent=RuleBasedStoryAgent())
        session.start_ideation("一件旧收音机寻找主人")
        palette = session.expand_selected()
        character = palette.options["character"][3]
        turning = palette.options["turning_point"][1]
        session.choose_element("character", character.option_id)
        session.choose_element("turning_point", turning.option_id)

        story = session.generate_story()
        self.assertIn(character.content, story.story_text)
        self.assertIn(character.content, story.characters[0].description)
        self.assertIn(turning.content, story.story_text)
        self.assertEqual(
            "selected_element",
            story.field_sources["turning_point"].source_type,
        )

    def test_empty_model_cards_report_whole_fallback(self) -> None:
        class Completions:
            def create(self, **request):
                message = SimpleNamespace(content='{"cards": []}')
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        agent = OpenAIStoryAgent(client, "fake-model")
        batch = agent.generate_ideas("雨夜车站", round_number=1)
        self.assertEqual(8, len(batch.cards))
        self.assertTrue(agent.last_used_fallback)
        self.assertEqual("whole", agent.last_fallback_kind)
        self.assertEqual(1, agent.fallback_count)

    def test_json_extractor_returns_first_complete_object(self) -> None:
        content = '说明 {"first": 1} 后续还有 {"second": 2}'
        self.assertEqual(
            {"first": 1},
            OpenAIStoryAgent._extract_json_object(content),
        )


class CoreCliRegressionTests(unittest.TestCase):
    def test_cli_rejects_zero_index_and_choose_before_expand_without_crashing(self) -> None:
        commands = iter(["雨夜车站", "/pick 0", "/more -1", "/choose ending 1", "/quit"])
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            session = run_interactive(
                agent=RuleBasedStoryAgent(),
                target_seconds=30,
                output_dir=temp_dir,
                input_fn=lambda _: next(commands),
                output_fn=messages.append,
            )
        self.assertEqual(Stage.IDEATING, session.stage)
        self.assertTrue(any("序号必须在 1 到 8" in item for item in messages))
        self.assertTrue(any("请先输入 /expand" in item for item in messages))

    def test_require_live_text_treats_partial_or_whole_fill_as_failure(self) -> None:
        class Completions:
            def create(self, **request):
                message = SimpleNamespace(content='{"cards": []}')
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        agent = OpenAIStoryAgent(client, "fake-model")
        with self.assertRaisesRegex(LiveTextRequiredError, "whole本地降级"):
            run_interactive(
                agent=agent,
                target_seconds=30,
                require_live_text=True,
                input_fn=lambda _: "雨夜车站",
                output_fn=lambda _: None,
            )

    def test_cli_main_exits_nonzero_when_live_text_is_unavailable(self) -> None:
        with (
            patch.object(sys, "argv", ["guided-story-cli", "--require-live-text"]),
            patch(
                "guided_story_agent.cli.OpenAIStoryAgent.from_env",
                return_value=OpenAIStoryAgent(None, "offline"),
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            cli_main()
        self.assertNotEqual(0, raised.exception.code)


if __name__ == "__main__":
    unittest.main()
