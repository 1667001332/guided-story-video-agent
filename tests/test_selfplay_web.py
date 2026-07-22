from __future__ import annotations

import asyncio
import unittest
import warnings

from guided_story_agent import RuleBasedStoryAgent, Stage
from guided_story_agent.agent import OpenAIStoryAgent
from guided_story_agent.web_app import (
    build_app,
    card_grid_payload,
    render_video_with_progress,
    start_garden_view,
)


class CreativeGardenWebTests(unittest.TestCase):
    def test_start_handler_returns_eight_cards(self) -> None:
        session, update, selection, chat, status = start_garden_view(
            "校园悬疑", 30, RuleBasedStoryAgent()
        )
        self.assertEqual(8, len(card_grid_payload(session)["cards"]))
        self.assertEqual([], session.selected_idea_ids)
        self.assertIn("暂无", selection)
        self.assertEqual(2, len(chat))
        self.assertIn("离线演示模式", status)
        self.assertIsNotNone(update)

    def test_render_progress_rejects_unconfirmed_session(self) -> None:
        session, *_ = start_garden_view("校园悬疑", 30, RuleBasedStoryAgent())
        updates = list(render_video_with_progress(session, False))
        self.assertEqual(1, len(updates))
        self.assertIn("必须先确认", updates[0][2])

    def test_web_explicitly_labels_offline_fallback(self) -> None:
        session, _, _, _, status = start_garden_view(
            "情人节杀人案", 30, OpenAIStoryAgent(None, "offline-test")
        )
        self.assertEqual(8, len(session.current_batch.cards))
        self.assertIn("离线兜底", status)
        self.assertIn("不是 LLM 结果", status)

    def test_app_has_native_multiselect_grid_and_no_dataframe(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = build_app()
        self.addCleanup(app.close)
        card_grids = [
            component
            for component in app.blocks.values()
            if "idea-card-grid" in (getattr(component, "elem_classes", []) or [])
        ]
        self.assertEqual(1, len(card_grids))
        names = {component.__class__.__name__ for component in app.blocks.values()}
        self.assertNotIn("Dataframe", names)
        api_names = {dependency.api_name for dependency in app.fns.values() if dependency.api_name}
        for name in (
            "start_ideation",
            "select_ideas",
            "mix_selected",
            "auto_choose",
            "generate_draft",
            "back_to_ideas",
            "render_video",
        ):
            self.assertIn(name, api_names)

    def test_process_api_one_sentence_select_and_draft(self) -> None:
        from gradio.state_holder import SessionState

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = build_app()
        self.addCleanup(app.close)
        state = SessionState(app)
        indexes = {
            dependency.api_name: index
            for index, dependency in app.fns.items()
            if dependency.api_name
        }

        async def process() -> None:
            started = await app.process_api(
                indexes["start_ideation"], ["校园悬疑", 30], state=state
            )
            grid = started["data"][1]
            self.assertEqual(8, len(grid["choices"]))
            first_id = grid["choices"][0][1]
            selected = await app.process_api(
                indexes["select_ideas"], [None, [first_id]], state=state
            )
            self.assertIn(first_id, selected["data"][1]["value"])
            mixed = await app.process_api(indexes["mix_selected"], [None], state=state)
            self.assertEqual(8, len(mixed["data"][1]["choices"]))
            auto = await app.process_api(indexes["auto_choose"], [None], state=state)
            self.assertEqual(1, len(auto["data"][1]["value"]))
            drafted = await app.process_api(indexes["generate_draft"], [None], state=state)
            self.assertIn("场景 1", drafted["data"][1])
            state_id = app.fns[indexes["generate_draft"]].inputs[0]._id
            active = state[state_id]
            self.assertEqual(Stage.DRAFT_REVIEW, active.stage)
            await app.process_api(indexes["back_to_ideas"], [None], state=state)
            self.assertIsNotNone(active.draft)

        asyncio.run(process())

    def test_process_api_paid_gate_does_not_call_provider(self) -> None:
        from gradio.state_holder import SessionState

        app = build_app()
        self.addCleanup(app.close)
        state = SessionState(app)
        indexes = {
            dependency.api_name: index
            for index, dependency in app.fns.items()
            if dependency.api_name
        }

        async def process() -> None:
            await app.process_api(indexes["start_ideation"], ["雨夜车站", 30], state=state)
            await app.process_api(indexes["auto_choose"], [None], state=state)
            await app.process_api(indexes["generate_draft"], [None], state=state)
            await app.process_api(indexes["build_storyboard"], [None], state=state)
            await app.process_api(indexes["confirm_storyboard"], [None], state=state)
            blocked = await app.process_api(indexes["render_video"], [None, False], state=state)
            self.assertIn("费用确认", blocked["data"][2])

        asyncio.run(process())


if __name__ == "__main__":
    unittest.main()
