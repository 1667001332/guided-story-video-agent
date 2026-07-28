from __future__ import annotations

import unittest

from guided_story_agent.models import StoryFacts, StoryScene, StoryScript, to_plain_data
from guided_story_agent.storyboard import build_storyboard, fit_scenes_to_duration
from guided_story_agent.timing import (
    allocate_durations,
    allocate_weighted_durations,
    estimate_story_duration,
)


class TimingTests(unittest.TestCase):
    def test_scene_budget_never_merges_different_locations(self) -> None:
        scenes = [
            StoryScene(
                index,
                f"场景{index}",
                f"地点{index}",
                "夜",
                ["主角"],
                f"主角完成动作{index}",
                "",
                3,
            )
            for index in range(1, 7)
        ]
        with self.assertRaisesRegex(ValueError, "跨地点"):
            fit_scenes_to_duration(scenes, 15)

    def test_director_plan_only_inherits_frame_for_explicit_continuous_action(self) -> None:
        script = StoryScript(
            "递信",
            15,
            [
                StoryScene(
                    1,
                    "递信",
                    "雨夜站台",
                    "夜",
                    ["邮差"],
                    "邮差举起信封，随后把它递进窗口",
                    "",
                    15,
                    visible_action="邮差举起信封，随后把它递进窗口",
                )
            ],
            confirmed=True,
        )
        director_plan = [
            {
                "scene_id": 1,
                "kind": "action",
                "action": "邮差举起信封",
                "purpose": "建立动作起点",
                "transition_type": "same_scene_cut",
                "transition_reason": "换到近景看清信封",
                "inherit_previous_frame": False,
            },
            {
                "scene_id": 1,
                "kind": "action",
                "action": "随后把信封递进窗口",
                "purpose": "完成同一个递交动作",
                "transition_type": "continuous_action",
                "transition_reason": "同一物理动作的直接下一阶段",
                "inherit_previous_frame": True,
            },
        ]
        plan = build_storyboard(
            script,
            StoryFacts(),
            director_plan=director_plan,
        )
        self.assertFalse(plan.shots[0].inherit_previous_frame)
        self.assertTrue(plan.shots[1].inherit_previous_frame)
        self.assertEqual("same_scene_chain", plan.shots[1].continuity_mode)

        director_plan[0]["inherit_previous_frame"] = True
        with self.assertRaisesRegex(ValueError, "连续动作镜头"):
            build_storyboard(script, StoryFacts(), director_plan=director_plan)

        cut_script = StoryScript(
            "同地跳时",
            15,
            [
                StoryScene(
                    1,
                    "第一次等待",
                    "同一站台",
                    "夜",
                    ["邮差"],
                    "邮差等候列车",
                    "",
                    7,
                ),
                StoryScene(
                    2,
                    "数小时后",
                    "同一站台",
                    "夜",
                    ["邮差"],
                    "邮差从长椅上醒来",
                    "",
                    8,
                ),
            ],
            confirmed=True,
        )
        cut_plan = build_storyboard(
            cut_script,
            StoryFacts(),
            director_plan=[
                {
                    "scene_id": 1,
                    "kind": "action",
                    "action": "邮差等候列车",
                    "purpose": "交代第一次等待",
                    "transition_type": "same_scene_cut",
                    "transition_reason": "普通换机位",
                    "inherit_previous_frame": False,
                },
                {
                    "scene_id": 2,
                    "kind": "action",
                    "action": "数小时后邮差从长椅上醒来",
                    "purpose": "明确同地时间跳跃",
                    "transition_type": "scene_change",
                    "transition_reason": "地点相同但叙事时间已经跳跃",
                    "inherit_previous_frame": False,
                },
            ],
        )
        self.assertEqual("scene_change", cut_plan.shots[1].transition_type)
        self.assertIsNone(cut_plan.shots[1].previous_shot_id)

    def test_story_duration_estimate_scales_and_stays_within_bounds(self) -> None:
        short = estimate_story_duration("女孩发现一封没有署名的信。她把信交给老师。")
        long = estimate_story_duration(
            "。".join(f"事件{i}推动人物作出新的选择" for i in range(30)),
            character_count=4,
            location_count=5,
        )
        self.assertTrue(15 <= short < long <= 300)

    def test_30_45_60_second_plans_are_exact(self) -> None:
        for target in (30, 45, 60):
            script = StoryScript(
                f"{target}秒",
                target,
                [
                    StoryScene(
                        1,
                        "交接证据",
                        "雨夜车站",
                        "雨夜",
                        ["邮差", "警探"],
                        "邮差取出怀表，穿过站台，随后把怀表交给警探",
                        "末班车进站前，警探必须确认怀表的来源。",
                        target,
                        dialogue="我只剩这一班车的时间说明真相。",
                        props=["铜怀表"],
                        visible_action="邮差取出怀表，穿过站台，随后把怀表交给警探",
                        start_state="邮差站在雨中，右手握着铜怀表",
                        end_state="铜怀表已经交到警探手中",
                        emotional_change="邮差从迟疑转为决绝",
                    )
                ],
                confirmed=True,
            )
            plan = build_storyboard(script, StoryFacts(character_visuals="邮差：深蓝制服"))
            self.assertEqual(target, plan.total_duration)
            self.assertTrue(all(3 <= shot.duration <= 15 for shot in plan.shots))
            self.assertTrue(
                all(f"在{shot.duration}秒内" in shot.motion_prompt for shot in plan.shots)
            )

    def test_impossible_duration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allocate_durations(20, 10)

    def test_weighted_allocation_is_deterministic_and_not_front_loaded(self) -> None:
        first = allocate_weighted_durations(
            31,
            [1, 1, 1, 1, 1],
            keys=["a", "b", "c", "d", "e"],
        )
        second = allocate_weighted_durations(
            31,
            [1, 1, 1, 1, 1],
            keys=["a", "b", "c", "d", "e"],
        )
        self.assertEqual(first, second)
        self.assertEqual(31, sum(first))
        self.assertNotEqual(7, first[0])
        self.assertIn(7, first)

    def test_long_dialogue_and_complex_action_outweigh_simple_detail(self) -> None:
        scene = StoryScene(
            1,
            "供出真相",
            "审讯室",
            "深夜",
            ["证人", "警探"],
            "证人取出钥匙，打开铁盒，翻出照片，随后把照片推给警探",
            "",
            45,
            dialogue=(
                "我看见他在十一点四十分进入仓库，随后关掉所有灯，"
                "又把这张照片塞进铁盒。他威胁我，如果说出去就再也见不到家人。"
            ),
            props=["钥匙", "铁盒", "照片"],
            visible_action="证人取出钥匙，打开铁盒，翻出照片，随后把照片推给警探",
            start_state="铁盒锁着，照片仍在盒内",
            end_state="照片停在警探手边",
            emotional_change="证人从恐惧转为释然",
        )
        plan = build_storyboard(
            StoryScript("供词", 45, [scene], confirmed=True),
            StoryFacts(),
        )
        dialogue = next(shot for shot in plan.shots if shot.shot_kind == "dialogue")
        details = [shot for shot in plan.shots if shot.shot_kind == "detail"]
        simple_detail = min(details, key=lambda shot: shot.duration_weight)
        self.assertGreater(dialogue.duration_weight, simple_detail.duration_weight)
        self.assertGreater(dialogue.duration, simple_detail.duration)

    def test_same_storyboard_input_is_fully_deterministic(self) -> None:
        script = StoryScript(
            "重复生成",
            30,
            [
                StoryScene(
                    1,
                    "奔跑",
                    "雨夜街道",
                    "雨夜",
                    ["女孩"],
                    "女孩越过水洼，推开铁门，冲进亮灯的门厅",
                    "她必须赶在钟声结束前送到信。",
                    30,
                    visible_action="女孩越过水洼，推开铁门，冲进亮灯的门厅",
                    emotional_change="焦急转为如释重负",
                )
            ],
            confirmed=True,
        )
        first = build_storyboard(script, StoryFacts())
        second = build_storyboard(script, StoryFacts())
        self.assertEqual(to_plain_data(first), to_plain_data(second))

    def test_formal_storyboard_advances_weather_clothing_and_injury_state(self) -> None:
        script = StoryScript(
            "雨中伤员",
            30,
            [
                StoryScene(
                    1,
                    "包扎",
                    "雨夜站台",
                    "雨夜",
                    ["邮差"],
                    "邮差按住流血的左臂，拿起绷带完成包扎",
                    "",
                    30,
                    props=["绷带"],
                    visible_action="邮差按住流血的左臂，拿起绷带完成包扎",
                    start_state="邮差左臂受伤流血，绷带在长椅上",
                    end_state="绷带缠在邮差左臂，伤口不再流血",
                )
            ],
            confirmed=True,
        )
        plan = build_storyboard(
            script,
            StoryFacts(character_visuals="邮差：短发，深蓝制服，左臂受伤"),
        )
        self.assertTrue(all(shot.continuity_start_state.weather == "雨" for shot in plan.shots))
        self.assertIn("邮差", plan.shots[0].continuity_start_state.character_clothing)
        self.assertIn("邮差", plan.shots[0].continuity_start_state.character_injuries)
        for previous, current in zip(plan.shots, plan.shots[1:]):
            self.assertEqual(
                to_plain_data(previous.continuity_end_state),
                to_plain_data(current.continuity_start_state),
            )

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

    def test_same_scene_normally_cuts_camera_without_inheriting_last_frame(self) -> None:
        script = StoryScript(
            "审讯室切镜",
            30,
            [
                StoryScene(
                    1,
                    "证据出现",
                    "审讯室",
                    "深夜",
                    ["警探", "证人"],
                    "证人把怀表推向警探，警探观察表盖上的血迹",
                    "",
                    30,
                    dialogue="这块表不是我的。",
                    props=["铜怀表"],
                    visible_action="证人把怀表推向警探，警探观察表盖上的血迹",
                    start_state="两人隔桌对坐，怀表在证人手中",
                    end_state="怀表停在警探面前",
                    emotional_change="警探从怀疑转为警觉",
                )
            ],
            confirmed=True,
        )
        plan = build_storyboard(script, StoryFacts())
        normal_cuts = [
            shot
            for shot in plan.shots
            if shot.continuity_mode == "same_scene_reference"
        ]

        self.assertTrue(normal_cuts)
        self.assertTrue(all(not shot.inherit_previous_frame for shot in normal_cuts))
        self.assertTrue(
            all("正常切镜" in shot.video_prompt for shot in normal_cuts)
        )
        self.assertTrue(
            any(shot.transition_type == "insert_shot" for shot in normal_cuts)
        )

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
