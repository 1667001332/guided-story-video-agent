from __future__ import annotations

import unittest
from unittest.mock import patch

from guided_story_agent.agent import OpenAIStoryAgent, RuleBasedStoryAgent
from guided_story_agent.models import (
    StoryDraft,
    StoryFacts,
    StoryScene,
    StoryScript,
    to_plain_data,
)
from guided_story_agent.storyboard import (
    StoryboardTimingBudgetError,
    assess_director_plan_timing,
    build_storyboard,
    fit_scenes_to_duration,
)
from guided_story_agent.timing import (
    ShotTimingDemand,
    allocate_durations,
    allocate_weighted_durations,
    assess_shot_readable_minimum,
    count_sequential_action_phases,
    estimate_story_duration,
    plan_scene_durations,
)


class TimingTests(unittest.TestCase):
    @staticmethod
    def _readable_minimum(
        action: str,
        *,
        dialogue: str = "",
        narration: str = "",
    ):
        return assess_shot_readable_minimum(
            ShotTimingDemand(
                shot_kind="action",
                purpose="测试",
                priority=4,
                action=action,
                dialogue=dialogue,
                narration=narration,
                emotional_change="",
                scene_duration=15,
                scene_shot_count=1,
                narration_is_per_shot=True,
            )
        )

    def test_readable_minimum_distinguishes_sequence_parallelism_and_speech(self) -> None:
        simple = self._readable_minimum("女孩拿起信")
        concurrent = self._readable_minimum("女孩拿起信，同时窗外闪电")
        sequential = self._readable_minimum(
            "女孩拿起信，随后拆开信封，接着读完内容"
        )
        descriptive = self._readable_minimum("女孩拿起红色、破旧、带邮戳的信封")
        long_dialogue = self._readable_minimum(
            "女孩坐在桌边",
            dialogue="我已经核对了车次、时间和地址，这封信不可能来自昨天。",
        )

        self.assertGreaterEqual(
            concurrent.minimum_seconds,
            simple.minimum_seconds,
        )
        self.assertGreater(
            sequential.minimum_seconds,
            concurrent.minimum_seconds,
        )
        self.assertEqual(simple.minimum_seconds, descriptive.minimum_seconds)
        self.assertGreater(
            long_dialogue.minimum_seconds,
            simple.minimum_seconds,
        )
        self.assertEqual(
            long_dialogue.minimum_seconds,
            max(
                long_dialogue.provider_seconds,
                long_dialogue.visual_seconds,
                long_dialogue.speech_seconds,
            ),
        )

    def test_sequential_markers_share_one_conservative_parser(self) -> None:
        self.assertEqual(
            1,
            count_sequential_action_phases("女孩拿起信，同时窗外闪电"),
        )
        self.assertEqual(
            1,
            count_sequential_action_phases("女孩不再犹豫"),
        )
        self.assertEqual(
            4,
            count_sequential_action_phases(
                "主角起身，走到门口，打开门，跑出去"
            ),
        )
        self.assertEqual(
            4,
            count_sequential_action_phases(
                "主角弯腰，系好鞋带，背起书包，锁上房门"
            ),
        )
        self.assertEqual(
            3,
            count_sequential_action_phases(
                "He opens the letter. He runs outside. He falls."
            ),
        )
        self.assertEqual(
            4,
            count_sequential_action_phases(
                "He bends down, ties his shoes, shoulders his bag, locks the door"
            ),
        )
        self.assertEqual(
            2,
            count_sequential_action_phases(
                "人物完成最后一个可见动作，证据留在画面中央"
            ),
        )
        for marker in ("最后", "再", "并且", "转而"):
            with self.subTest(marker=marker):
                self.assertEqual(
                    2,
                    count_sequential_action_phases(
                        f"女孩拿起信，{marker}走向门口"
                    ),
                )

    def test_five_comma_chained_shots_cannot_collapse_to_equal_three_seconds(
        self,
    ) -> None:
        script = StoryScript(
            "逗号动作链",
            15,
            [
                StoryScene(
                    1,
                    "逃离",
                    "走廊",
                    "夜",
                    ["主角"],
                    "主角逃离走廊",
                    "",
                    15,
                    visible_action="主角逃离走廊",
                )
            ],
            confirmed=True,
        )
        director_plan = [
            {
                "scene_id": 1,
                "kind": "action",
                "action": "主角起身，走到门口，打开门，跑出去",
                "purpose": f"动作链{index}",
                "transition_type": "opening" if index == 1 else "same_scene_cut",
                "transition_reason": "推进逃离",
                "inherit_previous_frame": False,
            }
            for index in range(1, 6)
        ]
        assessment = assess_director_plan_timing(script, director_plan)

        self.assertEqual((5, 5, 5, 5, 5), assessment.minimum_durations)
        self.assertGreater(assessment.minimum_total, script.target_seconds)
        self.assertFalse(assessment.feasible)

    def test_weighted_allocation_supports_heterogeneous_content_floors(self) -> None:
        exact = allocate_weighted_durations(
            15,
            [1, 5, 2],
            minimums=[3, 5, 7],
            keys=["simple", "dialogue", "action"],
        )
        self.assertEqual([3, 5, 7], exact)

        with self.assertRaisesRegex(ValueError, "最低可读时长"):
            allocate_weighted_durations(
                15,
                [1, 5, 2],
                minimums=[3, 5, 8],
                keys=["simple", "dialogue", "action"],
            )

        allocated = allocate_weighted_durations(
            20,
            [1, 5, 2],
            minimums=[3, 5, 7],
            keys=["simple", "dialogue", "action"],
        )
        self.assertEqual(20, sum(allocated))
        self.assertTrue(
            all(
                actual >= floor
                for actual, floor in zip(allocated, [3, 5, 7])
            )
        )
        self.assertGreater(len(set(allocated)), 1)

    def test_dense_five_shot_plan_is_blocked_before_equal_three_second_allocation(
        self,
    ) -> None:
        script = StoryScript(
            "循环预言",
            15,
            [
                StoryScene(
                    1,
                    "预言与牺牲",
                    "旧车站",
                    "暴雨夜",
                    ["林远", "陈叔"],
                    "林远发现预言，试图救人，目睹事故，听完解释并牺牲自己",
                    "",
                    15,
                    dialogue=(
                        "这封信预告明天的死亡。只有烧掉它，循环才会结束，"
                        "否则下一个死去的人就是你。"
                    ),
                    visible_action=(
                        "林远发现预言，试图救人，目睹事故，听完解释并牺牲自己"
                    ),
                )
            ],
            confirmed=True,
        )
        director_plan = [
            {
                "scene_id": 1,
                "kind": "establish",
                "action": "林远推车进站，在墙边坐下，脚边露出旧邮包",
                "purpose": "发现邮包",
                "transition_type": "opening",
                "transition_reason": "开场",
                "inherit_previous_frame": False,
            },
            {
                "scene_id": 1,
                "kind": "detail",
                "action": "林远打开邮包，随后取出信封，接着展开信纸",
                "purpose": "揭示预言",
                "transition_type": "insert_shot",
                "transition_reason": "看清信件",
                "inherit_previous_frame": False,
            },
            {
                "scene_id": 1,
                "kind": "action",
                "action": "林远抓住陈叔，随后陈叔挣脱，接着广告牌坠落，最后时间重置",
                "purpose": "呈现事故",
                "transition_type": "same_scene_cut",
                "transition_reason": "推进事故",
                "inherit_previous_frame": False,
            },
            {
                "scene_id": 1,
                "kind": "dialogue",
                "action": "林远听老人解释循环规则",
                "purpose": "解释规则",
                "transition_type": "reverse_shot",
                "transition_reason": "交代信息",
                "inherit_previous_frame": False,
            },
            {
                "scene_id": 1,
                "kind": "action",
                "action": "闪电劈开裂缝，随后林远跃入，接着晨光出现，最后身影消散",
                "purpose": "完成牺牲",
                "transition_type": "same_scene_cut",
                "transition_reason": "完成结局",
                "inherit_previous_frame": False,
            },
        ]

        with self.assertRaises(StoryboardTimingBudgetError) as raised:
            build_storyboard(
                script,
                StoryFacts(),
                director_plan=director_plan,
            )

        assessment = raised.exception.assessment
        self.assertEqual(5, len(assessment.minimum_durations))
        self.assertGreater(assessment.minimum_total, script.target_seconds)
        self.assertFalse(assessment.feasible)

    def test_feasible_short_plan_keeps_content_driven_variable_durations(self) -> None:
        script = StoryScript(
            "站台真相",
            15,
            [
                StoryScene(
                    1,
                    "交出证据",
                    "雨夜站台",
                    "夜",
                    ["邮差", "警探"],
                    "邮差走近警探并交出信封",
                    "",
                    15,
                    dialogue="这封信记录了真正的到站时间，请你现在打开它。",
                    visible_action="邮差走近警探并交出信封",
                )
            ],
            confirmed=True,
        )
        plan = build_storyboard(
            script,
            StoryFacts(),
            director_plan=[
                {
                    "scene_id": 1,
                    "kind": "establish",
                    "action": "邮差站在雨夜站台入口",
                    "purpose": "建立交接空间",
                    "transition_type": "opening",
                    "transition_reason": "开场",
                    "inherit_previous_frame": False,
                },
                {
                    "scene_id": 1,
                    "kind": "dialogue",
                    "action": "邮差把信封递给警探并说出关键时间",
                    "purpose": "交代关键证据",
                    "transition_type": "reverse_shot",
                    "transition_reason": "呈现对白关系",
                    "inherit_previous_frame": False,
                },
                {
                    "scene_id": 1,
                    "kind": "reaction",
                    "action": "警探看清信封日期后抬头",
                    "purpose": "确认信息生效",
                    "transition_type": "reaction_cut",
                    "transition_reason": "呈现反应",
                    "inherit_previous_frame": False,
                },
            ],
        )

        durations = [shot.duration for shot in plan.shots]
        self.assertEqual(15, sum(durations))
        self.assertGreater(len(set(durations)), 1)
        self.assertGreater(plan.shots[1].duration, plan.shots[0].duration)
        self.assertTrue(
            all(
                shot.duration >= shot.minimum_readable_duration
                for shot in plan.shots
            )
        )

    def test_dialogue_must_use_dialogue_shots_and_is_split_without_duplication(
        self,
    ) -> None:
        script = StoryScript(
            "两封信",
            15,
            [
                StoryScene(
                    1,
                    "交代线索",
                    "站台",
                    "夜",
                    ["邮差"],
                    "邮差把两封信放到桌上",
                    "",
                    15,
                    dialogue="第一封信来自昨天。第二封信写着明天。",
                    visible_action="邮差把两封信放到桌上",
                )
            ],
            confirmed=True,
        )
        action_only = [
            {
                "scene_id": 1,
                "kind": "action",
                "action": "邮差把两封信放到桌上",
                "purpose": "展示线索",
                "transition_type": "opening",
                "transition_reason": "开场",
                "inherit_previous_frame": False,
            }
        ]
        with self.assertRaisesRegex(ValueError, "没有 dialogue 镜头"):
            assess_director_plan_timing(script, action_only)

        dialogue_plan = [
            {
                "scene_id": 1,
                "kind": "dialogue",
                "action": "邮差指向第一封信",
                "purpose": "说明第一封信",
                "transition_type": "opening",
                "transition_reason": "开场",
                "inherit_previous_frame": False,
            },
            {
                "scene_id": 1,
                "kind": "dialogue",
                "action": "邮差指向第二封信",
                "purpose": "说明第二封信",
                "transition_type": "reverse_shot",
                "transition_reason": "继续交代",
                "inherit_previous_frame": False,
            },
        ]
        plan = build_storyboard(
            script,
            StoryFacts(),
            director_plan=dialogue_plan,
        )
        assigned = [shot.dialogue for shot in plan.shots]
        self.assertTrue(all(assigned))
        self.assertTrue(all(value != script.scenes[0].dialogue for value in assigned))
        self.assertEqual(
            script.scenes[0].dialogue.replace(" ", ""),
            "".join(assigned).replace(" ", ""),
        )

        english_script = StoryScript(
            "Two letters",
            15,
            [
                StoryScene(
                    1,
                    "Explain",
                    "Platform",
                    "Night",
                    ["Courier"],
                    "The courier points to two letters.",
                    "",
                    15,
                    dialogue=(
                        "The first letter came yesterday. "
                        "The second names tomorrow."
                    ),
                    visible_action="The courier points to two letters.",
                )
            ],
            confirmed=True,
        )
        english_plan = build_storyboard(
            english_script,
            StoryFacts(),
            director_plan=dialogue_plan,
        )
        self.assertTrue(all(shot.dialogue for shot in english_plan.shots))
        self.assertEqual(
            english_script.scenes[0].dialogue.replace(" ", ""),
            "".join(shot.dialogue for shot in english_plan.shots).replace(" ", ""),
        )

    def test_single_shot_over_provider_capacity_requires_split(self) -> None:
        script = StoryScript(
            "长对白",
            15,
            [
                StoryScene(
                    1,
                    "解释",
                    "房间",
                    "夜",
                    ["讲述者"],
                    "讲述者解释全部真相",
                    "",
                    15,
                    dialogue="这是无法在一个镜头里清楚说完的长对白。" * 20,
                    visible_action="讲述者看向镜头",
                )
            ],
            confirmed=True,
        )
        director_plan = [
            {
                "scene_id": 1,
                "kind": "dialogue",
                "action": "讲述者看向镜头",
                "purpose": "解释真相",
                "transition_type": "opening",
                "transition_reason": "开场",
                "inherit_previous_frame": False,
            }
        ]
        assessment = assess_director_plan_timing(script, director_plan)

        self.assertEqual((1,), assessment.over_capacity_shots)
        self.assertFalse(assessment.feasible)

    def test_model_director_retries_once_with_timing_budget_feedback(self) -> None:
        script = StoryScript(
            "重规划",
            15,
            [
                StoryScene(
                    1,
                    "站台",
                    "车站",
                    "夜",
                    ["主角"],
                    "主角进入车站并找到信",
                    "",
                    15,
                )
            ],
            confirmed=True,
        )
        overloaded = [
            {
                "scene_id": 1,
                "kind": "action",
                "action": f"主角看见线索{index}，随后起身，接着走向门口",
                "purpose": f"推进信息{index}",
                "transition_type": "opening" if index == 1 else "same_scene_cut",
                "transition_reason": "推进事件",
                "inherit_previous_frame": False,
            }
            for index in range(1, 6)
        ]
        feasible = [
            {
                "scene_id": 1,
                "kind": "establish",
                "action": "主角站在车站入口",
                "purpose": "建立空间",
                "transition_type": "opening",
                "transition_reason": "开场",
                "inherit_previous_frame": False,
            },
            {
                "scene_id": 1,
                "kind": "action",
                "action": "主角走到长椅旁并拿起信",
                "purpose": "找到线索",
                "transition_type": "same_scene_cut",
                "transition_reason": "推进动作",
                "inherit_previous_frame": False,
            },
            {
                "scene_id": 1,
                "kind": "reaction",
                "action": "主角看清署名后停住",
                "purpose": "呈现结果",
                "transition_type": "reaction_cut",
                "transition_reason": "呈现反应",
                "inherit_previous_frame": False,
            },
        ]
        agent = OpenAIStoryAgent(client=object(), model="test-model")

        with patch.object(
            agent,
            "_json_completion",
            side_effect=[{"shots": overloaded}, {"shots": feasible}],
        ) as completion:
            result = agent.plan_storyboard(script, StoryFacts())

        self.assertEqual(feasible, result)
        self.assertEqual(2, completion.call_count)
        retry_payload = completion.call_args_list[1].args[1]
        self.assertIn("内容可读下限", retry_payload["timing_feedback"])

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

    def test_scene_timing_honors_model_weights_and_exact_total(self) -> None:
        scenes = [
            StoryScene(
                scene_id=1,
                title="铺垫",
                location="车站",
                time_of_day="夜晚",
                characters=["邮差"],
                action="邮差发现信封",
                narration="雨声中，他注意到信封上的日期。",
                duration=3,
                visible_action="邮差发现信封",
                duration_weight=1.0,
                duration_reason="建立悬念",
            ),
            StoryScene(
                scene_id=2,
                title="冲突",
                location="车站",
                time_of_day="夜晚",
                characters=["邮差"],
                action="邮差冲向站台并拦住列车",
                narration="他只有最后一次机会。",
                duration=3,
                visible_action="邮差冲向站台并拦住列车",
                duration_weight=3.0,
                duration_reason="核心冲突和不可逆选择",
            ),
            StoryScene(
                scene_id=3,
                title="结局",
                location="站台",
                time_of_day="夜晚",
                characters=["邮差"],
                action="邮差交出信件",
                narration="列车驶入雨幕。",
                duration=3,
                visible_action="邮差交出信件",
                duration_weight=2.0,
                duration_reason="完成因果落点",
            ),
        ]
        durations, weights, reasons = plan_scene_durations(scenes, 15)
        self.assertEqual([1.0, 3.0, 2.0], weights)
        self.assertEqual(15, sum(durations))
        self.assertGreater(durations[1], durations[0])
        self.assertGreater(durations[2], durations[0])
        self.assertEqual(3, len(reasons))

    def test_model_script_weights_survive_normalization(self) -> None:
        fallback = StoryScript(
            "fallback",
            15,
            [
                StoryScene(
                    1,
                    "旧场景",
                    "车站",
                    "夜",
                    ["邮差"],
                    "邮差等待",
                    "",
                    15,
                )
            ],
        )
        data = {
            "script": {
                "title": "加权剧本",
                "scenes": [
                    {
                        "title": "铺垫",
                        "location": "车站",
                        "time_of_day": "夜",
                        "characters": ["邮差"],
                        "visible_action": "邮差发现信封",
                        "duration_weight": 1.0,
                        "timing_reason": "建立悬念",
                    },
                    {
                        "title": "冲突",
                        "location": "车站",
                        "time_of_day": "夜",
                        "characters": ["邮差"],
                        "visible_action": "邮差冲向站台并拦住列车",
                        "duration_weight": 4.0,
                        "timing_reason": "核心冲突",
                    },
                    {
                        "title": "结局",
                        "location": "站台",
                        "time_of_day": "夜",
                        "characters": ["邮差"],
                        "visible_action": "邮差交出信件",
                        "duration_weight": 2.0,
                        "timing_reason": "因果落点",
                    },
                ],
            }
        }

        script = OpenAIStoryAgent._script_from_model(data, fallback, 15)

        self.assertEqual([1.0, 4.0, 2.0], [scene.duration_weight for scene in script.scenes])
        self.assertEqual(15, script.total_duration)
        self.assertGreater(script.scenes[1].duration, script.scenes[0].duration)
        self.assertEqual("核心冲突", script.scenes[1].duration_reason)

    def test_offline_short_script_does_not_use_three_equal_scene_buckets(self) -> None:
        story = StoryDraft(
            title="雨夜车站",
            logline="邮差在雨夜车站发现一封改变命运的信。",
            story_text="邮差在雨夜车站发现一封改变命运的信。",
            core_conflict="是否交出信件",
            ending="邮差交出信件并阻止事故。",
            tone="悬疑",
        )

        script = RuleBasedStoryAgent().generate_script(story, 15)

        self.assertEqual(15, script.total_duration)
        self.assertNotEqual([5, 5, 5], [scene.duration for scene in script.scenes])
        self.assertTrue(all(scene.duration >= 3 for scene in script.scenes))

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
