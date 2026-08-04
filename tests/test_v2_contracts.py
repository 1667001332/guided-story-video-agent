from __future__ import annotations

import pytest

from guided_story_agent.v2 import (
    CameraInstruction,
    CameraPlan,
    CharacterSheet,
    CharacterSheetEntry,
    ContinuityEntry,
    ContinuityPlan,
    CreativeBrief,
    DirectorOrchestrator,
    DirectorOutputRejected,
    FilmBeatPlan,
    MoviePlan,
    MusicPlan,
    NarrationPlan,
    ProviderCapabilities,
    RetakeRequest,
    ScenePlan,
    ScenePlanEntry,
    Script,
    ScriptScene,
    ShotPlan,
    Story,
    TimingEntry,
    TimingPlan,
    TransitionPlan,
    VideoJob,
    validate_movie_plan,
    validate_video_job,
)


def make_plan(*, timing: tuple[float, float] = (4.0, 6.0)) -> MoviePlan:
    scenes = (
        ScriptScene(
            scene_id="s1",
            goal="发现异常",
            emotion="不安",
            importance="high",
            estimated_duration_weight=0.4,
            minimum_duration=3.0,
            camera_language="观察式长镜头",
            motion_type="slow_push",
            dialogue="",
            narration="",
            characters=("c1",),
            location="车站",
            continuity_requirements=("保持雨幕方向",),
            transition="cut",
            timing_reason="建立异常和空间关系",
            action="角色发现站牌上的陌生时间",
        ),
        ScriptScene(
            scene_id="s2",
            goal="做出选择",
            emotion="决绝",
            importance="high",
            estimated_duration_weight=0.6,
            minimum_duration=4.0,
            camera_language="贴近人物的主观视角",
            motion_type="handheld_follow",
            dialogue="现在必须走了。",
            narration="",
            characters=("c1",),
            location="车站",
            continuity_requirements=("手中的票保持可见",),
            transition="match_cut",
            timing_reason="给决定和动作更多时间",
            action="角色攥紧车票并走向即将关闭的车门",
        ),
    )
    return MoviePlan(
        plan_id="plan-1",
        story=Story("末班车", "一个人发现时间不对", "他必须在雨夜做出选择"),
        script=Script("末班车", "一个人发现时间不对", scenes),
        scene_plan=ScenePlan(
            (
                ScenePlanEntry("s1", "让观众先注意异常", "角色站在站牌下"),
                ScenePlanEntry("s2", "把选择变成可见动作", "角色从站台走向车门"),
            )
        ),
        camera_plan=CameraPlan(
            (
                CameraInstruction("s1", "wide", "eye", "35mm", "slow_push", "留出雨幕", "观察式"),
                CameraInstruction("s2", "medium", "low", "50mm", "follow", "压缩车门空间", "主观式"),
            )
        ),
        timing_plan=TimingPlan(
            target_duration_seconds=10,
            entries=(TimingEntry("s1", timing[0], "建立异常"), TimingEntry("s2", timing[1], "完成选择")),
        ),
        continuity_plan=ContinuityPlan(
            (
                ContinuityEntry("s1", ("雨幕方向一致",)),
                ContinuityEntry("s2", ("车票仍在右手",)),
            )
        ),
        character_sheet=CharacterSheet(
            (
                CharacterSheetEntry(
                    "c1",
                    "主角",
                    "雨夜旅客",
                    "protagonist",
                    "深色外套与湿发",
                    "保持右手持票",
                    ("c1-ref",),
                    "深色外套",
                ),
            )
        ),
        narration_plan=NarrationPlan(False, "zh-CN", ""),
        music_plan=MusicPlan("低频环境声"),
        transition_plan=TransitionPlan(),
        shot_plan=(
            ShotPlan(
                "shot-1",
                "s1",
                1,
                timing[0],
                "建立异常",
                "角色发现站牌上的陌生时间",
                "主角与站牌",
                "观察异常",
                "slow_push",
                "雨夜顶光",
                "雨幕与站牌同框",
                ("c1",),
                ("站牌",),
                "",
                "",
                "",
                "opening",
                "cut",
                ("雨幕方向一致",),
                ("陌生时间清晰可见",),
                ("观众识别异常",),
            ),
            ShotPlan(
                "shot-2",
                "s2",
                2,
                timing[1],
                "完成选择",
                "角色攥紧车票并走向即将关闭的车门",
                "主角与车门",
                "选择变成动作",
                "handheld_follow",
                "车站冷光",
                "车票和车门形成前后景",
                ("c1",),
                ("车票",),
                "",
                "现在必须走了。",
                "现在必须走了。",
                "cut",
                "ending",
                ("车票仍在右手",),
                ("车门关闭前的动作清晰",),
                ("选择完成且人物连续",),
            ),
        ),
        film_beats=(
            FilmBeatPlan(
                beat_id="beat-1",
                order=1,
                scene_id="s1",
                shot_ids=("shot-1",),
                dramatic_purpose="setup",
                narrative_function="setup",
                viewer_state_before="观众尚未知道异常",
                viewer_state_after="观众识别异常",
                emotion="不安",
                tension_level=0.4,
                visual_focus="陌生时间",
                required_audience_understanding="观众知道时间不对",
                required_evidence=("陌生时间清晰可见",),
                character_emotional_state=("c1:不安",),
                continuity_intent="雨幕方向与人物身份保持一致",
                transition_intent="从异常发现推进到选择",
                narration_intent="无旁白，以画面传达异常",
                music_intent="低频环境声逐渐显现",
                acceptance_criteria=("观众识别异常",),
                timing_weight=0.4,
            ),
            FilmBeatPlan(
                beat_id="beat-2",
                order=2,
                scene_id="s2",
                shot_ids=("shot-2",),
                dramatic_purpose="payoff",
                narrative_function="resolution",
                viewer_state_before="观众等待选择结果",
                viewer_state_after="观众理解角色已经做出选择",
                emotion="决绝",
                tension_level=0.8,
                visual_focus="车票与即将关闭的车门",
                required_audience_understanding="观众理解选择已完成",
                required_evidence=("车门关闭前的动作清晰",),
                character_emotional_state=("c1:决绝",),
                continuity_intent="车票仍在右手且空间方向连续",
                transition_intent="以动作完成结尾",
                narration_intent="无旁白，以动作完成信息",
                music_intent="环境声在结尾收束",
                acceptance_criteria=("选择完成且人物连续",),
                timing_weight=0.6,
            ),
        ),
        visual_style="cinematic rainy realism",
        review_criteria=("人物身份连续", "动作可辨识"),
        confirmed=True,
    )


def test_validator_rejects_timing_without_repairing_it() -> None:
    brief = CreativeBrief(10, "short film", "cinematic", "adult")
    plan = make_plan(timing=(3.0, 3.0))

    report = validate_movie_plan(plan, brief)

    assert not report.valid
    assert any("declared scene durations" in error for error in report.errors)
    assert [entry.duration_seconds for entry in plan.timing_plan.entries] == [3.0, 3.0]


def test_director_orchestrator_retries_with_structured_feedback() -> None:
    class FakeDirector:
        def __init__(self) -> None:
            self.feedback: list[str] = []

        def create_movie_plan(self, brief, direction, *, provider_capabilities=None, feedback=""):
            self.feedback.append(feedback)
            return make_plan(timing=(3.0, 3.0)) if len(self.feedback) == 1 else make_plan()

        def revise_movie_plan(self, *args, **kwargs):
            raise AssertionError("revision API should not be used for creation")

    agent = FakeDirector()
    plan = DirectorOrchestrator(agent, max_attempts=2).create_movie_plan(
        CreativeBrief(10, "short film", "cinematic", "adult"), "雨夜车站"
    )

    assert plan.plan_id == "plan-1"
    assert len(agent.feedback) == 2
    assert "declared scene durations" in agent.feedback[1]


def test_director_output_is_rejected_after_bounded_retries() -> None:
    class AlwaysInvalid:
        def create_movie_plan(self, *args, **kwargs):
            return make_plan(timing=(3.0, 3.0))

        def revise_movie_plan(self, *args, **kwargs):
            raise AssertionError

    with pytest.raises(DirectorOutputRejected) as caught:
        DirectorOrchestrator(AlwaysInvalid(), max_attempts=2).create_movie_plan(
            CreativeBrief(10, "short film", "cinematic", "adult"), "雨夜车站"
        )
    assert len(caught.value.attempts) == 2


def test_movie_plan_contains_no_provider_fields() -> None:
    plan = make_plan()
    payload = plan.__dict__ if hasattr(plan, "__dict__") else plan
    assert not hasattr(plan, "provider_prompts")
    assert "provider_prompt" not in str(payload)


def test_video_job_validation_is_provider_only() -> None:
    job = VideoJob(
        job_id="job-1",
        provider_key="fake",
        provider_prompt="compiled provider prompt",
        negative_prompt="no identity drift",
        duration_seconds=10,
        output_format="mp4",
        source_movie_plan_id="plan-1",
        compiler_version="v2-compiler/1",
        confirmed=True,
    )

    assert job.provider_prompt == "compiled provider prompt"
    assert validate_video_job(job, ProviderCapabilities("fake", max_duration_seconds=12)).valid
    rejected = validate_video_job(job, ProviderCapabilities("fake", max_duration_seconds=9))
    assert not rejected.valid
    assert any("exceeds Provider maximum" in error for error in rejected.errors)


def test_retake_contract_has_only_production_overlays() -> None:
    request = RetakeRequest(
        "retake-1",
        camera_requirements="更低机位",
        scene_ids=("s2",),
    )
    request.validate()
