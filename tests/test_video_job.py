from pathlib import Path
from shutil import rmtree
from unittest.mock import patch
from uuid import uuid4

import pytest

from guided_story_agent.agent import RuleBasedStoryAgent
from guided_story_agent.models import (
    CreativeBrief,
    ProviderCapabilities,
    VideoArtifact,
    VideoJob,
)
from guided_story_agent.rendering import VideoJobRenderer
from guided_story_agent.session import GuidedStorySession


class WholeVideoProvider:
    provider_name = "fake-whole"
    endpoint = "fake-whole-v1"
    capabilities = ProviderCapabilities()

    def __init__(self):
        self.calls = []

    def generate_video(self, job, output_dir, **kwargs):
        self.calls.append(job)
        target = Path(output_dir) / "whole.mp4"
        target.write_bytes(b"fake-mp4")
        return VideoArtifact(
            "whole",
            1,
            self.provider_name,
            self.endpoint,
            "succeeded",
            str(target),
            "",
            job.target_seconds,
            job.prompt,
            "now",
        )


@pytest.fixture
def output_dir():
    path = Path("outputs") / f"_video_job_test_{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        rmtree(path, ignore_errors=True)


def test_video_job_renderer_submits_one_complete_job(output_dir):
    provider = WholeVideoProvider()
    job = VideoJob(title="完整片段", prompt="连续动作", target_seconds=47, confirmed=True)
    with patch("guided_story_agent.rendering.validate_mp4_file", return_value=True):
        manifest = VideoJobRenderer(provider).render(job, output_dir)
    assert manifest.status == "succeeded"
    assert manifest.generated_shots == [1]
    assert manifest.final_video_path.endswith("whole.mp4")
    assert len(provider.calls) == 1
    assert provider.calls[0].target_seconds == 47


def test_session_builds_direct_job_without_storyboard(output_dir):
    session = GuidedStorySession(CreativeBrief(target_seconds=30), RuleBasedStoryAgent())
    session.start_ideation("一只猫在暴雨中寻找灯塔")
    session.auto_choose()
    session.generate_story()
    session.confirm_story()
    session.generate_script()
    session.confirm_script()
    job = session.build_video_job()
    assert job.confirmed is True
    assert session.storyboard is None
    assert session.stage.value == "render_ready"
    provider = WholeVideoProvider()
    with patch("guided_story_agent.rendering.validate_mp4_file", return_value=True):
        manifest = session.render_confirmed_video(
            VideoJobRenderer(provider), output_dir / "video"
        )
    assert manifest.status == "succeeded"
    assert session.stage.value == "completed"


def test_video_job_prompt_omits_narration_by_default():
    from guided_story_agent.video_job import build_video_job

    session = GuidedStorySession(CreativeBrief(target_seconds=30), RuleBasedStoryAgent())
    session.start_ideation("一只猫在暴雨中寻找灯塔")
    session.auto_choose()
    session.generate_story()
    session.confirm_story()
    session.generate_script()
    session.confirm_script()
    script = session.script
    story = session.story
    facts = session._story_facts()

    plain = build_video_job(script, story=story, facts=facts)
    assert "旁白" not in plain.prompt
    assert plain.narration.strip()

    voiced = build_video_job(
        script, story=story, facts=facts, include_narration_in_prompt=True
    )
    assert "旁白" in voiced.prompt


def test_provider_duration_limits_are_adapter_capabilities():
    from guided_story_agent.video_provider import AgnesVideoProvider

    capabilities = AgnesVideoProvider(api_key="test").capabilities
    assert capabilities.min_duration_seconds == 3
    assert capabilities.max_duration_seconds == 15
    assert capabilities.supports_long_video is False


def test_uncertain_whole_video_submission_is_not_retried_automatically(output_dir):
    from guided_story_agent.video_provider import VideoSubmissionUncertainError

    class UncertainProvider(WholeVideoProvider):
        def generate_video(self, *args, **kwargs):
            self.calls.append(args[0])
            raise VideoSubmissionUncertainError("lost response", operation_id="op-1")

    provider = UncertainProvider()
    job = VideoJob(title="完整片段", prompt="连续动作", target_seconds=30, confirmed=True)
    first = VideoJobRenderer(provider).render(job, output_dir)
    second = VideoJobRenderer(provider).render(job, output_dir)
    assert first.status == "submission_uncertain"
    assert second.status == "submission_uncertain"
    assert len(provider.calls) == 1
