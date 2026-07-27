from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guided_story_agent.models import StoryboardPlan, StoryboardShot, VideoArtifact
from guided_story_agent.narration import (
    EdgeNarrationSynthesizer,
    NarrationArtifact,
    NarrationUnavailable,
    build_srt,
    narration_text_from_timeline,
    normalize_narration_timeline,
)
from guided_story_agent.rendering import (
    StoryRenderer,
    mux_audio_and_subtitles,
    probe_media_duration,
)
from guided_story_agent.video_provider import (
    AgnesVideoProvider,
    VideoProviderNotConfigured,
    sanitize_remote_url,
    validate_mp4_file,
)


def make_plan() -> StoryboardPlan:
    shots = [
        StoryboardShot(
            shot_id=i,
            scene_id=i,
            duration=6,
            character="邮差",
            location="车站",
            visual=f"动作{i}",
            action=f"动作{i}",
            camera="medium",
            lighting="night",
            mood="mystery",
            narration=f"旁白{i}",
            video_prompt=f"prompt {i}",
            negative_prompt="bad",
        )
        for i in range(1, 6)
    ]
    return StoryboardPlan(
        "测试", 30, shots, "\n".join(f"旁白{i}" for i in range(1, 6)), confirmed=True
    )


def make_single_shot_plan() -> StoryboardPlan:
    plan = make_plan()
    plan.shots = plan.shots[:1]
    plan.target_seconds = plan.shots[0].duration
    plan.narration_text = ""
    return plan


class NarrationRenderingTests(unittest.TestCase):
    def test_remote_url_redaction_removes_credentials_signature_and_fragment(self) -> None:
        self.assertEqual(
            "https://cdn.example.test/shot.mp4",
            sanitize_remote_url(
                "https://user:password@cdn.example.test/shot.mp4?token=secret#private"
            ),
        )

    def test_srt_uses_shot_timeline(self) -> None:
        content = build_srt(make_plan())
        self.assertIn("00:00:00,000 --> 00:00:06,000", content)
        self.assertIn("00:00:24,000 --> 00:00:30,000", content)

    def test_same_scene_narration_tts_and_srt_share_one_timeline(self) -> None:
        plan = make_plan()
        plan.shots = plan.shots[:3]
        plan.target_seconds = 18
        for shot in plan.shots:
            shot.scene_id = 1
            shot.narration = "这句旁白只能出现一次。"
        plan.narration_text = "旧的重复旁白"
        normalize_narration_timeline(plan)

        captured: list[str] = []

        class Communicate:
            def __init__(self, text, voice, *, rate):
                del voice, rate
                captured.append(text)

            def save_sync(self, target):
                Path(target).write_bytes(b"audio")

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(sys.modules, {"edge_tts": SimpleNamespace(Communicate=Communicate)}),
        ):
            artifact = EdgeNarrationSynthesizer().synthesize(plan, temp)
            subtitle = Path(artifact.subtitle_path).read_text(encoding="utf-8")

        self.assertEqual(["这句旁白只能出现一次。"], captured)
        self.assertEqual("这句旁白只能出现一次。", plan.narration_text)
        self.assertEqual(plan.narration_text, narration_text_from_timeline(plan))
        self.assertEqual(1, subtitle.count("这句旁白只能出现一次。"))
        self.assertNotIn("动作2", subtitle)

    def test_generated_storyboard_assigns_scene_narration_only_once(self) -> None:
        from guided_story_agent import CreativeBrief, RuleBasedStoryAgent
        from guided_story_agent.session import GuidedStorySession

        session = GuidedStorySession(
            CreativeBrief(target_seconds=30),
            RuleBasedStoryAgent(),
        )
        session.start_ideation("雨夜车站")
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        plan = session.build_storyboard()

        self.assertEqual(
            plan.narration_text,
            "\n".join(shot.narration for shot in plan.shots if shot.narration),
        )
        for scene_id in {shot.scene_id for shot in plan.shots}:
            values = [
                shot.narration
                for shot in plan.shots
                if shot.scene_id == scene_id and shot.narration
            ]
            self.assertEqual(len(values), len(set(values)))

    def test_missing_api_key_fails_before_submit(self) -> None:
        called = []
        provider = AgnesVideoProvider(
            api_key="", submit_fn=lambda payload: called.append(payload) or {}
        )
        with self.assertRaises(VideoProviderNotConfigured):
            provider.generate_shot(make_plan().shots[0], "outputs")
        self.assertEqual([], called)

    def test_video_from_env_prefers_generic_config(self) -> None:
        environment = {
            "VIDEO_PROVIDER": "agnes",
            "VIDEO_API_KEY": "generic-video-key",
            "VIDEO_API_ROOT": "https://video.example.test",
            "VIDEO_MODEL": "mentor-video-model",
            "VIDEO_TIMEOUT": "90",
            "VIDEO_POLL_INTERVAL": "2",
            "VIDEO_MAX_POLL_SECONDS": "600",
            "AGNES_API_KEY": "legacy-video-key",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv"),
        ):
            provider = AgnesVideoProvider.from_env()

        self.assertEqual("generic-video-key", provider.api_key)
        self.assertEqual("https://video.example.test", provider.api_root)
        self.assertEqual("mentor-video-model", provider.model)
        self.assertEqual(90.0, provider.timeout)
        self.assertEqual(2.0, provider.poll_interval)
        self.assertEqual(600.0, provider.max_poll_seconds)
        self.assertEqual("VIDEO_*", provider.config_source)

    def test_video_from_env_keeps_agnes_legacy_compatibility(self) -> None:
        environment = {
            "AGNES_API_KEY": "legacy-video-key",
            "AGNES_API_ROOT": "https://legacy.example.test",
            "AGNES_VIDEO_MODEL": "legacy-video-model",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv"),
        ):
            provider = AgnesVideoProvider.from_env()

        self.assertEqual("legacy-video-key", provider.api_key)
        self.assertEqual("https://legacy.example.test", provider.api_root)
        self.assertEqual("legacy-video-model", provider.model)
        self.assertEqual("AGNES_* (legacy)", provider.config_source)

    def test_unsupported_video_provider_fails_before_submit(self) -> None:
        called = []
        environment = {
            "VIDEO_PROVIDER": "vidu",
            "VIDEO_API_KEY": "video-key",
            "VIDEO_MODEL": "video-model",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv"),
        ):
            provider = AgnesVideoProvider.from_env()
        provider._submit_fn = lambda payload: called.append(payload) or {}

        with self.assertRaisesRegex(VideoProviderNotConfigured, "VIDEO_PROVIDER=vidu"):
            provider.generate_shot(make_plan().shots[0], "outputs")
        self.assertEqual([], called)

    def test_renderer_prepares_narration_then_generates_and_assembles(self) -> None:
        events: list[str] = []

        class Narration:
            def synthesize(self, plan, output_dir):
                events.append("narration")
                target = Path(output_dir)
                audio = target / "voice.mp3"
                srt = target / "voice.srt"
                audio.write_bytes(b"audio")
                srt.write_text("subtitle", encoding="utf-8")
                return NarrationArtifact(str(audio), str(srt))

        class Provider:
            endpoint = "fake-model"
            provider_name = "fake"

            def generate_shot(self, shot, output_dir, *, attempt=1, progress_callback=None):
                events.append(f"shot-{shot.shot_id}")
                path = Path(output_dir) / f"{shot.shot_id}.mp4"
                path.write_bytes(b"video")
                return VideoArtifact(
                    f"a{shot.shot_id}",
                    shot.shot_id,
                    "fake",
                    self.endpoint,
                    "succeeded",
                    str(path),
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                    attempt=attempt,
                )

        def assemble(paths, output):
            events.append("assemble")
            Path(output).write_bytes(b"silent")
            return str(output)

        def finalize(video, audio, subtitle, output):
            events.append("finalize")
            Path(output).write_bytes(b"final")
            return str(output)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            renderer = StoryRenderer(
                Provider(), narration=Narration(), assembler=assemble, finalizer=finalize
            )
            manifest = renderer.render(make_plan(), temp_dir)
            self.assertTrue(Path(manifest.final_video_path).is_file())
        self.assertEqual("narration", events[0])
        self.assertEqual("succeeded", manifest.status)
        self.assertEqual([1, 2, 3, 4, 5], manifest.generated_shots)
        self.assertEqual("finalize", events[-1])

    def test_renderer_keeps_successes_and_retries_only_failed_shots(self) -> None:
        class Narration:
            def synthesize(self, plan, output_dir):
                return NarrationArtifact("", "")

        class Provider:
            endpoint = "fake"
            provider_name = "fake"

            def generate_shot(self, shot, output_dir, **kwargs):
                if shot.shot_id == 3:
                    raise RuntimeError("boom")
                path = Path(output_dir) / f"{shot.shot_id}.mp4"
                path.write_bytes(b"video")
                return VideoArtifact(
                    f"a{shot.shot_id}",
                    shot.shot_id,
                    "fake",
                    self.endpoint,
                    "succeeded",
                    str(path),
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                    attempt=kwargs.get("attempt", 1),
                )

        class RecoveredProvider(Provider):
            def generate_shot(self, shot, output_dir, **kwargs):
                path = Path(output_dir) / f"{shot.shot_id}.mp4"
                path.write_bytes(b"video")
                return VideoArtifact(
                    f"recovered-{shot.shot_id}",
                    shot.shot_id,
                    "fake",
                    self.endpoint,
                    "succeeded",
                    str(path),
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                    attempt=kwargs.get("attempt", 1),
                )

        def assemble(paths, output):
            Path(output).write_bytes(b"silent")
            return str(output)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            plan = make_plan()
            manifest = StoryRenderer(Provider(), narration=Narration()).render(plan, temp_dir)
            self.assertTrue((Path(temp_dir) / "render_manifest.json").is_file())
            self.assertEqual("failed", manifest.status)
            self.assertEqual([3], manifest.failed_shots)
            self.assertEqual([1, 2, 4, 5], manifest.generated_shots)

            recovered = StoryRenderer(
                RecoveredProvider(),
                narration=Narration(),
                assembler=assemble,
            ).render(plan, temp_dir)
            self.assertEqual("succeeded", recovered.status)
            self.assertEqual([1, 2, 4, 5], recovered.reused_shots)
            self.assertEqual([3], recovered.generated_shots)

    def test_timeout_resumes_existing_remote_task_without_resubmit(self) -> None:
        submissions: list[dict[str, object]] = []
        statuses = iter(
            [
                {"status": "running"},
                {
                    "status": "completed",
                    "video_url": "https://cdn.example.test/shot.mp4?token=secret#fragment",
                },
            ]
        )
        clock = iter([0.0, 0.0, 2.0, 3.0, 3.0])

        def download(_url: str, target: Path) -> None:
            target.write_bytes(b"fake-video")

        provider = AgnesVideoProvider(
            api_key="test-key",
            max_poll_seconds=1,
            poll_interval=0,
            submit_fn=lambda payload: submissions.append(payload) or {"video_id": "job-1"},
            status_fn=lambda _video_id: next(statuses),
            download_fn=download,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: next(clock),
        )

        class Narration:
            def synthesize(self, plan, output_dir):
                return NarrationArtifact("", "")

        def assemble(_paths, output):
            Path(output).write_bytes(b"assembled")
            return str(output)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("guided_story_agent.video_provider.validate_mp4_file", return_value=True),
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            plan = make_single_shot_plan()
            renderer = StoryRenderer(provider, narration=Narration(), assembler=assemble)
            pending = renderer.render(plan, temp_dir)
            self.assertEqual("pending", pending.status)
            self.assertEqual("job-1", plan.artifacts[0].request_id)

            completed = renderer.render(plan, temp_dir)

        self.assertEqual("succeeded", completed.status)
        self.assertEqual(1, len(submissions))
        self.assertEqual("job-1", plan.artifacts[0].request_id)
        self.assertEqual("https://cdn.example.test/shot.mp4", plan.artifacts[0].remote_url)

    def test_uncertain_submit_response_is_never_retried_automatically(self) -> None:
        submissions = 0

        def submit(_payload):
            nonlocal submissions
            submissions += 1
            raise TimeoutError("response timed out")

        class Narration:
            def synthesize(self, plan, output_dir):
                return NarrationArtifact("", "")

        provider = AgnesVideoProvider(
            api_key="test-key",
            submit_fn=submit,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = make_single_shot_plan()
            renderer = StoryRenderer(provider, narration=Narration())
            first = renderer.render(plan, temp_dir)
            second = renderer.render(plan, temp_dir)

        self.assertEqual("submission_uncertain", first.status)
        self.assertEqual("submission_uncertain", second.status)
        self.assertEqual("submission_uncertain", plan.artifacts[0].status)
        self.assertTrue(plan.artifacts[0].request_id.startswith("local-submit-"))
        self.assertEqual(1, submissions)

    def test_terminal_remote_failure_allows_a_new_submission(self) -> None:
        submissions: list[str] = []
        statuses = iter(
            [
                {"status": "failed", "error": "rejected"},
                {
                    "status": "completed",
                    "video_url": "https://cdn.example.test/shot.mp4",
                },
            ]
        )

        def submit(_payload):
            request_id = f"job-{len(submissions) + 1}"
            submissions.append(request_id)
            return {"video_id": request_id}

        def download(_url: str, target: Path) -> None:
            target.write_bytes(b"fake-video")

        provider = AgnesVideoProvider(
            api_key="test-key",
            submit_fn=submit,
            status_fn=lambda _video_id: next(statuses),
            download_fn=download,
            sleep_fn=lambda _seconds: None,
        )

        class Narration:
            def synthesize(self, plan, output_dir):
                return NarrationArtifact("", "")

        def assemble(_paths, output):
            Path(output).write_bytes(b"assembled")
            return str(output)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("guided_story_agent.video_provider.validate_mp4_file", return_value=True),
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            plan = make_single_shot_plan()
            renderer = StoryRenderer(provider, narration=Narration(), assembler=assemble)
            failed = renderer.render(plan, temp_dir)
            self.assertEqual("failed", failed.status)
            self.assertEqual("job-1", plan.artifacts[0].request_id)

            completed = renderer.render(plan, temp_dir)

        self.assertEqual("succeeded", completed.status)
        self.assertEqual(["job-1", "job-2"], submissions)

    def test_invalid_cached_video_is_not_reused(self) -> None:
        calls: list[int] = []

        class Narration:
            def synthesize(self, plan, output_dir):
                return NarrationArtifact("", "")

        class Provider:
            endpoint = "fake"
            provider_name = "fake"

            def generate_shot(self, shot, output_dir, **kwargs):
                calls.append(shot.shot_id)
                target = Path(output_dir) / "replacement.mp4"
                target.write_bytes(b"replacement")
                return VideoArtifact(
                    "replacement",
                    shot.shot_id,
                    "fake",
                    self.endpoint,
                    "succeeded",
                    str(target),
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                )

        def assemble(_paths, output):
            Path(output).write_bytes(b"assembled")
            return str(output)

        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "bad.mp4"
            bad_path.write_text("<html>not video</html>", encoding="utf-8")
            plan = make_single_shot_plan()
            shot = plan.shots[0]
            plan.artifacts.append(
                VideoArtifact(
                    "bad",
                    shot.shot_id,
                    "fake",
                    "fake",
                    "succeeded",
                    str(bad_path),
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                )
            )

            def validation(path):
                return Path(path).name == "replacement.mp4"

            with patch(
                "guided_story_agent.rendering.validate_mp4_file",
                side_effect=validation,
            ):
                manifest = StoryRenderer(
                    Provider(),
                    narration=Narration(),
                    assembler=assemble,
                ).render(plan, temp_dir)

        self.assertEqual("succeeded", manifest.status)
        self.assertEqual([1], calls)
        self.assertEqual("failed", plan.artifacts[0].status)
        self.assertEqual([1], manifest.generated_shots)

    def test_missing_narration_is_an_explicit_warning_and_keeps_subtitles(self) -> None:
        finalized: list[tuple[str, str]] = []

        class Narration:
            def synthesize(self, plan, output_dir):
                subtitle = Path(output_dir) / "narration.srt"
                subtitle.write_text(build_srt(plan), encoding="utf-8")
                plan.subtitle_path = str(subtitle)
                raise NarrationUnavailable("未安装 edge-tts")

        class Provider:
            endpoint = "fake"
            provider_name = "fake"

            def generate_shot(self, shot, output_dir, **kwargs):
                target = Path(output_dir) / "shot.mp4"
                target.write_bytes(b"video")
                return VideoArtifact(
                    "shot",
                    shot.shot_id,
                    "fake",
                    self.endpoint,
                    "succeeded",
                    str(target),
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                )

        def assemble(_paths, output):
            Path(output).write_bytes(b"assembled")
            return str(output)

        def finalize(_video, audio, subtitle, output):
            finalized.append((audio, subtitle))
            Path(output).write_bytes(b"final")
            return str(output)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = StoryRenderer(
                Provider(),
                narration=Narration(),
                assembler=assemble,
                finalizer=finalize,
            ).render(make_single_shot_plan(), temp_dir)

        self.assertEqual("succeeded_with_warnings", manifest.status)
        self.assertIn("edge-tts", manifest.error)
        self.assertEqual("", finalized[0][0])
        self.assertTrue(finalized[0][1].endswith("narration.srt"))

    def test_mp4_validation_rejects_non_video_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "download.mp4"
            target.write_text("<html>gateway error</html>", encoding="utf-8")
            self.assertFalse(validate_mp4_file(target))

    def test_mux_fails_closed_when_video_duration_cannot_be_probed(self) -> None:
        with (
            patch("guided_story_agent.rendering.shutil.which", return_value="ffmpeg"),
            patch(
                "guided_story_agent.rendering.probe_media_duration",
                return_value=None,
            ),
            patch.object(Path, "is_file", return_value=True),
            patch("guided_story_agent.rendering.subprocess.run") as run,
            self.assertRaisesRegex(RuntimeError, "ffprobe"),
        ):
            mux_audio_and_subtitles("video.mp4", "audio.mp3", "", "final.mp4")

        run.assert_not_called()

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "需要本地 ffmpeg/ffprobe",
    )
    def test_mux_pads_short_audio_to_video_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir)
            video = target / "video.mp4"
            audio = target / "audio.wav"
            output = target / "final.mp4"
            try:
                subprocess.run(
                    [
                        shutil.which("ffmpeg"),
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=black:s=160x90:r=10:d=2",
                        "-c:v",
                        "mpeg4",
                        "-an",
                        str(video),
                    ],
                    capture_output=True,
                    timeout=60,
                    check=True,
                )
                subprocess.run(
                    [
                        shutil.which("ffmpeg"),
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1000:sample_rate=16000:duration=0.3",
                        str(audio),
                    ],
                    capture_output=True,
                    timeout=60,
                    check=True,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.skipTest(f"本地 ffmpeg 不支持测试编码器：{exc}")

            mux_audio_and_subtitles(str(video), str(audio), "", output)
            duration = probe_media_duration(output)

        self.assertIsNotNone(duration)
        self.assertGreater(duration, 1.8)


if __name__ == "__main__":
    unittest.main()
