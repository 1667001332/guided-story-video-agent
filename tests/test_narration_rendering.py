from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from guided_story_agent.models import StoryboardPlan, StoryboardShot, VideoArtifact
from guided_story_agent.narration import NarrationArtifact, build_srt
from guided_story_agent.rendering import StoryRenderer
from guided_story_agent.video_provider import AgnesVideoProvider, VideoProviderNotConfigured


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
    return StoryboardPlan("测试", 30, shots, "\n".join(f"旁白{i}" for i in range(1, 6)), confirmed=True)


class NarrationRenderingTests(unittest.TestCase):
    def test_srt_uses_shot_timeline(self) -> None:
        content = build_srt(make_plan())
        self.assertIn("00:00:00,000 --> 00:00:06,000", content)
        self.assertIn("00:00:24,000 --> 00:00:30,000", content)

    def test_missing_api_key_fails_before_submit(self) -> None:
        called = []
        provider = AgnesVideoProvider(api_key="", submit_fn=lambda payload: called.append(payload) or {})
        with self.assertRaises(VideoProviderNotConfigured):
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
                    f"a{shot.shot_id}", shot.shot_id, "fake", self.endpoint, "succeeded",
                    str(path), "", shot.duration, shot.video_prompt, "now", attempt=attempt
                )

        def assemble(paths, output):
            events.append("assemble")
            Path(output).write_bytes(b"silent")
            return str(output)

        def finalize(video, audio, subtitle, output):
            events.append("finalize")
            Path(output).write_bytes(b"final")
            return str(output)

        with tempfile.TemporaryDirectory() as temp_dir:
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

        with tempfile.TemporaryDirectory() as temp_dir:
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


if __name__ == "__main__":
    unittest.main()
