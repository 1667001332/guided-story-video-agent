from __future__ import annotations

import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guided_story_agent import CreativeBrief, RuleBasedStoryAgent, Stage
from guided_story_agent.continuity import (
    build_input_fingerprint,
    freeze_confirmed_visual_inputs,
    resolve_reference_assets,
)
from guided_story_agent.models import (
    ContinuityState,
    ProviderCapabilities,
    StoryboardPlan,
    StoryboardShot,
    VideoArtifact,
    VisualAsset,
    VisualBible,
    VisualReference,
    to_plain_data,
)
from guided_story_agent.narration import NarrationArtifact
from guided_story_agent.rendering import StoryRenderer, extract_last_frame
from guided_story_agent.session import GuidedStorySession
from guided_story_agent.video_provider import (
    AgnesVideoProvider,
    LocalPublicImagePublisher,
    build_public_image_url_resolver,
)
from guided_story_agent.web_app import _render_evidence_summary, _storyboard_markdown


def make_shot(
    shot_id: int,
    *,
    scene_id: int = 1,
    location: str = "车站",
    mode: str = "independent",
    previous_shot_id: int | None = None,
) -> StoryboardShot:
    state = ContinuityState(
        character_appearance={"邮差": "短发，年轻面孔"},
        character_clothing={"邮差": "深蓝制服"},
        character_positions={"邮差": "站台中央"},
        character_held_props={"邮差": ["铜怀表"]},
        prop_positions={"铜怀表": "邮差右手"},
        location=location,
        time_of_day="雨夜",
        weather="雨",
        key_light_direction="画面左侧",
    )
    return StoryboardShot(
        shot_id=shot_id,
        scene_id=scene_id,
        duration=3,
        character="邮差",
        location=location,
        visual=f"动作 {shot_id}",
        action=f"动作 {shot_id}",
        camera="medium",
        lighting="night",
        mood="mystery",
        narration="",
        video_prompt=f"prompt {shot_id}",
        negative_prompt="bad",
        continuity_mode=mode,
        previous_shot_id=previous_shot_id,
        transition_type=(
            "continuous_action"
            if mode == "same_scene_chain"
            else "same_scene_cut"
            if mode == "same_scene_reference"
            else "independent"
        ),
        inherit_previous_frame=mode == "same_scene_chain",
        continuity_start_state=deepcopy(state),
        continuity_end_state=deepcopy(state),
    )


def make_chain_plan() -> StoryboardPlan:
    return StoryboardPlan(
        title="连续性测试",
        target_seconds=6,
        shots=[
            make_shot(1),
            make_shot(2, mode="same_scene_chain", previous_shot_id=1),
        ],
        narration_text="",
        confirmed=True,
    )


class EmptyNarration:
    def synthesize(self, plan, output_dir):
        return NarrationArtifact("", "")


class FakeVisualProvider:
    provider_name = "fake-visual"
    endpoint = "fake-visual-v1"
    capabilities = ProviderCapabilities(True, True, True, True)

    def __init__(self, *, fail_shots: set[int] | None = None) -> None:
        self.calls: list[StoryboardShot] = []
        self.fail_shots = fail_shots or set()

    def generate_shot(self, shot, output_dir, *, attempt=1, **kwargs):
        del kwargs
        self.calls.append(deepcopy(shot))
        if shot.shot_id in self.fail_shots:
            raise RuntimeError("provider failed")
        target = Path(output_dir) / f"shot-{shot.shot_id}-{len(self.calls)}.mp4"
        initial_bytes = (
            Path(shot.initial_frame_path).read_bytes()
            if shot.initial_frame_path and Path(shot.initial_frame_path).is_file()
            else b""
        )
        target.write_bytes(
            f"{shot.video_prompt}|{shot.initial_frame_path}|".encode("utf-8")
            + initial_bytes
        )
        return VideoArtifact(
            artifact_id=f"artifact-{shot.shot_id}-{len(self.calls)}",
            shot_id=shot.shot_id,
            provider=self.provider_name,
            model=self.endpoint,
            status="succeeded",
            local_path=str(target),
            remote_url="",
            duration=shot.duration,
            prompt=shot.video_prompt,
            created_at="now",
            attempt=attempt,
        )


class MockAgnesProvider(AgnesVideoProvider):
    """Capture official Agnes payloads without submitting any remote task."""

    def __init__(self, publisher: LocalPublicImagePublisher) -> None:
        super().__init__(api_key="fake", image_publisher=publisher)
        self.payloads: list[dict[str, object]] = []
        self.calls: list[StoryboardShot] = []

    def generate_shot(self, shot, output_dir, *, attempt=1, **kwargs):
        del kwargs
        self.calls.append(deepcopy(shot))
        self.payloads.append(self._build_payload(shot))
        target = Path(output_dir) / f"agnes-shot-{shot.shot_id}.mp4"
        target.write_bytes(
            f"{shot.video_prompt}|{shot.initial_frame_url}".encode("utf-8")
        )
        return VideoArtifact(
            artifact_id=f"agnes-{shot.shot_id}",
            shot_id=shot.shot_id,
            provider=self.provider_name,
            model=self.endpoint,
            status="succeeded",
            local_path=str(target),
            remote_url="",
            duration=shot.duration,
            prompt=shot.video_prompt,
            created_at="now",
            attempt=attempt,
        )


def fake_extract(video_path: str | Path, output_path: str | Path) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(video_path).read_bytes() + b"|last-frame")
    return str(target)


def fake_assemble(paths: list[str], output_path: str | Path) -> str:
    target = Path(output_path)
    target.write_bytes("|".join(paths).encode("utf-8"))
    return str(target)


def confirmed_reference(
    path: str | Path,
    *,
    usage: str,
    reference_id: str = "confirmed-ref",
) -> VisualReference:
    candidate = Path(path)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else ""
    return VisualReference(
        reference_id=reference_id,
        path=str(path),
        usage=usage,
        content_digest=digest,
        content_summary=f"sha256:{digest[:12]}" if digest else "",
        confirmed=True,
    )


def shot_num_frames(shot: StoryboardShot) -> int:
    return shot.duration * 24 + 1


class VideoContinuityTests(unittest.TestCase):
    def test_old_storyboard_loads_with_safe_continuity_defaults(self) -> None:
        shot = make_shot(1)
        old_shot = {
            key: value
            for key, value in to_plain_data(shot).items()
            if not key.startswith("continuity_")
            and key
            not in {
                "reference_image_paths",
                "initial_frame_path",
                "previous_shot_id",
                "seed",
                "generated_first_frame_path",
                "generated_last_frame_path",
            }
        }
        loaded = GuidedStorySession._storyboard_from(
            {
                "title": "old",
                "target_seconds": 3,
                "shots": [old_shot],
                "narration_text": "",
                "confirmed": True,
            }
        )
        self.assertEqual("independent", loaded.shots[0].continuity_mode)
        self.assertEqual([], loaded.shots[0].reference_image_paths)
        self.assertEqual([], loaded.shots[0].confirmed_visual_inputs)
        self.assertEqual("", loaded.shots[0].initial_frame_url)
        self.assertIsInstance(loaded.shots[0].continuity_start_state, ContinuityState)

    def test_asset_ids_resolve_to_existing_deduplicated_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "hero.png"
            image.write_bytes(b"png")
            plan = StoryboardPlan(
                "refs",
                3,
                [make_shot(1)],
                "",
                visual_bible=VisualBible(
                    assets=[
                        VisualAsset(
                            "character-01",
                            "character",
                            "邮差",
                            "深蓝制服",
                            [str(image), str(image)],
                        )
                    ]
                ),
            )
            plan.shots[0].reference_asset_ids = ["character-01"]
            diagnostics = freeze_confirmed_visual_inputs(plan)

        self.assertEqual([], diagnostics)
        self.assertEqual([str(image.resolve())], plan.shots[0].reference_image_paths)
        self.assertEqual(
            "identity_reference",
            plan.shots[0].confirmed_visual_inputs[0].usage,
        )
        self.assertNotEqual(
            "start_frame",
            plan.shots[0].confirmed_visual_inputs[0].usage,
        )

    def test_missing_and_invalid_reference_images_are_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            invalid = Path(temp) / "hero.txt"
            invalid.write_text("not image", encoding="utf-8")
            plan = StoryboardPlan(
                "refs",
                3,
                [make_shot(1)],
                "",
                visual_bible=VisualBible(
                    assets=[
                        VisualAsset(
                            "character-01",
                            "character",
                            "邮差",
                            "深蓝制服",
                            [str(invalid), str(Path(temp) / "missing.png")],
                        )
                    ]
                ),
            )
            plan.shots[0].reference_asset_ids = ["character-01"]
            diagnostics = resolve_reference_assets(plan)

        self.assertEqual([], plan.shots[0].reference_image_paths)
        self.assertTrue(any("不支持的图片类型" in item for item in diagnostics))
        self.assertTrue(any("不存在" in item for item in diagnostics))

    def test_same_scene_uses_previous_generated_last_frame(self) -> None:
        provider = FakeVisualProvider()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(make_chain_plan(), temp)

        self.assertEqual("succeeded", manifest.status)
        self.assertEqual("", provider.calls[0].initial_frame_path)
        self.assertTrue(provider.calls[1].initial_frame_path.endswith("shot_001_last.png"))

    def test_same_scene_camera_cut_does_not_inherit_previous_frame(self) -> None:
        provider = FakeVisualProvider()
        plan = StoryboardPlan(
            "normal cut",
            6,
            [
                make_shot(1),
                make_shot(
                    2,
                    mode="same_scene_reference",
                    previous_shot_id=1,
                ),
            ],
            "",
            confirmed=True,
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, temp)

        self.assertEqual("succeeded_with_warnings", manifest.status)
        self.assertFalse(provider.calls[1].inherit_previous_frame)
        self.assertEqual("", provider.calls[1].initial_frame_path)
        self.assertEqual("same_scene_reference", provider.calls[1].continuity_mode)

    def test_new_location_never_inherits_previous_last_frame(self) -> None:
        provider = FakeVisualProvider()
        plan = StoryboardPlan(
            "locations",
            6,
            [
                make_shot(1),
                make_shot(2, scene_id=2, location="仓库", mode="independent"),
            ],
            "",
            confirmed=True,
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, temp)

        self.assertEqual("", provider.calls[1].initial_frame_path)
        self.assertIsNone(provider.calls[1].previous_shot_id)

    def test_first_shot_has_no_previous_dependency(self) -> None:
        plan = make_chain_plan()
        self.assertIsNone(plan.shots[0].previous_shot_id)
        self.assertEqual("independent", plan.shots[0].continuity_mode)

    def test_extract_last_frame_uses_stable_decode_command(self) -> None:
        calls: list[list[str]] = []

        def runner(command, **kwargs):
            del kwargs
            calls.append(command)
            Path(command[-1]).write_bytes(b"frame")
            return SimpleNamespace(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "clip.mp4"
            video.write_bytes(b"video")
            output = Path(temp) / "frames" / "shot_001_last.png"
            result = extract_last_frame(
                video,
                output,
                runner=runner,
                ffmpeg_path="ffmpeg",
            )

        self.assertIn("-update", calls[0])
        self.assertTrue(result.endswith("shot_001_last.png"))

    def test_extract_last_frame_reports_ffmpeg_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "clip.mp4"
            video.write_bytes(b"video")
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                extract_last_frame(
                    video,
                    Path(temp) / "last.png",
                    runner=lambda *args, **kwargs: SimpleNamespace(
                        returncode=1,
                        stderr="decode failed",
                    ),
                    ffmpeg_path="ffmpeg",
                )

    def test_upstream_failure_causes_dependency_failure_without_fake_path(self) -> None:
        provider = FakeVisualProvider(fail_shots={1})
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(make_chain_plan(), temp)

        self.assertEqual([1], [call.shot_id for call in provider.calls])
        self.assertEqual([2], manifest.dependency_failed_shots)
        shot_two = next(item for item in manifest.artifacts if item.shot_id == 2)
        self.assertEqual("", shot_two.initial_frame_path)

    def test_upstream_failure_can_degrade_to_confirmed_fixed_reference(self) -> None:
        provider = FakeVisualProvider(fail_shots={1})
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            reference = Path(temp) / "confirmed.png"
            reference.write_bytes(b"confirmed")
            plan = make_chain_plan()
            plan.shots[1].confirmed_visual_inputs = [
                confirmed_reference(reference, usage="start_frame")
            ]
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, temp)

        self.assertEqual([1, 2], [call.shot_id for call in provider.calls])
        self.assertEqual("new_scene_reference", provider.calls[1].continuity_mode)
        self.assertEqual(str(reference.resolve()), provider.calls[1].initial_frame_path)
        self.assertNotIn(2, manifest.dependency_failed_shots)

    def test_reference_and_upstream_changes_invalidate_reuse(self) -> None:
        provider = FakeVisualProvider()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            plan = make_chain_plan()
            renderer = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            )
            first = renderer.render(plan, temp)
            second = renderer.render(plan, temp)
            upstream = Path(plan.artifacts[0].generated_last_frame_path)
            upstream.write_bytes(upstream.read_bytes() + b"|changed")
            third = renderer.render(plan, temp)

        self.assertEqual([1, 2], first.generated_shots)
        self.assertEqual([1, 2], second.reused_shots)
        self.assertEqual([2], third.generated_shots)
        self.assertEqual([1], third.reused_shots)
        self.assertEqual(3, len(provider.calls))

    def test_old_manifest_loads_with_new_evidence_defaults(self) -> None:
        manifest = GuidedStorySession._manifest_from(
            {
                "status": "succeeded",
                "output_dir": "old",
                "generated_shots": [1],
                "artifacts": [],
            }
        )
        self.assertEqual([], manifest.dependency_failed_shots)
        self.assertEqual([], manifest.unreferenced_fallback_shots)
        self.assertEqual("", manifest.render_run_id)

    def test_text_only_provider_marks_unreferenced_fallback(self) -> None:
        class TextProvider(FakeVisualProvider):
            capabilities = ProviderCapabilities(True, False, False, True)

        provider = TextProvider()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            reference = Path(temp) / "hero.png"
            reference.write_bytes(b"hero")
            plan = StoryboardPlan(
                "fallback",
                3,
                [make_shot(1, mode="new_scene_reference")],
                "",
                confirmed=True,
            )
            plan.shots[0].reference_image_paths = [str(reference)]
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, temp)

        self.assertEqual([1], manifest.unreferenced_fallback_shots)
        self.assertEqual("", provider.calls[0].initial_frame_path)
        self.assertEqual([], provider.calls[0].reference_image_paths)

    def test_legacy_provider_without_capabilities_cannot_silently_break_chain(self) -> None:
        class LegacyProvider:
            provider_name = "legacy"
            endpoint = "legacy-v1"

            def __init__(self) -> None:
                self.calls: list[StoryboardShot] = []

            def generate_shot(self, shot, output_dir, *, attempt=1, **kwargs):
                del kwargs
                self.calls.append(deepcopy(shot))
                target = Path(output_dir) / f"legacy-{shot.shot_id}.mp4"
                target.write_bytes(b"legacy")
                return VideoArtifact(
                    f"legacy-{shot.shot_id}",
                    shot.shot_id,
                    self.provider_name,
                    self.endpoint,
                    "succeeded",
                    str(target),
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                    attempt=attempt,
                )

        provider = LegacyProvider()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(make_chain_plan(), temp)

        self.assertEqual([1], [shot.shot_id for shot in provider.calls])
        self.assertEqual([2], manifest.dependency_failed_shots)
        self.assertTrue(
            any(
                "未声明视觉能力" in message
                for message in manifest.artifacts[1].continuity_diagnostics
            )
        )

    def test_reference_only_provider_receives_upstream_frame_as_reference(self) -> None:
        class ReferenceProvider(FakeVisualProvider):
            capabilities = ProviderCapabilities(True, False, True, False)

        provider = ReferenceProvider()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(make_chain_plan(), temp)

        self.assertEqual("succeeded", manifest.status)
        self.assertEqual("", provider.calls[1].initial_frame_path)
        self.assertTrue(
            provider.calls[1].reference_image_paths[0].endswith(
                "shot_001_last.png"
            )
        )

    def test_hard_continuity_mismatch_blocks_provider_call(self) -> None:
        provider = FakeVisualProvider()
        plan = make_chain_plan()
        plan.shots[1].continuity_start_state.character_clothing["邮差"] = "红色大衣"
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, temp)

        self.assertEqual([1], [call.shot_id for call in provider.calls])
        self.assertEqual([2], manifest.dependency_failed_shots)
        self.assertIn("人物服装", manifest.error)

    def test_unconfirmed_plan_never_calls_provider(self) -> None:
        provider = FakeVisualProvider()
        plan = make_chain_plan()
        plan.confirmed = False
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "尚未确认"):
                StoryRenderer(provider).render(plan, temp)
        self.assertEqual([], provider.calls)

    def test_changed_input_does_not_resubmit_existing_pending_task(self) -> None:
        class PendingProvider(FakeVisualProvider):
            def generate_shot(self, shot, output_dir, *, attempt=1, **kwargs):
                del output_dir, kwargs
                self.calls.append(deepcopy(shot))
                return VideoArtifact(
                    "pending",
                    shot.shot_id,
                    self.provider_name,
                    self.endpoint,
                    "pending",
                    "",
                    "",
                    shot.duration,
                    shot.video_prompt,
                    "now",
                    request_id="remote-pending",
                    attempt=attempt,
                )

        provider = PendingProvider()
        with tempfile.TemporaryDirectory() as temp:
            plan = StoryboardPlan(
                "pending",
                3,
                [make_shot(1, mode="new_scene_reference")],
                "",
                confirmed=True,
            )
            plan.shots[0].seed = 1
            renderer = StoryRenderer(provider, narration=EmptyNarration())
            first = renderer.render(plan, temp)
            plan.shots[0].seed = 2
            second = renderer.render(plan, temp)

        self.assertEqual("pending", first.status)
        self.assertEqual("pending", second.status)
        self.assertEqual(1, len(provider.calls))

    def test_changed_input_does_not_resubmit_submission_uncertain_task(self) -> None:
        provider = FakeVisualProvider()
        shot = make_shot(1)
        plan = StoryboardPlan("uncertain", 3, [shot], "", confirmed=True)
        plan.artifacts.append(
            VideoArtifact(
                "uncertain",
                1,
                provider.provider_name,
                provider.endpoint,
                "submission_uncertain",
                "",
                "",
                shot.duration,
                shot.video_prompt,
                "now",
                request_id="local-submit-unknown",
                input_fingerprint="old-version",
            )
        )
        shot.seed = 999
        with tempfile.TemporaryDirectory() as temp:
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
            ).render(plan, temp)

        self.assertEqual("submission_uncertain", manifest.status)
        self.assertEqual([], provider.calls)

    def test_agnes_payload_uses_only_documented_image_and_seed_fields(self) -> None:
        provider = AgnesVideoProvider(
            api_key="test",
            image_url_resolver=lambda path: f"https://assets.example/{Path(path).name}",
        )
        shot = make_shot(1, mode="new_scene_reference")
        shot.initial_frame_path = "hero.png"
        shot.seed = 42
        payload = provider._build_payload(shot)

        self.assertEqual("https://assets.example/hero.png", payload["image"])
        self.assertEqual(42, payload["seed"])
        self.assertNotIn("reference_images", payload)

    def test_identity_reference_is_never_promoted_to_agnes_image(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            root = Path(temp) / "public"
            root.mkdir()
            identity = Path(temp) / "identity.png"
            identity.write_bytes(b"identity")
            shot = make_shot(1, mode="new_scene_reference")
            shot.confirmed_visual_inputs = [
                confirmed_reference(
                    identity,
                    usage="identity_reference",
                    reference_id="hero-identity",
                )
            ]
            plan = StoryboardPlan("identity", 3, [shot], "", confirmed=True)
            freeze_confirmed_visual_inputs(plan)
            provider = MockAgnesProvider(
                LocalPublicImagePublisher(root, "https://assets.example/video")
            )
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, Path(temp) / "private-output")

        self.assertNotIn("image", provider.payloads[0])
        self.assertEqual([1], manifest.unreferenced_fallback_shots)
        self.assertIn("identity_reference", {
            item.usage for item in manifest.artifacts[0].confirmed_visual_inputs
        })

    def test_new_scene_uses_only_explicit_confirmed_start_frame(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            root = Path(temp) / "public"
            root.mkdir()
            identity = Path(temp) / "identity.png"
            start = Path(temp) / "scene-start.png"
            identity.write_bytes(b"identity")
            start.write_bytes(b"scene-start")
            shot = make_shot(1, mode="new_scene_reference")
            shot.confirmed_visual_inputs = [
                confirmed_reference(
                    identity,
                    usage="identity_reference",
                    reference_id="hero-identity",
                ),
                confirmed_reference(
                    start,
                    usage="start_frame",
                    reference_id="scene-start",
                ),
            ]
            plan = StoryboardPlan("start", 3, [shot], "", confirmed=True)
            freeze_confirmed_visual_inputs(plan)
            provider = MockAgnesProvider(
                LocalPublicImagePublisher(root, "https://assets.example/video")
            )
            StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, Path(temp) / "private-output")

        self.assertIn("image", provider.payloads[0])
        self.assertIn("shot_001_start", str(provider.payloads[0]["image"]))
        self.assertNotIn("identity.png", str(provider.payloads[0]["image"]))

    def test_public_image_resolver_stays_inside_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "public"
            root.mkdir()
            image = root / "角色 图.png"
            image.write_bytes(b"image")
            outside = Path(temp) / "outside.png"
            outside.write_bytes(b"outside")
            resolver = build_public_image_url_resolver(
                root,
                "https://cdn.example/assets",
            )
            self.assertEqual(
                "https://cdn.example/assets/%E8%A7%92%E8%89%B2%20%E5%9B%BE.png",
                resolver(str(image)),
            )
            with self.assertRaisesRegex(ValueError, "不在 VIDEO_REFERENCE_ROOT"):
                resolver(str(outside))

    def test_private_output_last_frame_is_atomically_staged_under_public_root(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            root = Path(temp) / "public"
            root.mkdir()
            output = Path(temp) / "private-output"
            provider = MockAgnesProvider(
                LocalPublicImagePublisher(root, "https://assets.example/video")
            )
            plan = make_chain_plan()
            plan.shots[0].seed = 101
            plan.shots[1].seed = 202
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, output)

            first = next(item for item in manifest.artifacts if item.shot_id == 1)
            second = next(item for item in manifest.artifacts if item.shot_id == 2)
            self.assertTrue(Path(first.generated_last_frame_path).is_relative_to(output))
            self.assertTrue(Path(first.published_last_frame_path).is_relative_to(root))
            self.assertFalse(output.is_relative_to(root))
            self.assertEqual(first.published_last_frame_url, provider.payloads[1]["image"])
            self.assertEqual(first.generated_last_frame_path, second.initial_frame_source_path)
            self.assertEqual(first.published_last_frame_path, second.initial_frame_path)
            self.assertEqual(202, provider.payloads[1]["seed"])
            self.assertEqual(shot_num_frames(plan.shots[1]), provider.payloads[1]["num_frames"])
            self.assertTrue(second.input_fingerprint)

    def test_reused_last_frame_url_is_rebuilt_with_current_public_base(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            target = Path(temp) / "private-output"
            old_provider = MockAgnesProvider(
                LocalPublicImagePublisher(
                    Path(temp) / "old-public",
                    "https://old-assets.example/video",
                )
            )
            plan = make_chain_plan()
            first = StoryRenderer(
                old_provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, target)
            old_url = next(
                item.published_last_frame_url
                for item in first.artifacts
                if item.shot_id == 1
            )
            self.assertTrue(old_url.startswith("https://old-assets.example/video/"))

            new_provider = MockAgnesProvider(
                LocalPublicImagePublisher(
                    Path(temp) / "new-public",
                    "https://new-assets.example/video",
                )
            )
            second = StoryRenderer(
                new_provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(plan, target)

        self.assertIn(1, second.reused_shots)
        self.assertEqual([2], [shot.shot_id for shot in new_provider.calls])
        self.assertTrue(
            str(new_provider.payloads[0]["image"]).startswith(
                "https://new-assets.example/video/"
            )
        )
        refreshed = next(item for item in second.artifacts if item.shot_id == 1)
        self.assertTrue(
            refreshed.published_last_frame_url.startswith(
                "https://new-assets.example/video/"
            )
        )
        self.assertNotEqual(old_url, refreshed.published_last_frame_url)

    def test_publication_failure_blocks_downstream_provider_call(self) -> None:
        class FailingPublisherProvider(FakeVisualProvider):
            capabilities = ProviderCapabilities(True, True, False, True, True)

            def prepare_image_input(self, source_path, *, run_id, label):
                del source_path, run_id, label
                raise RuntimeError("public root is read-only")

        provider = FailingPublisherProvider()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = StoryRenderer(
                provider,
                narration=EmptyNarration(),
                assembler=fake_assemble,
                frame_extractor=fake_extract,
            ).render(make_chain_plan(), temp)

        self.assertEqual([1], [shot.shot_id for shot in provider.calls])
        self.assertEqual([2], manifest.dependency_failed_shots)
        self.assertIn("公网暂存失败", manifest.artifacts[0].continuity_diagnostics[0])

    def test_fingerprint_includes_visual_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "hero.png"
            image.write_bytes(b"one")
            shot = make_shot(1, mode="new_scene_reference")
            shot.reference_image_paths = [str(image)]
            first = build_input_fingerprint(shot, provider="fake", model="v1")
            image.write_bytes(b"two")
            second = build_input_fingerprint(shot, provider="fake", model="v1")
        self.assertNotEqual(first, second)

    def test_all_render_inputs_invalidate_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "identity.png"
            upstream = Path(temp) / "last.png"
            reference.write_bytes(b"identity-v1")
            upstream.write_bytes(b"last-v1")
            shot = make_shot(2, mode="same_scene_chain", previous_shot_id=1)
            shot.seed = 77
            shot.duration = 6
            shot.reference_image_paths = [str(reference)]
            shot.initial_frame_path = str(upstream)
            baseline = build_input_fingerprint(shot, provider="fake", model="v1")

            changed_seed = deepcopy(shot)
            changed_seed.seed = 78
            changed_duration = deepcopy(shot)
            changed_duration.duration = 7
            changed_mode = deepcopy(shot)
            changed_mode.continuity_mode = "new_scene_reference"
            self.assertNotEqual(
                baseline,
                build_input_fingerprint(changed_seed, provider="fake", model="v1"),
            )
            self.assertNotEqual(
                baseline,
                build_input_fingerprint(changed_duration, provider="fake", model="v1"),
            )
            self.assertNotEqual(
                baseline,
                build_input_fingerprint(changed_mode, provider="fake", model="v1"),
            )
            self.assertNotEqual(
                baseline,
                build_input_fingerprint(shot, provider="fake", model="v2"),
            )
            reference.write_bytes(b"identity-v2")
            changed_reference = build_input_fingerprint(
                shot,
                provider="fake",
                model="v1",
            )
            self.assertNotEqual(baseline, changed_reference)
            reference.write_bytes(b"identity-v1")
            upstream.write_bytes(b"last-v2")
            changed_upstream = build_input_fingerprint(
                shot,
                provider="fake",
                model="v1",
            )
            self.assertNotEqual(baseline, changed_upstream)

    def test_confirmed_visual_substitution_invalidates_confirmation(self) -> None:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=15),
            RuleBasedStoryAgent(),
        )
        session.start_ideation("雨夜车站")
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        plan = session.build_storyboard()
        provider = FakeVisualProvider()
        with tempfile.TemporaryDirectory() as temp:
            start = Path(temp) / "start.png"
            start.write_bytes(b"confirmed-content")
            plan.shots[0].confirmed_visual_inputs = [
                confirmed_reference(start, usage="start_frame")
            ]
            session.confirm_storyboard()
            start.write_bytes(b"replaced-content")
            with self.assertRaisesRegex(RuntimeError, "确认已失效"):
                session.render_confirmed_plan(
                    StoryRenderer(provider, narration=EmptyNarration()),
                    Path(temp) / "output",
                )

        self.assertFalse(session.storyboard.confirmed)
        self.assertEqual([], provider.calls)

    def test_retake_assigns_a_new_seed_and_requires_reconfirmation(self) -> None:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=15),
            RuleBasedStoryAgent(),
        )
        session.start_ideation("车站交接")
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        plan = session.build_storyboard()
        original_seed = plan.shots[0].seed
        session.update_storyboard_shot(
            plan.shots[0].shot_id,
            {"retake_instruction": "动作更克制"},
        )
        self.assertNotEqual(original_seed, session.storyboard.shots[0].seed)
        self.assertFalse(session.storyboard.confirmed)
        self.assertEqual(Stage.STORYBOARD_REVIEW, session.stage)

    def test_confirmation_signature_covers_seed_duration_mode_and_visual_metadata(
        self,
    ) -> None:
        shot = make_shot(1)
        shot.seed = 10
        shot.confirmed_visual_inputs = [
            VisualReference(
                "ref-1",
                "identity.png",
                "identity_reference",
                "digest-1",
                "summary-1",
                True,
            )
        ]
        plan = StoryboardPlan("signature", 3, [shot], "", confirmed=True)
        baseline = GuidedStorySession._storyboard_confirmation_signature(plan)
        for field, value in (
            ("seed", 11),
            ("duration", 4),
            ("continuity_mode", "new_scene_reference"),
        ):
            changed = deepcopy(plan)
            setattr(changed.shots[0], field, value)
            self.assertNotEqual(
                baseline,
                GuidedStorySession._storyboard_confirmation_signature(changed),
            )
        changed_visual = deepcopy(plan)
        changed_visual.shots[0].confirmed_visual_inputs[0].usage = "start_frame"
        self.assertNotEqual(
            baseline,
            GuidedStorySession._storyboard_confirmation_signature(changed_visual),
        )

    def test_full_confirmed_agnes_chain_uses_real_staged_last_frame_without_api(
        self,
    ) -> None:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=15),
            RuleBasedStoryAgent(),
        )
        session.start_ideation("雨夜车站，一名邮差必须交出怀表")
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        plan = session.build_storyboard()
        chained_shot = next(
            shot
            for shot in plan.shots
            if shot.continuity_mode == "same_scene_reference"
        )
        plan = session.update_storyboard_shot(
            chained_shot.shot_id,
            {
                "continuity_mode": "same_scene_chain",
                "transition_type": "continuous_action",
                "transition_reason": "测试显式连续动作末帧继承",
                "inherit_previous_frame": True,
            },
        )
        session.confirm_storyboard()
        self.assertTrue(any(shot.inherit_previous_frame for shot in plan.shots))

        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            root = Path(temp) / "public"
            root.mkdir()
            provider = MockAgnesProvider(
                LocalPublicImagePublisher(root, "https://assets.example/video")
            )
            manifest = session.render_confirmed_plan(
                StoryRenderer(
                    provider,
                    narration=EmptyNarration(),
                    assembler=fake_assemble,
                    frame_extractor=fake_extract,
                ),
                Path(temp) / "private-output",
            )

        chained_index = next(
            index
            for index, shot in enumerate(plan.shots)
            if shot.inherit_previous_frame
        )
        payload = provider.payloads[chained_index]
        call = provider.calls[chained_index]
        artifact = manifest.artifacts[chained_index]
        self.assertIn("image", payload)
        self.assertEqual(call.initial_frame_url, payload["image"])
        self.assertEqual(call.seed, payload["seed"])
        self.assertEqual(shot_num_frames(call), payload["num_frames"])
        self.assertTrue(artifact.input_fingerprint)
        self.assertTrue(artifact.initial_frame_source_path.endswith("_last.png"))
        self.assertTrue(artifact.initial_frame_path)
        self.assertTrue(artifact.initial_frame_url)

    def test_storyboard_ui_shows_mode_assets_and_start_reference(self) -> None:
        plan = make_chain_plan()
        markdown = _storyboard_markdown(plan)
        self.assertIn("连续性模式", markdown)
        self.assertIn("same_scene_chain", markdown)
        self.assertIn("镜头 1 的生成末帧", markdown)
        self.assertIn("时长理由", markdown)
        self.assertIn("Seed", markdown)

    def test_video_summary_distinguishes_all_evidence_states(self) -> None:
        summary = _render_evidence_summary(
            GuidedStorySession._manifest_from(
                {
                    "status": "failed",
                    "output_dir": "out",
                    "generated_shots": [1, 2],
                    "reused_shots": [3],
                    "dependency_failed_shots": [4],
                    "unreferenced_fallback_shots": [5],
                    "final_video_path": "final.mp4",
                }
            )
        )
        self.assertIn("重新生成 2", summary)
        self.assertIn("复用 1", summary)
        self.assertIn("依赖失败 1", summary)
        self.assertIn("无参考回退 1", summary)
        self.assertIn("成片 1", summary)

    def test_session_accepts_runtime_frame_evidence_without_invalidating_confirmation(
        self,
    ) -> None:
        session = GuidedStorySession(
            CreativeBrief(target_seconds=15),
            RuleBasedStoryAgent(),
        )
        session.start_ideation("雨夜车站")
        session.generate_story()
        session.confirm_story()
        session.generate_script()
        session.confirm_script()
        session.build_storyboard()
        session.confirm_storyboard()
        provider = FakeVisualProvider()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("guided_story_agent.rendering.validate_mp4_file", return_value=True),
        ):
            manifest = session.render_confirmed_plan(
                StoryRenderer(
                    provider,
                    narration=EmptyNarration(),
                    assembler=fake_assemble,
                    frame_extractor=fake_extract,
                ),
                temp,
            )

        self.assertIn(manifest.status, {"succeeded", "succeeded_with_warnings"})
        self.assertEqual(Stage.COMPLETED, session.stage)
        self.assertTrue(session.storyboard.confirmed)
        self.assertTrue(
            any(item.generated_last_frame_path for item in session.storyboard.artifacts)
        )


if __name__ == "__main__":
    unittest.main()
