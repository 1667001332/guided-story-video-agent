from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guided_story_agent import RuleBasedStoryAgent
from guided_story_agent.batch_test import (
    BatchCase,
    _run_id,
    default_cases_source,
    load_cases,
    main,
    run_batch,
)
from guided_story_agent.models import ProviderCapabilities, RenderManifest, VideoArtifact
from guided_story_agent.narration import NarrationArtifact
from guided_story_agent.rendering import StoryRenderer


def fake_last_frame(video_path, output_path):
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(video_path).read_bytes() + b"|last")
    return str(target)


class BatchTestTests(unittest.TestCase):
    def test_load_jsonl_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "cases.jsonl"
            source.write_text(
                "\n".join(
                    [
                        '{"id":"a","direction":"校园悬疑","target_seconds":30}',
                        '{"id":"b","direction":"未来城市"}',
                    ]
                ),
                encoding="utf-8",
            )
            cases = load_cases(source)
        self.assertEqual(["a", "b"], [case.case_id for case in cases])
        self.assertEqual(30, cases[0].target_seconds)
        self.assertIsNone(cases[1].target_seconds)

    def test_default_cases_are_loaded_from_package_resource(self) -> None:
        cases = load_cases(default_cases_source())
        self.assertGreaterEqual(len(cases), 1)
        self.assertTrue(cases[0].direction)

    def test_offline_batch_writes_summary_csv_and_case_artifacts(self) -> None:
        cases = [
            BatchCase("mystery", "校园悬疑", 30),
            BatchCase("science", "未来城市", 45),
        ]
        with tempfile.TemporaryDirectory() as temp:
            result = run_batch(
                cases=cases,
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                require_live_text=False,
                progress_callback=None,
            )
            target = Path(temp)
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            with (target / "results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            transcript = json.loads(
                (Path(result["results"][0]["output_dir"]) / "transcript.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(2, summary["total_runs"])
        self.assertEqual(2, summary["executed_runs"])
        self.assertEqual(2, summary["succeeded"])
        self.assertEqual(0, summary["failed"])
        self.assertEqual(2, len(rows))
        self.assertEqual("校园悬疑", transcript["direction"])
        self.assertEqual(2, len(result["results"]))

    def test_resume_skips_successful_run(self) -> None:
        cases = [BatchCase("resume", "雨夜车站", 30)]
        with tempfile.TemporaryDirectory() as temp:
            run_batch(
                cases=cases,
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                require_live_text=False,
                progress_callback=None,
            )

            factory_calls = 0

            def identity_probe():
                nonlocal factory_calls
                factory_calls += 1
                return RuleBasedStoryAgent()

            resumed = run_batch(
                cases=cases,
                output_dir=temp,
                agent_factory=identity_probe,
                require_live_text=False,
                resume=True,
                progress_callback=None,
            )

        self.assertEqual(1, factory_calls)
        self.assertEqual(1, resumed["summary"]["skipped_by_resume"])
        self.assertEqual(0, resumed["summary"]["executed_runs"])
        self.assertEqual(1, resumed["summary"]["succeeded"])
        self.assertTrue(resumed["results"][0]["resume_identity_verified"])

    def test_resume_rejects_changed_direction_and_provider_model(self) -> None:
        class ModelAgent(RuleBasedStoryAgent):
            provider_name = "fake-provider"

            def __init__(self, model: str) -> None:
                self.model = model

        with tempfile.TemporaryDirectory() as temp:
            run_batch(
                cases=[BatchCase("same", "旧方向", 30)],
                output_dir=temp,
                agent_factory=lambda: ModelAgent("model-a"),
                require_live_text=False,
                progress_callback=None,
            )
            rerun = run_batch(
                cases=[BatchCase("same", "新方向", 30)],
                output_dir=temp,
                agent_factory=lambda: ModelAgent("model-b"),
                require_live_text=False,
                resume=True,
                progress_callback=None,
            )

        self.assertEqual(0, rerun["summary"]["skipped_by_resume"])
        self.assertEqual(1, rerun["summary"]["executed_runs"])
        self.assertEqual(
            1,
            rerun["summary"]["resume_rejections"]["identity_mismatch"],
        )
        self.assertEqual("新方向", rerun["results"][0]["direction"])
        self.assertEqual(
            "model-b",
            rerun["results"][0]["run_identity"]["text"]["model"],
        )

    def test_legacy_result_without_identity_is_not_resumed(self) -> None:
        cases = [BatchCase("legacy", "雨夜车站", 30)]
        with tempfile.TemporaryDirectory() as temp:
            first = run_batch(
                cases=cases,
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                require_live_text=False,
                progress_callback=None,
            )
            run_dir = Path(first["results"][0]["attempt_dir"]).parent
            result_path = run_dir / "result.json"
            legacy = json.loads(result_path.read_text(encoding="utf-8"))
            legacy.pop("run_identity")
            legacy.pop("run_identity_hash")
            result_path.write_text(
                json.dumps(legacy, ensure_ascii=False),
                encoding="utf-8",
            )

            rerun = run_batch(
                cases=cases,
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                require_live_text=False,
                resume=True,
                progress_callback=None,
            )

        self.assertEqual(0, rerun["summary"]["skipped_by_resume"])
        self.assertEqual(
            1,
            rerun["summary"]["resume_rejections"]["legacy_missing_identity"],
        )

    def test_strict_resume_does_not_accept_previous_offline_success(self) -> None:
        cases = [BatchCase("strict", "雨夜车站", 30)]
        with tempfile.TemporaryDirectory() as temp:
            run_batch(
                cases=cases,
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                require_live_text=False,
                progress_callback=None,
            )
            strict = run_batch(
                cases=cases,
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                require_live_text=True,
                resume=True,
                progress_callback=None,
            )

        self.assertEqual(0, strict["summary"]["skipped_by_resume"])
        self.assertEqual(1, strict["summary"]["failed"])
        self.assertEqual(
            1,
            strict["summary"]["resume_rejections"]["identity_mismatch"],
        )

    def test_failed_attempt_has_existing_output_and_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = run_batch(
                cases=[BatchCase("failure", "雨夜车站", 30)],
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                require_live_text=True,
                progress_callback=None,
            )
            failed = result["results"][0]
            output_dir = Path(failed["output_dir"])
            attempt_dir = Path(failed["attempt_dir"])
            diagnostic = json.loads((attempt_dir / "attempt.json").read_text(encoding="utf-8"))
            output_exists = output_dir.is_dir()
            attempt_exists = attempt_dir.is_dir()

        self.assertEqual("failed", failed["status"])
        self.assertTrue(output_exists)
        self.assertTrue(attempt_exists)
        self.assertEqual("failed", diagnostic["status"])
        self.assertEqual("story", diagnostic["phase"])

    def test_video_retry_reuses_successful_shots_in_same_plan_and_directory(self) -> None:
        calls: list[tuple[int, int, str]] = []
        failed_once = False

        class Narration:
            def synthesize(self, plan, output_dir):
                return NarrationArtifact("", "")

        class Provider:
            endpoint = "fake-video-model"
            provider_name = "fake-video"
            capabilities = ProviderCapabilities(True, True, False, False)

            def generate_shot(self, shot, output_dir, **kwargs):
                nonlocal failed_once
                attempt = int(kwargs.get("attempt", 1))
                calls.append((shot.shot_id, attempt, str(output_dir)))
                if shot.shot_id == 2 and not failed_once:
                    failed_once = True
                    raise RuntimeError("transient")
                path = Path(output_dir) / f"{shot.shot_id}-{attempt}.mp4"
                path.write_bytes(b"video")
                return VideoArtifact(
                    artifact_id=f"{shot.shot_id}-{attempt}",
                    shot_id=shot.shot_id,
                    provider=self.provider_name,
                    model=self.endpoint,
                    status="succeeded",
                    local_path=str(path),
                    remote_url="",
                    duration=shot.duration,
                    prompt=shot.video_prompt,
                    created_at="now",
                    attempt=attempt,
                )

        provider = Provider()

        def assemble(paths, output):
            Path(output).write_bytes(b"joined")
            return str(output)

        def renderer_factory():
            return StoryRenderer(
                provider,
                narration=Narration(),
                assembler=assemble,
                frame_extractor=fake_last_frame,
            )

        agent_factory_calls = 0

        def agent_factory():
            nonlocal agent_factory_calls
            agent_factory_calls += 1
            return RuleBasedStoryAgent()

        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "guided_story_agent.rendering.validate_mp4_file",
                return_value=True,
            ),
        ):
            result = run_batch(
                cases=[BatchCase("render-retry", "雨夜车站", 30)],
                output_dir=temp,
                agent_factory=agent_factory,
                require_live_text=False,
                render=True,
                renderer_factory=renderer_factory,
                retries=1,
                progress_callback=None,
            )

        second_attempt_calls = [shot_id for shot_id, attempt, _ in calls if attempt == 2]
        successful_first_calls = [
            shot_id for shot_id, attempt, _ in calls if attempt == 1 and shot_id != 2
        ]
        self.assertEqual(
            "succeeded",
            result["results"][0]["status"],
            result["results"][0],
        )
        self.assertEqual(1, agent_factory_calls)
        self.assertIn(2, second_attempt_calls)
        self.assertTrue(set(successful_first_calls).isdisjoint(second_attempt_calls))
        self.assertEqual(
            len(successful_first_calls),
            len(result["results"][0]["bench"]["reused_shots"]),
        )
        self.assertEqual(1, len({output_dir for _, _, output_dir in calls}))

    def test_render_warning_is_success_but_pending_is_failure(self) -> None:
        class Provider:
            provider_name = "fake"
            endpoint = "fake"

        class Renderer:
            provider = Provider()

            def __init__(self, status: str) -> None:
                self.status = status

            def render(self, plan, output_dir):
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                return RenderManifest(
                    status=self.status,
                    output_dir=str(output_dir),
                    error="provider warning" if "warning" in self.status else "",
                )

        with tempfile.TemporaryDirectory() as temp:
            warned = run_batch(
                cases=[BatchCase("warned", "雨夜车站", 30)],
                output_dir=Path(temp) / "warned",
                agent_factory=RuleBasedStoryAgent,
                render=True,
                renderer_factory=lambda: Renderer("succeeded_with_warnings"),
                require_live_text=False,
                progress_callback=None,
            )
            pending = run_batch(
                cases=[BatchCase("pending", "雨夜车站", 30)],
                output_dir=Path(temp) / "pending",
                agent_factory=RuleBasedStoryAgent,
                render=True,
                renderer_factory=lambda: Renderer("pending"),
                require_live_text=False,
                progress_callback=None,
            )

        self.assertEqual(1, warned["summary"]["succeeded"])
        self.assertEqual(1, warned["summary"]["succeeded_with_warnings"])
        self.assertEqual("provider warning", warned["results"][0]["warning"])
        self.assertEqual(1, pending["summary"]["failed"])
        self.assertEqual("RenderFailed", pending["results"][0]["error_type"])

    def test_pending_video_retry_reuses_request_id(self) -> None:
        calls: list[tuple[int, str | None, str]] = []

        class Narration:
            def synthesize(self, plan, output_dir):
                return NarrationArtifact("", "")

        class Provider:
            endpoint = "fake-pending-model"
            provider_name = "fake-pending"
            capabilities = ProviderCapabilities(True, True, False, False)

            def generate_shot(
                self,
                shot,
                output_dir,
                *,
                attempt=1,
                progress_callback=None,
                resume_request_id=None,
            ):
                del progress_callback
                calls.append((shot.shot_id, resume_request_id, str(output_dir)))
                if shot.shot_id == 1 and resume_request_id is None:
                    return VideoArtifact(
                        artifact_id="pending-1",
                        shot_id=shot.shot_id,
                        provider=self.provider_name,
                        model=self.endpoint,
                        status="pending",
                        local_path="",
                        remote_url="",
                        duration=shot.duration,
                        prompt=shot.video_prompt,
                        created_at="now",
                        request_id="request-1",
                        attempt=attempt,
                    )
                path = Path(output_dir) / f"{shot.shot_id}-{attempt}.mp4"
                path.write_bytes(b"video")
                return VideoArtifact(
                    artifact_id=f"{shot.shot_id}-{attempt}",
                    shot_id=shot.shot_id,
                    provider=self.provider_name,
                    model=self.endpoint,
                    status="succeeded",
                    local_path=str(path),
                    remote_url="",
                    duration=shot.duration,
                    prompt=shot.video_prompt,
                    created_at="now",
                    request_id=resume_request_id,
                    attempt=attempt,
                )

        provider = Provider()

        def assemble(paths, output):
            Path(output).write_bytes(b"joined")
            return str(output)

        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "guided_story_agent.rendering.validate_mp4_file",
                return_value=True,
            ),
        ):
            result = run_batch(
                cases=[BatchCase("pending-retry", "雨夜车站", 30)],
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                render=True,
                renderer_factory=lambda: StoryRenderer(
                provider,
                narration=Narration(),
                assembler=assemble,
                frame_extractor=fake_last_frame,
            ),
                require_live_text=False,
                retries=1,
                progress_callback=None,
            )

        shot_one_calls = [
            resume_request_id for shot_id, resume_request_id, _ in calls if shot_id == 1
        ]
        self.assertEqual("succeeded", result["results"][0]["status"])
        self.assertEqual([None, "request-1"], shot_one_calls)
        self.assertEqual(1, len({output_dir for _, _, output_dir in calls}))

    def test_resume_in_new_batch_call_continues_pending_request(self) -> None:
        calls: list[str | None] = []

        class Narration:
            def synthesize(self, plan, output_dir):
                return NarrationArtifact("", "")

        class Provider:
            endpoint = "fake-resume-model"
            provider_name = "fake-resume"
            capabilities = ProviderCapabilities(True, True, False, False)

            def generate_shot(
                self,
                shot,
                output_dir,
                *,
                attempt=1,
                progress_callback=None,
                resume_request_id=None,
            ):
                del progress_callback
                calls.append(resume_request_id)
                if shot.shot_id == 1 and resume_request_id is None:
                    return VideoArtifact(
                        artifact_id="pending-restart",
                        shot_id=shot.shot_id,
                        provider=self.provider_name,
                        model=self.endpoint,
                        status="pending",
                        local_path="",
                        remote_url="",
                        duration=shot.duration,
                        prompt=shot.video_prompt,
                        created_at="now",
                        request_id="request-restart",
                        attempt=attempt,
                    )
                path = Path(output_dir) / f"resumed-{shot.shot_id}.mp4"
                path.write_bytes(b"video")
                return VideoArtifact(
                    artifact_id=f"resumed-{shot.shot_id}",
                    shot_id=shot.shot_id,
                    provider=self.provider_name,
                    model=self.endpoint,
                    status="succeeded",
                    local_path=str(path),
                    remote_url="",
                    duration=shot.duration,
                    prompt=shot.video_prompt,
                    created_at="now",
                    request_id=resume_request_id,
                    attempt=attempt,
                )

        provider = Provider()

        def renderer():
            return StoryRenderer(
                provider,
                narration=Narration(),
                assembler=lambda paths, output: str(output),
                frame_extractor=fake_last_frame,
            )

        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                "guided_story_agent.rendering.validate_mp4_file",
                return_value=True,
            ),
        ):
            first = run_batch(
                cases=[BatchCase("restart", "雨夜车站", 15)],
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                render=True,
                renderer_factory=renderer,
                require_live_text=False,
                progress_callback=None,
            )
            second = run_batch(
                cases=[BatchCase("restart", "雨夜车站", 15)],
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                render=True,
                renderer_factory=renderer,
                require_live_text=False,
                resume=True,
                progress_callback=None,
            )

        self.assertEqual("failed", first["results"][0]["status"])
        self.assertEqual("succeeded", second["results"][0]["status"])
        self.assertTrue(second["results"][0]["resumed_incomplete"])
        self.assertEqual([None, "request-restart"], calls[:2])

    def test_resume_rejects_changed_pipeline_fingerprint(self) -> None:
        cases = [BatchCase("pipeline", "雨夜车站", 30)]
        with tempfile.TemporaryDirectory() as temp:
            with patch(
                "guided_story_agent.batch_test._pipeline_fingerprint",
                return_value={
                    "package_version": "0.5.1",
                    "content_sha256": "old",
                },
            ):
                run_batch(
                    cases=cases,
                    output_dir=temp,
                    agent_factory=RuleBasedStoryAgent,
                    require_live_text=False,
                    progress_callback=None,
                )
            with patch(
                "guided_story_agent.batch_test._pipeline_fingerprint",
                return_value={
                    "package_version": "0.5.1",
                    "content_sha256": "new",
                },
            ):
                rerun = run_batch(
                    cases=cases,
                    output_dir=temp,
                    agent_factory=RuleBasedStoryAgent,
                    require_live_text=False,
                    resume=True,
                    progress_callback=None,
                )

        self.assertEqual(0, rerun["summary"]["skipped_by_resume"])
        self.assertEqual(
            1,
            rerun["summary"]["resume_rejections"]["identity_mismatch"],
        )

    def test_run_ids_do_not_collide_after_sanitizing(self) -> None:
        left = _run_id(1, "a/b", 1, 1)
        right = _run_id(1, "a?b", 1, 1)
        self.assertNotEqual(left.casefold(), right.casefold())

    def test_csv_prefixes_formula_like_user_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_batch(
                cases=[BatchCase("formula", '=HYPERLINK("bad")', 30)],
                output_dir=temp,
                agent_factory=RuleBasedStoryAgent,
                require_live_text=False,
                progress_callback=None,
            )
            with (Path(temp) / "results.csv").open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                row = next(csv.DictReader(handle))

        self.assertTrue(row["direction"].startswith("'="))

    def test_cli_rejects_invalid_repeat_before_running(self) -> None:
        with (
            patch("sys.argv", ["guided-story-batch", "--repeat", "0"]),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(2, raised.exception.code)


if __name__ == "__main__":
    unittest.main()
