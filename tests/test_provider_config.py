from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guided_story_agent.agent import OpenAIStoryAgent
from guided_story_agent.provider_config import VideoProviderConfig
from guided_story_agent.video_provider import AgnesVideoProvider, VideoGenerationError


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> SimpleNamespace:
        self.requests.append(request)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class JsonRepairTests(unittest.TestCase):
    def test_invalid_json_gets_one_structured_repair_attempt(self) -> None:
        client = _FakeClient([
            "这是一段普通文字，不是 JSON。",
            '{"ok": true}',
        ])
        agent = OpenAIStoryAgent(client, "test-model")
        result = agent._json_completion("idea_divergence.md", {"direction": "test"})
        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, len(client.chat.completions.requests))
        self.assertIn("结构化输出修复器", str(client.chat.completions.requests[1]))
        self.assertTrue(agent.last_model_response_preview)

    def test_fenced_markdown_json_is_parsed_without_repair(self) -> None:
        client = _FakeClient(['```json\n{"ok": true}\n```'])
        agent = OpenAIStoryAgent(client, "test-model")
        result = agent._json_completion("idea_divergence.md", {"direction": "test"})
        self.assertEqual({"ok": True}, result)
        self.assertEqual(1, len(client.chat.completions.requests))

    def test_json_mode_disabled_never_sends_response_format(self) -> None:
        client = _FakeClient(['{"ok": true}'])
        agent = OpenAIStoryAgent(client, "test-model", json_mode="disabled")
        agent._json_completion("idea_divergence.md", {"direction": "test"})
        self.assertNotIn("response_format", client.chat.completions.requests[0])

    def test_json_mode_required_fails_fast_on_provider_rejection(self) -> None:
        class _RejectingCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **request: object) -> None:
                self.calls += 1
                raise RuntimeError("model does not support response_format")

        class _RejectingClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=_RejectingCompletions())

        client = _RejectingClient()
        agent = OpenAIStoryAgent(client, "test-model", json_mode="required")
        with self.assertRaises(RuntimeError):
            agent._json_completion("idea_divergence.md", {"direction": "test"})
        self.assertEqual(1, client.chat.completions.calls)

    def test_json_mode_auto_downgrades_once_when_response_format_rejected(self) -> None:
        class _DowngradingCompletions:
            def __init__(self) -> None:
                self.calls = 0
                self.requests: list[dict[str, object]] = []

            def create(self, **request: object) -> SimpleNamespace:
                self.calls += 1
                self.requests.append(request)
                if "response_format" in request:
                    raise RuntimeError("response_format is not supported")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))
                    ]
                )

        class _DowngradingClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=_DowngradingCompletions())

        client = _DowngradingClient()
        agent = OpenAIStoryAgent(client, "test-model", json_mode="auto")
        result = agent._json_completion("idea_divergence.md", {"direction": "test"})
        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, client.chat.completions.calls)
        self.assertIn("response_format", client.chat.completions.requests[0])
        self.assertNotIn("response_format", client.chat.completions.requests[1])


class VideoConfigTests(unittest.TestCase):
    def test_video_config_reads_network_retry_values(self) -> None:
        values = {
            "VIDEO_API_KEY": "video-key",
            "VIDEO_API_ROOT": "https://video.example",
            "VIDEO_MODEL": "video-model",
            "VIDEO_NETWORK_RETRIES": "3",
            "VIDEO_RETRY_BACKOFF": "0.5",
        }
        with patch("guided_story_agent.provider_config.load_dotenv"), patch.dict(
            os.environ, values, clear=True
        ):
            config = VideoProviderConfig.from_env(
                default_api_root="https://apihub.agnes-ai.com",
                default_model="agnes-video-v2.0",
            )
        self.assertEqual(3, config.network_retries)
        self.assertEqual(0.5, config.retry_backoff)
        self.assertTrue(config.configured)


class _FakeUrlopenResponse:
    def __init__(self, body: bytes) -> None:
        self._stream = io.BytesIO(body)

    def __enter__(self) -> _FakeUrlopenResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self, *args: int) -> bytes:
        return self._stream.read(*args)


class VideoRetryTests(unittest.TestCase):
    def test_status_request_retries_transient_network_errors(self) -> None:
        provider = AgnesVideoProvider(api_key="k", network_retries=2, retry_backoff=0.01)
        calls = {"n": 0}
        body = json.dumps({"status": "completed"}).encode("utf-8")

        def fake_urlopen(request: object, timeout: float) -> _FakeUrlopenResponse:
            calls["n"] += 1
            if calls["n"] < 3:
                raise OSError("temporary network failure")
            return _FakeUrlopenResponse(body)

        with patch(
            "guided_story_agent.video_provider.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = provider._status("vid-1")
        self.assertEqual("completed", result["status"])
        self.assertEqual(3, calls["n"])

    def test_submit_post_is_never_retried(self) -> None:
        provider = AgnesVideoProvider(api_key="k", network_retries=3, retry_backoff=0.01)
        calls = {"n": 0}

        def failing_urlopen(request: object, timeout: float) -> _FakeUrlopenResponse:
            calls["n"] += 1
            raise OSError("network down")

        with patch(
            "guided_story_agent.video_provider.urllib.request.urlopen",
            side_effect=failing_urlopen,
        ):
            with self.assertRaises(VideoGenerationError):
                provider._submit({"model": "m"})
        self.assertEqual(1, calls["n"])

    def test_download_retries_transient_errors_then_succeeds(self) -> None:
        provider = AgnesVideoProvider(api_key="k", network_retries=2, retry_backoff=0.01)
        calls = {"n": 0}

        def flaky_urlopen(request: object, timeout: float) -> _FakeUrlopenResponse:
            calls["n"] += 1
            if calls["n"] < 3:
                raise OSError("temporary")
            return _FakeUrlopenResponse(b"video-bytes")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shot.mp4"
            with patch(
                "guided_story_agent.video_provider.urllib.request.urlopen",
                side_effect=flaky_urlopen,
            ):
                provider._download_file("https://example.com/v.mp4", target)
            self.assertEqual(3, calls["n"])
            self.assertEqual(b"video-bytes", target.read_bytes())

    def test_poll_interval_floor_prevents_busy_loop(self) -> None:
        provider = AgnesVideoProvider(api_key="k", poll_interval=0)
        self.assertGreaterEqual(provider.poll_interval, 0.1)


if __name__ == "__main__":
    unittest.main()
