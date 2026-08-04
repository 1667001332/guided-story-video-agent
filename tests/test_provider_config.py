from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from guided_story_agent.agent import OpenAIStoryAgent
from guided_story_agent.provider_config import (
    TextProviderConfig,
    VideoProviderConfig,
)


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


class ProviderConfigTests(unittest.TestCase):
    def test_text_blank_generic_values_do_not_hide_legacy_config(self) -> None:
        environment = {
            "TEXT_PROVIDER": "openai_compatible",
            "TEXT_API_KEY": "",
            "TEXT_BASE_URL": "https://template.example/v1",
            "TEXT_MODEL": "template-model",
            "DEEPSEEK_API_KEY": "legacy-text-key",
            "DEEPSEEK_TEXT_MODEL": "legacy-text-model",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv"),
        ):
            config = TextProviderConfig.from_env()

        self.assertTrue(config.configured)
        self.assertEqual("legacy-text-key", config.api_key)
        self.assertEqual("legacy-text-model", config.model)
        self.assertEqual("DEEPSEEK_* (legacy)", config.source)

    def test_explicit_text_offline_wins_over_legacy_key(self) -> None:
        environment = {
            "TEXT_PROVIDER": "offline",
            "TEXT_API_KEY": "",
            "DEEPSEEK_API_KEY": "legacy-text-key",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv"),
        ):
            config = TextProviderConfig.from_env()

        self.assertFalse(config.configured)
        self.assertEqual("offline", config.provider)
        self.assertIn("已关闭", config.error)
        self.assertEqual("TEXT_*", config.source)

    def test_video_blank_generic_values_do_not_hide_legacy_config(self) -> None:
        environment = {
            "VIDEO_PROVIDER": "agnes",
            "VIDEO_API_KEY": "",
            "VIDEO_API_ROOT": "https://template.example",
            "VIDEO_MODEL": "template-video-model",
            "AGNES_API_KEY": "legacy-video-key",
            "AGNES_VIDEO_MODEL": "legacy-video-model",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv"),
        ):
            config = VideoProviderConfig.from_env(
                default_api_root="https://default.example",
                default_model="default-video-model",
            )

        self.assertTrue(config.configured)
        self.assertEqual("legacy-video-key", config.api_key)
        self.assertEqual("legacy-video-model", config.model)
        self.assertEqual("AGNES_* (legacy)", config.source)

    def test_explicit_video_disabled_wins_over_legacy_key(self) -> None:
        environment = {
            "VIDEO_PROVIDER": "disabled",
            "VIDEO_API_KEY": "",
            "AGNES_API_KEY": "legacy-video-key",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv"),
        ):
            config = VideoProviderConfig.from_env(
                default_api_root="https://default.example",
                default_model="default-video-model",
            )

        self.assertFalse(config.configured)
        self.assertEqual("disabled", config.provider)
        self.assertIn("已关闭", config.error)
        self.assertEqual("VIDEO_*", config.source)

    def test_default_dotenv_is_exactly_current_working_directory(self) -> None:
        expected = (Path.cwd() / ".env").resolve()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv") as loader,
        ):
            TextProviderConfig.from_env()

        loader.assert_called_once_with(dotenv_path=expected, override=False)

    def test_explicit_dotenv_path_is_resolved_and_used(self) -> None:
        requested = Path("config") / "mentor.env"
        expected = requested.expanduser().resolve()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("guided_story_agent.provider_config.load_dotenv") as loader,
        ):
            VideoProviderConfig.from_env(
                default_api_root="https://default.example",
                default_model="default-video-model",
                dotenv_path=requested,
            )

        loader.assert_called_once_with(dotenv_path=expected, override=False)

    def test_text_timeout_rejects_non_finite_or_non_positive_values(self) -> None:
        for value in ("nan", "inf", "0", "-1"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {
                        "TEXT_API_KEY": "test-key",
                        "TEXT_MODEL": "test-model",
                        "TEXT_TIMEOUT": value,
                    },
                    clear=True,
                ),
                patch("guided_story_agent.provider_config.load_dotenv"),
            ):
                config = TextProviderConfig.from_env()
                self.assertFalse(config.configured)
                self.assertIn("TEXT_TIMEOUT", config.error)

    def test_video_polling_numbers_must_be_finite_and_in_range(self) -> None:
        settings = (
            ("VIDEO_TIMEOUT", "0"),
            ("VIDEO_POLL_INTERVAL", "-1"),
            ("VIDEO_POLL_INTERVAL", "nan"),
            ("VIDEO_MAX_POLL_SECONDS", "inf"),
        )
        for name, value in settings:
            with (
                self.subTest(name=name, value=value),
                patch.dict(
                    os.environ,
                    {
                        "VIDEO_API_KEY": "test-key",
                        "VIDEO_MODEL": "test-model",
                        name: value,
                    },
                    clear=True,
                ),
                patch("guided_story_agent.provider_config.load_dotenv"),
            ):
                config = VideoProviderConfig.from_env(
                    default_api_root="https://default.example",
                    default_model="default-video-model",
                )
                self.assertFalse(config.configured)
                self.assertIn(name, config.error)

    def test_video_network_retry_settings_are_validated(self) -> None:
        settings = (
            ("VIDEO_NETWORK_RETRIES", "not-an-integer"),
            ("VIDEO_NETWORK_RETRIES", "11"),
            ("VIDEO_RETRY_BACKOFF", "-1"),
            ("VIDEO_RETRY_BACKOFF", "nan"),
        )
        for name, value in settings:
            with (
                self.subTest(name=name, value=value),
                patch.dict(
                    os.environ,
                    {
                        "VIDEO_API_KEY": "test-key",
                        "VIDEO_MODEL": "test-model",
                        name: value,
                    },
                    clear=True,
                ),
                patch("guided_story_agent.provider_config.load_dotenv"),
            ):
                config = VideoProviderConfig.from_env(
                    default_api_root="https://default.example",
                    default_model="default-video-model",
                )
                self.assertFalse(config.configured)
                self.assertIn(name, config.error)

    def test_video_reference_mapping_requires_root_and_base_url_together(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "VIDEO_API_KEY": "test-key",
                    "VIDEO_MODEL": "test-model",
                    "VIDEO_REFERENCE_ROOT": "public-assets",
                },
                clear=True,
            ),
            patch("guided_story_agent.provider_config.load_dotenv"),
        ):
            config = VideoProviderConfig.from_env(
                default_api_root="https://default.example",
                default_model="default-video-model",
            )

        self.assertFalse(config.configured)
        self.assertIn("必须同时填写", config.error)


if __name__ == "__main__":
    unittest.main()
