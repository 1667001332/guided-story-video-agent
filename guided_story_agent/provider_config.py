from __future__ import annotations

import math
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


TEXT_PROVIDER_ALIASES = {
    "openai": "openai_compatible",
    "openai-compatible": "openai_compatible",
    "openai_compatible": "openai_compatible",
    "deepseek": "openai_compatible",
    "agnes": "openai_compatible",
    "offline": "offline",
    "disabled": "offline",
}
TEXT_JSON_MODES = {"auto", "required", "disabled"}
VIDEO_PROVIDER_ALIASES = {
    "agnes": "agnes",
    "disabled": "disabled",
    "offline": "disabled",
}


def _value(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _load_project_dotenv(dotenv_path: str | os.PathLike[str] | None) -> Path:
    target = (
        Path(dotenv_path).expanduser().resolve()
        if dotenv_path is not None
        else (Path.cwd() / ".env").resolve()
    )
    load_dotenv(dotenv_path=target, override=False)
    return target


def _number(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: str, default: int) -> int:
    if not str(value).strip():
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


@dataclass(frozen=True, slots=True)
class TextProviderConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: float
    json_mode: str
    source: str
    error: str = ""

    @property
    def configured(self) -> bool:
        return not self.error and self.provider != "offline"

    @classmethod
    def from_env(
        cls,
        *,
        dotenv_path: str | os.PathLike[str] | None = None,
    ) -> TextProviderConfig:
        _load_project_dotenv(dotenv_path)
        provider_setting = _value("TEXT_PROVIDER").lower()
        provider_raw = provider_setting or "openai_compatible"
        provider = TEXT_PROVIDER_ALIASES.get(provider_raw, provider_raw)
        explicit_mode = bool(provider_setting)
        generic_key_present = bool(_value("TEXT_API_KEY"))
        if (
            generic_key_present
            or (explicit_mode and provider == "offline")
            or (explicit_mode and provider not in {"openai_compatible", "offline"})
        ):
            provider = TEXT_PROVIDER_ALIASES.get(provider_raw, provider_raw)
            config = cls(
                provider=provider,
                api_key=_value("TEXT_API_KEY"),
                base_url=_value("TEXT_BASE_URL"),
                model=_value("TEXT_MODEL"),
                timeout=_number(_value("TEXT_TIMEOUT", "120"), 120),
                json_mode=_value("TEXT_JSON_MODE", "auto").lower(),
                source="TEXT_*",
            )
            return config._validated(provider_raw)

        deepseek_key = _value("DEEPSEEK_API_KEY")
        if deepseek_key:
            config = cls(
                provider="openai_compatible",
                api_key=deepseek_key,
                base_url=_value("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=_value("DEEPSEEK_TEXT_MODEL", "deepseek-v4-pro"),
                timeout=_number(_value("DEEPSEEK_TIMEOUT", "120"), 120),
                json_mode=_value("TEXT_JSON_MODE", "auto").lower(),
                source="DEEPSEEK_* (legacy)",
            )
            return config._validated("deepseek")

        agnes_key = _value("AGNES_API_KEY")
        if agnes_key:
            config = cls(
                provider="openai_compatible",
                api_key=agnes_key,
                base_url=_value("AGNES_LLM_BASE_URL", "https://apihub.agnes-ai.com/v1"),
                model=_value("AGNES_TEXT_MODEL", "agnes-2.0-flash"),
                timeout=_number(_value("AGNES_TIMEOUT", "120"), 120),
                json_mode=_value("TEXT_JSON_MODE", "auto").lower(),
                source="AGNES_* (legacy)",
            )
            return config._validated("agnes")

        if explicit_mode:
            config = cls(
                provider=provider,
                api_key="",
                base_url=_value("TEXT_BASE_URL"),
                model=_value("TEXT_MODEL"),
                timeout=_number(_value("TEXT_TIMEOUT", "120"), 120),
                json_mode=_value("TEXT_JSON_MODE", "auto").lower(),
                source="TEXT_*",
            )
            return config._validated(provider_raw)

        return cls(
            provider="openai_compatible",
            api_key="",
            base_url="",
            model=_value("TEXT_MODEL") or _value("DEEPSEEK_TEXT_MODEL") or "unconfigured",
            timeout=120,
            json_mode="auto",
            source="none",
            error=("未配置 TEXT_API_KEY，也未发现旧版 DEEPSEEK_API_KEY 或 AGNES_API_KEY。"),
        )

    def _validated(self, provider_raw: str) -> TextProviderConfig:
        if self.provider not in {"openai_compatible", "offline"}:
            return self._with_error(
                f"TEXT_PROVIDER={provider_raw} 暂不支持；当前支持 openai_compatible 或 offline。"
            )
        if self.provider == "offline":
            return self._with_error("TEXT_PROVIDER 已设为 offline，真实文本 API 已关闭。")
        if self.json_mode not in TEXT_JSON_MODES:
            return self._with_error("TEXT_JSON_MODE 必须是 auto、required 或 disabled。")
        if not self.api_key:
            return self._with_error(f"{self.source} 缺少文本 API Key；请填写 TEXT_API_KEY。")
        if not self.model:
            return self._with_error(f"{self.source} 缺少模型 ID；请填写 TEXT_MODEL。")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            return self._with_error("TEXT_TIMEOUT 必须是大于 0 的有限数字。")
        return self

    def _with_error(self, error: str) -> TextProviderConfig:
        return TextProviderConfig(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            json_mode=self.json_mode,
            source=self.source,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class VideoProviderConfig:
    provider: str
    api_key: str
    api_root: str
    model: str
    timeout: float
    poll_interval: float
    max_poll_seconds: float
    source: str
    reference_root: str = ""
    reference_base_url: str = ""
    network_retries: int = 2
    retry_backoff: float = 1.0
    error: str = ""

    @property
    def configured(self) -> bool:
        return not self.error and self.provider != "disabled"

    @classmethod
    def from_env(
        cls,
        *,
        default_api_root: str,
        default_model: str,
        dotenv_path: str | os.PathLike[str] | None = None,
    ) -> VideoProviderConfig:
        _load_project_dotenv(dotenv_path)
        provider_setting = _value("VIDEO_PROVIDER").lower()
        provider_raw = provider_setting or "agnes"
        provider = VIDEO_PROVIDER_ALIASES.get(provider_raw, provider_raw)
        explicit_mode = bool(provider_setting)
        generic_key_present = bool(_value("VIDEO_API_KEY"))
        if (
            generic_key_present
            or (explicit_mode and provider == "disabled")
            or (explicit_mode and provider not in {"agnes", "disabled"})
        ):
            provider = VIDEO_PROVIDER_ALIASES.get(provider_raw, provider_raw)
            config = cls(
                provider=provider,
                api_key=_value("VIDEO_API_KEY"),
                api_root=_value("VIDEO_API_ROOT", default_api_root),
                model=_value("VIDEO_MODEL", default_model),
                timeout=_number(_value("VIDEO_TIMEOUT", "120"), 120),
                poll_interval=_number(_value("VIDEO_POLL_INTERVAL", "5"), 5),
                max_poll_seconds=_number(_value("VIDEO_MAX_POLL_SECONDS", "900"), 900),
                source="VIDEO_*",
                reference_root=_value("VIDEO_REFERENCE_ROOT"),
                reference_base_url=_value("VIDEO_REFERENCE_BASE_URL"),
                network_retries=_integer(_value("VIDEO_NETWORK_RETRIES", "2"), 2),
                retry_backoff=_number(_value("VIDEO_RETRY_BACKOFF", "1"), 1),
            )
            return config._validated(provider_raw)

        agnes_key = _value("AGNES_API_KEY")
        if agnes_key:
            config = cls(
                provider="agnes",
                api_key=agnes_key,
                api_root=_value("AGNES_API_ROOT", default_api_root),
                model=_value("AGNES_VIDEO_MODEL", default_model),
                timeout=_number(_value("AGNES_TIMEOUT", "120"), 120),
                poll_interval=_number(_value("AGNES_POLL_INTERVAL", "5"), 5),
                max_poll_seconds=_number(_value("AGNES_MAX_POLL_SECONDS", "900"), 900),
                source="AGNES_* (legacy)",
                reference_root=_value("VIDEO_REFERENCE_ROOT"),
                reference_base_url=_value("VIDEO_REFERENCE_BASE_URL"),
                network_retries=_integer(_value("VIDEO_NETWORK_RETRIES", "2"), 2),
                retry_backoff=_number(_value("VIDEO_RETRY_BACKOFF", "1"), 1),
            )
            return config._validated("agnes")

        config = cls(
            provider=provider,
            api_key="",
            api_root=_value("VIDEO_API_ROOT", default_api_root),
            model=_value("VIDEO_MODEL", default_model),
            timeout=_number(_value("VIDEO_TIMEOUT", "120"), 120),
            poll_interval=_number(_value("VIDEO_POLL_INTERVAL", "5"), 5),
            max_poll_seconds=_number(_value("VIDEO_MAX_POLL_SECONDS", "900"), 900),
            source="VIDEO_*" if explicit_mode else "none",
            reference_root=_value("VIDEO_REFERENCE_ROOT"),
            reference_base_url=_value("VIDEO_REFERENCE_BASE_URL"),
            network_retries=_integer(_value("VIDEO_NETWORK_RETRIES", "2"), 2),
            retry_backoff=_number(_value("VIDEO_RETRY_BACKOFF", "1"), 1),
        )
        return config._validated("agnes")

    def _validated(self, provider_raw: str) -> VideoProviderConfig:
        if self.provider not in {"agnes", "disabled"}:
            return self._with_error(
                f"VIDEO_PROVIDER={provider_raw} 暂未实现；当前支持 agnes 或 disabled。"
            )
        if self.provider == "disabled":
            return self._with_error("VIDEO_PROVIDER 已设为 disabled，付费视频 API 已关闭。")
        if not self.api_key:
            return self._with_error(f"{self.source} 缺少视频 API Key；请填写 VIDEO_API_KEY。")
        if not self.model:
            return self._with_error(f"{self.source} 缺少视频模型 ID；请填写 VIDEO_MODEL。")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            return self._with_error("VIDEO_TIMEOUT 必须是大于 0 的有限数字。")
        if not math.isfinite(self.poll_interval) or self.poll_interval < 0:
            return self._with_error("VIDEO_POLL_INTERVAL 必须是大于等于 0 的有限数字。")
        if not math.isfinite(self.max_poll_seconds) or self.max_poll_seconds <= 0:
            return self._with_error("VIDEO_MAX_POLL_SECONDS 必须是大于 0 的有限数字。")
        if (
            isinstance(self.network_retries, bool)
            or not isinstance(self.network_retries, int)
            or not 0 <= self.network_retries <= 10
        ):
            return self._with_error("VIDEO_NETWORK_RETRIES 必须是 0 到 10 之间的整数。")
        if not math.isfinite(self.retry_backoff) or self.retry_backoff < 0:
            return self._with_error("VIDEO_RETRY_BACKOFF 必须是大于等于 0 的有限数字。")
        if bool(self.reference_root) != bool(self.reference_base_url):
            return self._with_error(
                "VIDEO_REFERENCE_ROOT 与 VIDEO_REFERENCE_BASE_URL 必须同时填写或同时留空。"
            )
        if self.reference_base_url:
            parsed = urllib.parse.urlsplit(self.reference_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return self._with_error(
                    "VIDEO_REFERENCE_BASE_URL 必须是有效的 http(s) URL。"
                )
        return self

    def _with_error(self, error: str) -> VideoProviderConfig:
        return VideoProviderConfig(
            provider=self.provider,
            api_key=self.api_key,
            api_root=self.api_root,
            model=self.model,
            timeout=self.timeout,
            poll_interval=self.poll_interval,
            max_poll_seconds=self.max_poll_seconds,
            source=self.source,
            reference_root=self.reference_root,
            reference_base_url=self.reference_base_url,
            network_retries=self.network_retries,
            retry_backoff=self.retry_backoff,
            error=error,
        )
