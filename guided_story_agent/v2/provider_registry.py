"""Provider runtime registry with explicit, fail-closed assembly."""

from __future__ import annotations

from typing import Any

from .provider_runtime import ProviderRuntime


class ProviderNotRegisteredError(RuntimeError):
    pass


class ProviderRegistrationError(ValueError):
    pass


class ProviderRuntimeRegistry:
    """In-memory assembly registry.

    The registry is deliberately not serializable and never stores secrets as
    part of runtime state.  Applications re-inject adapters on recovery.
    """

    def __init__(self, providers: dict[str, ProviderRuntime] | None = None) -> None:
        self._providers: dict[str, ProviderRuntime] = {}
        for key, runtime in (providers or {}).items():
            self.register(key, runtime)

    def register(
        self,
        provider_or_key: ProviderRuntime | str,
        runtime: ProviderRuntime | None = None,
        *,
        replace: bool = False,
    ) -> None:
        if runtime is None:
            runtime = provider_or_key  # type: ignore[assignment]
            key = str(getattr(runtime, "provider_key", "")).strip()
        else:
            key = str(provider_or_key).strip()
        if not key:
            raise ProviderRegistrationError("provider_key is required")
        if not hasattr(runtime, "capabilities") or not callable(runtime.capabilities):
            raise ProviderRegistrationError(f"ProviderRuntime {key!r} has no capabilities()")
        runtime_key = str(getattr(runtime, "provider_key", key)).strip()
        if runtime_key and runtime_key != key:
            raise ProviderRegistrationError(f"ProviderRuntime key mismatch: registry={key}, runtime={runtime_key}")
        if key in self._providers and not replace:
            raise ProviderRegistrationError(f"ProviderRuntime already registered: {key}")
        self._providers[key] = runtime

    def unregister(self, provider_key: str) -> ProviderRuntime:
        key = str(provider_key).strip()
        try:
            return self._providers.pop(key)
        except KeyError as exc:
            raise ProviderNotRegisteredError(f"ProviderRuntime 未注册：{key}") from exc

    def get(self, provider_key: str) -> ProviderRuntime:
        key = str(provider_key).strip()
        try:
            return self._providers[key]
        except KeyError as exc:
            raise ProviderNotRegisteredError(f"ProviderRuntime 未注册：{key}；Runtime fail-closed。") from exc

    def require(self, provider_key: str) -> ProviderRuntime:
        return self.get(provider_key)

    def contains(self, provider_key: str) -> bool:
        return str(provider_key).strip() in self._providers

    def list_provider_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def keys(self) -> tuple[str, ...]:
        return self.list_provider_keys()

    def get_capabilities(self, provider_key: str) -> Any:
        return self.get(provider_key).capabilities()

    @classmethod
    def with_fake(
        cls,
        *,
        provider_keys: tuple[str, ...] = ("fake",),
        scenario: Any = None,
        clock: Any = None,
    ) -> "ProviderRuntimeRegistry":
        from .fake_provider_runtime import FakeProviderRuntime

        registry = cls()
        for key in provider_keys:
            profile = "offline" if key == "offline-v2" else ""
            registry.register(FakeProviderRuntime(scenario, provider_key=key, provider_profile=profile, clock=clock))
        return registry

    @classmethod
    def with_mock_http(cls, runtime: ProviderRuntime | None = None) -> "ProviderRuntimeRegistry":
        if runtime is None:
            from .mock_http_provider_runtime import MockHttpProviderRuntime
            from .mock_http_transport import MockHttpTransport

            runtime = MockHttpProviderRuntime(MockHttpTransport())
        registry = cls()
        registry.register(runtime)
        return registry


__all__ = ["ProviderNotRegisteredError", "ProviderRegistrationError", "ProviderRuntimeRegistry"]
