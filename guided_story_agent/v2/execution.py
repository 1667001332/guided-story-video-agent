"""Execution-layer contracts for the V2 compiler/provider boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Capabilities declared by one provider adapter.

    These values constrain compilation; they never rewrite a MoviePlan.  The
    first ``provider_key`` field is retained as the stable V2 identifier while
    ``provider_name`` is exposed as a readable compatibility alias.
    """

    provider_key: str = "generic"
    provider_profile: str = ""
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    supports_long_video: bool = True
    supports_multi_scene_prompt: bool = True
    supports_reference_images: bool = False
    supports_character_reference: bool = False
    supports_audio: bool = True
    supports_subtitles: bool = False
    supported_aspect_ratios: tuple[str, ...] = ()
    supported_resolutions: tuple[str, ...] = ()
    supported_fps: tuple[float, ...] = ()
    output_formats: tuple[str, ...] = ("mp4",)
    provider_name: str = ""

    def __post_init__(self) -> None:
        key = self.provider_key.strip() or self.provider_name.strip() or "generic"
        name = self.provider_name.strip() or key
        if self.provider_key == "generic" and self.provider_name.strip():
            key = self.provider_name.strip()
        object.__setattr__(self, "provider_key", key)
        object.__setattr__(self, "provider_name", name)

    @property
    def supports_image_references(self) -> bool:
        return self.supports_reference_images

    @property
    def supports_character_references(self) -> bool:
        return self.supports_character_reference


@dataclass(frozen=True, slots=True)
class CompilationOptions:
    """Explicit execution choices supplied to the compiler.

    Options are execution concerns, not creative defaults.  Empty optional
    values remain empty; the compiler never invents missing creative data.
    """

    aspect_ratio: str = "16:9"
    resolution: str = ""
    fps: float | None = None
    output_format: str = "mp4"
    negative_prompt: str = ""
    references: tuple[str, ...] = ()
    character_references: tuple[str, ...] = ()
    continuity_references: tuple[str, ...] = ()
    provider_profile: str = ""
    compiler_version: str = "v2-compiler/1"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompileError:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class CompileDiagnostic:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class CompileResult:
    video_job: "VideoJob | None"
    errors: tuple[CompileError, ...] = ()
    warnings: tuple[CompileDiagnostic, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.video_job is not None and not self.errors


@dataclass(frozen=True, slots=True)
class RetakeRequest:
    """Allowed retake scope; it deliberately has no story/script fields."""

    retake_id: str
    camera_requirements: str = ""
    lighting_requirements: str = ""
    performance_requirements: str = ""
    shot_language_requirements: str = ""
    scene_ids: tuple[str, ...] = ()
    reason: str = ""

    def validate(self) -> None:
        if not self.retake_id.strip():
            raise ValueError("retake_id is required")
        if not any(
            value.strip()
            for value in (
                self.camera_requirements,
                self.lighting_requirements,
                self.performance_requirements,
                self.shot_language_requirements,
            )
        ):
            raise ValueError("RetakeRequest must add at least one production requirement")


@dataclass(frozen=True, slots=True)
class VideoJob:
    """One complete provider execution request.

    The prompt fields are produced by a Compiler from a MovieIR.  This class
    never joins story fields or invents provider language.
    """

    job_id: str
    provider_key: str
    provider_prompt: str
    negative_prompt: str
    duration_seconds: float
    output_format: str
    aspect_ratio: str = "16:9"
    resolution: str = ""
    fps: float | None = None
    references: tuple[str, ...] = ()
    character_references: tuple[str, ...] = ()
    continuity_references: tuple[str, ...] = ()
    source_movie_plan_id: str = ""
    source_movie_ir_id: str = ""
    compiler_version: str = ""
    provider_profile: str = ""
    execution_units: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confirmed: bool = False
    source_film_ir_id: str = ""
    source_movie_plan_version: int = 0
    source_movie_plan_fingerprint: str = ""
    source_movie_plan_lineage_token: str = ""
    source_film_ir_fingerprint: str = ""
    source_movie_ir_fingerprint: str = ""
    video_job_fingerprint: str = ""
    schema_version: str = "v2-video-job/1"

    def __post_init__(self) -> None:
        if not self.video_job_fingerprint:
            # Import lazily to keep the historical execution module usable as
            # a leaf contract and to avoid a module cycle.
            from .execution_fingerprint import video_job_fingerprint

            object.__setattr__(self, "video_job_fingerprint", video_job_fingerprint(self))

    @property
    def fingerprint(self) -> str:
        return self.video_job_fingerprint

    def to_dict(self) -> dict[str, Any]:
        from .models import as_plain_data

        return as_plain_data(self)

    @property
    def prompt(self) -> str:
        """Read-only compatibility alias for provider_prompt."""

        return self.provider_prompt

    @property
    def reference_paths(self) -> tuple[str, ...]:
        return self.references

    @property
    def character_reference_paths(self) -> tuple[str, ...]:
        return self.character_references

@dataclass(frozen=True, slots=True)
class ProviderJob:
    provider_key: str
    request_id: str
    status: str
    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_error: str = ""
