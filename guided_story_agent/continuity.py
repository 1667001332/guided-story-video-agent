from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .models import (
    ContinuityState,
    StoryboardPlan,
    StoryboardShot,
    VisualReference,
)


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CONTINUITY_MODES = {
    "independent",
    "same_scene_chain",
    "same_scene_reference",
    "new_scene_reference",
}
TRANSITION_TYPES = {
    "opening",
    "continuous_action",
    "same_scene_cut",
    "reverse_shot",
    "insert_shot",
    "reaction_cut",
    "scene_change",
    "montage",
    "independent",
}
VISUAL_REFERENCE_USAGES = {
    "identity_reference",
    "location_reference",
    "prop_reference",
    "scene_reference",
    "start_frame",
}


@dataclass(slots=True)
class ContinuityCheck:
    hard_errors: list[str]
    warnings: list[str]


def resolve_reference_assets(
    plan: StoryboardPlan,
    *,
    base_dir: str | Path | None = None,
    freeze: bool = False,
) -> list[str]:
    """Resolve explicit visual references without promoting identity art to a frame."""
    root = Path(base_dir or Path.cwd()).expanduser().resolve()
    assets = {asset.asset_id: asset for asset in plan.visual_bible.assets}
    diagnostics: list[str] = []

    for shot in plan.shots:
        resolved_references: list[VisualReference] = []
        seen: set[str] = set()
        shot_messages: list[str] = []
        candidates = [
            reference
            for reference in shot.confirmed_visual_inputs
            if reference.binding_kind != "asset"
        ]
        if not candidates:
            for index, raw_path in enumerate(shot.reference_image_paths, start=1):
                candidates.append(
                    VisualReference(
                        reference_id=f"legacy-shot-{shot.shot_id}-{index}",
                        path=raw_path,
                        usage="scene_reference",
                    )
                )
        for asset_id in shot.reference_asset_ids:
            asset = assets.get(asset_id)
            if asset is None:
                shot_messages.append(f"引用资产不存在：{asset_id}")
                continue
            asset_references = [
                VisualReference(
                    reference_id=reference.reference_id,
                    path=reference.path,
                    usage=reference.usage,
                    content_digest=reference.content_digest,
                    content_summary=reference.content_summary,
                    confirmed=reference.confirmed,
                    binding_kind="asset",
                    binding_id=asset.asset_id,
                )
                for reference in asset.references
            ]
            legacy_usage = _legacy_usage_for_asset(asset.kind)
            asset_references.extend(
                VisualReference(
                    reference_id=f"{asset.asset_id}-legacy-{index}",
                    path=raw_path,
                    usage=legacy_usage,
                    binding_kind="asset",
                    binding_id=asset.asset_id,
                )
                for index, raw_path in enumerate(asset.reference_images, start=1)
            )
            if not asset_references:
                shot_messages.append(f"资产 {asset_id} 尚未绑定参考图")
                continue
            candidates.extend(asset_references)
        for reference in candidates:
            if reference.usage not in VISUAL_REFERENCE_USAGES:
                shot_messages.append(
                    f"参考图 {reference.reference_id or '未命名'} 用途无效："
                    f"{reference.usage or '空'}"
                )
                continue
            candidate, error = _resolve_local_image(reference.path, root)
            if error:
                shot_messages.append(
                    f"参考图 {reference.reference_id or '未命名'}：{error}"
                )
                continue
            key = f"{candidate!s}|{reference.usage}".casefold()
            if key in seen:
                continue
            seen.add(key)
            evidence = _file_evidence(str(candidate))
            digest = str(evidence.get("sha256", ""))
            if (
                reference.content_digest
                and reference.content_digest != digest
                and not freeze
            ):
                shot_messages.append(
                    f"已确认参考图内容发生变化：{reference.reference_id or candidate.name}"
                )
                continue
            resolved_references.append(
                VisualReference(
                    reference_id=(
                        reference.reference_id
                        or f"shot-{shot.shot_id}-{len(resolved_references) + 1}"
                    ),
                    path=str(candidate),
                    usage=reference.usage,
                    content_digest=digest,
                    content_summary=(
                        reference.content_summary
                        or f"sha256:{digest[:12]}，{evidence.get('size', 0)} bytes"
                    ),
                    confirmed=bool(reference.confirmed or freeze),
                    binding_kind=reference.binding_kind,
                    binding_id=reference.binding_id,
                )
            )
        shot.confirmed_visual_inputs = resolved_references
        shot.reference_image_paths = [
            reference.path
            for reference in resolved_references
            if reference.confirmed and reference.usage != "start_frame"
        ]
        shot.continuity_diagnostics = _unique(
            [*shot.continuity_diagnostics, *shot_messages]
        )
        diagnostics.extend(
            f"镜头 {shot.shot_id}：{message}" for message in shot_messages
        )
    return diagnostics


def freeze_confirmed_visual_inputs(
    plan: StoryboardPlan,
    *,
    base_dir: str | Path | None = None,
) -> list[str]:
    """Freeze selected image IDs, usages and content digests at confirmation."""
    return resolve_reference_assets(plan, base_dir=base_dir, freeze=True)


def verify_confirmed_visual_inputs(plan: StoryboardPlan) -> list[str]:
    """Detect path/content substitution after a storyboard was confirmed."""
    errors: list[str] = []
    for shot in plan.shots:
        for reference in shot.confirmed_visual_inputs:
            if not reference.confirmed:
                continue
            evidence = _file_evidence(reference.path)
            if evidence.get("status") != "ready":
                errors.append(
                    f"镜头 {shot.shot_id} 的已确认视觉输入不存在："
                    f"{reference.reference_id}"
                )
                continue
            digest = str(evidence.get("sha256", ""))
            if not reference.content_digest or reference.content_digest != digest:
                errors.append(
                    f"镜头 {shot.shot_id} 的已确认视觉输入内容已变化："
                    f"{reference.reference_id}"
                )
    return errors


def confirmed_start_frame(shot: StoryboardShot) -> VisualReference | None:
    """Return only an explicitly confirmed start-frame reference."""
    return next(
        (
            reference
            for reference in shot.confirmed_visual_inputs
            if reference.confirmed and reference.usage == "start_frame"
        ),
        None,
    )


def assign_continuity_modes(shots: list[StoryboardShot]) -> None:
    """Keep normal camera cuts independent; chain frames only when explicitly requested."""
    previous: StoryboardShot | None = None
    for shot in shots:
        if previous is None:
            shot.previous_shot_id = None
            shot.inherit_previous_frame = False
            shot.transition_type = "opening"
            shot.transition_reason = "开场镜头，独立建立机位和构图"
            shot.continuity_mode = (
                "new_scene_reference" if shot.reference_asset_ids else "independent"
            )
        elif same_scene(previous, shot) and shot.inherit_previous_frame:
            shot.previous_shot_id = previous.shot_id
            shot.continuity_mode = "same_scene_chain"
            shot.transition_type = "continuous_action"
            shot.transition_reason = (
                shot.transition_reason
                or "动作被拆成连续过程，使用上一镜头真实末帧作为当前首帧"
            )
        elif same_scene(previous, shot):
            shot.previous_shot_id = previous.shot_id
            shot.inherit_previous_frame = False
            shot.continuity_mode = "same_scene_reference"
            if shot.transition_type not in TRANSITION_TYPES or shot.transition_type in {
                "opening",
                "scene_change",
                "continuous_action",
                "independent",
            }:
                shot.transition_type = "same_scene_cut"
            shot.transition_reason = (
                shot.transition_reason
                or "同一场景正常切换机位；继承人物和场景状态，但不继承上一镜头构图"
            )
        else:
            shot.previous_shot_id = None
            shot.inherit_previous_frame = False
            shot.transition_type = "scene_change"
            shot.transition_reason = "地点、场景或时段变化，使用新场景自己的构图和参考输入"
            shot.continuity_mode = (
                "new_scene_reference" if shot.reference_asset_ids else "independent"
            )
        previous = shot


def same_scene(previous: StoryboardShot, current: StoryboardShot) -> bool:
    previous_state = previous.continuity_end_state
    current_state = current.continuity_start_state
    previous_time = previous_state.time_of_day.strip()
    current_time = current_state.time_of_day.strip()
    time_matches = not previous_time or not current_time or previous_time == current_time
    return (
        previous.scene_id == current.scene_id
        and previous.location.strip() == current.location.strip()
        and time_matches
    )


def validate_continuity_boundary(
    previous: StoryboardShot,
    current: StoryboardShot,
) -> ContinuityCheck:
    """Compare the previous end state with the current start state."""
    if current.continuity_mode not in {"same_scene_chain", "same_scene_reference"}:
        return ContinuityCheck([], [])

    before = previous.continuity_end_state
    after = current.continuity_start_state
    hard_errors: list[str] = []
    warnings: list[str] = []

    for field_name, label in (
        ("location", "地点"),
        ("time_of_day", "时段"),
        ("weather", "天气"),
        ("key_light_direction", "主光方向"),
    ):
        left = str(getattr(before, field_name, "") or "").strip()
        right = str(getattr(after, field_name, "") or "").strip()
        if left and right and left != right:
            hard_errors.append(
                f"镜头 {previous.shot_id}→{current.shot_id} 的{label}不连续："
                f"“{left}”→“{right}”"
            )

    for field_name, label in (
        ("character_appearance", "人物外观"),
        ("character_clothing", "人物服装"),
        ("character_injuries", "人物伤势"),
    ):
        hard_errors.extend(
            _compare_mapping(
                getattr(before, field_name),
                getattr(after, field_name),
                previous.shot_id,
                current.shot_id,
                label,
            )
        )

    for field_name, label in (
        ("character_positions", "人物位置"),
        ("character_emotions", "人物情绪"),
        ("character_knowledge", "人物已知信息"),
        ("character_held_props", "人物持有道具"),
        ("prop_positions", "道具位置"),
    ):
        warnings.extend(
            _compare_mapping(
                getattr(before, field_name),
                getattr(after, field_name),
                previous.shot_id,
                current.shot_id,
                label,
            )
        )

    return ContinuityCheck(_unique(hard_errors), _unique(warnings))


def build_input_fingerprint(
    shot: StoryboardShot,
    *,
    provider: str,
    model: str,
) -> str:
    """Hash all generation inputs, including upstream and fixed visual evidence."""
    paths = [shot.initial_frame_path, *shot.reference_image_paths]
    payload = {
        "provider": provider,
        "model": model,
        "shot_id": shot.shot_id,
        "duration": shot.duration,
        "prompt": shot.video_prompt,
        "negative_prompt": shot.negative_prompt,
        "continuity_mode": shot.continuity_mode,
        "transition_type": shot.transition_type,
        "transition_reason": shot.transition_reason,
        "inherit_previous_frame": shot.inherit_previous_frame,
        "previous_shot_id": shot.previous_shot_id,
        "seed": shot.seed,
        "initial_frame_url": shot.initial_frame_url,
        "confirmed_visual_inputs": [
            {
                "reference_id": reference.reference_id,
                "usage": reference.usage,
                "content_digest": reference.content_digest,
                "content_summary": reference.content_summary,
                "confirmed": reference.confirmed,
                "binding_kind": reference.binding_kind,
                "binding_id": reference.binding_id,
            }
            for reference in shot.confirmed_visual_inputs
        ],
        "visual_inputs": [_file_evidence(path) for path in paths if path],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def continuity_state_to_dict(state: ContinuityState) -> dict[str, object]:
    return {
        "character_appearance": dict(state.character_appearance),
        "character_clothing": dict(state.character_clothing),
        "character_positions": dict(state.character_positions),
        "character_emotions": dict(state.character_emotions),
        "character_knowledge": {
            key: list(value) for key, value in state.character_knowledge.items()
        },
        "character_injuries": dict(state.character_injuries),
        "character_held_props": {
            key: list(value) for key, value in state.character_held_props.items()
        },
        "prop_positions": dict(state.prop_positions),
        "location": state.location,
        "time_of_day": state.time_of_day,
        "weather": state.weather,
        "key_light_direction": state.key_light_direction,
    }


def _resolve_local_image(raw_path: str, root: Path) -> tuple[Path, str]:
    value = str(raw_path or "").strip()
    if not value:
        return root, "参考图路径为空"
    direct_path = Path(value).expanduser()
    parsed = urlparse(value)
    if direct_path.is_absolute():
        candidate = direct_path
    elif parsed.scheme and parsed.scheme.lower() != "file":
        return root, "参考图必须是本地文件，不能直接写远程 URL"
    else:
        candidate = Path(parsed.path if parsed.scheme == "file" else value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return candidate, f"不支持的图片类型：{candidate.suffix or '无扩展名'}"
    if not candidate.is_file():
        return candidate, f"参考图不存在：{candidate}"
    if candidate.stat().st_size <= 0:
        return candidate, f"参考图为空文件：{candidate}"
    return candidate, ""


def _legacy_usage_for_asset(kind: str) -> str:
    return {
        "character": "identity_reference",
        "location": "location_reference",
        "prop": "prop_reference",
        "scene": "scene_reference",
    }.get(str(kind).strip().lower(), "scene_reference")


def _file_evidence(raw_path: str) -> dict[str, object]:
    path = Path(raw_path)
    evidence: dict[str, object] = {"path": str(path)}
    if not path.is_file():
        evidence["status"] = "missing"
        return evidence
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    evidence["status"] = "ready"
    evidence["size"] = path.stat().st_size
    evidence["sha256"] = digest.hexdigest()
    return evidence


def _compare_mapping(
    before: dict[str, object],
    after: dict[str, object],
    previous_shot_id: int,
    current_shot_id: int,
    label: str,
) -> list[str]:
    messages: list[str] = []
    for key in sorted(set(before).intersection(after)):
        left = before[key]
        right = after[key]
        if _has_value(left) and _has_value(right) and left != right:
            messages.append(
                f"镜头 {previous_shot_id}→{current_shot_id} 的{label}"
                f"“{key}”发生未解释变化：{left!r}→{right!r}"
            )
    return messages


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
