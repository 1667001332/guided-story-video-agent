from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ShotTimingDemand:
    """Structured, auditable inputs for one shot's duration demand."""

    shot_kind: str
    purpose: str
    priority: int
    action: str
    dialogue: str
    narration: str
    emotional_change: str
    scene_duration: int
    scene_shot_count: int


_BASE_SECONDS = {
    "establish": 3.8,
    "detail": 3.2,
    "action": 4.4,
    "dialogue": 3.5,
    "reaction": 4.2,
    "transition": 3.4,
}


def estimate_shot_duration(demand: ShotTimingDemand) -> tuple[float, float, str]:
    """Estimate content need without turning shot kinds into fixed durations."""
    kind = demand.shot_kind if demand.shot_kind in _BASE_SECONDS else "action"
    base = _BASE_SECONDS[kind]
    action_steps = _count_action_steps(demand.action)
    action_increment = max(0, action_steps - 1) * 0.85
    complex_markers = sum(
        marker in demand.action
        for marker in (
            "同时",
            "随后",
            "突然",
            "转身",
            "穿过",
            "追逐",
            "争夺",
            "打开",
            "取出",
            "递给",
            "倒下",
        )
    )
    action_increment += complex_markers * 0.35

    dialogue_chars = _spoken_character_count(demand.dialogue)
    dialogue_increment = 0.0
    if dialogue_chars:
        dialogue_increment = dialogue_chars / 4.2
        dialogue_increment += max(0, _pause_count(demand.dialogue) - 1) * 0.35

    narration_chars = _spoken_character_count(demand.narration)
    narration_share = narration_chars / max(1, demand.scene_shot_count)
    narration_increment = narration_share / 5.0

    emotion_increment = 0.0
    if demand.emotional_change.strip():
        emotion_increment = 1.1 if kind == "reaction" else 0.45

    combined_text = f"{demand.purpose} {demand.action}"
    role_labels: list[str] = []
    role_increment = 0.0
    for label, markers, increment in (
        ("建立空间", ("建立", "交代空间", "空间关系"), 0.7),
        ("揭示信息", ("揭示", "证据", "真相", "发现", "确认"), 1.0),
        ("高潮", ("高潮", "决战", "爆发", "生死", "关键转折"), 1.4),
        ("转场", ("转场", "衔接", "状态变化"), 0.45),
    ):
        if any(marker in combined_text for marker in markers):
            role_labels.append(label)
            role_increment += increment

    scene_share = max(3.0, float(demand.scene_duration)) / max(
        1, demand.scene_shot_count
    )
    content_estimate = (
        base
        + action_increment
        + dialogue_increment
        + narration_increment
        + emotion_increment
        + role_increment
    )
    estimated = 0.8 * content_estimate + 0.2 * scene_share
    estimated = min(15.0, max(3.0, estimated))
    priority_multiplier = 1.0 + max(0, demand.priority - 3) * 0.06
    weight = max(0.1, (estimated - 2.0) * priority_multiplier)

    reasons = [f"{kind}基础{base:.1f}秒"]
    if action_steps > 1 or complex_markers:
        reasons.append(f"{action_steps}个动作步骤/复杂动作")
    if dialogue_chars:
        reasons.append(f"对白{dialogue_chars}字")
    if narration_chars:
        reasons.append(f"场景旁白{narration_chars}字")
    if demand.emotional_change.strip():
        reasons.append("保留情绪停顿")
    reasons.extend(role_labels)
    reasons.append(f"场景原始{max(0, int(demand.scene_duration))}秒")
    reasons.append(f"优先级{demand.priority}")
    return round(estimated, 3), round(weight, 6), "；".join(reasons)


def allocate_durations(
    target_seconds: int,
    count: int,
    *,
    minimum: int = 3,
    maximum: int = 15,
) -> list[int]:
    return allocate_weighted_durations(
        target_seconds,
        [1.0] * count,
        minimum=minimum,
        maximum=maximum,
        keys=[f"shot-{index}" for index in range(count)],
    )


def allocate_weighted_durations(
    target_seconds: int,
    weights: Sequence[float],
    *,
    minimum: int = 3,
    maximum: int = 15,
    keys: Sequence[str] | None = None,
) -> list[int]:
    """Bounded largest-remainder allocation with deterministic non-positional ties."""
    target = int(target_seconds)
    count = len(weights)
    if count < 1:
        raise ValueError("镜头数量必须大于 0。")
    if minimum < 0 or maximum < minimum:
        raise ValueError("单镜头时长上下限无效。")
    if target < minimum * count or target > maximum * count:
        raise ValueError("目标时长无法在当前镜头数量和单镜头限制下分配。")
    normalized = [
        float(value) if math.isfinite(float(value)) and float(value) > 0 else 0.1
        for value in weights
    ]
    tie_keys = list(keys) if keys is not None else [str(index) for index in range(count)]
    if len(tie_keys) != count:
        raise ValueError("时长分配 keys 数量必须与权重数量一致。")

    capacity = float(maximum - minimum)
    remaining = float(target - minimum * count)
    extras = [0.0] * count
    active = set(range(count))
    while active and remaining > 1e-9:
        total_weight = sum(normalized[index] for index in active)
        if total_weight <= 0:
            total_weight = float(len(active))
        proposed = {
            index: remaining * normalized[index] / total_weight
            for index in active
        }
        capped = [
            index
            for index, share in proposed.items()
            if extras[index] + share >= capacity - 1e-9
        ]
        if not capped:
            for index, share in proposed.items():
                extras[index] += share
            remaining = 0.0
            break
        for index in capped:
            available = capacity - extras[index]
            extras[index] += available
            remaining -= available
            active.remove(index)

    floors = [min(maximum - minimum, int(math.floor(value + 1e-9))) for value in extras]
    durations = [minimum + value for value in floors]
    remainder = target - sum(durations)
    candidates = sorted(
        range(count),
        key=lambda index: (
            -(extras[index] - floors[index]),
            _stable_tie_rank(tie_keys[index]),
        ),
    )
    while remainder > 0:
        changed = False
        for index in candidates:
            if durations[index] >= maximum:
                continue
            durations[index] += 1
            remainder -= 1
            changed = True
            if remainder == 0:
                break
        if not changed:
            raise ValueError("时长余数无法在单镜头上限内分配。")
    if sum(durations) != target or any(
        value < minimum or value > maximum for value in durations
    ):
        raise ValueError("时长分配未满足总时长或单镜头边界。")
    return durations


def estimate_story_duration(
    story_text: str,
    *,
    character_count: int = 1,
    location_count: int = 1,
    minimum: int = 15,
    maximum: int = 300,
) -> int:
    """Estimate film duration from narrative density without a fixed beat template."""
    compact = re.sub(r"\s+", "", story_text)
    if not compact:
        return minimum
    events = [
        item
        for item in re.split(r"[。！？；]+", compact)
        if len(item.strip()) >= 4
    ]
    reading_seconds = math.ceil(len(compact) / 4.5)
    event_seconds = (
        10
        + len(events) * 6
        + max(0, int(character_count) - 1) * 3
        + max(0, int(location_count) - 1) * 5
    )
    estimate = max(reading_seconds, event_seconds)
    rounded = int(round(estimate / 5.0) * 5)
    return min(maximum, max(minimum, rounded))


def _count_action_steps(action: str) -> int:
    parts = [
        item
        for item in re.split(
            r"[，,；;。！？!?\n]|然后|随后|接着|同时|并且|再",
            action,
        )
        if item.strip()
    ]
    return max(1, len(parts))


def _spoken_character_count(text: str) -> int:
    return len(re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()—…-]", "", text))


def _pause_count(text: str) -> int:
    return len(re.findall(r"[，。！？；：,…—]", text))


def _stable_tie_rank(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()
