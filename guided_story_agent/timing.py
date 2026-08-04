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
    narration_is_per_shot: bool = False


@dataclass(frozen=True, slots=True)
class ReadableShotMinimum:
    """Hard timing floor derived from visible action and spoken material."""

    provider_seconds: int
    visual_seconds: int
    speech_seconds: int
    minimum_seconds: int
    reason: str
    max_duration_seconds: int = 15

    @property
    def requires_split(self) -> bool:
        return self.minimum_seconds > self.max_duration_seconds


@dataclass(frozen=True, slots=True)
class ShotTimingProfile:
    """Planning-time per-shot duration bounds.

    Defaults preserve the original Agnes adapter bounds (3–15 seconds per
    generation).  A provider with different bounds injects its own profile at
    storyboard planning time, so the pipeline never hardcodes 3/15 again.
    """

    min_duration_seconds: int = 3
    max_duration_seconds: int = 15

    def minimum_shot_count(self, target_seconds: int) -> int:
        """Fewest shots that can physically carry ``target_seconds``."""
        return max(1, math.ceil(target_seconds / self.max_duration_seconds))

    def maximum_shot_count(self, target_seconds: int) -> int:
        """Most shots that fit into ``target_seconds`` at the minimum duration."""
        return max(1, target_seconds // self.min_duration_seconds)


_BASE_SECONDS = {
    "establish": 3.8,
    "detail": 3.2,
    "action": 4.4,
    "dialogue": 3.5,
    "reaction": 4.2,
    "transition": 3.4,
}

_READABLE_CJK_CHARS_PER_SECOND = 5.5
_READABLE_LATIN_WORDS_PER_SECOND = 2.7
_VISUAL_SETUP_SECONDS = 1.25
_VISUAL_SEQUENCE_SECONDS = 0.75
_SEQUENTIAL_MARKERS = (
    "随后",
    "接着",
    "紧接着",
    "然后",
    "随即",
    "继而",
    "之后",
    "转而",
    "并且",
)
_CONCURRENT_PREFIXES = (
    "同时",
    "与此同时",
    "一边",
    "伴随",
    "while ",
    "as ",
)


def estimate_shot_duration(
    demand: ShotTimingDemand,
    *,
    timing_profile: ShotTimingProfile | None = None,
) -> tuple[float, float, str]:
    """Estimate content need without turning shot kinds into fixed durations."""
    profile = timing_profile or ShotTimingProfile()
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
    narration_share = (
        narration_chars
        if demand.narration_is_per_shot
        else narration_chars / max(1, demand.scene_shot_count)
    )
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
    estimated = min(
        float(profile.max_duration_seconds),
        max(float(profile.min_duration_seconds), estimated),
    )
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


def estimate_scene_duration_weight(
    *,
    title: str,
    action: str,
    dialogue: str,
    narration: str,
    emotional_change: str,
) -> tuple[float, str]:
    """Estimate a scene's fallback pacing weight from its actual content."""

    demand = ShotTimingDemand(
        shot_kind="action",
        purpose=title,
        priority=5,
        action=action,
        dialogue=dialogue,
        narration=narration,
        emotional_change=emotional_change,
        scene_duration=3,
        scene_shot_count=1,
        narration_is_per_shot=True,
    )
    _, weight, reason = estimate_shot_duration(demand)
    return weight, reason


def plan_scene_durations(
    scenes: Sequence[object],
    target_seconds: int,
    *,
    minimum: int = 3,
    maximum: int | None = None,
    enforce_readable_minimums: bool = False,
) -> tuple[list[int], list[float], list[str]]:
    """Use model pacing weights, with a content-based offline fallback.

    Scene planning only needs a provider-safe scalar floor by default.  The
    stricter content/readability floor belongs to shot planning, where a scene
    can be split into multiple shots instead of making a short target impossible.
    """

    if not scenes:
        raise ValueError("至少需要一个剧本场景")
    upper = int(maximum if maximum is not None else target_seconds)
    weights: list[float] = []
    reasons: list[str] = []
    minimums: list[int] = []
    for scene in scenes:
        action = str(getattr(scene, "visible_action", "") or getattr(scene, "action", ""))
        dialogue = str(getattr(scene, "dialogue", ""))
        narration = str(getattr(scene, "narration", ""))
        emotion = str(getattr(scene, "emotional_change", ""))
        title = str(getattr(scene, "title", ""))
        provided = getattr(scene, "duration_weight", 0.0)
        try:
            provided_weight = float(provided)
        except (TypeError, ValueError):
            provided_weight = 0.0
        demand = ShotTimingDemand(
            shot_kind="action",
            purpose=title,
            priority=5,
            action=action,
            dialogue=dialogue,
            narration=narration,
            emotional_change=emotion,
            scene_duration=3,
            scene_shot_count=1,
            narration_is_per_shot=True,
        )
        if enforce_readable_minimums:
            readable = assess_shot_readable_minimum(demand, provider_minimum=minimum)
            minimums.append(max(minimum, readable.minimum_seconds))
        else:
            minimums.append(minimum)
        if math.isfinite(provided_weight) and provided_weight > 0:
            weights.append(provided_weight)
            reasons.append(str(getattr(scene, "duration_reason", "") or "LLM 提供的剧情节奏权重"))
        else:
            _, fallback_weight, fallback_reason = estimate_shot_duration(demand)
            weights.append(fallback_weight)
            reasons.append(f"离线时长估计：{fallback_reason}")
    durations = allocate_weighted_durations(
        int(target_seconds),
        weights,
        minimum=minimum,
        maximum=upper,
        minimums=minimums,
        keys=[f"scene-{index}" for index in range(len(weights))],
    )
    return durations, weights, reasons


def assess_shot_readable_minimum(
    demand: ShotTimingDemand,
    *,
    provider_minimum: int = 3,
    provider_maximum: int = 15,
) -> ReadableShotMinimum:
    """Calculate a content-specific floor without treating the preferred estimate as a gate.

    Sequential visual phases consume time, while simultaneous action does not add another
    full phase. Dialogue and narration share the audio timeline, so their readable times
    are added; the resulting speech can run in parallel with the visible action.
    """

    if (
        isinstance(provider_minimum, bool)
        or not isinstance(provider_minimum, int)
        or provider_minimum < 0
    ):
        raise ValueError("Provider 单镜头最低时长必须是非负整数。")
    action_phases = count_sequential_action_phases(demand.action)
    visual_seconds = max(
        1,
        int(
            math.ceil(
                _VISUAL_SETUP_SECONDS
                + _VISUAL_SEQUENCE_SECONDS * action_phases
                - 1e-9
            )
        ),
    )
    dialogue_seconds = _minimum_spoken_seconds(demand.dialogue)
    narration_seconds = _minimum_spoken_seconds(demand.narration)
    if not demand.narration_is_per_shot:
        narration_seconds /= max(1, int(demand.scene_shot_count))
    speech_seconds = (
        int(math.ceil(dialogue_seconds + narration_seconds - 1e-9))
        if dialogue_seconds or narration_seconds
        else 0
    )
    minimum_seconds = max(provider_minimum, visual_seconds, speech_seconds)
    reasons = [
        f"Provider下限{provider_minimum}秒",
        f"{action_phases}个顺序动作阶段需{visual_seconds}秒",
    ]
    if dialogue_seconds:
        reasons.append(f"对白至少{dialogue_seconds:.2f}秒")
    if narration_seconds:
        reasons.append(f"本镜旁白至少{narration_seconds:.2f}秒")
    reasons.append(f"内容可读下限{minimum_seconds}秒")
    return ReadableShotMinimum(
        provider_seconds=provider_minimum,
        visual_seconds=visual_seconds,
        speech_seconds=speech_seconds,
        minimum_seconds=minimum_seconds,
        reason="；".join(reasons),
        max_duration_seconds=provider_maximum,
    )


def count_sequential_action_phases(action: str) -> int:
    """Count sequential phases while ignoring descriptive lists and parallel clauses."""

    return len(split_sequential_action_phases(action))


def split_sequential_action_phases(action: str) -> list[str]:
    """Split hard boundaries and commas, except explicitly concurrent clauses."""

    cleaned = str(action or "").strip()
    if not cleaned:
        return [""]
    marker_pattern = "|".join(re.escape(marker) for marker in _SEQUENTIAL_MARKERS)
    ordinal_suffixes = "个|次|步|项|段|场|镜|句|刻|天|年|人|动作"
    pieces = [
        item.strip(" ，,")
        for item in re.split(
            (
                r"[。！？!?；;\n]+|(?<!\d)\.(?!\d)"
                rf"|(?=(?:{marker_pattern}))"
                rf"|(?=最后(?!一?(?:{ordinal_suffixes})))"
                rf"|(?=最终(?!一?(?:{ordinal_suffixes})))"
                r"|(?=(?<!不)再(?:次|度)?)"
                r"|(?=\b(?:and\s+then|then|next|afterwards|subsequently|finally)\b)"
            ),
            cleaned,
            flags=re.IGNORECASE,
        )
        if item.strip(" ，,")
    ]
    phases: list[str] = []
    for piece in pieces:
        comma_clauses = [
            clause.strip()
            for clause in re.split(r"[，,]", piece)
            if clause.strip()
        ]
        if not comma_clauses:
            continue
        current = comma_clauses[0]
        for clause in comma_clauses[1:]:
            lowered = clause.lower()
            if lowered.startswith(_CONCURRENT_PREFIXES):
                current = f"{current}，{clause}"
                continue
            phases.append(current)
            current = clause
        phases.append(current)
    return phases or [cleaned]


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
    minimums: Sequence[int] | None = None,
    keys: Sequence[str] | None = None,
) -> list[int]:
    """Allocate exact integer seconds above scalar or per-shot lower bounds."""
    target = int(target_seconds)
    count = len(weights)
    if count < 1:
        raise ValueError("镜头数量必须大于 0。")
    if minimum < 0 or maximum < minimum:
        raise ValueError("单镜头时长上下限无效。")
    if minimums is None:
        lower_bounds = [minimum] * count
    else:
        if len(minimums) != count:
            raise ValueError("逐镜最低可读时长数量必须与权重数量一致。")
        lower_bounds = []
        for value in minimums:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("逐镜最低可读时长必须是整数。")
            if value < minimum:
                raise ValueError("逐镜最低可读时长不能低于单镜头下限。")
            lower_bounds.append(value)
    if any(value > maximum for value in lower_bounds):
        raise ValueError(
            f"存在最低可读时长超过 {maximum} 秒的镜头，需要拆分镜头内容。"
        )
    minimum_total = sum(lower_bounds)
    if target < minimum_total:
        raise ValueError(
            f"目标时长 {target} 秒低于镜头内容的最低可读时长 "
            f"{minimum_total} 秒。"
        )
    if target > maximum * count:
        raise ValueError("目标时长无法在当前镜头数量和单镜头限制下分配。")
    normalized = [
        float(value) if math.isfinite(float(value)) and float(value) > 0 else 0.1
        for value in weights
    ]
    tie_keys = list(keys) if keys is not None else [str(index) for index in range(count)]
    if len(tie_keys) != count:
        raise ValueError("时长分配 keys 数量必须与权重数量一致。")

    capacities = [float(maximum - value) for value in lower_bounds]
    remaining = float(target - minimum_total)
    extras = [0.0] * count
    active = {index for index, capacity in enumerate(capacities) if capacity > 0}
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
            if extras[index] + share >= capacities[index] - 1e-9
        ]
        if not capped:
            for index, share in proposed.items():
                extras[index] += share
            remaining = 0.0
            break
        for index in capped:
            available = capacities[index] - extras[index]
            extras[index] += available
            remaining -= available
            active.remove(index)
    if remaining > 1e-9:
        raise ValueError("时长余量无法在单镜头上限内分配。")

    floors = [
        min(int(capacities[index]), int(math.floor(value + 1e-9)))
        for index, value in enumerate(extras)
    ]
    durations = [
        lower_bounds[index] + value
        for index, value in enumerate(floors)
    ]
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
        value < lower_bounds[index] or value > maximum
        for index, value in enumerate(durations)
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


def _minimum_spoken_seconds(text: str) -> float:
    source = str(text or "")
    cjk_characters = len(re.findall(r"[\u3400-\u9fff]", source))
    latin_words = len(
        re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", source)
    )
    return (
        cjk_characters / _READABLE_CJK_CHARS_PER_SECOND
        + latin_words / _READABLE_LATIN_WORDS_PER_SECOND
    )


def _pause_count(text: str) -> int:
    return len(re.findall(r"[，。！？；：,…—]", text))


def _stable_tie_rank(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()
