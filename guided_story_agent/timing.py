from __future__ import annotations

import math
import re


def allocate_durations(
    target_seconds: int,
    count: int,
    *,
    minimum: int = 3,
    maximum: int = 15,
) -> list[int]:
    target = int(target_seconds)
    if count < 1:
        raise ValueError("镜头数量必须大于 0。")
    if target < minimum * count or target > maximum * count:
        raise ValueError("目标时长无法在当前镜头数量和单镜头限制下分配。")
    base, remainder = divmod(target, count)
    durations = [base + (1 if index < remainder else 0) for index in range(count)]
    if any(value < minimum or value > maximum for value in durations):
        raise ValueError("分配结果超出单镜头时长限制。")
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
