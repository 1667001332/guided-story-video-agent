from __future__ import annotations


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
