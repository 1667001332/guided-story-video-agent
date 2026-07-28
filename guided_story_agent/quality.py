from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .models import ArtifactReview, StoryDraft, StoryScript, StoryboardPlan


_GENERIC_BIGRAMS = {
    "一个",
    "人物",
    "主角",
    "故事",
    "开始",
    "发生",
    "最后",
    "已经",
    "自己",
    "他们",
    "她们",
    "我们",
    "这个",
    "那个",
    "进行",
    "完成",
}
_CONTINUITY_MARKERS = (
    "承接",
    "继续",
    "随后",
    "此时",
    "紧接",
    "同一",
    "仍然",
    "已经",
    "刚刚",
    "因此",
)


def semantic_bigrams(value: str) -> set[str]:
    """Return stable Chinese bigrams and meaningful latin tokens for cheap alignment checks."""
    result: set[str] = set()
    for block in re.findall(r"[\u4e00-\u9fff]+", str(value or "")):
        if len(block) == 1:
            result.add(block)
            continue
        result.update(block[index : index + 2] for index in range(len(block) - 1))
    result.difference_update(_GENERIC_BIGRAMS)
    result.update(
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]{3,}", str(value or ""))
    )
    return result


def semantic_coverage(source: str, target: str) -> float:
    expected = semantic_bigrams(source)
    if not expected:
        return 1.0
    observed = semantic_bigrams(target)
    return len(expected & observed) / len(expected)


def review_script_against_story(
    story: StoryDraft,
    script: StoryScript,
    *,
    required_character_names: list[str] | None = None,
) -> ArtifactReview:
    review = ArtifactReview(artifact_type="script")
    script_text = "\n".join(
        " ".join(
            [
                scene.title,
                scene.location,
                " ".join(scene.characters),
                scene.visible_action or scene.action,
                scene.dialogue,
                scene.narration,
                " ".join(scene.props),
                scene.start_state,
                scene.end_state,
            ]
        )
        for scene in script.scenes
    )
    required_names = [
        name.strip()
        for name in (
            required_character_names
            if required_character_names is not None
            else [character.name for character in story.characters[:1]]
        )
        if name.strip()
    ]
    missing_names = [name for name in required_names if name not in script_text]
    if missing_names:
        review.hard_errors.append(
            "剧本遗漏已确认的关键人物：" + "、".join(missing_names)
        )

    conflict_coverage = semantic_coverage(story.core_conflict, script_text)
    ending_coverage = semantic_coverage(story.ending, script_text)
    location_coverage = (
        sum(location.name in script_text for location in story.locations)
        / max(1, len(story.locations))
    )
    review.scores.update(
        {
            "story_conflict_coverage": round(conflict_coverage, 3),
            "story_ending_coverage": round(ending_coverage, 3),
            "story_location_coverage": round(location_coverage, 3),
        }
    )
    if story.core_conflict.strip() and conflict_coverage < 0.12:
        review.hard_errors.append("剧本没有形成已确认故事的核心冲突")
    if story.ending.strip() and ending_coverage < 0.12:
        review.hard_errors.append("剧本没有落实已确认故事的结局")
    if story.locations and location_coverage == 0:
        review.warnings.append("剧本没有沿用故事中已确认的地点")

    scene_count = max(1, len(script.scenes))
    generic_title_count = sum(
        bool(re.fullmatch(r"(?:故事)?片段\s*\d+", scene.title.strip()))
        for scene in script.scenes
    )
    copied_narration_count = sum(
        bool(
            (scene.visible_action or scene.action).strip()
            and (scene.visible_action or scene.action).strip() == scene.narration.strip()
        )
        for scene in script.scenes
    )
    generic_location_count = sum(
        scene.location.strip() in {"故事核心场景", "连续的故事空间", "主要场景"}
        for scene in script.scenes
    )
    template_phrase_count = sum(
        any(
            phrase in " ".join(
                (
                    scene.visible_action or scene.action,
                    scene.dialogue,
                )
            )
            for phrase in (
                "人物必须面对这一核心冲突",
                "我已经作出了选择",
                "故事在行动完成后的余波中结束",
            )
        )
        for scene in script.scenes
    )
    review.scores.update(
        {
            "specific_scene_title_rate": round(1 - generic_title_count / scene_count, 3),
            "action_narration_separation": round(
                1 - copied_narration_count / scene_count, 3
            ),
            "specific_location_rate": round(1 - generic_location_count / scene_count, 3),
        }
    )
    if generic_title_count / scene_count >= 0.5:
        review.hard_errors.append("剧本场景标题仍是编号模板，没有表达具体事件")
    if copied_narration_count / scene_count >= 0.5:
        review.hard_errors.append("剧本把画面动作直接复制成旁白，尚未完成视听改编")
    if generic_location_count == len(script.scenes) and script.scenes:
        review.hard_errors.append("剧本所有场景都使用泛化地点，无法建立可拍摄空间")
    if template_phrase_count:
        review.hard_errors.append("剧本仍包含离线模板占位句")

    nonempty_state_scenes = sum(
        bool(scene.start_state.strip() and scene.end_state.strip())
        for scene in script.scenes
    )
    state_field_coverage = nonempty_state_scenes / max(1, len(script.scenes))
    bridge_count = 0
    for previous, current in zip(script.scenes, script.scenes[1:]):
        overlap = semantic_coverage(previous.end_state, current.start_state)
        marker = any(
            item in current.start_state
            for item in _CONTINUITY_MARKERS
        )
        if overlap >= 0.12 or marker:
            bridge_count += 1
    bridge_coverage = bridge_count / max(1, len(script.scenes) - 1)
    review.scores["scene_state_field_coverage"] = round(state_field_coverage, 3)
    review.scores["scene_state_bridge_coverage"] = round(bridge_coverage, 3)
    if state_field_coverage < 1:
        review.hard_errors.append("剧本存在缺少起止状态的场景")
    if len(script.scenes) > 1 and bridge_coverage < 0.5:
        review.warnings.append("部分相邻场景的状态承接仍不够明确")
    return review


def evaluate_storyboard_quality(plan: StoryboardPlan) -> dict[str, Any]:
    actions = [_normalize_action(shot.action) for shot in plan.shots]
    duplicate_pairs = 0
    comparisons = 0
    for left, right in zip(actions, actions[1:]):
        if not left or not right:
            continue
        comparisons += 1
        if SequenceMatcher(None, left, right).ratio() >= 0.82:
            duplicate_pairs += 1
    explicit_transitions = sum(
        shot.transition_type
        not in {"", "independent"}
        and bool(shot.transition_reason.strip())
        for shot in plan.shots
    )
    atomic_actions = sum(_action_clause_count(shot.action) <= 3 for shot in plan.shots)
    return {
        "storyboard_action_uniqueness": round(
            1.0 - duplicate_pairs / max(1, comparisons),
            3,
        ),
        "storyboard_transition_explicitness": round(
            explicit_transitions / max(1, len(plan.shots)),
            3,
        ),
        "storyboard_atomic_action_rate": round(
            atomic_actions / max(1, len(plan.shots)),
            3,
        ),
        "storyboard_duplicate_adjacent_pairs": duplicate_pairs,
    }


def build_human_review_template(
    story: StoryDraft,
    script: StoryScript,
    plan: StoryboardPlan,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "title": story.title,
        "instructions": "每项填写1到5分，并在notes中记录可复现的问题和镜头编号。",
        "scores": {
            "story_causal_continuity": None,
            "character_motivation": None,
            "script_story_fidelity": None,
            "scene_transition_clarity": None,
            "shot_information_gain": None,
            "character_visual_consistency": None,
            "location_prop_consistency": None,
            "audio_video_sync": None,
            "overall_watchability": None,
        },
        "notes": [],
        "evidence": {
            "story_version": story.version,
            "scene_count": len(script.scenes),
            "shot_count": len(plan.shots),
            "target_seconds": plan.target_seconds,
        },
    }


def _normalize_action(value: str) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").lower())


def _action_clause_count(value: str) -> int:
    return len(
        [
            item
            for item in re.split(r"[。！？；;，,]|然后|随后|接着|同时", str(value or ""))
            if item.strip()
        ]
    )
