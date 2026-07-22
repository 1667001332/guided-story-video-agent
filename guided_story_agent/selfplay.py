from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .agent import OpenAIStoryAgent, StoryAgent
from .models import CreativeBrief, to_plain_data
from .rendering import StoryRenderer
from .session import GuidedStorySession
from .video_provider import AgnesVideoProvider


def _idea_metrics(session: GuidedStorySession) -> tuple[float, float]:
    cards = session.current_batch.cards if session.current_batch else []
    fingerprints = [card.fingerprint for card in cards]
    unique = len(set(fingerprints))
    duplicate_rate = 1.0 - unique / max(1, len(cards))
    feature_sets = [
        {card.protagonist, card.central_conflict, card.tone, card.ending_direction}
        for card in cards
    ]
    pair_scores: list[float] = []
    for index, left in enumerate(feature_sets):
        for right in feature_sets[index + 1 :]:
            pair_scores.append(1.0 - len(left & right) / max(1, len(left | right)))
    diversity = sum(pair_scores) / max(1, len(pair_scores))
    return round(diversity, 3), round(duplicate_rate, 3)


def run_selfplay(
    *,
    agent: StoryAgent,
    target_seconds: int = 45,
    max_turns: int = 12,
    output_dir: str | Path,
    render: bool = False,
    renderer=None,
    require_live_text: bool = False,
) -> dict[str, object]:
    """Exercise the one-sentence path. ``max_turns`` remains CLI-compatible only."""
    del max_turns
    fallback_before = int(getattr(agent, "fallback_count", 0))
    if require_live_text and getattr(agent, "client", None) is None:
        raise RuntimeError("--require-live-text 已启用，但没有可用的真实文本 API 配置。")

    session = GuidedStorySession(CreativeBrief(target_seconds=target_seconds), agent=agent)
    direction = agent.simulate_creator_direction()
    ideas = session.start_ideation(direction)
    session.auto_choose()
    selected_snapshot = [to_plain_data(card) for card in session.selected_cards]
    draft = session.generate_draft()
    session.confirm_draft()
    storyboard = session.build_storyboard()
    session.confirm_storyboard()

    fallback_after = int(getattr(agent, "fallback_count", 0))
    if require_live_text and fallback_after > fallback_before:
        reason = str(getattr(agent, "last_fallback_reason", "未知原因"))
        raise RuntimeError(f"真实文本链路发生本地降级，测试判定失败：{reason}")

    diversity, duplicate_rate = _idea_metrics(session)
    selected_ids = {card["idea_id"] for card in selected_snapshot}
    recorded_ids = set(draft.field_sources["selected_ideas"].source_ids)
    disclosure_denominator = max(1, len(draft.ai_filled_fields))
    disclosed = sum(field in draft.field_sources for field in draft.ai_filled_fields)
    cameras = {shot.camera for shot in storyboard.shots}
    anchors = sum(bool(shot.visual_anchors) for shot in storyboard.shots)
    bench = {
        "schema_version": 3,
        "idea_count": len(ideas.cards),
        "idea_diversity": diversity,
        "duplicate_rate": duplicate_rate,
        "selection_retention": round(
            len(selected_ids & recorded_ids) / max(1, len(selected_ids)), 3
        ),
        "ai_fill_transparency": round(disclosed / disclosure_denominator, 3),
        "mandatory_followup_text_count": 0,
        "free_text_required_count": 1,
        "clicks_to_draft": 2,
        "target_seconds": target_seconds,
        "storyboard_seconds": storyboard.total_duration,
        "duration_within_tolerance": abs(storyboard.total_duration - target_seconds) <= 1,
        "visual_anchor_coverage": round(anchors / max(1, len(storyboard.shots)), 3),
        "shot_diversity": round(len(cameras) / max(1, len(storyboard.shots)), 3),
        "video_requested": bool(render),
        "text_api_mode": "live-required" if require_live_text else "fallback-allowed",
        "text_fallback_count": fallback_after - fallback_before,
    }

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "transcript.json": {
            "schema_version": 3,
            "direction": direction,
            "chat": to_plain_data(session.chat_history),
        },
        "ideas.json": {"schema_version": 3, "batches": to_plain_data(session.idea_batches)},
        "selection.json": {
            "schema_version": 3,
            "ideas": selected_snapshot,
            "elements": to_plain_data(session.selected_elements),
        },
        "draft.json": {"schema_version": 3, **to_plain_data(draft)},
        "storyboard.json": {"schema_version": 3, **to_plain_data(storyboard)},
        "prompt_log.json": {
            "schema_version": 3,
            "prompts": [
                {
                    "shot_id": shot.shot_id,
                    "positive_prompt": shot.video_prompt,
                    "negative_prompt": shot.negative_prompt,
                }
                for shot in storyboard.shots
            ],
        },
        "bench.json": bench,
    }
    for name, data in artifacts.items():
        (target / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    session.save(target / "session.json")

    if render:
        active_renderer = renderer or StoryRenderer(AgnesVideoProvider.from_env())
        manifest = session.render_confirmed_plan(active_renderer, target / "video")
        bench["render_status"] = manifest.status
        bench["failed_shots"] = manifest.failed_shots
        (target / "bench.json").write_text(
            json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"session": session, "output_dir": str(target), "bench": bench}


def main() -> None:
    parser = argparse.ArgumentParser(description="一句方向到分镜的创意花园自演测试")
    parser.add_argument("--target-seconds", type=int, default=45, choices=range(30, 61))
    parser.add_argument(
        "--max-turns",
        type=int,
        default=12,
        help="兼容 v0.2；v0.3 不再需要追问轮数",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--render", action="store_true", help="显式允许付费视频请求")
    parser.add_argument(
        "--require-live-text",
        action="store_true",
        help="文本 API 不可用或发生本地降级时判定失败",
    )
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"outputs/selfplay/{stamp}"
    result = run_selfplay(
        agent=OpenAIStoryAgent.from_env(),
        target_seconds=args.target_seconds,
        max_turns=args.max_turns,
        output_dir=output,
        render=args.render,
        require_live_text=args.require_live_text,
    )
    print(
        json.dumps(
            {"output_dir": result["output_dir"], "bench": result["bench"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
