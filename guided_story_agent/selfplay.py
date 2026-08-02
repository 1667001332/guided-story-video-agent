from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .agent import OpenAIStoryAgent, RuleBasedStoryAgent, StoryAgent
from .models import CreativeBrief, to_plain_data
from .quality import (
    build_human_review_template,
    evaluate_storyboard_quality,
    review_script_against_story,
)
from .rendering import VideoJobRenderer
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
    direction: str | None = None,
    target_seconds: int | None = None,
    max_turns: int = 12,
    output_dir: str | Path,
    render: bool = False,
    renderer=None,
    require_live_text: bool = False,
    llm_judge: bool = False,
) -> dict[str, object]:
    """Exercise the one-sentence path. ``max_turns`` remains CLI-compatible only."""
    del max_turns
    fallback_before = int(getattr(agent, "fallback_count", 0))
    if require_live_text and getattr(agent, "client", None) is None:
        raise RuntimeError("--require-live-text 已启用，但没有可用的真实文本 API 配置。")

    session = GuidedStorySession(
        CreativeBrief(
            target_seconds=target_seconds,
            duration_mode="custom" if target_seconds is not None else "auto",
        ),
        agent=agent,
    )
    active_direction = (direction or "").strip() or agent.simulate_creator_direction()
    ideas = session.start_ideation(active_direction)
    session.auto_choose()
    selected_snapshot = [to_plain_data(card) for card in session.selected_cards]
    story = session.generate_story()
    session.confirm_story()
    script = session.generate_script()
    session.confirm_script()
    storyboard = session.build_storyboard()
    session.confirm_storyboard()

    semantic_review = review_script_against_story(story, script)
    deterministic_quality = {
        **semantic_review.scores,
        **evaluate_storyboard_quality(storyboard),
        "quality_hard_error_count": len(semantic_review.hard_errors),
        "quality_warning_count": len(semantic_review.warnings),
    }
    judge_result: dict[str, object] = {}
    evaluator = getattr(agent, "evaluate_artifacts", None)
    if llm_judge and callable(evaluator):
        judge_result = dict(
            evaluator(
                story,
                script,
                to_plain_data(storyboard),
            )
            or {}
        )
    fallback_after = int(getattr(agent, "fallback_count", 0))
    if require_live_text and fallback_after > fallback_before:
        reason = str(getattr(agent, "last_fallback_reason", "未知原因"))
        raise RuntimeError(f"真实文本链路发生本地降级，测试判定失败：{reason}")

    diversity, duplicate_rate = _idea_metrics(session)
    selected_ids = {card["idea_id"] for card in selected_snapshot}
    recorded_ids = set(story.field_sources["selected_ideas"].source_ids)
    disclosure_denominator = max(1, len(story.ai_filled_fields))
    disclosed = sum(field in story.field_sources for field in story.ai_filled_fields)
    cameras = {shot.camera for shot in storyboard.shots}
    anchors = sum(bool(shot.visual_anchors) for shot in storyboard.shots)
    if require_live_text:
        text_api_mode = "live-required"
    elif isinstance(agent, RuleBasedStoryAgent) and not isinstance(agent, OpenAIStoryAgent):
        text_api_mode = "offline"
    else:
        text_api_mode = "fallback-allowed"
    bench = {
        "schema_version": GuidedStorySession.schema_version,
        "idea_count": len(ideas.cards),
        "idea_diversity": diversity,
        "duplicate_rate": duplicate_rate,
        "selection_retention": round(
            len(selected_ids & recorded_ids) / max(1, len(selected_ids)), 3
        ),
        "ai_fill_transparency": round(disclosed / disclosure_denominator, 3),
        "mandatory_followup_text_count": 0,
        "free_text_required_count": 1,
        "clicks_to_story": 2,
        "clicks_story_to_script": 1,
        "script_scene_count": len(script.scenes),
        "duration_mode": session.brief.duration_mode,
        "target_seconds": script.target_seconds,
        "storyboard_seconds": storyboard.total_duration,
        "duration_within_tolerance": (abs(storyboard.total_duration - script.target_seconds) <= 1),
        "visual_anchor_coverage": round(anchors / max(1, len(storyboard.shots)), 3),
        "shot_diversity": round(len(cameras) / max(1, len(storyboard.shots)), 3),
        "video_requested": bool(render),
        "text_api_mode": text_api_mode,
        "text_fallback_count": fallback_after - fallback_before,
        "text_provider": str(getattr(agent, "provider_name", type(agent).__name__)),
        "text_model": str(getattr(agent, "model", "rule-based")),
        **deterministic_quality,
        "llm_judge_enabled": bool(llm_judge),
        "llm_judge_scores": dict(judge_result.get("scores", {}))
        if isinstance(judge_result.get("scores"), dict)
        else {},
        "llm_judge_issue_count": len(judge_result.get("issues", []))
        if isinstance(judge_result.get("issues"), list)
        else 0,
    }

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "transcript.json": {
            "schema_version": GuidedStorySession.schema_version,
            "direction": active_direction,
            "chat": to_plain_data(session.chat_history),
        },
        "ideas.json": {
            "schema_version": GuidedStorySession.schema_version,
            "batches": to_plain_data(session.idea_batches),
        },
        "selection.json": {
            "schema_version": GuidedStorySession.schema_version,
            "ideas": selected_snapshot,
            "elements": to_plain_data(session.selected_elements),
        },
        "story.json": {
            "schema_version": GuidedStorySession.schema_version,
            **to_plain_data(story),
        },
        "script.json": {
            "schema_version": GuidedStorySession.schema_version,
            **to_plain_data(script),
        },
        "storyboard.json": {
            "schema_version": GuidedStorySession.schema_version,
            **to_plain_data(storyboard),
        },
        "prompt_log.json": {
            "schema_version": GuidedStorySession.schema_version,
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
        "quality_report.json": {
            "schema_version": 1,
            "deterministic": deterministic_quality,
            "hard_errors": semantic_review.hard_errors,
            "warnings": semantic_review.warnings,
            "llm_judge": judge_result,
        },
        "human_review.json": build_human_review_template(
            story,
            script,
            storyboard,
        ),
    }
    for name, data in artifacts.items():
        (target / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    session.save(target / "session.json")

    if render:
        # Keep the legacy storyboard in the benchmark artifacts, but exercise
        # the production path with one complete VideoJob.
        try:
            if isinstance(renderer, VideoJobRenderer):
                if session.video_job is None:
                    session.build_video_job()
                manifest = session.render_confirmed_video(renderer, target / "video")
            elif renderer is not None:
                # Preserve the legacy Storyboard renderer seam for offline
                # tests and callers that explicitly provide one.
                manifest = session.render_confirmed_plan(renderer, target / "video")
            else:
                session.build_video_job()
                manifest = session.render_confirmed_video(
                    VideoJobRenderer(AgnesVideoProvider.from_env()),
                    target / "video",
                )
            bench["render_status"] = manifest.status
            bench["render_warning"] = manifest.error
            bench["failed_shots"] = manifest.failed_shots
        finally:
            session.save(target / "session.json")
        (target / "bench.json").write_text(
            json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"session": session, "output_dir": str(target), "bench": bench}


def main() -> None:
    parser = argparse.ArgumentParser(description="一句方向到分镜的创意花园自演测试")
    parser.add_argument(
        "--target-seconds",
        type=int,
        default=None,
        help="自定义成片秒数（15–300）；省略时根据完整故事自动估算",
    )
    parser.add_argument(
        "--direction",
        default="",
        help="指定测试方向；省略时由自演创建者生成一句方向",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=12,
        help="兼容 v0.2；v0.3 不再需要追问轮数",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--render", action="store_true", help="显式允许付费视频请求")
    parser.add_argument(
        "--confirm-paid-video",
        default="",
        help="与 --render 同用时必须精确填写 RENDER",
    )
    parser.add_argument(
        "--require-live-text",
        action="store_true",
        help="文本 API 不可用或发生本地降级时判定失败",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="强制使用本地规则，不读取或请求真实文本 API",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="额外调用一次文本模型评价故事、剧本和分镜质量",
    )
    args = parser.parse_args()
    if args.render and args.confirm_paid_video != "RENDER":
        parser.error("--render 会调用付费视频 API，必须同时填写 --confirm-paid-video RENDER")
    if not args.render and args.confirm_paid_video:
        parser.error("--confirm-paid-video 只能与 --render 同时使用。")
    if args.offline and args.require_live_text:
        parser.error("--offline 与 --require-live-text 不能同时使用。")
    if args.offline and args.llm_judge:
        parser.error("--offline 不能启用 --llm-judge。")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or f"outputs/selfplay/{stamp}"
    result = run_selfplay(
        agent=RuleBasedStoryAgent() if args.offline else OpenAIStoryAgent.from_env(),
        direction=args.direction,
        target_seconds=args.target_seconds,
        max_turns=args.max_turns,
        output_dir=output,
        render=args.render,
        require_live_text=args.require_live_text,
        llm_judge=args.llm_judge,
    )
    print(
        json.dumps(
            {"output_dir": result["output_dir"], "bench": result["bench"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.render and result["bench"].get("render_status") not in {
        "succeeded",
        "succeeded_with_warnings",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
