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
    fallback_before = int(getattr(agent, "fallback_count", 0))
    if require_live_text and getattr(agent, "client", None) is None:
        raise RuntimeError("--require-live-text 已启用，但没有可用的真实文本 API 配置。")
    session = GuidedStorySession(CreativeBrief(target_seconds=target_seconds), agent=agent)
    question = session.current_question
    questions = [question]
    while session.valid_turns < max_turns and not session.can_build_outline:
        answer = agent.simulate_creator(question, session.contributions)
        result = session.submit_user_turn(answer, source="simulated_llm")
        question = result.next_question or session.current_question
        questions.append(question)
    if not session.can_build_outline:
        raise RuntimeError(f"自演在 {max_turns} 轮内没有完成故事大纲条件。")

    outline = session.build_outline()
    session.confirm_outline()
    detail_turns = 0
    while not session.can_build_script and detail_turns < 10:
        history = session.contributions + session.detail_contributions
        answer = agent.simulate_creator(session.current_question, history)
        session.answer_detail_question(answer, source="simulated_llm")
        questions.append(session.current_question)
        detail_turns += 1
    if not session.can_build_script:
        raise RuntimeError("自演在 10 轮制作追问内没有补齐剧本细节。")
    script = session.build_script()
    session.confirm_script()
    storyboard = session.build_storyboard()
    session.confirm_storyboard()

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    normalized_questions = [" ".join(item.split()) for item in questions if item.strip()]
    repeated = len(normalized_questions) - len(set(normalized_questions))
    extracted_pairs = {
        (field, value)
        for turn in session.contributions + session.detail_contributions
        for field, value in turn.extracted_facts.items()
    }
    retained = sum(
        1 for field, value in extracted_pairs if getattr(session.facts, field, "") == value
    )
    cameras = {shot.camera for shot in storyboard.shots}
    anchor_shots = sum(bool(shot.visual_anchors) for shot in storyboard.shots)
    outline_review = session.review_current_artifact("outline")
    fallback_after = int(getattr(agent, "fallback_count", 0))
    if require_live_text and fallback_after > fallback_before:
        reason = str(getattr(agent, "last_fallback_reason", "未知原因"))
        raise RuntimeError(f"真实文本链路发生本地降级，测试判定失败：{reason}")
    bench = {
        "schema_version": 2,
        "valid_story_turns": session.valid_turns,
        "has_opening": bool(session.facts.opening),
        "has_ending": bool(session.facts.ending),
        "outline_complete": not session.facts.missing_outline_fields(),
        "detail_complete": not session.facts.missing_detail_fields(),
        "target_seconds": target_seconds,
        "storyboard_seconds": storyboard.total_duration,
        "duration_within_tolerance": abs(storyboard.total_duration - target_seconds) <= 1,
        "video_requested": bool(render),
        "question_repetition_rate": round(
            repeated / max(1, len(normalized_questions)), 3
        ),
        "user_fact_retention": round(retained / max(1, len(extracted_pairs)), 3),
        "conflict_resolution_rate": 1.0 if not session.unresolved_conflicts else 0.0,
        "causal_completeness": outline_review.scores.get("causal_completeness", 0.0),
        "visual_anchor_coverage": round(anchor_shots / max(1, len(storyboard.shots)), 3),
        "shot_diversity": round(len(cameras) / max(1, len(storyboard.shots)), 3),
        "text_api_mode": "live-required" if require_live_text else "fallback-allowed",
        "text_fallback_count": fallback_after - fallback_before,
    }
    artifacts = {
        "transcript.json": {
            "schema_version": 2,
            "story": to_plain_data(session.contributions),
            "details": to_plain_data(session.detail_contributions),
        },
        "outline.json": {"schema_version": 2, **to_plain_data(outline)},
        "script.json": {"schema_version": 2, **to_plain_data(script)},
        "storyboard.json": {"schema_version": 2, **to_plain_data(storyboard)},
        "story_bible.json": {"schema_version": 2, **to_plain_data(session.story_bible)},
        "revisions.json": {"schema_version": 2, "revisions": to_plain_data(session.revisions)},
        "prompt_log.json": {
            "schema_version": 2,
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
        (target / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
    parser = argparse.ArgumentParser(description="LLM self-play for the guided story workflow")
    parser.add_argument("--target-seconds", type=int, default=45, choices=range(30, 61))
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--output", default="")
    parser.add_argument("--render", action="store_true", help="Explicitly allow paid video requests")
    parser.add_argument(
        "--require-live-text",
        action="store_true",
        help="Fail if the text API is unavailable or any text call falls back locally",
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
    print(json.dumps({"output_dir": result["output_dir"], "bench": result["bench"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
