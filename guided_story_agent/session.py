from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .agent import (
    ALL_FACT_FIELDS,
    DETAIL_FIELDS,
    OUTLINE_FIELDS,
    QUESTION_TEXT,
    RuleBasedStoryAgent,
    StoryAgent,
    readiness_for,
    select_next_gap,
)
from .models import (
    ArtifactReview,
    ArtifactRevision,
    CreativeBrief,
    CreativeSuggestion,
    CreatorContribution,
    FactEvidence,
    GuideTurnResult,
    RenderManifest,
    Stage,
    StoryBeat,
    StoryConflict,
    StoryFacts,
    StoryOutline,
    StoryScene,
    StoryScript,
    StoryboardPlan,
    StoryboardShot,
    VideoArtifact,
    to_plain_data,
)
from .storyboard import build_storyboard


class GuidedStorySession:
    """Single source of truth for web, CLI, self-play, and rendering."""

    schema_version = 2

    def __init__(
        self,
        brief: CreativeBrief | None = None,
        agent: StoryAgent | None = None,
    ) -> None:
        self.brief = brief or CreativeBrief()
        self.brief.validate()
        self.agent = agent or RuleBasedStoryAgent()
        self.stage = Stage.COLLECTING
        self.facts = StoryFacts(genre=self.brief.genre)
        self.contributions: list[CreatorContribution] = []
        self.detail_contributions: list[CreatorContribution] = []
        self.outline: StoryOutline | None = None
        self.script: StoryScript | None = None
        self.storyboard: StoryboardPlan | None = None
        self.render_manifest: RenderManifest | None = None
        self.expected_field = "opening"
        self._next_question = QUESTION_TEXT["opening"]
        self.pending_suggestions: list[CreativeSuggestion] = []
        self.unresolved_conflicts: list[StoryConflict] = []
        self.revisions: dict[str, list[ArtifactRevision]] = {}
        self.revision_cursor: dict[str, int] = {}

    @property
    def story_bible(self) -> StoryFacts:
        return self.facts

    @property
    def valid_turns(self) -> int:
        return len(self.contributions)

    @property
    def readiness_score(self) -> float:
        return readiness_for(self.facts, self.valid_turns)[0]

    @property
    def can_build_outline(self) -> bool:
        return (
            self.valid_turns >= 5
            and not self.facts.missing_outline_fields()
            and not self.unresolved_conflicts
        )

    @property
    def can_build_script(self) -> bool:
        return (
            self.stage == Stage.DETAILING
            and not self.facts.missing_detail_fields()
            and not self.unresolved_conflicts
        )

    @property
    def current_question(self) -> str:
        return self._next_question

    def submit_user_turn(self, text: str, *, source: str = "human") -> GuideTurnResult:
        if self.stage != Stage.COLLECTING:
            raise RuntimeError("当前阶段不再收集剧情方向。")
        return self._submit_coached_turn(text, source=source, phase="story")

    def answer_detail_question(self, text: str, *, source: str = "human") -> GuideTurnResult:
        if self.stage != Stage.DETAILING:
            raise RuntimeError("当前阶段不接受制作细节回答。")
        return self._submit_coached_turn(text, source=source, phase="production")

    def _submit_coached_turn(self, text: str, *, source: str, phase: str) -> GuideTurnResult:
        cleaned = " ".join(text.split())
        if not cleaned:
            return self._guide_result(False, "这句话是空的，请提供一个具体方向。", phase)
        history = self.contributions + self.detail_contributions
        target = self.contributions if phase == "story" else self.detail_contributions
        if any(item.text == cleaned for item in target):
            return self._guide_result(False, "这条内容已经记录过，请补充新的信息。", phase)

        data = self.agent.coach_turn(cleaned, self.facts, history, phase=phase)
        evidence = [self._evidence_from(item, cleaned) for item in data.get("extracted_facts", [])]
        conflicts = [self._conflict_from(item) for item in data.get("conflicts", [])]
        conflicts = [item for item in conflicts if item.field]
        if conflicts:
            self.unresolved_conflicts = conflicts
            self.pending_suggestions = self._suggestions_from(data.get("suggestions", []))
            self._next_question = str(data.get("next_question", "请澄清冲突后继续。"))
            return self._result_from_data(
                accepted=False,
                message="这条方向与已经记录的故事事实冲突，尚未写入故事圣经。",
                data=data,
                evidence=evidence,
                conflicts=conflicts,
                phase=phase,
            )

        allowed = OUTLINE_FIELDS + tuple(
            field for field in ALL_FACT_FIELDS if field not in OUTLINE_FIELDS
        ) if phase == "story" else DETAIL_FIELDS + (
            "dialogue_style", "camera_style", "visual_anchors"
        )
        extracted_map: dict[str, str] = {}
        for item in evidence:
            if item.field in allowed and item.value.strip():
                setattr(self.facts, item.field, item.value.strip())
                extracted_map[item.field] = item.value.strip()
        if not extracted_map:
            expected = select_next_gap(self.facts, phase, self.valid_turns)
            setattr(self.facts, expected, cleaned)
            extracted_map[expected] = cleaned
            evidence.append(FactEvidence(expected, cleaned, cleaned, 0.5))

        item = CreatorContribution(
            turn_id=len(history) + 1,
            text=cleaned,
            source=source,
            extracted_facts=extracted_map,
            fact_evidence=to_plain_data(evidence),
        )
        target.append(item)
        self.unresolved_conflicts = []
        self.expected_field = select_next_gap(self.facts, phase, self.valid_turns)
        self._next_question = str(data.get("next_question", "")).strip() or QUESTION_TEXT[self.expected_field]
        self.pending_suggestions = self._suggestions_from(data.get("suggestions", []))
        self._snapshot("story_bible", to_plain_data(self.facts), source_turn_ids=[item.turn_id])
        if phase == "story" and self.can_build_outline:
            message = "故事因果链已经完整。你可以继续补充，或主动生成大纲候选。"
            self._next_question = "故事条件已完整；请由你决定继续补充，还是主动生成大纲候选。"
        elif phase == "production" and self.can_build_script:
            message = "制作信息已经完整，可以生成定时剧本。"
            self._next_question = "制作信息已完整；请检查故事圣经，确认后生成定时剧本。"
        else:
            message = str(data.get("assistant_message", "")).strip() or "已记录这条创作方向。"
        return self._result_from_data(
            accepted=True,
            message=message,
            data=data,
            evidence=evidence,
            conflicts=[],
            phase=phase,
        )

    def request_suggestions(self) -> list[CreativeSuggestion]:
        if self.pending_suggestions:
            return deepcopy(self.pending_suggestions)
        field = select_next_gap(
            self.facts,
            "story" if self.stage == Stage.COLLECTING else "production",
            self.valid_turns,
        )
        builder = getattr(self.agent, "suggestions_for", None)
        raw = builder(field) if callable(builder) else RuleBasedStoryAgent().suggestions_for(field)
        self.pending_suggestions = self._suggestions_from(raw)
        return deepcopy(self.pending_suggestions)

    def apply_suggestion(self, suggestion_id: str) -> GuideTurnResult:
        match = next(
            (item for item in self.pending_suggestions if item.suggestion_id == suggestion_id),
            None,
        )
        if match is None:
            raise ValueError("建议不存在或已经过期。")
        if self.stage == Stage.COLLECTING:
            return self.submit_user_turn(match.content, source="accepted_suggestion")
        if self.stage == Stage.DETAILING:
            return self.answer_detail_question(match.content, source="accepted_suggestion")
        raise RuntimeError("当前阶段不能采用创作建议。")

    def update_story_bible(self, patch: dict[str, str]) -> StoryFacts:
        changed = False
        for field, value in patch.items():
            if field not in ALL_FACT_FIELDS:
                raise ValueError(f"不支持的故事圣经字段：{field}")
            cleaned = " ".join(str(value).split())
            if getattr(self.facts, field) != cleaned:
                setattr(self.facts, field, cleaned)
                changed = True
        if changed:
            self.unresolved_conflicts = [
                item for item in self.unresolved_conflicts if item.field not in patch
            ]
            self._snapshot("story_bible", to_plain_data(self.facts), user_feedback="manual edit")
            if self.outline and any(field in OUTLINE_FIELDS for field in patch):
                self.outline.confirmed = False
                self.stage = Stage.OUTLINE_REVIEW
                self._invalidate_after("outline")
                self._snapshot(
                    "outline",
                    to_plain_data(self.outline),
                    user_feedback="story bible changed after confirmation",
                )
            elif self.script and any(field in DETAIL_FIELDS for field in patch):
                self.script.confirmed = False
                self.stage = Stage.SCRIPT_REVIEW
                self._invalidate_after("script")
                self._snapshot(
                    "script",
                    to_plain_data(self.script),
                    user_feedback="story bible changed after confirmation",
                )
        return self.facts

    def build_outline(self) -> StoryOutline:
        if self.stage != Stage.COLLECTING:
            raise RuntimeError("当前阶段不能重新生成大纲。")
        if not self.can_build_outline:
            missing = "、".join(self.facts.missing_outline_fields()) or "至少五轮有效输入或冲突处理"
            raise RuntimeError(f"故事尚未达到大纲条件：{missing}。")
        self.outline = self.agent.build_outline(self.facts, self.contributions)
        durations = self._beat_durations()
        for beat, duration in zip(self.outline.beats, durations):
            beat.duration = duration
        self.stage = Stage.OUTLINE_REVIEW
        self._snapshot("outline", to_plain_data(self.outline), source_turn_ids=self.outline.source_turn_ids)
        return self.outline

    def update_outline(self, patch: dict[str, Any]) -> StoryOutline:
        if self.outline is None:
            raise RuntimeError("尚未生成大纲。")
        data = to_plain_data(self.outline)
        data.update(patch)
        data["confirmed"] = False
        self.outline = self._outline_from(data)
        self._invalidate_after("outline")
        self.stage = Stage.OUTLINE_REVIEW
        self._snapshot("outline", to_plain_data(self.outline), user_feedback="manual edit")
        return self.outline

    def revise_outline(self, feedback: str) -> StoryOutline:
        if self.outline is None or not feedback.strip():
            raise ValueError("需要已有大纲和具体修改意见。")
        revise = getattr(self.agent, "revise_artifact", None)
        revised = revise("outline", to_plain_data(self.outline), feedback) if callable(revise) else to_plain_data(self.outline)
        if "outline" in revised and isinstance(revised["outline"], dict):
            revised = revised["outline"]
        merged = to_plain_data(self.outline)
        merged.update(revised)
        self.outline = self._outline_from(merged)
        self.outline.confirmed = False
        self._invalidate_after("outline")
        self.stage = Stage.OUTLINE_REVIEW
        self._snapshot("outline", to_plain_data(self.outline), user_feedback=feedback)
        return self.outline

    def confirm_outline(self) -> None:
        if self.stage != Stage.OUTLINE_REVIEW or self.outline is None:
            raise RuntimeError("当前没有等待确认的大纲。")
        review = self.review_current_artifact("outline")
        if not review.can_confirm:
            raise RuntimeError("大纲仍有必须修复的问题：" + "；".join(review.hard_errors))
        self.outline.confirmed = True
        self.stage = Stage.DETAILING
        self.expected_field = select_next_gap(self.facts, "production")
        self._next_question = QUESTION_TEXT[self.expected_field]
        self._snapshot("outline", to_plain_data(self.outline), confirmed=True)

    def build_script(self) -> StoryScript:
        if not self.can_build_script or self.outline is None or not self.outline.confirmed:
            raise RuntimeError("请先确认大纲并补齐全部制作细节。")
        self.script = self.agent.build_script(self.outline, self.facts, self.brief.target_seconds)
        if abs(self.script.total_duration - self.brief.target_seconds) > 1:
            raise ValueError("剧本总时长与目标时长不一致。")
        self.stage = Stage.SCRIPT_REVIEW
        self._snapshot("script", to_plain_data(self.script))
        return self.script

    def update_script_scene(self, scene_id: int, patch: dict[str, Any]) -> StoryScript:
        if self.script is None:
            raise RuntimeError("尚未生成剧本。")
        scene = next((item for item in self.script.scenes if item.scene_id == int(scene_id)), None)
        if scene is None:
            raise ValueError("场景不存在。")
        for field, value in patch.items():
            if not hasattr(scene, field) or field == "scene_id":
                raise ValueError(f"不支持的场景字段：{field}")
            setattr(scene, field, value)
        self.script.confirmed = False
        self._invalidate_after("script")
        self.stage = Stage.SCRIPT_REVIEW
        self._snapshot("script", to_plain_data(self.script), user_feedback=f"edit scene {scene_id}")
        return self.script

    def revise_script(self, feedback: str) -> StoryScript:
        if self.script is None or not feedback.strip():
            raise ValueError("需要已有剧本和具体修改意见。")
        revise = getattr(self.agent, "revise_artifact", None)
        revised = revise("script", to_plain_data(self.script), feedback) if callable(revise) else to_plain_data(self.script)
        if "script" in revised and isinstance(revised["script"], dict):
            revised = revised["script"]
        merged = to_plain_data(self.script)
        merged.update(revised)
        self.script = self._script_from(merged)
        self.script.confirmed = False
        self._invalidate_after("script")
        self.stage = Stage.SCRIPT_REVIEW
        self._snapshot("script", to_plain_data(self.script), user_feedback=feedback)
        return self.script

    def confirm_script(self) -> None:
        if self.stage != Stage.SCRIPT_REVIEW or self.script is None:
            raise RuntimeError("当前没有等待确认的剧本。")
        review = self.review_current_artifact("script")
        if not review.can_confirm:
            raise RuntimeError("剧本仍有必须修复的问题：" + "；".join(review.hard_errors))
        self.script.confirmed = True
        self._snapshot("script", to_plain_data(self.script), confirmed=True)

    def build_storyboard(self) -> StoryboardPlan:
        if self.stage != Stage.SCRIPT_REVIEW or self.script is None or not self.script.confirmed:
            raise RuntimeError("请先确认剧本，再生成分镜。")
        self.storyboard = build_storyboard(self.script, self.facts)
        self.stage = Stage.STORYBOARD_REVIEW
        self._snapshot("storyboard", to_plain_data(self.storyboard))
        return self.storyboard

    def update_storyboard_shot(self, shot_id: int, patch: dict[str, Any]) -> StoryboardPlan:
        if self.storyboard is None:
            raise RuntimeError("尚未生成分镜。")
        shot = next((item for item in self.storyboard.shots if item.shot_id == int(shot_id)), None)
        if shot is None:
            raise ValueError("镜头不存在。")
        for field, value in patch.items():
            if not hasattr(shot, field) or field == "shot_id":
                raise ValueError(f"不支持的镜头字段：{field}")
            setattr(shot, field, value)
        self.storyboard.confirmed = False
        self.stage = Stage.STORYBOARD_REVIEW
        self._snapshot("storyboard", to_plain_data(self.storyboard), user_feedback=f"edit shot {shot_id}")
        return self.storyboard

    def revise_storyboard(self, feedback: str) -> StoryboardPlan:
        if self.storyboard is None or not feedback.strip():
            raise ValueError("需要已有分镜和具体修改意见。")
        revise = getattr(self.agent, "revise_artifact", None)
        revised = revise("storyboard", to_plain_data(self.storyboard), feedback) if callable(revise) else to_plain_data(self.storyboard)
        if "storyboard" in revised and isinstance(revised["storyboard"], dict):
            revised = revised["storyboard"]
        merged = to_plain_data(self.storyboard)
        merged.update(revised)
        self.storyboard = self._storyboard_from(merged)
        self.storyboard.confirmed = False
        self.stage = Stage.STORYBOARD_REVIEW
        self._snapshot("storyboard", to_plain_data(self.storyboard), user_feedback=feedback)
        return self.storyboard

    def confirm_storyboard(self) -> None:
        if self.stage != Stage.STORYBOARD_REVIEW or self.storyboard is None:
            raise RuntimeError("当前没有等待确认的分镜。")
        review = self.review_current_artifact("storyboard")
        if not review.can_confirm:
            raise RuntimeError("分镜仍有必须修复的问题：" + "；".join(review.hard_errors))
        self.storyboard.confirmed = True
        self.stage = Stage.RENDER_READY
        self._snapshot("storyboard", to_plain_data(self.storyboard), confirmed=True)

    def render_confirmed_plan(self, renderer, output_dir: str | Path) -> RenderManifest:
        if self.stage != Stage.RENDER_READY or self.storyboard is None or not self.storyboard.confirmed:
            raise RuntimeError("必须先确认完整分镜，才能调用视频生成。")
        self.render_manifest = renderer.render(self.storyboard, output_dir)
        if self.render_manifest.status == "succeeded":
            self.stage = Stage.COMPLETED
        return self.render_manifest

    def review_current_artifact(self, artifact_type: str | None = None) -> ArtifactReview:
        artifact_type = artifact_type or self._active_artifact_type()
        review = ArtifactReview(artifact_type=artifact_type)
        payload: dict[str, Any]
        if artifact_type == "outline" and self.outline:
            payload = to_plain_data(self.outline)
            for field in ("title", "opening", "protagonist_goal", "conflict", "ending"):
                if not str(payload.get(field, "")).strip():
                    review.hard_errors.append(f"大纲缺少 {field}")
            if len(self.outline.beats) != 5:
                review.hard_errors.append("大纲必须包含五个情节节点")
            if any(not beat.event.strip() or not beat.causal_link.strip() for beat in self.outline.beats):
                review.hard_errors.append("每个情节节点必须包含事件和因果承接")
            review.scores["causal_completeness"] = 1.0 if not review.hard_errors else 0.5
        elif artifact_type == "script" and self.script:
            payload = to_plain_data(self.script)
            if abs(self.script.total_duration - self.brief.target_seconds) > 1:
                review.hard_errors.append("剧本总时长不符合目标")
            for scene in self.script.scenes:
                if not (scene.visible_action or scene.action).strip():
                    review.hard_errors.append(f"场景 {scene.scene_id} 缺少可见动作")
            review.scores["filmability"] = 1.0 if not review.hard_errors else 0.5
        elif artifact_type == "storyboard" and self.storyboard:
            payload = to_plain_data(self.storyboard)
            if abs(self.storyboard.total_duration - self.brief.target_seconds) > 1:
                review.hard_errors.append("分镜总时长不符合目标")
            if not 5 <= len(self.storyboard.shots) <= 10:
                review.hard_errors.append("分镜数量必须在5到10之间")
            if any(not 3 <= shot.duration <= 15 for shot in self.storyboard.shots):
                review.hard_errors.append("存在超出3到15秒限制的镜头")
            cameras = {shot.camera for shot in self.storyboard.shots}
            if len(cameras) < min(3, len(self.storyboard.shots)):
                review.warnings.append("镜头景别变化不足")
            if any(not shot.visual_anchors for shot in self.storyboard.shots):
                review.warnings.append("部分镜头缺少视觉锚点")
            if any(not shot.character.strip() for shot in self.storyboard.shots):
                review.hard_errors.append("存在没有明确主体人物的镜头")
            if any(not shot.continuity_notes for shot in self.storyboard.shots[1:]):
                review.warnings.append("部分镜头缺少与前镜头的连续性说明")
            abrupt_spaces = sum(
                previous.location != current.location and not current.continuity_notes
                for previous, current in zip(
                    self.storyboard.shots, self.storyboard.shots[1:]
                )
            )
            if abrupt_spaces:
                review.warnings.append(f"有 {abrupt_spaces} 处空间变化没有承接说明")
            prop_tokens = [item.strip() for item in self.facts.props.replace("、", "，").split("，") if item.strip()]
            if prop_tokens and not any(
                any(token in " ".join(shot.visual_anchors) for token in prop_tokens)
                for shot in self.storyboard.shots
            ):
                review.warnings.append("关键道具没有进入视觉锚点")
            review.scores["shot_diversity"] = min(1.0, len(cameras) / 4)
            review.scores["visual_anchor_coverage"] = sum(
                bool(shot.visual_anchors) for shot in self.storyboard.shots
            ) / max(1, len(self.storyboard.shots))
        else:
            raise RuntimeError("当前没有可审查的产物。")
        remote_review = getattr(self.agent, "review_artifact", None)
        if callable(remote_review):
            data = remote_review(artifact_type, payload)
            review.warnings.extend(str(item) for item in data.get("warnings", []) if str(item))
        return review

    def undo_artifact(self, artifact_type: str | None = None) -> Any:
        artifact_type = artifact_type or self._active_artifact_type()
        cursor = self.revision_cursor.get(artifact_type, -1)
        if cursor <= 0:
            raise RuntimeError("没有更早的版本可以撤销。")
        cursor -= 1
        self.revision_cursor[artifact_type] = cursor
        return self._restore_revision(self.revisions[artifact_type][cursor])

    def redo_artifact(self, artifact_type: str | None = None) -> Any:
        artifact_type = artifact_type or self._active_artifact_type()
        cursor = self.revision_cursor.get(artifact_type, -1)
        history = self.revisions.get(artifact_type, [])
        if cursor < 0 or cursor >= len(history) - 1:
            raise RuntimeError("没有更新的版本可以重做。")
        cursor += 1
        self.revision_cursor[artifact_type] = cursor
        return self._restore_revision(history[cursor])

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "brief": to_plain_data(self.brief),
            "facts": to_plain_data(self.facts),
            "contributions": to_plain_data(self.contributions),
            "detail_contributions": to_plain_data(self.detail_contributions),
            "outline": to_plain_data(self.outline) if self.outline else None,
            "script": to_plain_data(self.script) if self.script else None,
            "storyboard": to_plain_data(self.storyboard) if self.storyboard else None,
            "render_manifest": to_plain_data(self.render_manifest) if self.render_manifest else None,
            "pending_suggestions": to_plain_data(self.pending_suggestions),
            "unresolved_conflicts": to_plain_data(self.unresolved_conflicts),
            "revisions": to_plain_data(self.revisions),
            "revision_cursor": self.revision_cursor,
        }

    @classmethod
    def load(cls, path: str | Path, *, agent: StoryAgent | None = None) -> GuidedStorySession:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        session = cls(CreativeBrief(**data.get("brief", {})), agent=agent)
        session.stage = Stage(data.get("stage", Stage.COLLECTING.value))
        session.facts = cls._facts_from(data.get("facts", {}))
        session.contributions = [CreatorContribution(**item) for item in data.get("contributions", [])]
        session.detail_contributions = [CreatorContribution(**item) for item in data.get("detail_contributions", [])]
        if data.get("outline"):
            session.outline = session._outline_from(data["outline"])
        if data.get("script"):
            session.script = session._script_from(data["script"])
        if data.get("storyboard"):
            session.storyboard = session._storyboard_from(data["storyboard"])
        if data.get("render_manifest"):
            manifest = data["render_manifest"]
            session.render_manifest = RenderManifest(
                status=str(manifest.get("status", "")),
                output_dir=str(manifest.get("output_dir", "")),
                generated_shots=[int(item) for item in manifest.get("generated_shots", [])],
                reused_shots=[int(item) for item in manifest.get("reused_shots", [])],
                failed_shots=[int(item) for item in manifest.get("failed_shots", [])],
                artifacts=[VideoArtifact(**item) for item in manifest.get("artifacts", [])],
                final_video_path=str(manifest.get("final_video_path", "")),
                audio_path=str(manifest.get("audio_path", "")),
                subtitle_path=str(manifest.get("subtitle_path", "")),
                error=str(manifest.get("error", "")),
            )
        session.pending_suggestions = session._suggestions_from(data.get("pending_suggestions", []))
        session.unresolved_conflicts = [session._conflict_from(item) for item in data.get("unresolved_conflicts", [])]
        for artifact_type, entries in data.get("revisions", {}).items():
            session.revisions[artifact_type] = [ArtifactRevision(**entry) for entry in entries]
        session.revision_cursor = {key: int(value) for key, value in data.get("revision_cursor", {}).items()}
        phase = "story" if session.stage == Stage.COLLECTING else "production"
        session.expected_field = select_next_gap(session.facts, phase, session.valid_turns)
        session._next_question = QUESTION_TEXT[session.expected_field]
        return session

    def _snapshot(
        self,
        artifact_type: str,
        payload: dict[str, Any],
        *,
        user_feedback: str = "",
        source_turn_ids: list[int] | None = None,
        confirmed: bool = False,
    ) -> ArtifactRevision:
        history = self.revisions.setdefault(artifact_type, [])
        cursor = self.revision_cursor.get(artifact_type, len(history) - 1)
        if cursor < len(history) - 1:
            del history[cursor + 1 :]
        revision = ArtifactRevision(
            artifact_type=artifact_type,
            version=len(history) + 1,
            payload=deepcopy(payload),
            parent_version=history[-1].version if history else None,
            user_feedback=user_feedback,
            source_turn_ids=source_turn_ids or [],
            confirmed=confirmed,
        )
        history.append(revision)
        self.revision_cursor[artifact_type] = len(history) - 1
        return revision

    def _restore_revision(self, revision: ArtifactRevision) -> Any:
        if revision.artifact_type == "story_bible":
            self.facts = self._facts_from(revision.payload)
            return self.facts
        if revision.artifact_type == "outline":
            self.outline = self._outline_from(revision.payload)
            self.stage = Stage.OUTLINE_REVIEW
            self._invalidate_after("outline")
            return self.outline
        if revision.artifact_type == "script":
            self.script = self._script_from(revision.payload)
            self.stage = Stage.SCRIPT_REVIEW
            self._invalidate_after("script")
            return self.script
        if revision.artifact_type == "storyboard":
            self.storyboard = self._storyboard_from(revision.payload)
            self.stage = Stage.STORYBOARD_REVIEW
            return self.storyboard
        raise ValueError("未知版本类型。")

    def _active_artifact_type(self) -> str:
        if self.stage in (Stage.OUTLINE_REVIEW, Stage.DETAILING):
            return "outline"
        if self.stage == Stage.SCRIPT_REVIEW:
            return "script"
        if self.stage in (Stage.STORYBOARD_REVIEW, Stage.RENDER_READY):
            return "storyboard"
        return "story_bible"

    def _invalidate_after(self, artifact_type: str) -> None:
        if artifact_type == "outline":
            self.script = None
            self.storyboard = None
            self.render_manifest = None
        elif artifact_type == "script":
            self.storyboard = None
            self.render_manifest = None

    def _beat_durations(self) -> list[int]:
        from .timing import allocate_durations

        return allocate_durations(self.brief.target_seconds, 5, minimum=3, maximum=15)

    def _guide_result(self, accepted: bool, message: str, phase: str) -> GuideTurnResult:
        score, missing = readiness_for(self.facts, self.valid_turns)
        return GuideTurnResult(
            accepted=accepted,
            assistant_message=message,
            next_question=self.current_question,
            suggestions=[item.content for item in self.pending_suggestions],
            valid_turns=self.valid_turns,
            missing_fields=missing if phase == "story" else self.facts.missing_detail_fields(),
            can_build_outline=self.can_build_outline,
            readiness_score=score,
            missing_critical_fields=missing,
        )

    def _result_from_data(
        self,
        *,
        accepted: bool,
        message: str,
        data: dict[str, Any],
        evidence: list[FactEvidence],
        conflicts: list[StoryConflict],
        phase: str,
    ) -> GuideTurnResult:
        score, missing = readiness_for(self.facts, self.valid_turns)
        return GuideTurnResult(
            accepted=accepted,
            assistant_message=message,
            next_question=self.current_question,
            suggestions=[item.content for item in self.pending_suggestions],
            valid_turns=self.valid_turns,
            missing_fields=missing if phase == "story" else self.facts.missing_detail_fields(),
            can_build_outline=self.can_build_outline,
            extracted_facts=evidence,
            conflicts=conflicts,
            readiness_score=score,
            missing_critical_fields=missing,
            recommended_action=str(data.get("recommended_action", "continue")),
            used_fallback=bool(data.get("used_fallback", False)),
        )

    @staticmethod
    def _evidence_from(item: Any, default_evidence: str) -> FactEvidence:
        if isinstance(item, FactEvidence):
            return item
        return FactEvidence(
            field=str(item.get("field", "")),
            value=str(item.get("value", "")),
            evidence=str(item.get("evidence", default_evidence)),
            confidence=float(item.get("confidence", 1.0)),
        )

    @staticmethod
    def _conflict_from(item: Any) -> StoryConflict:
        if isinstance(item, StoryConflict):
            return item
        return StoryConflict(
            field=str(item.get("field", "")),
            existing_value=str(item.get("existing_value", "")),
            proposed_value=str(item.get("proposed_value", "")),
            reason=str(item.get("reason", "")),
        )

    @staticmethod
    def _suggestions_from(items: list[Any]) -> list[CreativeSuggestion]:
        result = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, CreativeSuggestion):
                result.append(item)
            elif isinstance(item, dict) and str(item.get("content", "")).strip():
                result.append(
                    CreativeSuggestion(
                        suggestion_id=str(item.get("suggestion_id", f"suggestion-{index}")),
                        label=str(item.get("label", f"方向 {index}")),
                        content=str(item["content"]).strip(),
                        target_field=str(item.get("target_field", "")),
                    )
                )
        return result[:3]

    @staticmethod
    def _facts_from(data: dict[str, Any]) -> StoryFacts:
        allowed = StoryFacts.__dataclass_fields__
        return StoryFacts(**{key: str(value) for key, value in data.items() if key in allowed})

    @staticmethod
    def _outline_from(data: dict[str, Any]) -> StoryOutline:
        beats = [StoryBeat(**item) for item in data.get("beats", [])]
        fields = StoryOutline.__dataclass_fields__
        kwargs = {key: value for key, value in data.items() if key in fields and key != "beats"}
        kwargs.setdefault("source_turn_ids", [])
        return StoryOutline(**kwargs, beats=beats)

    @staticmethod
    def _script_from(data: dict[str, Any]) -> StoryScript:
        scenes = [StoryScene(**item) for item in data.get("scenes", [])]
        return StoryScript(
            title=str(data.get("title", "")),
            target_seconds=int(data.get("target_seconds", sum(item.duration for item in scenes))),
            scenes=scenes,
            confirmed=bool(data.get("confirmed", False)),
        )

    @staticmethod
    def _storyboard_from(data: dict[str, Any]) -> StoryboardPlan:
        shots = [StoryboardShot(**item) for item in data.get("shots", [])]
        artifacts = [VideoArtifact(**item) for item in data.get("artifacts", [])]
        return StoryboardPlan(
            title=str(data.get("title", "")),
            target_seconds=int(data.get("target_seconds", sum(item.duration for item in shots))),
            shots=shots,
            narration_text=str(data.get("narration_text", "")),
            confirmed=bool(data.get("confirmed", False)),
            audio_path=str(data.get("audio_path", "")),
            subtitle_path=str(data.get("subtitle_path", "")),
            artifacts=artifacts,
        )
