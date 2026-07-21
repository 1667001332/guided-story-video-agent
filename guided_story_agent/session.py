from __future__ import annotations

import json
from pathlib import Path

from .agent import DETAIL_FIELDS, OUTLINE_FIELDS, RuleBasedStoryAgent, StoryAgent
from .models import (
    CreativeBrief,
    CreatorContribution,
    GuideTurnResult,
    RenderManifest,
    Stage,
    StoryFacts,
    StoryOutline,
    StoryScript,
    StoryboardPlan,
    to_plain_data,
)
from .storyboard import build_storyboard


QUESTION_TEXT = {
    "opening": "请先用一句话写出故事开头：发生了什么异常事件？",
    "protagonist_goal": "主角是谁，他必须在短片结束前完成什么目标？",
    "conflict": "什么人、规则或危险正在阻止主角？",
    "development": "主角采取行动后，事情如何进一步升级？",
    "turning_point": "故事中最关键的发现或转折是什么？",
    "ending": "请明确故事最后发生什么，主角做出了怎样的选择？",
    "character_visuals": "主要人物长什么样？请给出可持续复用的服装和外形特征。",
    "scene_details": "故事具体发生在哪些场景、什么时间，整体色彩和天气如何？",
    "props": "哪些道具必须在不同镜头中保持一致？",
    "narration_style": "旁白采用什么人称、语气和节奏？",
    "transitions": "镜头之间用什么视觉线索承接，结尾如何呼应开头？",
}


class GuidedStorySession:
    def __init__(
        self,
        brief: CreativeBrief | None = None,
        agent: StoryAgent | None = None,
    ) -> None:
        self.brief = brief or CreativeBrief()
        self.brief.validate()
        self.agent = agent or RuleBasedStoryAgent()
        self.stage = Stage.COLLECTING
        self.facts = StoryFacts()
        self.contributions: list[CreatorContribution] = []
        self.detail_contributions: list[CreatorContribution] = []
        self.outline: StoryOutline | None = None
        self.script: StoryScript | None = None
        self.storyboard: StoryboardPlan | None = None
        self.render_manifest: RenderManifest | None = None
        self.expected_field = "opening"

    @property
    def valid_turns(self) -> int:
        return len(self.contributions)

    @property
    def can_build_outline(self) -> bool:
        return self.valid_turns >= 5 and not self.facts.missing_outline_fields()

    @property
    def can_build_script(self) -> bool:
        return self.stage == Stage.DETAILING and not self.facts.missing_detail_fields()

    @property
    def current_question(self) -> str:
        fallback = QUESTION_TEXT[self.expected_field]
        question_builder = getattr(self.agent, "next_question", None)
        if callable(question_builder):
            return question_builder(
                self.expected_field,
                self.facts,
                self.contributions + self.detail_contributions,
                fallback,
            )
        return fallback

    def submit_user_turn(self, text: str, *, source: str = "human") -> GuideTurnResult:
        if self.stage != Stage.COLLECTING:
            raise RuntimeError("当前阶段不再收集剧情方向。")
        cleaned = " ".join(text.split())
        if not cleaned:
            return self._guide_result(False, "这句话是空的，请给出一个具体剧情方向。")
        if any(item.text == cleaned for item in self.contributions):
            return self._guide_result(False, "这条方向已经记录过，请补充新的剧情信息。")
        extracted = self.agent.analyze_turn(
            cleaned, self.facts, self.expected_field, self.contributions
        )
        for key, value in extracted.items():
            if key in OUTLINE_FIELDS and value.strip():
                setattr(self.facts, key, value.strip())
        contribution = CreatorContribution(
            turn_id=len(self.contributions) + 1,
            text=cleaned,
            source=source,
            extracted_facts=extracted,
        )
        self.contributions.append(contribution)
        self.expected_field = self._next_outline_field()
        if self.can_build_outline:
            message = "故事已经从开头发展到明确结局。你可以继续补充，或点击“完成大纲”。"
        else:
            message = "已记录这条方向。我们继续补齐故事中仍然缺少的部分。"
        return self._guide_result(True, message)

    def build_outline(self) -> StoryOutline:
        if self.stage != Stage.COLLECTING:
            raise RuntimeError("当前阶段不能重新生成大纲。")
        if not self.can_build_outline:
            missing = "、".join(self.facts.missing_outline_fields()) or "至少五轮有效输入"
            raise RuntimeError(f"故事尚未达到大纲条件：{missing}。")
        self.outline = self.agent.build_outline(self.facts, self.contributions)
        self.stage = Stage.OUTLINE_REVIEW
        return self.outline

    def confirm_outline(self) -> None:
        if self.stage != Stage.OUTLINE_REVIEW or self.outline is None:
            raise RuntimeError("当前没有等待确认的大纲。")
        self.outline.confirmed = True
        self.stage = Stage.DETAILING
        self.expected_field = self._next_detail_field()

    def answer_detail_question(self, text: str, *, source: str = "human") -> GuideTurnResult:
        if self.stage != Stage.DETAILING:
            raise RuntimeError("当前阶段不接受剧本细节回答。")
        cleaned = " ".join(text.split())
        if not cleaned:
            return self._detail_result(False, "回答不能为空。")
        if any(item.text == cleaned for item in self.detail_contributions):
            return self._detail_result(False, "这条细节已经记录过，请补充新的信息。")
        extracted = self.agent.analyze_turn(
            cleaned,
            self.facts,
            self.expected_field,
            self.contributions + self.detail_contributions,
        )
        for key, value in extracted.items():
            if key in DETAIL_FIELDS and value.strip():
                setattr(self.facts, key, value.strip())
        item = CreatorContribution(
            turn_id=len(self.contributions) + len(self.detail_contributions) + 1,
            text=cleaned,
            source=source,
            extracted_facts=extracted,
        )
        self.detail_contributions.append(item)
        self.expected_field = self._next_detail_field()
        message = (
            "细节已完整，可以生成剧本。"
            if self.can_build_script
            else "已记录这条制作细节，我们继续补齐下一项。"
        )
        return self._detail_result(True, message)

    def build_script(self) -> StoryScript:
        if not self.can_build_script or self.outline is None or not self.outline.confirmed:
            raise RuntimeError("请先确认大纲并补齐全部制作细节。")
        self.script = self.agent.build_script(
            self.outline, self.facts, self.brief.target_seconds
        )
        if abs(self.script.total_duration - self.brief.target_seconds) > 1:
            raise ValueError("剧本总时长与目标时长不一致。")
        self.stage = Stage.SCRIPT_REVIEW
        return self.script

    def confirm_script(self) -> None:
        if self.stage != Stage.SCRIPT_REVIEW or self.script is None:
            raise RuntimeError("当前没有等待确认的剧本。")
        self.script.confirmed = True

    def build_storyboard(self) -> StoryboardPlan:
        if self.stage != Stage.SCRIPT_REVIEW or self.script is None or not self.script.confirmed:
            raise RuntimeError("请先确认剧本，再生成分镜。")
        self.storyboard = build_storyboard(self.script, self.facts)
        self.stage = Stage.STORYBOARD_REVIEW
        return self.storyboard

    def confirm_storyboard(self) -> None:
        if self.stage != Stage.STORYBOARD_REVIEW or self.storyboard is None:
            raise RuntimeError("当前没有等待确认的分镜。")
        self.storyboard.confirmed = True
        self.stage = Stage.RENDER_READY

    def render_confirmed_plan(self, renderer, output_dir: str | Path) -> RenderManifest:
        if self.stage != Stage.RENDER_READY or self.storyboard is None or not self.storyboard.confirmed:
            raise RuntimeError("必须先确认完整分镜，才能调用视频生成。")
        self.render_manifest = renderer.render(self.storyboard, output_dir)
        if self.render_manifest.status == "succeeded":
            self.stage = Stage.COMPLETED
        return self.render_manifest

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "stage": self.stage.value,
                    "brief": to_plain_data(self.brief),
                    "facts": to_plain_data(self.facts),
                    "contributions": to_plain_data(self.contributions),
                    "detail_contributions": to_plain_data(self.detail_contributions),
                    "outline": to_plain_data(self.outline) if self.outline else None,
                    "script": to_plain_data(self.script) if self.script else None,
                    "storyboard": to_plain_data(self.storyboard) if self.storyboard else None,
                    "render_manifest": to_plain_data(self.render_manifest) if self.render_manifest else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    def _next_outline_field(self) -> str:
        for name in OUTLINE_FIELDS:
            if not getattr(self.facts, name).strip():
                return name
        if self.valid_turns < 5:
            return "development"
        return "ending"

    def _next_detail_field(self) -> str:
        for name in DETAIL_FIELDS:
            if not getattr(self.facts, name).strip():
                return name
        return "transitions"

    def _guide_result(self, accepted: bool, message: str) -> GuideTurnResult:
        missing = self.facts.missing_outline_fields()
        question = "" if self.can_build_outline else self.current_question
        return GuideTurnResult(
            accepted=accepted,
            assistant_message=message,
            next_question=question,
            suggestions=self._suggestions(self.expected_field),
            valid_turns=self.valid_turns,
            missing_fields=missing,
            can_build_outline=self.can_build_outline,
        )

    def _detail_result(self, accepted: bool, message: str) -> GuideTurnResult:
        missing = self.facts.missing_detail_fields()
        return GuideTurnResult(
            accepted=accepted,
            assistant_message=message,
            next_question="" if self.can_build_script else self.current_question,
            suggestions=self._suggestions(self.expected_field),
            valid_turns=self.valid_turns,
            missing_fields=missing,
            can_build_outline=self.can_build_outline,
        )

    @staticmethod
    def _suggestions(field: str) -> list[str]:
        examples = {
            "opening": ["从一次异常来信开始", "从主角目睹意外开始"],
            "protagonist_goal": ["限时找到某个人", "阻止一个不可逆事件"],
            "conflict": ["可信赖的人隐瞒真相", "环境规则让行动变得危险"],
            "development": ["线索指向主角过去", "行动带来更严重后果"],
            "turning_point": ["敌人其实在保护主角", "主角就是事件的原因"],
            "ending": ["主角牺牲目标救下他人", "真相公开但留下代价"],
        }
        return examples.get(field, ["采用克制写实方案", "采用更强视觉符号方案"])
