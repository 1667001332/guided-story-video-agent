from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from .models import (
    CreatorContribution,
    FactEvidence,
    StoryBeat,
    StoryFacts,
    StoryOutline,
    StoryScene,
    StoryScript,
    to_plain_data,
)
from .timing import allocate_durations


OUTLINE_FIELDS = (
    "opening",
    "protagonist_goal",
    "conflict",
    "development",
    "turning_point",
    "ending",
)
DETAIL_FIELDS = (
    "character_visuals",
    "scene_details",
    "props",
    "narration_style",
    "transitions",
)
OPTIONAL_STORY_FIELDS = (
    "premise",
    "genre",
    "tone",
    "theme",
    "audience",
    "protagonist",
    "motivation",
    "stakes",
    "dialogue_style",
    "camera_style",
    "visual_anchors",
)
ALL_FACT_FIELDS = OPTIONAL_STORY_FIELDS + OUTLINE_FIELDS + DETAIL_FIELDS

QUESTION_TEXT = {
    "opening": "先别急着讲完整故事：故事开头的第一幅画面里发生了什么异常事件？",
    "protagonist_goal": "这件异常发生后，主角必须在短片结束前完成什么？",
    "conflict": "哪一个人、规则或危险最直接地阻止主角？失败会失去什么？",
    "development": "主角第一次采取行动后，局势怎样升级，而不是原地停留？",
    "turning_point": "什么发现会改变我们对前面事件的理解？",
    "ending": "最后一个画面发生什么？主角做出了什么不可撤回的选择？",
    "character_visuals": "为了让人物跨镜头一致，请固定主角的年龄感、发型和标志性服装。",
    "scene_details": "故事发生在哪些具体空间？时间、天气和主色调分别是什么？",
    "props": "哪些关键道具推动剧情，而且必须在多个镜头中保持一致？",
    "narration_style": "旁白采用什么人称、语气和节奏？哪些内容应改用对白？",
    "transitions": "镜头之间用哪个动作、道具或构图承接，结尾怎样呼应开场？",
}

SUGGESTION_TEMPLATES = {
    "opening": (
        "用一个不可能出现的物件打破日常",
        "让主角在公共场所发现只与自己有关的异常",
        "从一个已经发生、无法撤回的错误开始",
    ),
    "protagonist_goal": (
        "主角必须在倒计时结束前救下某个人",
        "主角必须证明一段被所有人否认的记忆",
        "主角必须决定是否交出改变命运的东西",
    ),
    "conflict": (
        "阻力来自一个动机合理的对手",
        "阻力来自世界中不能违反的规则",
        "阻力来自主角自己的秘密和错误判断",
    ),
    "development": (
        "第一次行动暂时成功，却带来更大的代价",
        "线索把主角引向一个更危险的空间",
        "盟友的选择迫使主角改变原计划",
    ),
    "turning_point": (
        "对手其实一直在保护主角",
        "主角追寻的目标正是灾难的原因",
        "关键证据证明主角的记忆并不可靠",
    ),
    "ending": (
        "主角牺牲最想保留的东西完成目标",
        "主角拒绝原目标，选择承担真实后果",
        "结尾回到开场意象，但含义完全改变",
    ),
    "character_visuals": (
        "用一件高辨识度外套作为人物锚点",
        "固定发型、年龄感和一个随身伤痕",
        "让两名主要人物具有明显不同的轮廓和色彩",
    ),
    "scene_details": (
        "限制在两个相邻空间，避免短片跳跃过大",
        "用天气变化表现情绪升级",
        "为每个空间确定一种主色和稳定光源",
    ),
    "props": (
        "选择一个既推动剧情又能做转场的道具",
        "让道具在结局中改变用途或含义",
        "只保留两个容易维持一致性的关键道具",
    ),
    "narration_style": (
        "第一人称克制旁白，只补充画面看不到的信息",
        "不用旁白，以短对白和环境声推进",
        "第三人称童话式旁白，与危险画面形成反差",
    ),
    "transitions": (
        "使用同一动作方向做匹配剪辑",
        "用关键道具特写连接前后空间",
        "让结尾复现开场构图并改变光线",
    ),
}


class StoryAgent(Protocol):
    def coach_turn(
        self,
        text: str,
        facts: StoryFacts,
        history: list[CreatorContribution],
        *,
        phase: str,
    ) -> dict[str, Any]: ...

    def build_outline(
        self, facts: StoryFacts, history: list[CreatorContribution]
    ) -> StoryOutline: ...

    def build_script(
        self, outline: StoryOutline, facts: StoryFacts, target_seconds: int
    ) -> StoryScript: ...

    def simulate_creator(self, question: str, history: list[CreatorContribution]) -> str: ...


def select_next_gap(facts: StoryFacts, phase: str, valid_turns: int = 0) -> str:
    fields = OUTLINE_FIELDS if phase == "story" else DETAIL_FIELDS
    for field in fields:
        if field == "turning_point" and facts.development.strip():
            continue
        if not getattr(facts, field).strip():
            return field
    if phase == "story" and valid_turns < 5:
        # The causal chain can be complete before the participation gate is met.
        # Ask for a distinct discovery instead of repeating the last answered gap.
        return "turning_point"
    return fields[-1]


def readiness_for(facts: StoryFacts, valid_turns: int) -> tuple[float, list[str]]:
    missing = facts.missing_outline_fields()
    required_units = 5
    completed = required_units - len(missing)
    field_score = max(0.0, completed / required_units)
    participation_score = min(1.0, valid_turns / 5)
    return round(0.8 * field_score + 0.2 * participation_score, 3), missing


class RuleBasedStoryAgent:
    """Deterministic coach used by tests and as a transparent offline fallback."""

    def coach_turn(
        self,
        text: str,
        facts: StoryFacts,
        history: list[CreatorContribution],
        *,
        phase: str,
    ) -> dict[str, Any]:
        expected = select_next_gap(facts, phase, len(history))
        extracted = self.analyze_turn(text, facts, expected, history)
        evidence = [
            FactEvidence(field=field, value=value, evidence=text, confidence=0.75)
            for field, value in extracted.items()
            if field in ALL_FACT_FIELDS and value.strip()
        ]
        preview = StoryFacts(**to_plain_data(facts))
        for item in evidence:
            setattr(preview, item.field, item.value)
        next_field = select_next_gap(preview, phase, len(history) + 1)
        score, missing = readiness_for(preview, len(history) + 1)
        understood = "、".join(item.field for item in evidence) or "新的创作方向"
        return {
            "assistant_message": f"我理解到你补充了：{understood}。",
            "extracted_facts": [to_plain_data(item) for item in evidence],
            "conflicts": [],
            "readiness_score": score,
            "missing_critical_fields": missing,
            "next_field": next_field,
            "next_question": QUESTION_TEXT[next_field],
            "suggestions": self.suggestions_for(next_field),
            "recommended_action": "build_outline" if score == 1.0 else "continue",
            "used_fallback": True,
        }

    def next_question(
        self,
        field: str,
        facts: StoryFacts,
        history: list[CreatorContribution],
        fallback_question: str,
    ) -> str:
        return QUESTION_TEXT.get(field, fallback_question)

    def analyze_turn(
        self,
        text: str,
        facts: StoryFacts,
        expected_field: str,
        history: list[CreatorContribution],
    ) -> dict[str, str]:
        cleaned = " ".join(text.split())
        extracted: dict[str, str] = {}
        denies_ending = expected_field == "ending" and any(
            token in cleaned
            for token in ("没有收尾", "没有结局", "尚未结局", "还没结局", "暂时没结局")
        )
        if expected_field in ALL_FACT_FIELDS and not denies_ending:
            extracted[expected_field] = cleaned
        if not history:
            extracted.setdefault("opening", cleaned)
            extracted.setdefault("premise", cleaned)
        rules = {
            "protagonist_goal": ("目标", "必须", "想要", "要在"),
            "conflict": ("冲突", "阻碍", "阻止他", "阻止她", "阻止主角", "威胁", "追捕", "困难"),
            "stakes": ("否则", "代价", "失去", "失败"),
            "development": ("随后", "于是", "升级", "越来越", "接着"),
            "turning_point": ("转折", "却发现", "原来", "真相"),
            "ending": ("结局", "最终", "最后", "结束"),
            "character_visuals": ("穿", "发型", "外形", "长相", "服装"),
            "scene_details": ("场景", "发生在", "雨夜", "白天", "色调"),
            "props": ("道具", "信封", "怀表", "钥匙", "手机"),
            "narration_style": ("旁白", "第一人称", "第三人称"),
            "dialogue_style": ("对白", "台词"),
            "camera_style": ("镜头", "摄影", "手持", "长镜头"),
            "transitions": ("转场", "剪辑", "呼应", "承接"),
        }
        for field, tokens in rules.items():
            if any(token in cleaned for token in tokens):
                extracted.setdefault(field, cleaned)
        return extracted

    def suggestions_for(self, field: str) -> list[dict[str, str]]:
        choices = SUGGESTION_TEMPLATES.get(field, ("补充更具体的信息",) * 3)
        return [
            {
                "suggestion_id": f"{field}-{index}",
                "label": f"方向 {index}",
                "content": content,
                "target_field": field,
            }
            for index, content in enumerate(choices, start=1)
        ]

    def build_outline(
        self, facts: StoryFacts, history: list[CreatorContribution]
    ) -> StoryOutline:
        opening = facts.opening.strip()
        title_seed = re.split(r"[，。！？]", opening)[0][:12] or "未命名短片"
        development = facts.development.strip() or facts.turning_point.strip()
        turning = facts.turning_point.strip() or development
        source_ids = [item.turn_id for item in history]
        beats = [
            StoryBeat(1, "开场钩子", opening, "异常事件触发故事", "日常被打破", 0, source_ids),
            StoryBeat(2, "目标与触发", facts.protagonist_goal, "主角被迫行动", "从犹豫到行动", 0, source_ids),
            StoryBeat(3, "冲突升级", facts.conflict, "行动遭遇阻力", "压力上升", 0, source_ids),
            StoryBeat(4, "发现或反转", turning, "新信息改变选择", "认知被颠覆", 0, source_ids),
            StoryBeat(5, "结局与落点", facts.ending, "选择造成不可逆结果", "主题落地", 0, source_ids),
        ]
        return StoryOutline(
            title=title_seed,
            logline=f"{facts.protagonist_goal}，但{facts.conflict}，最终走向{facts.ending}",
            opening=opening,
            protagonist_goal=facts.protagonist_goal.strip(),
            conflict=facts.conflict.strip(),
            development=development,
            turning_point=turning,
            ending=facts.ending.strip(),
            source_turn_ids=source_ids,
            beats=beats,
        )

    def build_script(
        self, outline: StoryOutline, facts: StoryFacts, target_seconds: int
    ) -> StoryScript:
        beats = outline.beats or self.build_outline(facts, []).beats
        durations = allocate_durations(target_seconds, len(beats), minimum=3, maximum=15)
        location = facts.scene_details.strip() or "故事的核心场景"
        character = facts.character_visuals.strip() or facts.protagonist.strip() or "外形统一的主角"
        scenes = []
        for index, (beat, duration) in enumerate(zip(beats, durations), start=1):
            beat.duration = duration
            scenes.append(
                StoryScene(
                    scene_id=index,
                    title=beat.purpose,
                    location=location,
                    time_of_day="连续时间",
                    characters=[character],
                    action=beat.event,
                    narration=beat.event,
                    duration=duration,
                    props=[facts.props] if facts.props.strip() else [],
                    visible_action=beat.event,
                    start_state=beat.causal_link,
                    end_state=beat.event,
                    emotional_change=beat.emotional_change,
                )
            )
        return StoryScript(title=outline.title, target_seconds=target_seconds, scenes=scenes)

    def simulate_creator(self, question: str, history: list[CreatorContribution]) -> str:
        if not history:
            return "暴雨夜，一名失忆的邮差在废弃车站收到一封写给明天的信。"
        answers = [
            (("目标", "必须完成", "必须在"), "邮差必须在午夜前找到收信人，阻止车站里即将发生的事故。"),
            (("冲突", "阻止主角"), "封锁车站的管理员阻止他进入站台，否则所有乘客会消失。"),
            (("升级", "采取行动"), "邮差沿旧时刻表寻找线索，却让停驶列车重新启动。"),
            (("发现", "转折", "理解"), "他发现收信人是十年前的自己，管理员一直在保护他。"),
            (("最后", "结局", "不可撤回"), "最后他烧掉信件、拉下制动，救下乘客并接受过去。"),
            (("外形", "服装"), "主角穿深蓝旧制服和邮差帽，管理员穿灰色长风衣。"),
            (("空间", "场景", "色调"), "故事发生在雨夜老车站、地下站台和停驶列车内，使用冷色霓虹。"),
            (("承接", "呼应", "构图"), "用怀表特写和列车灯做匹配剪辑，结尾回到空站台。"),
            (("道具",), "关键道具是湿信封、铜怀表和红色制动杆。"),
            (("旁白", "对白"), "第一人称克制旁白，只解释画面看不到的记忆。"),
        ]
        for keywords, answer in answers:
            if any(keyword in question for keyword in keywords):
                return answer
        return "我想让故事继续围绕邮差、信件和午夜列车推进，并保持悬疑但温暖。"


class OpenAIStoryAgent(RuleBasedStoryAgent):
    """OpenAI-compatible coach with explicit fallback telemetry."""

    def __init__(self, client: Any | None, model: str, prompt_dir: Path | None = None) -> None:
        self.client = client
        self.model = model
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts"
        self.last_used_fallback = False
        self.fallback_count = 0
        self.last_fallback_reason = ""

    @classmethod
    def from_env(cls) -> OpenAIStoryAgent:
        load_dotenv()
        api_key = os.getenv("AGNES_API_KEY", "").strip()
        model = os.getenv("AGNES_TEXT_MODEL", "agnes-2.0-flash").strip()
        if not api_key:
            return cls(None, model)
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("AGNES_LLM_BASE_URL", "https://apihub.agnes-ai.com/v1"),
                timeout=float(os.getenv("AGNES_TIMEOUT", "120")),
                max_retries=0,
            )
            return cls(client, model)
        except Exception as exc:
            agent = cls(None, model)
            agent._mark_fallback("client_init", exc)
            return agent

    def coach_turn(
        self,
        text: str,
        facts: StoryFacts,
        history: list[CreatorContribution],
        *,
        phase: str,
    ) -> dict[str, Any]:
        self.last_used_fallback = False
        if self.client is None:
            self._mark_fallback("coach_turn", "text API is not configured")
            return super().coach_turn(text, facts, history, phase=phase)
        fallback = super().coach_turn(text, facts, history, phase=phase)
        try:
            data = self._json_completion(
                "story_coach.md",
                {
                    "phase": phase,
                    "user_text": text,
                    "story_bible": to_plain_data(facts),
                    "history": [to_plain_data(item) for item in history],
                    "allowed_fields": list(ALL_FACT_FIELDS),
                    "minimum_user_turns": 5,
                    "fallback_question": fallback["next_question"],
                },
            )
            data["extracted_facts"] = self._clean_evidence(data.get("extracted_facts"), text)
            data["conflicts"] = self._clean_conflicts(data.get("conflicts"))
            preview = StoryFacts(**to_plain_data(facts))
            for item in data["extracted_facts"]:
                setattr(preview, item["field"], item["value"])
            local_gap = select_next_gap(preview, phase, len(history) + 1)
            if str(data.get("next_field", "")) != local_gap:
                data["next_question"] = QUESTION_TEXT[local_gap]
                data["suggestions"] = self.suggestions_for(local_gap)
            data["next_field"] = local_gap
            data["suggestions"] = self._clean_suggestions(data.get("suggestions"), local_gap)
            data["assistant_message"] = str(data.get("assistant_message", "")).strip() or fallback["assistant_message"]
            data["next_question"] = str(data.get("next_question", "")).strip() or fallback["next_question"]
            data["readiness_score"] = min(1.0, max(0.0, float(data.get("readiness_score", 0.0))))
            data["missing_critical_fields"] = [
                str(item) for item in data.get("missing_critical_fields", []) if str(item) in ALL_FACT_FIELDS or str(item) == "development_or_turning_point"
            ]
            data["recommended_action"] = str(data.get("recommended_action", "continue"))
            data["used_fallback"] = False
            return data
        except Exception as exc:
            self._mark_fallback("coach_turn", exc)
            return fallback

    def build_outline(
        self, facts: StoryFacts, history: list[CreatorContribution]
    ) -> StoryOutline:
        self.last_used_fallback = False
        fallback = super().build_outline(facts, history)
        if self.client is None:
            self._mark_fallback("build_outline", "text API is not configured")
            return fallback
        try:
            data = self._json_completion(
                "outline_writer.md",
                {"story_bible": to_plain_data(facts), "source_turn_ids": [item.turn_id for item in history]},
            )
            raw_beats = data.get("beats", [])
            if not isinstance(raw_beats, list) or len(raw_beats) != 5:
                raise ValueError("outline must contain five beats")
            beats = [
                StoryBeat(
                    beat_id=index,
                    purpose=str(raw["purpose"]).strip(),
                    event=str(raw["event"]).strip(),
                    causal_link=str(raw["causal_link"]).strip(),
                    emotional_change=str(raw["emotional_change"]).strip(),
                    duration=0,
                    source_turn_ids=[item.turn_id for item in history],
                )
                for index, raw in enumerate(raw_beats, start=1)
            ]
            return StoryOutline(
                title=str(data["title"]).strip(),
                logline=str(data["logline"]).strip(),
                opening=str(data.get("opening", facts.opening)).strip(),
                protagonist_goal=str(data.get("protagonist_goal", facts.protagonist_goal)).strip(),
                conflict=str(data.get("conflict", facts.conflict)).strip(),
                development=str(data.get("development", facts.development)).strip(),
                turning_point=str(data.get("turning_point", facts.turning_point)).strip(),
                ending=str(data.get("ending", facts.ending)).strip(),
                source_turn_ids=[item.turn_id for item in history],
                beats=beats,
            )
        except Exception as exc:
            self._mark_fallback("build_outline", exc)
            return fallback

    def build_script(
        self, outline: StoryOutline, facts: StoryFacts, target_seconds: int
    ) -> StoryScript:
        self.last_used_fallback = False
        fallback = super().build_script(outline, facts, target_seconds)
        if self.client is None:
            self._mark_fallback("build_script", "text API is not configured")
            return fallback
        try:
            durations = allocate_durations(target_seconds, len(outline.beats) or 5, minimum=3, maximum=15)
            data = self._json_completion(
                "script_writer.md",
                {
                    "outline": to_plain_data(outline),
                    "story_bible": to_plain_data(facts),
                    "durations": durations,
                },
            )
            raw_scenes = data.get("scenes")
            if not isinstance(raw_scenes, list) or len(raw_scenes) != len(durations):
                raise ValueError("script scene count does not match durations")
            scenes = []
            for index, (raw, duration) in enumerate(zip(raw_scenes, durations), start=1):
                scenes.append(
                    StoryScene(
                        scene_id=index,
                        title=str(raw["title"]).strip(),
                        location=str(raw["location"]).strip(),
                        time_of_day=str(raw.get("time_of_day", "连续时间")).strip(),
                        characters=[str(item).strip() for item in raw.get("characters", []) if str(item).strip()],
                        action=str(raw["visible_action"]).strip(),
                        narration=str(raw.get("narration", "")).strip(),
                        duration=duration,
                        dialogue=str(raw.get("dialogue", "")).strip(),
                        props=[str(item).strip() for item in raw.get("props", []) if str(item).strip()],
                        visible_action=str(raw["visible_action"]).strip(),
                        start_state=str(raw.get("start_state", "")).strip(),
                        end_state=str(raw.get("end_state", "")).strip(),
                        emotional_change=str(raw.get("emotional_change", "")).strip(),
                    )
                )
            return StoryScript(title=outline.title, target_seconds=target_seconds, scenes=scenes)
        except Exception as exc:
            self._mark_fallback("build_script", exc)
            return fallback

    def revise_artifact(self, artifact_type: str, payload: dict[str, Any], feedback: str) -> dict[str, Any]:
        self.last_used_fallback = False
        if self.client is None:
            self._mark_fallback("revise_artifact", "text API is not configured")
            return payload
        try:
            return self._json_completion(
                "quality_reviewer.md",
                {"task": "revise", "artifact_type": artifact_type, "artifact": payload, "feedback": feedback},
            )
        except Exception as exc:
            self._mark_fallback("revise_artifact", exc)
            return payload

    def review_artifact(self, artifact_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            return {}
        try:
            return self._json_completion(
                "quality_reviewer.md",
                {"task": "review", "artifact_type": artifact_type, "artifact": payload},
            )
        except Exception as exc:
            self._mark_fallback("review_artifact", exc)
            return {}

    def simulate_creator(self, question: str, history: list[CreatorContribution]) -> str:
        self.last_used_fallback = False
        if self.client is None:
            self._mark_fallback("simulate_creator", "text API is not configured")
            return super().simulate_creator(question, history)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._load_prompt("selfplay_creator.md")},
                    {"role": "user", "content": json.dumps({"question": question, "history": [item.text for item in history]}, ensure_ascii=False)},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("model returned empty creator response")
            return content
        except Exception as exc:
            self._mark_fallback("simulate_creator", exc)
            return super().simulate_creator(question, history)

    def _json_completion(self, prompt_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._load_prompt(prompt_name)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("model did not return JSON")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("model JSON must be an object")
        return data

    def _load_prompt(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8")

    def _mark_fallback(self, context: str, error: object) -> None:
        self.last_used_fallback = True
        self.fallback_count += 1
        self.last_fallback_reason = f"{context}: {type(error).__name__ if isinstance(error, Exception) else error}"

    @staticmethod
    def _clean_evidence(raw: Any, user_text: str) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            raise ValueError("extracted_facts must be a list")
        clean = []
        for item in raw:
            if not isinstance(item, dict) or item.get("field") not in ALL_FACT_FIELDS:
                continue
            value = str(item.get("value", "")).strip()
            if not value:
                continue
            clean.append(
                {
                    "field": item["field"],
                    "value": value,
                    "evidence": str(item.get("evidence", user_text)).strip() or user_text,
                    "confidence": min(1.0, max(0.0, float(item.get("confidence", 0.8)))),
                }
            )
        if not clean:
            raise ValueError("model did not extract any allowed facts")
        return clean

    @staticmethod
    def _clean_conflicts(raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        return [
            {
                "field": str(item.get("field", "")),
                "existing_value": str(item.get("existing_value", "")),
                "proposed_value": str(item.get("proposed_value", "")),
                "reason": str(item.get("reason", "")),
            }
            for item in raw
            if isinstance(item, dict) and item.get("field") in ALL_FACT_FIELDS
        ]

    def _clean_suggestions(self, raw: Any, target_field: str) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return self.suggestions_for(target_field)
        clean = []
        for index, item in enumerate(raw[:3], start=1):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if content:
                clean.append(
                    {
                        "suggestion_id": str(item.get("suggestion_id", f"{target_field}-{index}")),
                        "label": str(item.get("label", f"方向 {index}")),
                        "content": content,
                        "target_field": str(item.get("target_field", target_field)),
                    }
                )
        return clean or self.suggestions_for(target_field)
