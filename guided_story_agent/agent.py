from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from .models import CreatorContribution, StoryFacts, StoryOutline, StoryScene, StoryScript
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


class StoryAgent(Protocol):
    def next_question(
        self,
        field: str,
        facts: StoryFacts,
        history: list[CreatorContribution],
        fallback_question: str,
    ) -> str: ...

    def analyze_turn(
        self,
        text: str,
        facts: StoryFacts,
        expected_field: str,
        history: list[CreatorContribution],
    ) -> dict[str, str]: ...

    def build_outline(
        self, facts: StoryFacts, history: list[CreatorContribution]
    ) -> StoryOutline: ...

    def build_script(
        self, outline: StoryOutline, facts: StoryFacts, target_seconds: int
    ) -> StoryScript: ...

    def simulate_creator(self, question: str, history: list[CreatorContribution]) -> str: ...


class RuleBasedStoryAgent:
    """Deterministic fallback used by tests and when no text API is configured."""

    def next_question(
        self,
        field: str,
        facts: StoryFacts,
        history: list[CreatorContribution],
        fallback_question: str,
    ) -> str:
        return fallback_question

    def analyze_turn(
        self,
        text: str,
        facts: StoryFacts,
        expected_field: str,
        history: list[CreatorContribution],
    ) -> dict[str, str]:
        cleaned = " ".join(text.split())
        extracted: dict[str, str] = {}
        if expected_field in OUTLINE_FIELDS + DETAIL_FIELDS:
            extracted[expected_field] = cleaned
        if any(token in cleaned for token in ("结局", "最终", "最后", "结束")):
            extracted["ending"] = cleaned
        if any(token in cleaned for token in ("冲突", "阻止", "威胁", "追捕", "困难")):
            extracted.setdefault("conflict", cleaned)
        if any(token in cleaned for token in ("转折", "却发现", "原来", "真相")):
            extracted.setdefault("turning_point", cleaned)
        return extracted

    def build_outline(
        self, facts: StoryFacts, history: list[CreatorContribution]
    ) -> StoryOutline:
        opening = facts.opening.strip()
        title_seed = re.split(r"[，。！？]", opening)[0][:12] or "未命名短片"
        development = facts.development.strip() or facts.turning_point.strip()
        turning = facts.turning_point.strip() or development
        return StoryOutline(
            title=title_seed,
            logline=f"{facts.protagonist_goal}，但{facts.conflict}，最终走向{facts.ending}",
            opening=opening,
            protagonist_goal=facts.protagonist_goal.strip(),
            conflict=facts.conflict.strip(),
            development=development,
            turning_point=turning,
            ending=facts.ending.strip(),
            source_turn_ids=[item.turn_id for item in history],
        )

    def build_script(
        self, outline: StoryOutline, facts: StoryFacts, target_seconds: int
    ) -> StoryScript:
        durations = allocate_durations(target_seconds, 5, minimum=3, maximum=15)
        beats = [
            ("开场", outline.opening),
            ("目标与阻力", f"{outline.protagonist_goal}。{outline.conflict}"),
            ("发展", outline.development),
            ("转折", outline.turning_point),
            ("结局", outline.ending),
        ]
        location = facts.scene_details.strip() or "故事的核心场景"
        character = facts.character_visuals.strip() or "外形统一的主角"
        scenes = [
            StoryScene(
                scene_id=index,
                title=title,
                location=location,
                time_of_day="连续时间",
                characters=[character],
                action=content,
                narration=content,
                duration=durations[index - 1],
            )
            for index, (title, content) in enumerate(beats, start=1)
        ]
        return StoryScript(title=outline.title, target_seconds=target_seconds, scenes=scenes)

    def simulate_creator(self, question: str, history: list[CreatorContribution]) -> str:
        if not history:
            return "暴雨夜，一名失忆的邮差在废弃车站收到一封写给明天的信。"
        answers = [
            (("目标", "必须完成"), "邮差必须在午夜前找到收信人，阻止车站里即将发生的事故。"),
            (("冲突", "阻止主角"), "冲突来自封锁车站的管理员，他认定这封信会让过去重演。"),
            (("发展", "升级"), "邮差沿着旧时刻表寻找线索，逐渐发现每一站都对应自己失去的一段记忆。"),
            (("转折", "关键的发现"), "转折是他发现真正的收信人就是十年前的自己，而管理员一直在保护他。"),
            (("结局", "最后发生", "怎样的选择"), "最后他烧掉信件、拉下紧急制动，救下乘客，也接受了自己的过去。"),
            (("外形", "长什么样", "服装"), "主角穿深蓝旧制服、戴磨损邮差帽，管理员穿灰色长风衣，造型始终一致。"),
            (("场景", "发生在哪些", "色彩"), "故事发生在雨夜的老车站、地下站台和停驶列车内，冷色霓虹贯穿全片。"),
            (("道具",), "关键道具是湿透的信封、铜制怀表和红色紧急制动杆。"),
            (("旁白",), "旁白克制、低沉，用第一人称讲述，每句话短而清晰。"),
            (("承接", "呼应开头"), "镜头用怀表特写和列车灯光做匹配剪辑，结尾回到空站台。"),
        ]
        for keywords, answer in answers:
            if any(keyword in question for keyword in keywords):
                return answer
        return "让情节继续围绕邮差、信件和午夜列车推进，并保持悬疑但温暖的基调。"


class OpenAIStoryAgent(RuleBasedStoryAgent):
    """OpenAI-compatible story agent with deterministic local fallback."""

    def __init__(self, client: Any | None, model: str, prompt_dir: Path | None = None) -> None:
        self.client = client
        self.model = model
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts"
        self.last_used_fallback = False

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
        except Exception:
            client = None
        return cls(client, model)

    def analyze_turn(
        self,
        text: str,
        facts: StoryFacts,
        expected_field: str,
        history: list[CreatorContribution],
    ) -> dict[str, str]:
        if self.client is None:
            self.last_used_fallback = True
            return super().analyze_turn(text, facts, expected_field, history)
        payload = {
            "task": "extract_facts",
            "expected_field": expected_field,
            "known_facts": {name: getattr(facts, name) for name in OUTLINE_FIELDS + DETAIL_FIELDS},
            "user_text": text,
            "allowed_fields": list(OUTLINE_FIELDS + DETAIL_FIELDS),
        }
        try:
            data = self._json_completion("guide.md", payload)
            extracted = data.get("extracted_facts", {})
            if not isinstance(extracted, dict):
                raise ValueError("extracted_facts must be an object")
            clean = {
                key: str(value).strip()
                for key, value in extracted.items()
                if key in OUTLINE_FIELDS + DETAIL_FIELDS and str(value).strip()
            }
            if not clean:
                clean[expected_field] = " ".join(text.split())
            self.last_used_fallback = False
            return clean
        except Exception:
            self.last_used_fallback = True
            return super().analyze_turn(text, facts, expected_field, history)

    def build_outline(
        self, facts: StoryFacts, history: list[CreatorContribution]
    ) -> StoryOutline:
        if self.client is None:
            return super().build_outline(facts, history)
        try:
            data = self._json_completion(
                "author.md",
                {
                    "task": "build_outline",
                    "facts": {name: getattr(facts, name) for name in OUTLINE_FIELDS},
                    "source_turn_ids": [item.turn_id for item in history],
                },
            )
            outline = StoryOutline(
                title=str(data["title"]).strip(),
                logline=str(data["logline"]).strip(),
                opening=str(data["opening"]).strip(),
                protagonist_goal=str(data["protagonist_goal"]).strip(),
                conflict=str(data["conflict"]).strip(),
                development=str(data["development"]).strip(),
                turning_point=str(data["turning_point"]).strip(),
                ending=str(data["ending"]).strip(),
                source_turn_ids=[item.turn_id for item in history],
            )
            if not all((outline.title, outline.opening, outline.conflict, outline.ending)):
                raise ValueError("outline missing required values")
            return outline
        except Exception:
            self.last_used_fallback = True
            return super().build_outline(facts, history)

    def build_script(
        self, outline: StoryOutline, facts: StoryFacts, target_seconds: int
    ) -> StoryScript:
        if self.client is None:
            return super().build_script(outline, facts, target_seconds)
        try:
            durations = allocate_durations(target_seconds, 5, minimum=3, maximum=15)
            data = self._json_completion(
                "author.md",
                {
                    "task": "build_script",
                    "outline": outline.__dict__ if hasattr(outline, "__dict__") else {
                        name: getattr(outline, name)
                        for name in (
                            "title", "logline", "opening", "protagonist_goal", "conflict",
                            "development", "turning_point", "ending"
                        )
                    },
                    "details": {name: getattr(facts, name) for name in DETAIL_FIELDS},
                    "durations": durations,
                },
            )
            raw_scenes = data.get("scenes")
            if not isinstance(raw_scenes, list) or len(raw_scenes) != 5:
                raise ValueError("script must contain five scenes")
            scenes = []
            for index, raw in enumerate(raw_scenes, start=1):
                scenes.append(
                    StoryScene(
                        scene_id=index,
                        title=str(raw["title"]).strip(),
                        location=str(raw["location"]).strip(),
                        time_of_day=str(raw.get("time_of_day", "连续时间")).strip(),
                        characters=[str(item).strip() for item in raw.get("characters", []) if str(item).strip()],
                        action=str(raw["action"]).strip(),
                        narration=str(raw["narration"]).strip(),
                        duration=durations[index - 1],
                    )
                )
            return StoryScript(title=outline.title, target_seconds=target_seconds, scenes=scenes)
        except Exception:
            self.last_used_fallback = True
            return super().build_script(outline, facts, target_seconds)

    def simulate_creator(self, question: str, history: list[CreatorContribution]) -> str:
        if self.client is None:
            return super().simulate_creator(question, history)
        prompt = self._load_prompt("selfplay_creator.md")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps({
                        "question": question,
                        "history": [item.text for item in history],
                    }, ensure_ascii=False)},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception:
            pass
        self.last_used_fallback = True
        return super().simulate_creator(question, history)

    def _json_completion(self, prompt_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._load_prompt(prompt_name)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=2400,
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

    def next_question(
        self,
        field: str,
        facts: StoryFacts,
        history: list[CreatorContribution],
        fallback_question: str,
    ) -> str:
        if self.client is None:
            return fallback_question
        try:
            data = self._json_completion(
                "guide.md",
                {
                    "task": "write_next_question",
                    "target_field": field,
                    "known_facts": {
                        name: getattr(facts, name)
                        for name in OUTLINE_FIELDS + DETAIL_FIELDS
                    },
                    "recent_user_turns": [item.text for item in history[-4:]],
                    "fallback_question": fallback_question,
                },
            )
            question = str(data.get("question", "")).strip()
            if question:
                return question
        except Exception:
            self.last_used_fallback = True
        return fallback_question
