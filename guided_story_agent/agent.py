from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from .models import (
    CreatorContribution,
    DraftBundle,
    ElementOption,
    ElementPalette,
    IdeaBatch,
    IdeaCard,
    SourceAttribution,
    StoryBeat,
    StoryFacts,
    StoryOutline,
    StoryScene,
    StoryScript,
    to_plain_data,
)
from .timing import allocate_durations


IDEA_COUNT = 8
ELEMENT_KINDS = ("character", "conflict", "turning_point", "ending")


class StoryAgent(Protocol):
    def generate_ideas(
        self,
        direction: str,
        *,
        round_number: int,
        feedback: str = "",
        previous_cards: list[IdeaCard] | None = None,
        mode: str = "diverge",
        anchors: list[IdeaCard] | None = None,
    ) -> IdeaBatch: ...

    def expand_elements(self, direction: str, selected_cards: list[IdeaCard]) -> ElementPalette: ...

    def generate_draft(
        self,
        direction: str,
        selected_cards: list[IdeaCard],
        selected_elements: dict[str, ElementOption],
        target_seconds: int,
    ) -> DraftBundle: ...

    def revise_draft(self, draft: DraftBundle, feedback: str) -> DraftBundle: ...

    def simulate_creator_direction(self) -> str: ...


class RuleBasedStoryAgent:
    """Deterministic creative garden used by tests and offline fallback."""

    _concepts = (
        ("倒计时", "必须在最后一班车离开前完成一件不可能的事", "紧张温暖", "付出代价后得到和解"),
        ("时间错位", "收到来自明天的求救信息", "悬疑克制", "开场意象在结尾反转"),
        ("身份秘密", "发现最信任的人隐藏了真实身份", "情感悬疑", "主角主动保守秘密"),
        ("世界规则", "城市里每个人只能说一句真话", "奇幻冷幽默", "主角把唯一真话留给陌生人"),
        ("道德选择", "救一个人会让另一段记忆永远消失", "忧伤治愈", "主角选择承担而非逃避"),
        ("不可靠记忆", "主角追查的失踪者其实是过去的自己", "心理悬疑", "接受真相并停止循环"),
        ("物件奇迹", "一件普通物品开始实现未说出口的愿望", "轻奇幻", "物件失效但关系被修复"),
        (
            "平行选择",
            "同一空间出现两个做过不同选择的自己",
            "科幻诗意",
            "两个自己共同作出第三种选择",
        ),
        ("封闭空间", "所有人被困在只停留六分钟的房间", "高概念惊悚", "主角留下让别人离开"),
        ("误会喜剧", "一个无心谎言被全城当成重大预言", "荒诞喜剧", "真相更离谱却救了所有人"),
        ("关系反转", "对手其实一直替主角承担后果", "温暖反转", "主角放弃胜利选择道歉"),
        ("循环任务", "每天醒来都必须把同一封信送给不同的人", "浪漫悬疑", "最后收件人是主角自己"),
        ("声音谜题", "只有主角能听见城市即将消失的声音", "都市奇幻", "主角用沉默让城市留下"),
        ("交换代价", "每实现一个愿望就会失去一种感官", "黑色童话", "主角放弃最后愿望"),
        ("微小英雄", "最不起眼的人掌握唯一能阻止事故的细节", "现实励志", "没人知道英雄是谁"),
        ("善意骗局", "主角必须维持一个谎言直到日落", "温柔现实", "谎言被识破但善意被接受"),
    )
    _lenses = (
        "逆序叙事",
        "旁观者视角",
        "群像接力",
        "无对白表达",
        "单场景推进",
        "伪纪录观察",
        "双时间线交错",
        "关键物件视角",
    )

    def generate_ideas(
        self,
        direction: str,
        *,
        round_number: int,
        feedback: str = "",
        previous_cards: list[IdeaCard] | None = None,
        mode: str = "diverge",
        anchors: list[IdeaCard] | None = None,
    ) -> IdeaBatch:
        cleaned = " ".join(direction.split())
        if not cleaned:
            raise ValueError("请先给出一句方向。")
        anchors = anchors or []
        previous_cards = previous_cards or []
        offset = ((round_number - 1) * IDEA_COUNT) % len(self._concepts)
        concepts = [self._concepts[(offset + i) % len(self._concepts)] for i in range(IDEA_COUNT)]
        lens = self._lenses[(round_number - 3) % len(self._lenses)] if round_number > 2 else ""
        source_ids = [item.idea_id for item in anchors]
        cards = []
        anchor_text = " × ".join(item.title for item in anchors)
        for index, (axis, conflict, tone, ending) in enumerate(concepts, start=1):
            if mode == "similar" and anchors:
                axis = f"{anchors[0].title}·变体{index}"
                protagonist = anchors[0].protagonist
                hook = f"保留“{anchors[0].hook}”的核心吸引力，但改用{conflict}"
            elif mode == "mix" and anchors:
                protagonist = anchors[(index - 1) % len(anchors)].protagonist
                hook = f"融合{anchor_text}，重点采用{axis}结构"
            else:
                protagonist = self._protagonist(cleaned, index)
                hook = f"把“{cleaned}”变成{axis}故事"
            if lens:
                axis = f"{axis}·{lens}"
                hook = f"{hook}，并用{lens}重新组织因果"
            suffix = f"，并满足你的补充：{feedback}" if feedback else ""
            card = IdeaCard(
                idea_id=f"idea-r{round_number}-{index}",
                title=f"{axis}：{self._short_seed(cleaned)}",
                logline=f"{protagonist}因为{hook}，{conflict}{suffix}。",
                hook=hook,
                protagonist=protagonist,
                central_conflict=conflict,
                tone=tone,
                ending_direction=ending,
                source_idea_ids=source_ids,
                generation_kind=mode,
            )
            cards.append(card)
        return IdeaBatch(
            round=round_number,
            cards=cards,
            recommended_id=cards[0].idea_id,
            feedback=feedback,
            generation_kind=mode,
        )

    def expand_elements(self, direction: str, selected_cards: list[IdeaCard]) -> ElementPalette:
        card = (
            selected_cards[0]
            if selected_cards
            else self.generate_ideas(direction, round_number=1).cards[0]
        )
        source_ids = [item.idea_id for item in selected_cards]
        raw = {
            "character": (
                ("普通人视角", f"{card.protagonist}，能力普通但观察敏锐"),
                ("带秘密的主角", f"{card.protagonist}隐瞒着与事件有关的过去"),
                ("双主角", f"{card.protagonist}与立场相反的伙伴被迫合作"),
                ("非人主角", "让一件物品、动物或人工智能承担主角视角"),
            ),
            "conflict": (
                ("倒计时压力", "目标必须在明确倒计时结束前完成"),
                ("合理的对手", "对手的阻止行为有值得理解的理由"),
                ("规则代价", "每次接近目标都会失去同等重要的东西"),
                ("内外双重冲突", "外部危险迫使主角承认自己的错误"),
            ),
            "turning_point": (
                ("身份反转", "主角追寻的人其实一直以另一身份陪在身边"),
                ("目标反转", "原本想得到的东西正是灾难的来源"),
                ("记忆反转", "关键记忆来自别人，而非主角本人"),
                ("关系反转", "看似的敌人一直在承担保护主角的代价"),
            ),
            "ending": (
                ("牺牲式和解", card.ending_direction),
                ("开放余韵", "目标完成，但最后一个细节暗示事情尚未真正结束"),
                ("温暖闭环", "结尾复现开场画面，含义从孤独变成连接"),
                ("黑色幽默", "主角成功解决大问题，却立刻面对一个荒诞小麻烦"),
            ),
        }
        return ElementPalette(
            options={
                kind: [
                    ElementOption(
                        option_id=f"{kind}-{index}",
                        kind=kind,
                        title=title,
                        content=content,
                        source_idea_ids=source_ids,
                    )
                    for index, (title, content) in enumerate(items, start=1)
                ]
                for kind, items in raw.items()
            }
        )

    def generate_draft(
        self,
        direction: str,
        selected_cards: list[IdeaCard],
        selected_elements: dict[str, ElementOption],
        target_seconds: int,
    ) -> DraftBundle:
        card = (
            selected_cards[0]
            if selected_cards
            else self.generate_ideas(direction, round_number=1).cards[0]
        )
        chosen = {kind: item.content for kind, item in selected_elements.items()}
        protagonist = chosen.get("character", card.protagonist)
        conflict = chosen.get("conflict", card.central_conflict)
        turning = chosen.get("turning_point", f"主角发现：{card.hook}")
        ending = chosen.get("ending", card.ending_direction)
        opening = f"{direction}。第一幅画面立刻出现异常：{card.hook}。"
        goal = f"{protagonist}必须解决由“{direction}”引发的问题"
        development = f"主角第一次行动后，{conflict}，局势因此升级"
        source_ids = [item.idea_id for item in selected_cards]
        beats = [
            StoryBeat(1, "开场钩子", opening, "异常打破日常", "惊讶", 0, []),
            StoryBeat(2, "目标触发", goal, "异常迫使主角行动", "决心", 0, []),
            StoryBeat(3, "冲突升级", conflict, "行动带来更大代价", "压力", 0, []),
            StoryBeat(4, "关键反转", turning, "新信息改变原计划", "震动", 0, []),
            StoryBeat(5, "结局落点", ending, "主角作出不可逆选择", "释然", 0, []),
        ]
        durations = allocate_durations(target_seconds, 5, minimum=3, maximum=15)
        for beat, duration in zip(beats, durations):
            beat.duration = duration
        outline = StoryOutline(
            title=card.title,
            logline=card.logline,
            opening=opening,
            protagonist_goal=goal,
            conflict=conflict,
            development=development,
            turning_point=turning,
            ending=ending,
            source_turn_ids=[],
            beats=beats,
        )
        scenes = [
            StoryScene(
                scene_id=beat.beat_id,
                title=beat.purpose,
                location="故事核心场景",
                time_of_day="连续时间",
                characters=[protagonist],
                action=beat.event,
                visible_action=beat.event,
                narration=beat.event,
                dialogue="" if beat.beat_id != 5 else "我终于知道该留下什么了。",
                props=["贯穿故事的关键物件"],
                start_state=beat.causal_link,
                end_state=beat.event,
                emotional_change=beat.emotional_change,
                duration=beat.duration,
            )
            for beat in beats
        ]
        script = StoryScript(title=card.title, target_seconds=target_seconds, scenes=scenes)
        source_type = "selected_card" if selected_cards else "ai_fill"
        fields = {
            "opening": opening,
            "protagonist": protagonist,
            "conflict": conflict,
            "turning_point": turning,
            "ending": ending,
        }
        field_sources = {}
        ai_filled = []
        for field, value in fields.items():
            if field == "opening":
                origin = "user"
            elif field in selected_elements:
                origin = "selected_element"
            else:
                origin = source_type
            if origin == "ai_fill":
                ai_filled.append(field)
            field_sources[field] = SourceAttribution(
                field=field,
                source_type=origin,
                value=value,
                source_ids=source_ids,
            )
        production_fills = {
            "scene_details": "故事核心场景",
            "props": "贯穿故事的关键物件",
            "narration": "克制旁白",
            "dialogue": "简短可表演对白",
            "transitions": "动作与物件匹配剪辑",
        }
        for field, value in production_fills.items():
            ai_filled.append(field)
            field_sources[field] = SourceAttribution(
                field=field,
                source_type="ai_fill",
                value=value,
                source_ids=[],
            )
        return DraftBundle(
            outline=outline,
            script=script,
            field_sources=field_sources,
            ai_filled_fields=ai_filled,
        )

    def revise_draft(self, draft: DraftBundle, feedback: str) -> DraftBundle:
        if not feedback.strip():
            raise ValueError("请用一句话说明想怎样修改。")
        revised = deepcopy(draft)
        revised.version += 1
        revised.script.confirmed = False
        revised.outline.confirmed = False
        revised.script.scenes[
            0
        ].action = f"根据“{feedback.strip()}”调整：{revised.script.scenes[0].action}"
        revised.script.scenes[0].visible_action = revised.script.scenes[0].action
        return revised

    def build_outline(self, facts: StoryFacts, history: list[CreatorContribution]) -> StoryOutline:
        direction = facts.premise or facts.opening or "一个尚未命名的故事"
        return self.generate_draft(direction, [], {}, 45).outline

    def build_script(
        self, outline: StoryOutline, facts: StoryFacts, target_seconds: int
    ) -> StoryScript:
        return self.generate_draft(outline.logline or outline.title, [], {}, target_seconds).script

    def simulate_creator_direction(self) -> str:
        return "暴雨夜，一名邮差在废弃车站收到一封写给明天的信。"

    def simulate_creator(self, question: str, history: list[CreatorContribution]) -> str:
        return self.simulate_creator_direction()

    @staticmethod
    def _short_seed(direction: str) -> str:
        return re.split(r"[，。！？]", direction)[0][:12] or "未命名方向"

    @staticmethod
    def _protagonist(direction: str, index: int) -> str:
        roles = ("谨慎的年轻人", "隐瞒秘密的快递员", "即将离开的学生", "失去记忆的老人")
        return roles[(index - 1) % len(roles)]


class OpenAIStoryAgent(RuleBasedStoryAgent):
    """Agnes/OpenAI-compatible creative agent with explicit fallback telemetry."""

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

    def generate_ideas(
        self,
        direction: str,
        *,
        round_number: int,
        feedback: str = "",
        previous_cards: list[IdeaCard] | None = None,
        mode: str = "diverge",
        anchors: list[IdeaCard] | None = None,
    ) -> IdeaBatch:
        self.last_used_fallback = False
        fallback = super().generate_ideas(
            direction,
            round_number=round_number,
            feedback=feedback,
            previous_cards=previous_cards,
            mode=mode,
            anchors=anchors,
        )
        if self.client is None:
            self._mark_fallback("generate_ideas", "text API is not configured")
            return fallback
        prompt = {
            "diverge": "idea_divergence.md",
            "similar": "idea_similarity.md",
            "mix": "idea_mixer.md",
        }.get(mode, "idea_divergence.md")
        try:
            data = self._json_completion(
                prompt,
                {
                    "direction": direction,
                    "feedback": feedback,
                    "round": round_number,
                    "anchors": to_plain_data(anchors or []),
                    "previous_cards": to_plain_data(previous_cards or []),
                    "required_count": IDEA_COUNT,
                },
            )
            cards = self._cards_from(
                data.get("cards", []),
                fallback,
                round_number,
                mode,
                previous_cards or [],
            )
            recommended = str(data.get("recommended_id", ""))
            if recommended not in {card.idea_id for card in cards}:
                recommended = cards[0].idea_id
            return IdeaBatch(
                round=round_number,
                cards=cards,
                recommended_id=recommended,
                feedback=feedback,
                generation_kind=mode,
            )
        except Exception as exc:
            self._mark_fallback("generate_ideas", exc)
            return fallback

    def expand_elements(self, direction: str, selected_cards: list[IdeaCard]) -> ElementPalette:
        self.last_used_fallback = False
        fallback = super().expand_elements(direction, selected_cards)
        if self.client is None:
            self._mark_fallback("expand_elements", "text API is not configured")
            return fallback
        try:
            data = self._json_completion(
                "element_expansion.md",
                {"direction": direction, "selected_cards": to_plain_data(selected_cards)},
            )
            options: dict[str, list[ElementOption]] = {}
            for kind in ELEMENT_KINDS:
                raw = data.get("options", {}).get(kind, [])
                if not isinstance(raw, list) or len(raw) != 4:
                    raise ValueError(f"{kind} must contain four options")
                options[kind] = [
                    ElementOption(
                        option_id=f"{kind}-{index}",
                        kind=kind,
                        title=str(item["title"]).strip(),
                        content=str(item["content"]).strip(),
                        source_idea_ids=[card.idea_id for card in selected_cards],
                    )
                    for index, item in enumerate(raw, 1)
                ]
            return ElementPalette(options=options)
        except Exception as exc:
            self._mark_fallback("expand_elements", exc)
            return fallback

    def generate_draft(
        self,
        direction: str,
        selected_cards: list[IdeaCard],
        selected_elements: dict[str, ElementOption],
        target_seconds: int,
    ) -> DraftBundle:
        self.last_used_fallback = False
        fallback = super().generate_draft(
            direction, selected_cards, selected_elements, target_seconds
        )
        if self.client is None:
            self._mark_fallback("generate_draft", "text API is not configured")
            return fallback
        try:
            durations = allocate_durations(target_seconds, 5, minimum=3, maximum=15)
            data = self._json_completion(
                "draft_writer.md",
                {
                    "direction": direction,
                    "selected_cards": to_plain_data(selected_cards),
                    "selected_elements": to_plain_data(selected_elements),
                    "durations": durations,
                },
            )
            return self._draft_from(data, fallback, durations)
        except Exception as exc:
            self._mark_fallback("generate_draft", exc)
            return fallback

    def revise_draft(self, draft: DraftBundle, feedback: str) -> DraftBundle:
        self.last_used_fallback = False
        fallback = super().revise_draft(draft, feedback)
        if self.client is None:
            self._mark_fallback("revise_draft", "text API is not configured")
            return fallback
        try:
            data = self._json_completion(
                "draft_rewriter.md",
                {"draft": to_plain_data(draft), "feedback": feedback},
            )
            durations = [scene.duration for scene in draft.script.scenes]
            revised = self._draft_from(data, fallback, durations)
            revised.version = draft.version + 1
            return revised
        except Exception as exc:
            self._mark_fallback("revise_draft", exc)
            return fallback

    def simulate_creator_direction(self) -> str:
        if self.client is None:
            self._mark_fallback("simulate_creator_direction", "text API is not configured")
            return super().simulate_creator_direction()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._load_prompt("selfplay_creator.md")},
                    {"role": "user", "content": "只给出一句模糊但有画面感的短片方向。"},
                ],
                temperature=0.7,
                max_tokens=120,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("model returned empty direction")
            return content
        except Exception as exc:
            self._mark_fallback("simulate_creator_direction", exc)
            return super().simulate_creator_direction()

    def _cards_from(
        self,
        raw: Any,
        fallback: IdeaBatch,
        round_number: int,
        mode: str,
        previous_cards: list[IdeaCard],
    ) -> list[IdeaCard]:
        if not isinstance(raw, list):
            return fallback.cards
        result: list[IdeaCard] = []
        seen = {card.fingerprint for card in previous_cards}
        for index, item in enumerate(raw[:IDEA_COUNT], 1):
            try:
                card = IdeaCard(
                    idea_id=f"idea-r{round_number}-{index}",
                    title=str(item["title"]).strip(),
                    logline=str(item["logline"]).strip(),
                    hook=str(item["hook"]).strip(),
                    protagonist=str(item["protagonist"]).strip(),
                    central_conflict=str(item["central_conflict"]).strip(),
                    tone=str(item["tone"]).strip(),
                    ending_direction=str(item["ending_direction"]).strip(),
                    source_idea_ids=[str(value) for value in item.get("source_idea_ids", [])],
                    generation_kind=mode,
                )
            except (KeyError, TypeError):
                continue
            if not all((card.title, card.logline, card.hook)) or card.fingerprint in seen:
                continue
            seen.add(card.fingerprint)
            result.append(card)
        for card in fallback.cards:
            if len(result) == IDEA_COUNT:
                break
            if card.fingerprint not in seen:
                card.idea_id = f"idea-r{round_number}-{len(result) + 1}"
                result.append(card)
                seen.add(card.fingerprint)
        return result

    @staticmethod
    def _draft_from(
        data: dict[str, Any], fallback: DraftBundle, durations: list[int]
    ) -> DraftBundle:
        raw_outline = data.get("outline", {})
        raw_scenes = data.get("script", {}).get("scenes", [])
        if len(raw_scenes) != 5:
            raise ValueError("draft must contain five scenes")
        beats = []
        for index, item in enumerate(raw_outline.get("beats", []), 1):
            beats.append(
                StoryBeat(
                    beat_id=index,
                    purpose=str(item["purpose"]),
                    event=str(item["event"]),
                    causal_link=str(item["causal_link"]),
                    emotional_change=str(item["emotional_change"]),
                    duration=durations[index - 1],
                )
            )
        if len(beats) != 5:
            raise ValueError("outline must contain five beats")
        outline = StoryOutline(
            title=str(raw_outline["title"]),
            logline=str(raw_outline["logline"]),
            opening=str(raw_outline["opening"]),
            protagonist_goal=str(raw_outline["protagonist_goal"]),
            conflict=str(raw_outline["conflict"]),
            development=str(raw_outline["development"]),
            turning_point=str(raw_outline["turning_point"]),
            ending=str(raw_outline["ending"]),
            source_turn_ids=[],
            beats=beats,
        )
        scenes = []
        for index, (item, duration) in enumerate(zip(raw_scenes, durations), 1):
            scenes.append(
                StoryScene(
                    scene_id=index,
                    title=str(item["title"]),
                    location=str(item["location"]),
                    time_of_day=str(item.get("time_of_day", "连续时间")),
                    characters=[str(value) for value in item.get("characters", [])],
                    action=str(item["visible_action"]),
                    visible_action=str(item["visible_action"]),
                    dialogue=str(item.get("dialogue", "")),
                    narration=str(item.get("narration", "")),
                    props=[str(value) for value in item.get("props", [])],
                    start_state=str(item.get("start_state", "")),
                    end_state=str(item.get("end_state", "")),
                    emotional_change=str(item.get("emotional_change", "")),
                    duration=duration,
                )
            )
        field_sources = deepcopy(fallback.field_sources)
        ai_filled = [
            str(value) for value in data.get("ai_filled_fields", fallback.ai_filled_fields)
        ]
        return DraftBundle(
            outline=outline,
            script=StoryScript(
                title=outline.title,
                target_seconds=sum(durations),
                scenes=scenes,
            ),
            field_sources=field_sources,
            ai_filled_fields=ai_filled,
        )

    def _json_completion(self, prompt_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._load_prompt(prompt_name)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.65 if "idea_" in prompt_name else 0.25,
            max_tokens=4000,
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

    def _mark_fallback(self, operation: str, reason: object) -> None:
        self.last_used_fallback = True
        self.fallback_count += 1
        self.last_fallback_reason = f"{operation}: {reason}"
