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
    StoryCharacter,
    StoryDraft,
    StoryFacts,
    StoryLocation,
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

    def generate_story(
        self,
        direction: str,
        selected_cards: list[IdeaCard],
        selected_elements: dict[str, ElementOption],
    ) -> StoryDraft: ...

    def revise_story(self, story: StoryDraft, feedback: str) -> StoryDraft: ...

    def generate_script(self, story: StoryDraft, target_seconds: int) -> StoryScript: ...

    def revise_script(
        self, story: StoryDraft, script: StoryScript, feedback: str
    ) -> StoryScript: ...

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
    _mystery_concepts = (
        (
            "死后送达的玫瑰",
            "替同事值夜班的花店店员",
            "死者在死亡十分钟后亲自下单了一束玫瑰",
            "订单将在午夜自动删除，她必须赶在警察带走手机前找到真正收花人",
            "都市悬疑",
            "收花人正是店员自己，而订单是死者留给她的求救证据",
        ),
        (
            "第九份不在场证明",
            "专拍情侣合照的街头摄影师",
            "八名嫌疑人的照片里都出现了同一只不属于现场的红气球",
            "照片时间只相差一分钟，摄影师必须证明其中一张照片来自预先录制的街景",
            "本格推理",
            "所谓不在场证明其实是凶手为真正目击者准备的保护伞",
        ),
        (
            "零点告白",
            "准备在毕业前关闭节目的校园电台主持人",
            "死者预先录好的告白在案发后准时播出，并直接叫出主持人的名字",
            "下一段录音将在六分钟后公开凶手，但有人正在切断整栋楼的电源",
            "校园惊悚",
            "录音没有说出名字，而是用一句只有主持人听得懂的话指出共犯",
        ),
        (
            "被剪掉的那句话",
            "替警方修复语音的声音剪辑师",
            "求救录音中每次出现“我爱你”都会少掉同一个人的呼吸声",
            "她必须在嫌疑人销毁母带前，用环境回声还原案发房间和说话顺序",
            "技术悬疑",
            "被剪掉的并非凶手声音，而是死者主动隐瞒的保护对象",
        ),
        (
            "最后一颗巧克力",
            "第一次独立出现场的法医实习生",
            "死者手中的巧克力写着凶手名字，但糖纸上的字迹会被体温融化",
            "所有人都以为这是死亡留言，她却发现巧克力是在死者遇害后才被握进手里",
            "冷峻反转",
            "名字指向的不是凶手，而是唯一能证明死者真正死亡时间的人",
        ),
        (
            "两人份的晚餐",
            "记得每一位顾客门牌号的外卖骑手",
            "独居死者家门口却摆着两双湿鞋和一份从未下单的双人套餐",
            "骑手必须在凶手取回保温袋前，沿配送轨迹找出被调换的案发地点",
            "现实犯罪",
            "真正的谋杀现场不是公寓，而是骑手刚刚经过的停运餐厅",
        ),
        (
            "心形盲区",
            "商场监控室里即将退休的保安",
            "所有摄像头同时转向情侣时，画面中央形成了六秒钟的心形盲区",
            "保安必须只靠玻璃倒影和人群移动，还原凶手如何穿过封锁线",
            "密室推理",
            "制造盲区的人想掩盖的不是杀人过程，而是受害者仍活着离开的瞬间",
        ),
        (
            "丘比特的最后一箭",
            "在商场扮演丘比特的兼职演员",
            "他的道具箭上出现血迹，而监控显示案发时只有“丘比特”接近过死者",
            "同一套服装共有三件，他必须在巡游结束前找出谁交换了翅膀和箭袋",
            "黑色幽默悬疑",
            "真正的凶手没有穿丘比特服，交换服装的人是在替目击者争取逃跑时间",
        ),
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
        if mode == "diverge" and self._is_mystery(cleaned):
            return self._mystery_batch(cleaned, round_number, feedback)
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
                hook = f"“{cleaned}”最平静的一刻，{conflict}"
            if lens:
                axis = f"{axis}·{lens}"
                hook = f"{hook}，并用{lens}重新组织因果"
            suffix = f"，并满足你的补充：{feedback}" if feedback else ""
            card = IdeaCard(
                idea_id=f"idea-r{round_number}-{index}",
                title=f"{axis}·{self._short_seed(cleaned)}",
                logline=(
                    f"{protagonist}卷入“{cleaned}”：{conflict}{suffix}；故事最终通向{ending}。"
                ),
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

    def _mystery_batch(self, direction: str, round_number: int, feedback: str) -> IdeaBatch:
        lens = "" if round_number == 1 else self._lenses[(round_number - 2) % len(self._lenses)]
        preference = f"；同时遵守你的偏好：{feedback}" if feedback else ""
        cards = []
        for index, (title, protagonist, hook, conflict, tone, ending) in enumerate(
            self._mystery_concepts, start=1
        ):
            if lens:
                title = f"{title}·{lens}"
                hook = f"{hook}，线索将用{lens}呈现"
            cards.append(
                IdeaCard(
                    idea_id=f"idea-r{round_number}-{index}",
                    title=title,
                    logline=(
                        f"{protagonist}卷入“{direction}”，发现{hook}；{conflict}{preference}。"
                    ),
                    hook=hook,
                    protagonist=protagonist,
                    central_conflict=conflict,
                    tone=tone,
                    ending_direction=ending,
                    generation_kind="diverge",
                )
            )
        return IdeaBatch(
            round=round_number,
            cards=cards,
            recommended_id=cards[0].idea_id,
            feedback=feedback,
            generation_kind="diverge",
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

    def generate_story(
        self,
        direction: str,
        selected_cards: list[IdeaCard],
        selected_elements: dict[str, ElementOption],
    ) -> StoryDraft:
        card = (
            selected_cards[0]
            if selected_cards
            else self.generate_ideas(direction, round_number=1).cards[0]
        )
        chosen = {kind: item.content for kind, item in selected_elements.items()}
        protagonist = chosen.get("character", card.protagonist)
        conflict = chosen.get("conflict", card.central_conflict)
        turning = chosen.get("turning_point", f"真正的线索藏在“{card.hook}”背后")
        ending = chosen.get("ending", card.ending_direction)
        story_text = (
            f"{direction}。{protagonist}原本只想维持平静，却在最普通的一刻发现"
            f"{card.hook}。这个异常不是偶然，它迫使主角立即作出选择。\n\n"
            f"主角沿着异常留下的痕迹行动，逐渐确认{conflict}。每一次接近答案，"
            "现实中的代价都会变得更具体，身边的人与原先相信的规则也开始动摇。\n\n"
            f"当主角以为已经找到解决办法时，事情发生变化：{turning}。此前看似无关的"
            "人物、地点和物件因此连成一条完整线索，主角也终于明白真正需要面对的并不是"
            "表面的危险，而是自己一直回避的选择。\n\n"
            f"主角承担选择带来的后果，并让故事走向{ending}。开场出现过的关键意象再次出现，"
            "但它的含义已经改变，故事在行动完成后的余波中结束。"
        )
        source_ids = [item.idea_id for item in selected_cards]
        source_type = "selected_card" if selected_cards else "ai_fill"
        fields = {
            "protagonist": protagonist,
            "core_conflict": conflict,
            "turning_point": turning,
            "ending": ending,
        }
        field_sources: dict[str, SourceAttribution] = {}
        ai_filled_fields: list[str] = []
        for field, value in fields.items():
            option_kind = "character" if field == "protagonist" else field
            if option_kind in selected_elements:
                origin = "selected_element"
            else:
                origin = source_type
            field_sources[field] = SourceAttribution(
                field=field,
                source_type=origin,
                value=value,
                source_ids=source_ids,
            )
            if origin == "ai_fill":
                ai_filled_fields.append(field)
        for field in ("story_text", "characters", "locations", "visual_anchors"):
            field_sources[field] = SourceAttribution(
                field=field,
                source_type="ai_fill",
                value="由故事生成阶段补全",
            )
            ai_filled_fields.append(field)
        return StoryDraft(
            title=card.title,
            logline=card.logline,
            story_text=story_text,
            characters=[
                StoryCharacter(
                    name=protagonist,
                    description=f"{protagonist}必须在压力下完成一次不可回避的选择。",
                    visual_identity="保持统一的脸部、发型、服装与标志性随身物件",
                )
            ],
            locations=[
                StoryLocation(
                    name="故事核心场景",
                    description=f"承载“{direction}”主要行动的连续空间",
                    visual_identity="保持统一的空间结构、主色和光线方向",
                )
            ],
            tone=card.tone,
            theme="人在代价面前如何作出真正属于自己的选择",
            core_conflict=conflict,
            ending=ending,
            visual_anchors=["主角固定服装", "贯穿故事的关键物件", "统一场景主色"],
            field_sources=field_sources,
            ai_filled_fields=ai_filled_fields,
        )

    def revise_story(self, story: StoryDraft, feedback: str) -> StoryDraft:
        if not feedback.strip():
            raise ValueError("请用一句话说明想怎样修改故事。")
        revised = deepcopy(story)
        revised.story_text = (
            f"{story.story_text}\n\n根据创作者的新方向，故事进一步调整为：{feedback.strip()}。"
        )
        revised.version = story.version + 1
        revised.confirmed = False
        return revised

    def generate_script(self, story: StoryDraft, target_seconds: int) -> StoryScript:
        scene_count = max(3, round(int(target_seconds) / 9))
        durations = allocate_durations(target_seconds, scene_count, minimum=3, maximum=15)
        events = [
            item.strip()
            for item in re.split(r"(?<=[。！？])", story.story_text)
            if item.strip()
        ]
        if not events:
            events = [story.logline]
        locations = story.locations or [
            StoryLocation("故事核心场景", "连续的故事空间", "统一空间与光线")
        ]
        character_names = [item.name for item in story.characters] or ["主角"]
        scenes: list[StoryScene] = []
        for index, duration in enumerate(durations, 1):
            event = events[min(len(events) - 1, (index - 1) * len(events) // scene_count)]
            location = locations[min(len(locations) - 1, (index - 1) * len(locations) // scene_count)]
            scenes.append(
                StoryScene(
                    scene_id=index,
                    title=f"故事片段 {index}",
                    location=location.name,
                    time_of_day="连续时间",
                    characters=list(character_names),
                    action=event,
                    visible_action=event,
                    narration=event,
                    duration=duration,
                    dialogue="" if index < scene_count else "我已经作出了选择。",
                    props=list(story.visual_anchors[:2]),
                    start_state="承接上一场的动作与人物状态",
                    end_state=event,
                    emotional_change=story.tone,
                )
            )
        return StoryScript(title=story.title, target_seconds=target_seconds, scenes=scenes)

    def revise_script(
        self, story: StoryDraft, script: StoryScript, feedback: str
    ) -> StoryScript:
        if not feedback.strip():
            raise ValueError("请用一句话说明想怎样修改剧本。")
        revised = deepcopy(script)
        for scene in revised.scenes:
            scene.visible_action = (
                f"按照“{feedback.strip()}”调整表演和镜头动作："
                f"{scene.visible_action or scene.action}"
            )
            scene.action = scene.visible_action
        revised.confirmed = False
        return revised

    def generate_draft(
        self,
        direction: str,
        selected_cards: list[IdeaCard],
        selected_elements: dict[str, ElementOption],
        target_seconds: int,
    ) -> DraftBundle:
        story = self.generate_story(direction, selected_cards, selected_elements)
        script = self.generate_script(story, target_seconds)
        beats = [
            StoryBeat(
                beat_id=scene.scene_id,
                purpose=scene.title,
                event=scene.visible_action or scene.action,
                causal_link=scene.start_state,
                emotional_change=scene.emotional_change,
                duration=scene.duration,
            )
            for scene in script.scenes
        ]
        opening = script.scenes[0].visible_action or script.scenes[0].action
        development = "；".join(
            scene.visible_action or scene.action for scene in script.scenes[1:-1]
        )
        outline = StoryOutline(
            title=story.title,
            logline=story.logline,
            opening=opening,
            protagonist_goal=story.characters[0].description if story.characters else story.logline,
            conflict=story.core_conflict,
            development=development,
            turning_point=script.scenes[-2].visible_action if len(script.scenes) > 1 else development,
            ending=story.ending,
            source_turn_ids=[],
            beats=beats,
        )
        return DraftBundle(
            outline=outline,
            script=script,
            field_sources=deepcopy(story.field_sources),
            ai_filled_fields=list(story.ai_filled_fields),
        )

    def revise_draft(self, draft: DraftBundle, feedback: str) -> DraftBundle:
        if not feedback.strip():
            raise ValueError("请用一句话说明想怎样修改。")
        story = StoryDraft(
            title=draft.outline.title,
            logline=draft.outline.logline,
            story_text="；".join(scene.visible_action or scene.action for scene in draft.script.scenes),
            core_conflict=draft.outline.conflict,
            ending=draft.outline.ending,
        )
        revised = deepcopy(draft)
        revised.script = self.revise_script(story, draft.script, feedback)
        revised.version += 1
        revised.outline.confirmed = False
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

    @staticmethod
    def _is_mystery(direction: str) -> bool:
        return any(
            keyword in direction
            for keyword in ("杀人", "谋杀", "命案", "凶案", "案件", "侦探", "推理")
        )


class OpenAIStoryAgent(RuleBasedStoryAgent):
    """OpenAI-compatible creative agent with explicit fallback telemetry."""

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
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        legacy_api_key = os.getenv("AGNES_API_KEY", "").strip()
        use_deepseek = bool(deepseek_api_key)
        api_key = deepseek_api_key or legacy_api_key
        if use_deepseek:
            model = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-v4-pro").strip()
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            timeout = os.getenv("DEEPSEEK_TIMEOUT", "120")
        else:
            model = os.getenv("AGNES_TEXT_MODEL", "agnes-2.0-flash").strip()
            base_url = os.getenv(
                "AGNES_LLM_BASE_URL", "https://apihub.agnes-ai.com/v1"
            )
            timeout = os.getenv("AGNES_TIMEOUT", "120")
        if not api_key:
            return cls(None, os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-v4-pro").strip())
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(timeout),
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

    def generate_story(
        self,
        direction: str,
        selected_cards: list[IdeaCard],
        selected_elements: dict[str, ElementOption],
    ) -> StoryDraft:
        self.last_used_fallback = False
        fallback = super().generate_story(direction, selected_cards, selected_elements)
        if self.client is None:
            self._mark_fallback("generate_story", "text API is not configured")
            return fallback
        try:
            data = self._json_completion(
                "story_writer.md",
                {
                    "direction": direction,
                    "selected_cards": to_plain_data(selected_cards),
                    "selected_elements": to_plain_data(selected_elements),
                },
            )
            story = self._story_from(data, fallback)
        except Exception as exc:
            self._mark_fallback("generate_story", exc)
            return fallback
        return self._review_story_continuity(story, operation="generate_story_review")

    def revise_story(self, story: StoryDraft, feedback: str) -> StoryDraft:
        self.last_used_fallback = False
        fallback = super().revise_story(story, feedback)
        if self.client is None:
            self._mark_fallback("revise_story", "text API is not configured")
            return fallback
        try:
            data = self._json_completion(
                "story_rewriter.md",
                {"story": to_plain_data(story), "feedback": feedback},
            )
            revised = self._story_from(data, fallback)
            revised.version = story.version + 1
        except Exception as exc:
            self._mark_fallback("revise_story", exc)
            return fallback
        reviewed = self._review_story_continuity(
            revised,
            operation="revise_story_review",
        )
        reviewed.version = story.version + 1
        return reviewed

    def generate_script(self, story: StoryDraft, target_seconds: int) -> StoryScript:
        self.last_used_fallback = False
        fallback = super().generate_script(story, target_seconds)
        if self.client is None:
            self._mark_fallback("generate_script", "text API is not configured")
            return fallback
        try:
            data = self._json_completion(
                "script_writer.md",
                {"confirmed_story": to_plain_data(story), "target_seconds": target_seconds},
            )
            script = self._script_from_model(data, fallback, target_seconds)
        except Exception as exc:
            self._mark_fallback("generate_script", exc)
            return fallback
        return self._review_script_continuity(
            story,
            script,
            operation="generate_script_review",
        )

    def revise_script(
        self, story: StoryDraft, script: StoryScript, feedback: str
    ) -> StoryScript:
        self.last_used_fallback = False
        fallback = super().revise_script(story, script, feedback)
        if self.client is None:
            self._mark_fallback("revise_script", "text API is not configured")
            return fallback
        try:
            data = self._json_completion(
                "script_rewriter.md",
                {
                    "confirmed_story": to_plain_data(story),
                    "script": to_plain_data(script),
                    "feedback": feedback,
                },
            )
            revised = self._script_from_model(data, fallback, script.target_seconds)
        except Exception as exc:
            self._mark_fallback("revise_script", exc)
            return fallback
        return self._review_script_continuity(
            story,
            revised,
            operation="revise_script_review",
        )

    def generate_draft(
        self,
        direction: str,
        selected_cards: list[IdeaCard],
        selected_elements: dict[str, ElementOption],
        target_seconds: int,
    ) -> DraftBundle:
        story = self.generate_story(direction, selected_cards, selected_elements)
        script = self.generate_script(story, target_seconds)
        fallback = RuleBasedStoryAgent().generate_draft(
            direction, selected_cards, selected_elements, target_seconds
        )
        fallback.script = script
        fallback.outline.title = story.title
        fallback.outline.logline = story.logline
        fallback.outline.conflict = story.core_conflict
        fallback.outline.ending = story.ending
        fallback.field_sources = deepcopy(story.field_sources)
        fallback.ai_filled_fields = list(story.ai_filled_fields)
        return fallback

    def revise_draft(self, draft: DraftBundle, feedback: str) -> DraftBundle:
        return super().revise_draft(draft, feedback)

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
            if (
                not all(
                    (
                        card.title,
                        card.logline,
                        card.hook,
                        card.protagonist,
                        card.central_conflict,
                        card.ending_direction,
                    )
                )
                or self._is_weak_card(card)
                or card.fingerprint in seen
            ):
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
    def _is_weak_card(card: IdeaCard) -> bool:
        meta_patterns = (
            r"把[“\"].+?[”\"]变成",
            r"围绕.+?展开",
            r"某个故事",
            r"主角遇到困难",
        )
        combined = f"{card.title} {card.logline} {card.hook}"
        return (
            any(re.search(pattern, combined) for pattern in meta_patterns)
            or len(card.protagonist) < 4
            or len(card.hook) < 10
            or len(card.central_conflict) < 12
            or len(card.ending_direction) < 8
        )

    @staticmethod
    def _story_from(data: dict[str, Any], fallback: StoryDraft) -> StoryDraft:
        raw = data.get("story", data)
        story_text = str(raw.get("story_text", "")).strip()
        if len(story_text) < 120:
            raise ValueError("story_text is too short to support script adaptation")
        characters = [
            StoryCharacter(
                name=str(item.get("name", "")).strip(),
                description=str(item.get("description", "")).strip(),
                visual_identity=str(item.get("visual_identity", "")).strip(),
            )
            for item in raw.get("characters", [])
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]
        locations = [
            StoryLocation(
                name=str(item.get("name", "")).strip(),
                description=str(item.get("description", "")).strip(),
                visual_identity=str(item.get("visual_identity", "")).strip(),
            )
            for item in raw.get("locations", [])
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]
        return StoryDraft(
            title=str(raw.get("title", fallback.title)).strip() or fallback.title,
            logline=str(raw.get("logline", fallback.logline)).strip() or fallback.logline,
            story_text=story_text,
            characters=characters or deepcopy(fallback.characters),
            locations=locations or deepcopy(fallback.locations),
            tone=str(raw.get("tone", fallback.tone)).strip(),
            theme=str(raw.get("theme", fallback.theme)).strip(),
            core_conflict=str(raw.get("core_conflict", fallback.core_conflict)).strip(),
            ending=str(raw.get("ending", fallback.ending)).strip(),
            visual_anchors=[
                str(value).strip()
                for value in raw.get("visual_anchors", fallback.visual_anchors)
                if str(value).strip()
            ],
            field_sources=deepcopy(fallback.field_sources),
            ai_filled_fields=[
                str(value)
                for value in data.get("ai_filled_fields", fallback.ai_filled_fields)
            ],
        )

    @staticmethod
    def _script_from_model(
        data: dict[str, Any], fallback: StoryScript, target_seconds: int
    ) -> StoryScript:
        raw = data.get("script", data)
        raw_scenes = raw.get("scenes", [])
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise ValueError("script must contain at least one scene")
        durations = allocate_durations(
            target_seconds,
            len(raw_scenes),
            minimum=1,
            maximum=target_seconds,
        )
        scenes: list[StoryScene] = []
        for index, (item, duration) in enumerate(zip(raw_scenes, durations), 1):
            if not isinstance(item, dict):
                raise ValueError("each script scene must be an object")
            visible_action = str(
                item.get("visible_action", item.get("action", ""))
            ).strip()
            if not visible_action:
                raise ValueError("each script scene needs a visible action")
            scenes.append(
                StoryScene(
                    scene_id=index,
                    title=str(item.get("title", f"场景 {index}")).strip(),
                    location=str(item.get("location", "故事核心场景")).strip(),
                    time_of_day=str(item.get("time_of_day", "连续时间")).strip(),
                    characters=[str(value) for value in item.get("characters", [])],
                    action=visible_action,
                    visible_action=visible_action,
                    dialogue=str(item.get("dialogue", "")).strip(),
                    narration=str(item.get("narration", "")).strip(),
                    props=[str(value) for value in item.get("props", [])],
                    start_state=str(item.get("start_state", "")).strip(),
                    end_state=str(item.get("end_state", "")).strip(),
                    emotional_change=str(item.get("emotional_change", "")).strip(),
                    duration=duration,
                )
            )
        return StoryScript(
            title=str(raw.get("title", fallback.title)).strip() or fallback.title,
            target_seconds=target_seconds,
            scenes=scenes,
        )

    def _review_story_continuity(
        self,
        story: StoryDraft,
        *,
        operation: str,
    ) -> StoryDraft:
        try:
            data = self._json_completion(
                "story_continuity_reviewer.md",
                {"story": to_plain_data(story)},
            )
            reviewed = self._story_from(data, story)
            reviewed.field_sources = deepcopy(story.field_sources)
            reviewed.ai_filled_fields = list(story.ai_filled_fields)
            return reviewed
        except Exception as exc:
            self._mark_fallback(operation, exc)
            return story

    def _review_script_continuity(
        self,
        story: StoryDraft,
        script: StoryScript,
        *,
        operation: str,
    ) -> StoryScript:
        try:
            data = self._json_completion(
                "script_continuity_reviewer.md",
                {
                    "confirmed_story": to_plain_data(story),
                    "script": to_plain_data(script),
                    "target_seconds": script.target_seconds,
                },
            )
            return self._script_from_model(data, script, script.target_seconds)
        except Exception as exc:
            self._mark_fallback(operation, exc)
            return script

    def _json_completion(self, prompt_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if prompt_name.startswith("story_"):
            max_tokens = 8000
        elif prompt_name.startswith("script_"):
            max_tokens = 6000
        else:
            max_tokens = 4000
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._load_prompt(prompt_name)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": (
                0.7
                if prompt_name in {"story_writer.md", "story_rewriter.md"}
                else 0.2
                if "continuity_reviewer" in prompt_name
                else 0.65
                if "idea_" in prompt_name
                else 0.35
            ),
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            message = str(exc).lower()
            if "response_format" not in message and "json_object" not in message:
                raise
            request.pop("response_format")
            response = self.client.chat.completions.create(**request)
        content = (response.choices[0].message.content or "").strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise ValueError("model did not return a JSON object") from exc
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
