from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

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
from .provider_config import TextProviderConfig
from .storyboard import assess_director_plan_timing
from .timing import (
    ShotTimingProfile,
    count_sequential_action_phases,
    plan_scene_durations,
)


IDEA_COUNT = 8
ELEMENT_KINDS = ("character", "conflict", "turning_point", "ending")
MAX_PREVIOUS_CARDS = 64
MAX_LLM_PAYLOAD_CHARS = 250_000
MAX_LLM_RESPONSE_CHARS = 1_000_000
MAX_REPAIR_INPUT_CHARS = 30_000


def _natural_join(values: list[str], separator: str) -> str:
    return separator.join(dict.fromkeys(value.strip() for value in values if value.strip()))


def _fit_offline_spoken_material(
    dialogue: str,
    narration: str,
    duration: int,
) -> tuple[str, str]:
    """Keep deterministic fallback speech inside its scene's audible window."""

    capacity = max(0, int(max(0, duration) * 4.5))

    def clipped(value: str, limit: int) -> str:
        cleaned = " ".join(str(value or "").split())
        if not cleaned or limit <= 0:
            return ""
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[: max(1, limit - 1)].rstrip('，,；;：:')}…"

    fitted_dialogue = clipped(dialogue, capacity)
    remaining = max(0, capacity - len(fitted_dialogue))
    fitted_narration = clipped(narration, remaining)
    return fitted_dialogue, fitted_narration


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

    def generate_script(
        self,
        story: StoryDraft,
        target_seconds: int,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript: ...

    def revise_script(
        self,
        story: StoryDraft,
        script: StoryScript,
        feedback: str,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript: ...

    def plan_storyboard(
        self,
        script: StoryScript,
        facts: StoryFacts,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> list[dict[str, Any]] | None: ...

    def evaluate_artifacts(
        self,
        story: StoryDraft,
        script: StoryScript,
        storyboard: dict[str, Any],
    ) -> dict[str, Any]: ...

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
        cards = (
            list(selected_cards)
            if selected_cards
            else [self.generate_ideas(direction, round_number=1).cards[0]]
        )
        card = cards[0]
        chosen = {kind: item.content for kind, item in selected_elements.items()}
        protagonist = chosen.get("character", card.protagonist)
        card_protagonists = list(
            dict.fromkeys(
                [protagonist, *(item.protagonist for item in cards[1:])]
            )
        )
        conflict = chosen.get(
            "conflict",
            _natural_join([item.central_conflict for item in cards], "，同时必须面对"),
        )
        turning = chosen.get(
            "turning_point",
            "此前分散的异常终于连成一条线索："
            + _natural_join([item.hook for item in cards], "；"),
        )
        ending = chosen.get(
            "ending",
            _natural_join([item.ending_direction for item in cards], "，并由此"),
        )
        supporting_arrival = ""
        if len(card_protagonists) > 1:
            supporting_arrival = (
                f"{_natural_join(card_protagonists[1:], '、')}因各自掌握的异常线索进入事件，"
                f"与{protagonist}形成目标一致但方法冲突的临时同盟。"
            )
        story_text = (
            f"{direction}。{protagonist}原本只想维持平静，却在最普通的一刻发现"
            f"{card.hook}。这个异常不是偶然，它迫使{protagonist}立即作出选择。"
            f"{supporting_arrival}\n\n"
            f"{_natural_join(card_protagonists, '与')}沿着异常留下的痕迹行动，逐渐确认"
            f"{conflict}。每一次接近答案，"
            "现实中的代价都会变得更具体，身边的人与原先相信的规则也开始动摇。\n\n"
            f"当{protagonist}以为已经找到解决办法时，事情发生变化：{turning}。"
            "此前看似无关的人物、地点和物件因此连成一条完整线索，众人也终于明白真正"
            "需要面对的并不是"
            "表面的危险，而是自己一直回避的选择。\n\n"
            f"{protagonist}承担选择带来的后果，并让故事走向{ending}。"
            "开场出现过的关键意象再次出现，"
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
        story_corpus = f"{direction} {card.title} {card.logline} {card.hook}"
        if "玫瑰" in story_corpus and ("花店" in story_corpus or "订单" in story_corpus):
            locations = [
                StoryLocation(
                    name="夜班花店",
                    description="收银台、花束操作台与临街橱窗构成主要行动空间",
                    visual_identity="冷白顶灯、红玫瑰、午夜电子钟",
                ),
                StoryLocation(
                    name="花店配送台",
                    description="堆放订单标签和包装材料的狭窄后场",
                    visual_identity="手机屏幕、撕裂标签、低照度侧光",
                ),
            ]
            visual_anchors = ["死者订单手机", "红玫瑰花束", "午夜倒计时"]
        else:
            locations = [
                StoryLocation(
                    name=f"{self._short_seed(direction)}主要行动地",
                    description=f"承载“{direction}”主要行动的连续空间",
                    visual_identity="保持统一的空间结构、主色和光线方向",
                )
            ]
            visual_anchors = ["主角固定服装", "贯穿故事的关键物件", "统一场景主色"]
        return StoryDraft(
            title=" × ".join(item.title for item in cards),
            logline=_natural_join([item.logline for item in cards], "；"),
            story_text=story_text,
            characters=[
                StoryCharacter(
                    name=name,
                    description=f"{name}必须在压力下完成一次不可回避的选择。",
                    visual_identity="保持统一的脸部、发型、服装与标志性随身物件",
                )
                for name in card_protagonists
            ],
            locations=locations,
            tone=card.tone,
            theme="人在代价面前如何作出真正属于自己的选择",
            core_conflict=conflict,
            ending=ending,
            visual_anchors=visual_anchors,
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

    def generate_script(
        self,
        story: StoryDraft,
        target_seconds: int,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript:
        profile = timing_profile or ShotTimingProfile()
        # Keep very short offline scripts from degenerating into three equal
        # buckets; the scene count is a pacing choice, not a fixed template.
        scene_count = max(2, round(int(target_seconds) / 9))
        character_names = [item.name for item in story.characters] or ["主角"]
        protagonist = character_names[0]
        locations, props, beats = self._offline_script_material(story, protagonist)
        if scene_count == 1:
            beat_indexes = [len(beats) - 1]
        elif scene_count == 2:
            beat_indexes = [0, len(beats) - 1]
        else:
            interior_count = scene_count - 2
            interior = [
                1
                + round(
                    offset * max(0, len(beats) - 3) / max(1, interior_count - 1)
                )
                for offset in range(interior_count)
            ]
            interior[-1] = max(1, len(beats) - 2)
            beat_indexes = [0, *interior, len(beats) - 1]
        scenes: list[StoryScene] = []
        for index, beat_index in enumerate(beat_indexes, 1):
            title, location_index, action, dialogue, narration, end_state = beats[beat_index]
            location = locations[min(location_index, len(locations) - 1)]
            previous_state = scenes[-1].end_state if scenes else f"{protagonist}尚未发现异常"
            scenes.append(
                StoryScene(
                    scene_id=index,
                    title=title,
                    location=location.name,
                    time_of_day="连续时间",
                    characters=list(character_names),
                    action=action,
                    visible_action=action,
                    narration=narration,
                    duration=3,
                    dialogue=dialogue,
                    props=list(props),
                    start_state=f"承接上一场：{previous_state}",
                    end_state=end_state,
                    emotional_change=f"{story.tone}：局势向下一步行动推进",
                )
            )
        durations, weights, reasons = plan_scene_durations(
            scenes,
            target_seconds,
            minimum=profile.min_duration_seconds,
            maximum=target_seconds,
        )
        for scene, duration, weight, reason in zip(scenes, durations, weights, reasons):
            scene.duration = duration
            scene.duration_weight = weight
            scene.duration_reason = reason
            scene.dialogue, scene.narration = _fit_offline_spoken_material(
                scene.dialogue,
                scene.narration,
                duration,
            )
        return StoryScript(title=story.title, target_seconds=target_seconds, scenes=scenes)

    @staticmethod
    def _offline_script_material(
        story: StoryDraft,
        protagonist: str,
    ) -> tuple[list[StoryLocation], list[str], list[tuple[str, int, str, str, str, str]]]:
        """Create a filmable offline script without pretending abstract prose is a shot."""
        corpus = " ".join(
            (story.title, story.logline, story.story_text, story.core_conflict, story.ending)
        )
        if "玫瑰" in corpus and ("订单" in corpus or "花店" in corpus):
            locations = [
                StoryLocation("夜班花店", "临街花店的收银台与操作台", "冷白顶灯、红色玫瑰"),
                StoryLocation("花店配送台", "堆放订单与包装纸的后场", "手机屏幕与标签特写"),
                StoryLocation("花店门外警戒线", "警车灯映入橱窗的街道", "蓝红闪光与湿地面"),
            ]
            props = ["死者订单手机", "红玫瑰花束"]
            conflict = story.core_conflict.strip() or "订单将在午夜删除，手机即将被警方带走"
            ending = story.ending.strip() or "收花人就是店员，订单保存着求救证据"
            beats = [
                (
                    "夜班来单",
                    0,
                    f"{protagonist}独自在收银台清点玫瑰，墙上时钟逼近午夜，订单手机突然亮起。",
                    "",
                    "情人节夜班原本平静。",
                    "手机收到一笔异常玫瑰订单",
                ),
                (
                    "死亡后的订单",
                    0,
                    f"{protagonist}放大订单时间，又对照手机里的死者遇害通报；两个时间相差十分钟。",
                    "死人怎么会下单？",
                    "",
                    "她确认订单生成于死者死亡之后",
                ),
                (
                    "留下证据",
                    1,
                    f"{protagonist}立即截屏并抄下订单号，删除倒计时在手机顶端跳动。",
                    "",
                    conflict,
                    "她决定在手机被带走前找到收花人",
                ),
                (
                    "追查收花人",
                    1,
                    f"{protagonist}翻查配送标签和夜班记录，把订单号、地址与死者姓名逐项连线。",
                    "",
                    "",
                    "配送记录指向一张被撕掉姓名的标签",
                ),
                (
                    "警方到门外",
                    2,
                    f"警车灯扫过橱窗；{protagonist}看见警员走近，迅速把订单截图传到自己的手机。",
                    "再给我一分钟。",
                    "",
                    "警方即将收走订单手机",
                ),
                (
                    "标签背面的线索",
                    1,
                    f"{protagonist}把撕裂标签贴回花束包装，背面的笔画拼成她自己的姓氏。",
                    "",
                    "",
                    "她发现收花人的姓名可能与自己有关",
                ),
                (
                    "收件人揭晓",
                    0,
                    f"{protagonist}重新打开订单详情，在收件人栏看到自己的全名，手里的玫瑰停在半空。",
                    "这束花……是给我的。",
                    "",
                    "收花人确认是她本人",
                ),
                (
                    "求救信息",
                    0,
                    f"{protagonist}拆开花束，在订单备注对应的花枝中找到存储卡，并插入手机。",
                    "",
                    ending,
                    "订单与花束共同指向死者留下的证据",
                ),
                (
                    "赶在删除前",
                    2,
                    f"倒计时归零前，{protagonist}把截图和存储卡交给警员；原订单随即从屏幕消失。",
                    "证据不在订单里，在花里。",
                    "",
                    "警方接收了未被删除的求救证据",
                ),
                (
                    "玫瑰送达",
                    0,
                    f"天将亮时，{protagonist}把那束红玫瑰放在空收银台上，手机里的证据上传完成。",
                    "",
                    "死后送达的花，终于完成了它真正的投递。",
                    ending,
                ),
            ]
            return locations, props, beats

        usable_locations = [
            item for item in story.locations if item.name.strip() != "故事核心场景"
        ]
        locations = usable_locations or [
            StoryLocation(
                f"{protagonist}所在的主要场所",
                "由故事事件决定的具体行动空间",
                "保持空间结构和光线方向一致",
            )
        ]
        props = [
            item
            for item in story.visual_anchors
            if item.strip() and item not in {"主角固定服装", "贯穿故事的关键物件", "统一场景主色"}
        ][:2] or ["关键证据"]
        conflict = story.core_conflict.strip() or story.logline.strip()
        ending = story.ending.strip() or "主角完成关键行动并承担结果"
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[。！？])", story.story_text)
            if item.strip()
        ]
        concrete = sentences or [story.logline]
        selected = [
            concrete[round(i * (len(concrete) - 1) / 5)]
            for i in range(6)
        ]
        beats = [
            ("异常出现", 0, f"{protagonist}在日常动作中发现第一个异常细节。", "", selected[0], "异常被主角注意"),
            ("核对线索", 0, f"{protagonist}拿起关键证据，逐项核对时间、人物和地点。", "", selected[1], "异常被证实并非偶然"),
            ("作出决定", 0, f"{protagonist}保存证据并离开原位，开始主动追查。", "", conflict, "主角确立明确目标"),
            ("推进调查", 0, f"{protagonist}沿着证据留下的痕迹行动，并排除一个错误答案。", "", selected[2], "线索指向新的对象"),
            ("阻力逼近", 0, f"外部阻力进入现场，{protagonist}抢在证据失效前完成下一步。", "", selected[3], "行动代价变得具体"),
            ("线索反转", 0, f"{protagonist}重新排列已有证据，发现此前忽略的对应关系。", "", selected[4], "旧线索获得相反含义"),
            ("正面选择", 0, f"{protagonist}面对阻止者，当场执行无法撤回的决定。", "", conflict, "主角承担选择后果"),
            (
                "结果落地",
                0,
                f"{protagonist}完成最后一个可见动作，关键证据留在画面中央。",
                "",
                f"{conflict}；{selected[5]}",
                ending,
            ),
        ]
        return locations, props, beats

    def revise_script(
        self,
        story: StoryDraft,
        script: StoryScript,
        feedback: str,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript:
        del timing_profile
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

    def plan_storyboard(
        self,
        script: StoryScript,
        facts: StoryFacts,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> list[dict[str, Any]] | None:
        del script, facts, timing_profile
        return None

    def evaluate_artifacts(
        self,
        story: StoryDraft,
        script: StoryScript,
        storyboard: dict[str, Any],
    ) -> dict[str, Any]:
        del story, script, storyboard
        return {}

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
            turning_point=script.scenes[-2].visible_action
            if len(script.scenes) > 1
            else development,
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
            story_text="；".join(
                scene.visible_action or scene.action for scene in draft.script.scenes
            ),
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

    def __init__(
        self,
        client: Any | None,
        model: str,
        prompt_dir: Path | None = None,
        *,
        provider_name: str = "openai_compatible",
        config_source: str = "constructor",
        json_mode: str = "auto",
        configuration_error: str = "",
        allow_artifact_fallback: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.provider_name = provider_name
        self.config_source = config_source
        self.json_mode = json_mode
        self.configuration_error = configuration_error
        self.allow_artifact_fallback = allow_artifact_fallback
        self.last_used_fallback = False
        self.last_fallback_kind = ""
        self.fallback_count = 0
        self.last_fallback_reason = ""

    @classmethod
    def from_env(cls, *, allow_artifact_fallback: bool = False) -> OpenAIStoryAgent:
        config = TextProviderConfig.from_env()
        model = config.model or "unconfigured"
        if not config.configured:
            return cls(
                None,
                model,
                provider_name=config.provider,
                config_source=config.source,
                json_mode=config.json_mode,
                configuration_error=config.error,
                allow_artifact_fallback=allow_artifact_fallback,
            )
        try:
            from openai import OpenAI

            client_args: dict[str, Any] = {
                "api_key": config.api_key,
                "timeout": config.timeout,
                "max_retries": 0,
            }
            if config.base_url:
                client_args["base_url"] = config.base_url
            client = OpenAI(**client_args)
            return cls(
                client,
                model,
                provider_name=config.provider,
                config_source=config.source,
                json_mode=config.json_mode,
                allow_artifact_fallback=allow_artifact_fallback,
            )
        except Exception as exc:
            agent = cls(
                None,
                model,
                provider_name=config.provider,
                config_source=config.source,
                json_mode=config.json_mode,
                configuration_error=f"文本客户端初始化失败：{exc}",
                allow_artifact_fallback=allow_artifact_fallback,
            )
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
        self._reset_fallback_status()
        fallback = super().generate_ideas(
            direction,
            round_number=round_number,
            feedback=feedback,
            previous_cards=previous_cards,
            mode=mode,
            anchors=anchors,
        )
        if self.client is None:
            self._mark_fallback("generate_ideas", self._unavailable_reason)
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
                    "previous_cards": to_plain_data((previous_cards or [])[-MAX_PREVIOUS_CARDS:]),
                    "required_count": IDEA_COUNT,
                },
            )
            cards, local_fill_count = self._cards_from(
                data.get("cards", []),
                fallback,
                round_number,
                mode,
                previous_cards or [],
            )
            if local_fill_count:
                fallback_kind = "whole" if local_fill_count == IDEA_COUNT else "partial"
                self._mark_fallback(
                    "generate_ideas",
                    f"模型创意不足，使用本地创意补齐 {local_fill_count}/{IDEA_COUNT} 张",
                    fallback_kind=fallback_kind,
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
        self._reset_fallback_status()
        fallback = super().expand_elements(direction, selected_cards)
        if self.client is None:
            self._mark_fallback("expand_elements", self._unavailable_reason)
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
        self._reset_fallback_status()
        fallback = super().generate_story(direction, selected_cards, selected_elements)
        if self.client is None:
            self._mark_fallback("generate_story", self._unavailable_reason)
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
            self._handle_artifact_failure("generate_story", exc)
            return fallback
        return self._review_story_continuity(
            story,
            operation="generate_story_review",
            selected_cards=selected_cards,
            selected_elements=selected_elements,
        )

    def revise_story(self, story: StoryDraft, feedback: str) -> StoryDraft:
        self._reset_fallback_status()
        fallback = super().revise_story(story, feedback)
        if self.client is None:
            self._mark_fallback("revise_story", self._unavailable_reason)
            return fallback
        try:
            data = self._json_completion(
                "story_rewriter.md",
                {"story": to_plain_data(story), "feedback": feedback},
            )
            revised = self._story_from(data, fallback)
            revised.version = story.version + 1
        except Exception as exc:
            self._handle_artifact_failure("revise_story", exc)
            return fallback
        reviewed = self._review_story_continuity(
            revised,
            operation="revise_story_review",
        )
        reviewed.version = story.version + 1
        return reviewed

    def generate_script(
        self,
        story: StoryDraft,
        target_seconds: int,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript:
        profile = timing_profile or ShotTimingProfile()
        self._reset_fallback_status()
        fallback = super().generate_script(story, target_seconds, timing_profile=profile)
        if self.client is None:
            self._mark_fallback("generate_script", self._unavailable_reason)
            return fallback
        try:
            data = self._json_completion(
                "script_writer.md",
                {
                    "confirmed_story": to_plain_data(story),
                    "required_constraints": self._script_constraints(story),
                    "target_seconds": target_seconds,
                    "maximum_scenes": profile.maximum_shot_count(target_seconds),
                },
            )
            script = self._script_from_model(
                data, fallback, target_seconds, timing_profile=profile
            )
        except Exception as exc:
            self._handle_artifact_failure("generate_script", exc)
            return fallback
        return self._review_script_continuity(
            story,
            script,
            operation="generate_script_review",
            timing_profile=profile,
        )

    def revise_script(
        self,
        story: StoryDraft,
        script: StoryScript,
        feedback: str,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript:
        profile = timing_profile or ShotTimingProfile()
        self._reset_fallback_status()
        fallback = super().revise_script(
            story, script, feedback, timing_profile=profile
        )
        if self.client is None:
            self._mark_fallback("revise_script", self._unavailable_reason)
            return fallback
        try:
            data = self._json_completion(
                "script_rewriter.md",
                {
                    "confirmed_story": to_plain_data(story),
                    "required_constraints": self._script_constraints(story),
                    "script": to_plain_data(script),
                    "feedback": feedback,
                },
            )
            revised = self._script_from_model(
                data, fallback, script.target_seconds, timing_profile=profile
            )
        except Exception as exc:
            self._handle_artifact_failure("revise_script", exc)
            return fallback
        return self._review_script_continuity(
            story,
            revised,
            operation="revise_script_review",
            timing_profile=profile,
        )

    def plan_storyboard(
        self,
        script: StoryScript,
        facts: StoryFacts,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> list[dict[str, Any]] | None:
        profile = timing_profile or ShotTimingProfile()
        self._reset_fallback_status()
        if self.client is None:
            self._mark_fallback("plan_storyboard", self._unavailable_reason)
            return None
        try:
            timing_feedback = ""
            for attempt in range(3):
                data = self._json_completion(
                    "storyboard_director.md",
                    {
                        "confirmed_script": to_plain_data(script),
                        "story_facts": to_plain_data(facts),
                        "target_seconds": script.target_seconds,
                        "minimum_shots": profile.minimum_shot_count(
                            script.target_seconds
                        ),
                        "maximum_shots": profile.maximum_shot_count(
                            script.target_seconds
                        ),
                        "timing_feedback": timing_feedback,
                    },
                    timing_profile=profile,
                )
                shots = data.get("shots")
                if not isinstance(shots, list) or not all(
                    isinstance(item, dict) for item in shots
                ):
                    error = ValueError("storyboard director must return a shots array")
                else:
                    result = [dict(item) for item in shots]
                    try:
                        self._validate_director_plan(
                            script, result, timing_profile=profile
                        )
                        assessment = assess_director_plan_timing(
                            script, result, timing_profile=profile
                        )
                        if not assessment.feasible:
                            raise ValueError(assessment.feedback())
                        return result
                    except ValueError as exc:
                        error = exc
                if attempt < 2:
                    timing_feedback = str(error)
                    continue
                raise error
        except Exception as exc:
            self._handle_artifact_failure("plan_storyboard", exc)
            return None

    def evaluate_artifacts(
        self,
        story: StoryDraft,
        script: StoryScript,
        storyboard: dict[str, Any],
    ) -> dict[str, Any]:
        self._reset_fallback_status()
        if self.client is None:
            self._mark_fallback("quality_judge", self._unavailable_reason)
            return {}
        try:
            data = self._json_completion(
                "quality_judge.md",
                {
                    "story": to_plain_data(story),
                    "script": to_plain_data(script),
                    "storyboard": storyboard,
                },
            )
            scores = data.get("scores")
            issues = data.get("issues", [])
            if not isinstance(scores, dict) or not isinstance(issues, list):
                raise ValueError("quality judge must return scores and issues")
            normalized_scores: dict[str, float] = {}
            for key, value in scores.items():
                score = float(value)
                if not 0 <= score <= 1:
                    raise ValueError("quality judge scores must be between 0 and 1")
                normalized_scores[str(key)] = round(score, 3)
            return {
                "scores": normalized_scores,
                "issues": [str(item) for item in issues if str(item).strip()],
                "summary": str(data.get("summary", "")).strip(),
            }
        except Exception as exc:
            self._mark_fallback("quality_judge", exc)
            return {}

    @staticmethod
    def _validate_director_plan(
        script: StoryScript,
        shots: list[dict[str, Any]],
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> None:
        profile = timing_profile or ShotTimingProfile()
        minimum = profile.minimum_shot_count(script.target_seconds)
        maximum = profile.maximum_shot_count(script.target_seconds)
        if not minimum <= len(shots) <= maximum:
            raise ValueError(f"director shot count must be between {minimum} and {maximum}")
        scene_ids = [scene.scene_id for scene in script.scenes]
        planned_ids: list[int] = []
        allowed_kinds = {
            "establish",
            "action",
            "detail",
            "dialogue",
            "reaction",
            "transition",
        }
        allowed_transitions = {
            "opening",
            "scene_change",
            "same_scene_cut",
            "continuous_action",
            "insert_shot",
            "reverse_shot",
            "reaction_cut",
        }
        for index, shot in enumerate(shots):
            scene_id = int(shot.get("scene_id", 0))
            planned_ids.append(scene_id)
            if scene_id not in scene_ids:
                raise ValueError("director plan references an unknown scene")
            if str(shot.get("kind", "")) not in allowed_kinds:
                raise ValueError("director plan contains an unsupported shot kind")
            action = str(shot.get("action", "")).strip()
            if not action:
                raise ValueError("director plan contains an empty action")
            if not str(shot.get("purpose", "")).strip():
                raise ValueError("director plan contains an empty purpose")
            if count_sequential_action_phases(action) > 3:
                raise ValueError(
                    "director plan contains a shot with too many sequential action phases"
                )
            transition = str(shot.get("transition_type", ""))
            if transition not in allowed_transitions:
                raise ValueError("director plan contains an unsupported transition")
            inherit = shot.get("inherit_previous_frame", False)
            if not isinstance(inherit, bool):
                raise ValueError("inherit_previous_frame must be boolean")
            if inherit and (index == 0 or transition != "continuous_action"):
                raise ValueError("only a non-opening continuous action may inherit a frame")
        order = {scene_id: index for index, scene_id in enumerate(scene_ids)}
        if planned_ids != sorted(planned_ids, key=order.__getitem__):
            raise ValueError("director plan changes script scene order")
        if set(scene_ids) - set(planned_ids):
            raise ValueError("director plan omits a script scene")

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
        self._reset_fallback_status()
        if self.client is None:
            self._mark_fallback("simulate_creator_direction", self._unavailable_reason)
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
    ) -> tuple[list[IdeaCard], int]:
        raw_items = raw if isinstance(raw, list) else []
        result: list[IdeaCard] = []
        seen = {card.fingerprint for card in previous_cards}
        for index, item in enumerate(raw_items[:IDEA_COUNT], 1):
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
        model_card_count = len(result)
        for card in fallback.cards:
            if len(result) == IDEA_COUNT:
                break
            if card.fingerprint not in seen:
                local_card = deepcopy(card)
                local_card.idea_id = f"idea-r{round_number}-{len(result) + 1}"
                result.append(local_card)
                seen.add(local_card.fingerprint)
        return result, max(0, len(result) - model_card_count)

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
                str(value) for value in data.get("ai_filled_fields", fallback.ai_filled_fields)
            ],
        )

    @staticmethod
    def _script_from_model(
        data: dict[str, Any],
        fallback: StoryScript,
        target_seconds: int,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript:
        raw = data.get("script", data)
        raw_scenes = raw.get("scenes", [])
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise ValueError("script must contain at least one scene")
        scenes: list[StoryScene] = []
        for index, item in enumerate(raw_scenes, 1):
            if not isinstance(item, dict):
                raise ValueError("each script scene must be an object")
            visible_action = str(item.get("visible_action", item.get("action", ""))).strip()
            if not visible_action:
                raise ValueError("each script scene needs a visible action")
            characters = item.get("characters", [])
            props = item.get("props", [])
            if not isinstance(characters, list) or not isinstance(props, list):
                raise ValueError("script scene characters and props must be arrays")
            fallback_scene = fallback.scenes[index - 1] if index <= len(fallback.scenes) else None
            raw_weight = item.get(
                "duration_weight",
                getattr(fallback_scene, "duration_weight", 0.0),
            )
            try:
                duration_weight = float(raw_weight)
            except (TypeError, ValueError):
                duration_weight = 0.0
            if not math.isfinite(duration_weight) or duration_weight < 0:
                raise ValueError("duration_weight must be a finite non-negative number")
            scenes.append(
                StoryScene(
                    scene_id=index,
                    title=str(item.get("title", f"场景 {index}")).strip(),
                    location=str(item.get("location", "故事核心场景")).strip(),
                    time_of_day=str(item.get("time_of_day", "连续时间")).strip(),
                    characters=[str(value).strip() for value in characters if str(value).strip()],
                    action=visible_action,
                    visible_action=visible_action,
                    dialogue=str(item.get("dialogue", "")).strip(),
                    narration=str(item.get("narration", "")).strip(),
                    props=[str(value).strip() for value in props if str(value).strip()],
                    start_state=str(item.get("start_state", "")).strip(),
                    end_state=str(item.get("end_state", "")).strip(),
                    emotional_change=str(item.get("emotional_change", "")).strip(),
                    duration_weight=duration_weight,
                    duration_reason=str(
                        item.get(
                            "timing_reason",
                            item.get(
                                "duration_reason",
                                getattr(fallback_scene, "duration_reason", ""),
                            ),
                        )
                    ).strip(),
                    duration=1,
                )
            )
        profile = timing_profile or ShotTimingProfile()
        if len(scenes) <= profile.maximum_shot_count(target_seconds):
            OpenAIStoryAgent._normalize_script_durations(
                scenes, target_seconds, timing_profile=profile
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
        selected_cards: list[IdeaCard] | None = None,
        selected_elements: dict[str, ElementOption] | None = None,
    ) -> StoryDraft:
        try:
            data = self._json_completion(
                "story_continuity_reviewer.md",
                {
                    "story": to_plain_data(story),
                    "immutable_selected_cards": to_plain_data(selected_cards or []),
                    "immutable_selected_elements": to_plain_data(selected_elements or {}),
                },
            )
            reviewed = self._story_from(data, story)
            reviewed.field_sources = deepcopy(story.field_sources)
            reviewed.ai_filled_fields = list(story.ai_filled_fields)
            return reviewed
        except Exception as exc:
            self._handle_artifact_failure(operation, exc, fallback_kind="partial")
            return story

    def _review_script_continuity(
        self,
        story: StoryDraft,
        script: StoryScript,
        *,
        operation: str,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript:
        profile = timing_profile or ShotTimingProfile()
        reviewed = script
        try:
            data = self._json_completion(
                "script_continuity_reviewer.md",
                {
                    "confirmed_story": to_plain_data(story),
                    "required_constraints": self._script_constraints(story),
                    "script": to_plain_data(script),
                    "target_seconds": script.target_seconds,
                    "maximum_scenes": profile.maximum_shot_count(
                        script.target_seconds
                    ),
                },
            )
            reviewed = self._script_from_model(
                data, script, script.target_seconds, timing_profile=profile
            )
        except Exception as exc:
            self._handle_artifact_failure(operation, exc, fallback_kind="partial")
        return self._repair_script_budget(
            story, reviewed, operation=operation, timing_profile=profile
        )

    def _repair_script_budget(
        self,
        story: StoryDraft,
        script: StoryScript,
        *,
        operation: str,
        timing_profile: ShotTimingProfile | None = None,
    ) -> StoryScript:
        profile = timing_profile or ShotTimingProfile()
        maximum = profile.maximum_shot_count(script.target_seconds)
        if len(script.scenes) <= maximum:
            self._normalize_script_durations(
                script.scenes,
                script.target_seconds,
                timing_profile=profile,
            )
            return script
        try:
            data = self._json_completion(
                "script_compressor.md",
                {
                    "confirmed_story": to_plain_data(story),
                    "required_constraints": self._script_constraints(story),
                    "script": to_plain_data(script),
                    "target_seconds": script.target_seconds,
                    "maximum_scenes": maximum,
                },
            )
            compressed = self._script_from_model(
                data,
                script,
                script.target_seconds,
                timing_profile=profile,
            )
            if len(compressed.scenes) > maximum:
                raise ValueError("compressed script still exceeds maximum_scenes")
            self._normalize_script_durations(
                compressed.scenes,
                compressed.target_seconds,
                timing_profile=profile,
            )
            return compressed
        except Exception as exc:
            self._handle_artifact_failure(
                f"{operation}_scene_budget",
                exc,
                fallback_kind="whole",
            )
            return RuleBasedStoryAgent().generate_script(
                story, script.target_seconds, timing_profile=profile
            )

    @staticmethod
    def _script_constraints(story: StoryDraft) -> dict[str, Any]:
        return {
            "character_names": [
                character.name
                for character in story.characters
                if character.name.strip()
            ],
            "character_identities": [
                f"{character.name}：{character.description}".strip("：")
                for character in story.characters
                if character.name.strip() or character.description.strip()
            ],
            "core_conflict": story.core_conflict,
            "ending": story.ending,
        }

    @staticmethod
    def _normalize_script_durations(
        scenes: list[StoryScene],
        target_seconds: int,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> None:
        profile = timing_profile or ShotTimingProfile()
        durations, weights, reasons = plan_scene_durations(
            scenes,
            target_seconds,
            minimum=profile.min_duration_seconds,
            maximum=target_seconds,
        )
        for scene, duration, weight, reason in zip(scenes, durations, weights, reasons):
            scene.duration = duration
            scene.duration_weight = weight
            scene.duration_reason = reason

    @property
    def provider_label(self) -> str:
        return f"{self.provider_name} · {self.model}"

    @property
    def _unavailable_reason(self) -> str:
        return self.configuration_error or "文本 API 未配置。"

    def _json_completion(
        self,
        prompt_name: str,
        payload: dict[str, Any],
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> dict[str, Any]:
        profile = timing_profile or ShotTimingProfile()
        if prompt_name.startswith("story_") or prompt_name.startswith("storyboard_"):
            max_tokens = 8000
        elif prompt_name.startswith("script_"):
            max_tokens = 6000
        else:
            max_tokens = 4000
        payload_text = json.dumps(payload, ensure_ascii=False)
        if len(payload_text) > MAX_LLM_PAYLOAD_CHARS:
            raise ValueError(
                f"文本模型输入过长（{len(payload_text)} 字符），请缩短方向或减少历史轮次"
            )
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._render_prompt(prompt_name, profile)},
                {"role": "user", "content": payload_text},
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
        }
        if self.json_mode != "disabled":
            request["response_format"] = {"type": "json_object"}
        response = self._create_completion(request)
        content = (response.choices[0].message.content or "").strip()
        if len(content) > MAX_LLM_RESPONSE_CHARS:
            raise ValueError("model JSON response is too large")
        if not content:
            response = self._create_completion(request)
            content = (response.choices[0].message.content or "").strip()
            if len(content) > MAX_LLM_RESPONSE_CHARS:
                raise ValueError("model JSON response is too large")
        try:
            data = self._extract_json_object(content)
        except ValueError as original_error:
            try:
                data = self._repair_json_object(
                    prompt_name,
                    payload_text,
                    content,
                    max_tokens,
                    timing_profile=profile,
                )
            except Exception as repair_error:
                raise ValueError(
                    f"{original_error}; JSON repair failed: {repair_error}"
                ) from original_error
        if not isinstance(data, dict):
            raise ValueError("model JSON must be an object")
        return data

    def _create_completion(self, request: dict[str, Any]) -> Any:
        try:
            return self.client.chat.completions.create(**request)
        except Exception as exc:
            message = str(exc).lower()
            can_retry_without_json_mode = self.json_mode == "auto" and (
                "response_format" in message
                or "json_object" in message
                or ("json" in message and ("unsupported" in message or "not support" in message))
            )
            if not can_retry_without_json_mode:
                raise
            fallback_request = dict(request)
            fallback_request.pop("response_format", None)
            return self.client.chat.completions.create(**fallback_request)

    def _repair_json_object(
        self,
        prompt_name: str,
        payload_text: str,
        content: str,
        max_tokens: int,
        *,
        timing_profile: ShotTimingProfile | None = None,
    ) -> dict[str, Any]:
        profile = timing_profile or ShotTimingProfile()
        repair_request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是结构化输出修复器。只返回一个合法的 JSON 对象，"
                        "不要 Markdown、不要解释、不要额外文字。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "原任务要求：\n"
                        f"{self._render_prompt(prompt_name, profile)}\n\n"
                        "原始输入：\n"
                        f"{payload_text[:MAX_REPAIR_INPUT_CHARS]}\n\n"
                        "模型原始回答：\n"
                        f"{content[:MAX_REPAIR_INPUT_CHARS]}\n\n"
                        "请将模型原始回答转换为符合原任务要求的 JSON 对象。\n"
                        "结构要求：严格保持“原任务要求”中定义的 JSON 顶层与嵌套结构"
                        "（键名、层级、数组/对象类型都不得改变），只修复内容，"
                        "不要发明新的顶层字段。"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if self.json_mode != "disabled":
            repair_request["response_format"] = {"type": "json_object"}
        response = self._create_completion(repair_request)
        repaired = (response.choices[0].message.content or "").strip()
        return self._extract_json_object(repaired)

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, character in enumerate(content):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(content, index)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    return candidate
            raise ValueError("model did not return a JSON object")
        if not isinstance(data, dict):
            raise ValueError("model JSON must be an object")
        return data

    def _load_prompt(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8")

    def _render_prompt(
        self,
        name: str,
        timing_profile: ShotTimingProfile | None = None,
    ) -> str:
        """Load a prompt and substitute provider-bound duration tokens."""
        profile = timing_profile or ShotTimingProfile()
        return (
            self._load_prompt(name)
            .replace("{MIN_DURATION_SECONDS}", str(profile.min_duration_seconds))
            .replace("{MAX_DURATION_SECONDS}", str(profile.max_duration_seconds))
        )

    def _reset_fallback_status(self) -> None:
        self.last_used_fallback = False
        self.last_fallback_kind = ""
        self.last_fallback_reason = ""

    def _mark_fallback(
        self,
        operation: str,
        reason: object,
        *,
        fallback_kind: str = "whole",
    ) -> None:
        self.last_used_fallback = True
        self.last_fallback_kind = fallback_kind
        self.fallback_count += 1
        self.last_fallback_reason = f"{operation}: {reason}"

    def _handle_artifact_failure(
        self,
        operation: str,
        reason: object,
        *,
        fallback_kind: str = "whole",
    ) -> None:
        self._mark_fallback(operation, reason, fallback_kind=fallback_kind)
        if self.client is not None and not self.allow_artifact_fallback:
            raise RuntimeError(
                f"真实文本模型生成失败（{self.provider_label}，{operation}）：{reason}"
            ) from reason if isinstance(reason, BaseException) else None
