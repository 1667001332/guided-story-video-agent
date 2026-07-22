# v0.2 研究与设计依据

本项目只借鉴公开的架构和交互思想，没有复制闭源产品代码。

## 参考模式

- [Dramatron](https://github.com/google-deepmind/dramatron)：借鉴概念、人物地点、情节节点、场景对白的分层生成，以及逐层扩展和人类重写。
- [DeepMind Dramatron 研究](https://deepmind.google/research/publications/13609/)：强调系统是共同写作工具而非自动作者，因此建议被设计成选项，不直接写入事实。
- [Sudowrite Story Bible](https://docs.sudowrite.com/using-sudowrite/1ow1qkGqof9rtcyGnrWUBS/what-is-story-bible/jmWepHcQdJetNrE991fjJC)：借鉴可编辑故事圣经、人物卡和场景卡作为统一上下文；本项目另加原句证据和本地冲突门禁。
- [Story2Board](https://github.com/DavidDinkevich/Story2Board)：借鉴逐面板提示词、主体参考和一致性检查，并落入每个镜头的视觉锚点与连续性说明。
- [Storyboarder](https://github.com/wonderunit/storyboarder)：借鉴轻量编辑与撤销/重做体验，不采用其桌面绘图实现。
- [LTX Studio](https://website.ltx.studio/)：借鉴人物、场景、道具复用和逐镜头 Retake 的产品思路，不涉及其闭源代码。
- [CHI 2024 人机共写研究](https://arxiv.org/abs/2402.11723)：研究提示过强的 AI 主导会影响作者所有感，因此系统每轮只提一个问题，三条建议均可拒绝。

## 本项目的具体流程

1. 用户写开头或自由想法。
2. 单次教练调用提取带证据事实、诊断冲突和最高优先级缺口。
3. 本地状态机过滤字段并保护正式事实；AI 只提交候选更新。
4. 用户自由回答、采用建议或忽略建议，至少有效参与五轮。
5. 因果链完整且没有冲突后，用户主动生成五节点大纲候选。
6. 用户编辑、局部重写、撤销、重做、质量检查并确认大纲。
7. 系统继续补齐人物、场景、道具、旁白对白和镜头承接。
8. 生成定时可拍摄剧本并确认，再生成动态数量分镜。
9. 分镜检查因果、时长、主体、空间承接、视觉锚点和镜头变化。
10. 用户确认分镜并再次确认费用后，才允许视频生成。

## 与 v0.1 的本质差异

v0.1 的问题选择主要由固定字段缺失顺序决定，且产物以只读 JSON 为主。v0.2 将诊断、证据、冲突、建议、版本和审阅提升为核心领域对象；网页、CLI 和自演只调用同一个 `GuidedStorySession`，界面不复制剧情逻辑。
