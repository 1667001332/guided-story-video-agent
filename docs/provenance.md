# 代码来源与边界

## v0.4 故事优先重构

完整故事、结构化人物地点资料、故事确认门禁、自适应场景剧本以及当前 schema v5 均在本仓库重新实现。schema v4 仍作为历史格式只读兼容；Dramatron 只提供“故事要素先于逐场剧本”的公开架构启发；本项目没有复制其 Notebook 代码、提示词样例或文本标记解析。

## v0.3 创意花园重构

创意卡、相似扩展、混合、故事零件、AI 补全来源、简化 CLI、网页与新 Bench 均在本仓库重新实现。Sudowrite、Dramatron、Story2Board、Storyboarder 和 LTX Studio 仅作为公开交互与架构参考，未复制其代码；具体依据见 `docs/research-basis.md`。

本仓库拥有独立 Git 历史，不是 `interactive-movie-agent` 的子目录或 Git 分支。

从 `interactive-movie-agent v0.1.0` 选择性迁移并重新适配的能力：

- Agnes 异步视频任务的提交、轮询和下载流程；
- `StoryboardShot`、`VideoArtifact`、`RenderManifest` 的结构化思路；
- 逐镜头生成、失败记录、成功产物复用和 FFmpeg 拼接；
- 所有付费视频请求必须位于显式确认之后的安全边界。

明确没有迁移的旧版能力：

- `WorldState` 与世界预设；
- 一句话导演命令解析和冲突校验；
- 互动剧情节点、分支和旧版长期记忆；
- 旧版 Gradio 页面和 Batch Case。

新仓库的原创实现包括低压力创意状态机、恰好 8 卡校验、1–3 张来源保留、AI 补全透明标签、完整故事确认、故事/剧本连续性二次审查、故事驱动的自动时长估算、15–300秒自定义时长、自适应场景剧本、视觉圣经、内容驱动镜头规划、三段式镜头 Prompt、失败镜头续跑、旁白/SRT 预生成和一句话自演测试。v0.2/v0.3 的接口名称仅保留为弃用兼容包装，不参与 v0.4 主流程。

相关开源设计参考：

- PenShot：剧本到视频模型友好的定时镜头、结构化提示词和连续性设计。
- Toonflow：策划、编剧、分镜、出片的阶段化产品流程。
- VibeFrame：plan、build、inspect、render 分层，供应商解耦、生成报告和断点恢复。
- ShortGPT：旁白、字幕和视频编辑流水线。

本仓库不直接打包或复制上述第三方仓库的源文件。

## 当前主流程的边界

当前主流程采用 `StoryScript → VideoJob → VideoProvider.generate_video → VideoArtifact`。
`StoryboardPlan` 与逐镜头 `StoryRenderer` 只作为旧会话和旧测试的兼容层，不再是新视频任务的必要中间层。Provider 的时长上限、是否需要内部切片以及具体 API 字段均属于 Provider 适配器能力；核心不会把 Agnes 的 3–15 秒限制传播到剧本或会话模型。
