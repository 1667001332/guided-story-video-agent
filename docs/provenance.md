# 代码来源与边界

## v0.2 共创重构

动态故事教练、故事圣经版本、可编辑工作坊、人工 CLI 和新 Bench 均在本仓库重新实现。Dramatron、Sudowrite、Story2Board、Storyboarder 和 LTX Studio 仅作为公开交互与架构参考，未复制其代码；具体依据见 `docs/research-basis.md`。

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

新仓库新增的原创实现包括动态创作引导状态机、至少五轮与结局门禁、制作细节盘问、定时剧本、旁白/SRT 预生成和双角色 LLM 自演测试。

相关开源设计参考：

- PenShot：剧本到视频模型友好的定时镜头、结构化提示词和连续性设计。
- Toonflow：策划、编剧、分镜、出片的阶段化产品流程。
- ShortGPT：旁白、字幕和视频编辑流水线。

本仓库不直接打包或复制上述第三方仓库的源文件。
