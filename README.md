# AI 故事、剧本与视频生成 Agent

输入一个故事灵感，系统可以依次生成创意方案、完整故事、剧本和分镜，并调用视频 API 生成成片。故事、剧本和分镜均可修改，并在进入下一阶段前由用户确认。

## 功能

- 根据一句话生成 8 个故事创意
- 支持选择、混合、换一批和生成相似创意
- 生成完整故事并检查人物、事件和因果是否连贯
- 根据用户意见修改故事并保存历史版本
- 将确认后的故事改编成剧本
- 自动估算成片时长，也可设置 15–300 秒的目标时长
- 检查剧本中相邻场景的时间、地点、人物和状态衔接
- 根据剧本生成视觉设定和分镜
- 按动作、对白、旁白、情绪停顿和叙事职责分配 3–15 秒镜头时长
- 保存每个镜头的时长理由、权重、seed、首帧、动作、结束帧和视频提示词
- 区分人物、地点、道具、场景参考图与明确的 `start_frame`
- 在分镜页上传参考图、选择用途、绑定镜头或资产、删除并冻结视觉输入
- 同场景镜头生成后提取真实末帧，安全暂存后作为下一镜头首帧
- 换地点时停止继承上一镜头；上游失败时只允许使用已确认 `start_frame` 降级
- 调用视频 API 按镜头依赖顺序生成视频
- 使用包含参考图和上游末帧内容的指纹安全复用镜头
- 让镜头旁白、TTS 和 SRT 共用同一条不重复的旁白时间轴，并使用 FFmpeg 拼接成片
- 保存已取得的任务 ID、成功镜头和会话状态；取得任务 ID 后发生中断时，只恢复待完成或失败镜头
- 提交响应无法确认时停止自动重提，避免同一镜头被重复计费
- 支持离线演示、真实文本 API 测试和真实视频 API 测试

## 工作流程

```text
故事灵感
→ 创意选择
→ 完整故事
→ 用户确认故事
→ 剧本
→ 用户确认剧本
→ 分镜
→ 用户确认分镜和费用
→ 生成视频
```

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- 生成最终成片时需要安装带 `ffprobe` 的 FFmpeg

在项目根目录执行：

```powershell
python -m pip install -e ".[web,narration]"
Copy-Item .env.example .env
notepad .env
```

启动网页：

```powershell
python -m guided_story_agent.web_app
```

浏览器打开：

```text
http://127.0.0.1:7860/
```

如果没有配置 API Key，文本生成会使用本地离线方案，不发送网络请求。页面会显示当前结果来自真实文本模型、离线模式还是 API 失败后的本地兜底。

## API 配置

配置默认从命令执行目录下的 `.env` 读取。不要把包含 API Key 的 `.env`
上传到 GitHub 或发送给其他人。

### 文本模型

当前支持 OpenAI `chat/completions` 兼容接口，包括提供兼容接口的第三方平台。

```env
TEXT_PROVIDER=openai_compatible
TEXT_API_KEY=
TEXT_BASE_URL=
TEXT_MODEL=
TEXT_TIMEOUT=120
TEXT_JSON_MODE=auto
```

需要填写 `TEXT_API_KEY` 和 `TEXT_MODEL`。使用第三方平台时，还需要填写该平台提供的 `TEXT_BASE_URL`。

原生非 OpenAI 协议的 Gemini、Claude 等平台目前需要单独实现 Provider 适配器。

### 视频模型

当前只实现了 Agnes 视频 Provider。

```env
VIDEO_PROVIDER=agnes
VIDEO_API_KEY=
VIDEO_API_ROOT=https://apihub.agnes-ai.com
VIDEO_MODEL=agnes-video-v2.0
VIDEO_TIMEOUT=120
VIDEO_POLL_INTERVAL=5
VIDEO_MAX_POLL_SECONDS=900
```

视频 API 只会在故事、剧本和分镜全部确认，并再次确认费用后调用。

Agnes 官方接口支持 `image` 图生视频和 `seed`，但 `image` 必须是公网可访问
URL。同时配置 `VIDEO_REFERENCE_ROOT` / `VIDEO_REFERENCE_BASE_URL` 后，项目会把
当前渲染任务需要的单张首帧或真实末帧原子暂存到公开根目录下的独立运行目录；
`VIDEO_OUTPUT_DIR` 不需要位于公开根目录内，项目也不会暴露整个仓库。静态服务器或
CDN 仍需由部署方负责把该公开根目录映射到所填 URL。复用旧任务末帧时，会根据当前
`VIDEO_REFERENCE_BASE_URL` 重新发布并生成 URL，不沿用 Manifest 中的旧地址。

Agnes 的 `image` 只会接收明确确认的 `start_frame` 或同场景上一镜头的真实末帧。
人物定妆照、地点图和道具图不会被自动当作首帧；Agnes 不支持通用身份参考图时，
该能力缺失会写入 UI 和清单。没有安全首帧时，同场景链会明确报告
`dependency_failed`，不会悄悄退回文生视频；新场景允许的文本回退会标记在
`unreferenced_fallback_shots`。

Web 分镜页会把上传图片复制到 `VISUAL_INPUT_DIR`（默认
`outputs/visual_inputs`），再由用户选择用途并绑定到镜头或视觉资产。添加、删除或
重新确认视觉输入都会使原分镜确认失效，必须重新确认后才能进入付费生成。

Agnes 参数依据其[官方 Video V2.0 文档](https://agnes-ai.com/en/docs/agnes-video-v20)；
项目没有添加文档中不存在的私有字段。

完整的 API 交接和测试步骤见 [师兄 API 测试指南](docs/mentor-testing-guide.md)。

## 测试运行

### 离线测试

运行完整的故事、剧本和分镜流程，不调用真实 API：

```powershell
python scripts/run_guided_story_selfplay.py --offline --output outputs/selfplay_auto
```

### 真实文本 API 测试

要求所有文本请求都通过真实接口完成，发生本地兜底时测试失败：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --output outputs/live_text --require-live-text
```

### 真实视频 API 测试

以下命令会调用付费视频接口：

```powershell
python scripts/run_guided_story_selfplay.py `
  --target-seconds 30 `
  --output outputs/live_video `
  --require-live-text `
  --render `
  --confirm-paid-video RENDER
```

测试结果保存在指定的输出目录中，包括故事、剧本、分镜、调用记录、会话状态和评测数据。

### 批量离线测试

默认使用安装包内置的 12 条题目，不调用任何远端服务：

```powershell
guided-story-batch --offline --max-cases 2 --output outputs/offline_batch
```

`--resume` 只会复用输入、时长、文本模式、模型、视频设置以及代码和
Prompt 指纹均一致的结果；配置或实现变化时会重新执行。未完成的视频会加载
原会话、原任务 ID 和原输出目录继续处理。批量视频仍需同时提供 `--render` 和
`--confirm-paid-video RENDER`。

## 命令行模式

除网页外，也可以使用命令行：

```powershell
guided-story-cli --offline
```

严格测试真实文本接口时改用 `guided-story-cli --require-live-text`。
如需允许命令行调用视频接口，启动时增加 `--render`。真正调用前仍需在终端输入 `RENDER` 确认。

## 自动测试

```powershell
python -m pip install -e ".[web,narration,dev]"
python -m pytest -q
python -m unittest discover -s tests -v
python -m compileall -q guided_story_agent scripts tests
python -m ruff check .
git diff --check
python scripts/verify_installed_package.py
```

## 主要文件

```text
guided_story_agent/
  agent.py             文本模型调用、Prompt 加载和离线兜底
  session.py           创作流程、状态和确认规则
  storyboard.py        剧本到分镜的转换
  continuity.py        参考资产解析、连续性校验和输入指纹
  video_provider.py    视频 API 适配
  rendering.py         依赖渲染、末帧提取、续跑和成片拼接
  web_app.py           Gradio 网页
  prompts/             随安装包发布的创意、故事、剧本和审查 Prompt
  resources/           随安装包发布的默认批测题目

tests/                 自动测试
docs/                  测试指南、研究依据和版本说明
```

## 当前限制

- 视频生成目前只支持 Agnes
- Agnes 的本地图生视频需要调用方另外提供安全的公网图片托管和 URL 解析器
- 跨镜头链路能把真实末帧传入下一镜头，但最终一致性仍取决于视频模型本身
- 原生非 OpenAI 协议的文本平台尚未适配
- 不包含账号系统、数据库、多人协作和商业任务队列
- 当前不支持角色口型同步
- 离线测试不会验证真实 API 的鉴权、稳定性、费用和生成质量；这些仍需使用
  测试者自己的账号单独验收

研究与设计说明见 [研究与设计依据](docs/research-basis.md)，代码来源和迁移边界见 [来源说明](docs/provenance.md)。
