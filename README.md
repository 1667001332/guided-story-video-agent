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
- 保存每个镜头的首帧、动作、结束帧和视频提示词
- 调用视频 API 逐镜头生成视频
- 生成旁白和字幕，并使用 FFmpeg 拼接成片
- 保存视频生成进度，再次运行时只重试失败镜头
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
- 生成最终成片时需要安装 FFmpeg

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

配置写在当前项目根目录的 `.env` 中。不要把包含 API Key 的 `.env` 上传到 GitHub 或发送给其他人。

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

完整的 API 交接和测试步骤见 [师兄 API 测试指南](docs/mentor-testing-guide.md)。

## 测试运行

### 离线测试

运行完整的故事、剧本和分镜流程，不调用真实 API：

```powershell
python scripts/run_guided_story_selfplay.py --output outputs/selfplay_auto
```

### 真实文本 API 测试

要求所有文本请求都通过真实接口完成，发生本地兜底时测试失败：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --output outputs/live_text --require-live-text
```

### 真实视频 API 测试

以下命令会调用付费视频接口：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --output outputs/live_video --require-live-text --render
```

测试结果保存在指定的输出目录中，包括故事、剧本、分镜、调用记录、会话状态和评测数据。

## 命令行模式

除网页外，也可以使用命令行：

```powershell
guided-story-cli --require-live-text
```

如需允许命令行调用视频接口，启动时增加 `--render`。真正调用前仍需在终端输入 `RENDER` 确认。

## 自动测试

```powershell
python -m pytest -q
python -m compileall -q guided_story_agent scripts tests
python -m ruff check .
git diff --check
```

## 主要文件

```text
guided_story_agent/
  agent.py             文本模型调用、Prompt 加载和离线兜底
  session.py           创作流程、状态和确认规则
  storyboard.py        剧本到分镜的转换
  video_provider.py    视频 API 适配
  rendering.py         镜头生成、续跑和成片拼接
  web_app.py           Gradio 网页

prompts/               创意、故事、剧本和审查 Prompt
tests/                 自动测试
docs/                  测试指南、研究依据和版本说明
```

## 当前限制

- 视频生成目前只支持 Agnes
- 原生非 OpenAI 协议的文本平台尚未适配
- 不包含账号系统、数据库、多人协作和商业任务队列
- 当前不支持角色口型同步
- 真实 API 的稳定性、费用和生成质量需要使用各自的账号单独测试

研究与设计说明见 [研究与设计依据](docs/research-basis.md)，代码来源和迁移边界见 [来源说明](docs/provenance.md)。
