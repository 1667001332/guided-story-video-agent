# 师兄 API 测试指南

本指南用于 `v0.5.1` 的真实 API 验收。请使用测试者自己的 API Key，不要接收或复制项目作者的 `.env`。

## 1. 安装

```powershell
git clone https://github.com/1667001332/guided-story-video-agent.git
Set-Location guided-story-video-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[web,dev]"
Copy-Item .env.example .env
notepad .env
```

## 2. 文本 API

当前文本适配器要求 Provider 兼容 OpenAI `chat/completions`：

```env
TEXT_PROVIDER=openai_compatible
TEXT_API_KEY=测试者自己的Key
TEXT_BASE_URL=平台提供的兼容地址
TEXT_MODEL=平台提供的准确模型ID
TEXT_TIMEOUT=120
TEXT_JSON_MODE=auto
```

- `TEXT_API_KEY` 和 `TEXT_MODEL` 必填。
- OpenAI 官方接口可将 `TEXT_BASE_URL` 留空；其他平台应填写平台文档中的兼容地址。
- `TEXT_JSON_MODE=auto` 适合大多数兼容平台。
- 原生 Gemini、Claude 等非 OpenAI 协议不能直接使用，应先增加对应 Provider 适配器。

先运行严格验收，任何离线兜底都会判定失败：

```powershell
python scripts/run_guided_story_selfplay.py --output outputs/live_text --require-live-text
```

通过后启动网页：

```powershell
python -m guided_story_agent.web_app
```

网页成功提示会显示实际 Provider 和模型；失败提示会显示缺少的变量、未支持的协议或请求错误。

## 3. 视频 API

视频测试会产生费用，文本验收通过后再进行。当前只实现 Agnes：

```env
VIDEO_PROVIDER=agnes
VIDEO_API_KEY=测试者自己的Key
VIDEO_API_ROOT=https://apihub.agnes-ai.com
VIDEO_MODEL=agnes-video-v2.0
```

使用其他视频平台时不要直接填写其 Key；应先实现对应视频 Provider。

## 4. 建议测试内容

1. 普通剧情、悬疑、爱情和科幻各测试至少两次。
2. 检查人物动机、信息来源、证据链和相邻场景衔接。
3. 分别测试自动时长和15–300秒自定义时长。
4. 确认长故事可以生成超过12个镜头。
5. 记录输入文本、网页状态提示、完整错误和 `outputs/live_text` 结果。

反馈时不要提交 `.env`、API Key 或包含凭据的截图。
