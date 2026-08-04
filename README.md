# AI 故事到剧本到视频 Agent

一句话方向 → 8 张创意卡 → 用户挑选 → 完整故事 → 剧本 → 确认 → 完整视频。
视频以单个 VideoJob 提交,Provider 在自己适配器内部决定是否分段;核心流程不再把
分镜固定成 3–15 秒切片。

## 特性

- 一句话生成 8 张创意卡,支持挑选、相似、混合和刷新
- 故事提供人物、事件、冲突、反转、结局等可组合零件
- 允许用户反馈修改故事和剧本,保留历史版本
- 已确认故事约束剧本,不把不同地点/时段硬拼成同一场景
- 剧本按内容时长编排,而不是机械切成等长镜头
- **多视频 Provider 架构**:`VideoProvider` 协议 + 注册表,`VIDEO_PROVIDER` 一键切换
- **时长边界参数化**:`ShotTimingProfile` 由 Provider capabilities 自动推导,
  规划、分镜、门禁不再硬编码 3–15 秒
- **Web/CLI 切换 Provider**:网页下拉框 + CLI `--video-provider`
- 旁白默认不进视频 prompt(避免模型机械念白),角色对白保留
- JSON 结构化修复器:解析失败自动发起一次修复,空响应自动重试
- 分镜支持人物、地点、道具、参考图绑定,并确认 `start_frame`
- 视频提交前持久化提交意图;状态查询与下载可重试,POST 永不自动重试
- 提交响应不确定时停止自动重提,可在网页登记后台真实任务 ID
- 网页会话自动保存,可恢复上次进度
- 支持离线演示、真实文本 API 测试和真实视频 API 验收

## 创作流程

```text
一句话方向
  → 创意卡(挑选/相似/混合)
  → 故事零件(人物/冲突/反转/结局)
  → 用户确认故事
  → 剧本
  → 用户确认剧本
  → 完整视频任务
  → 用户确认费用
  → 真实视频
```

## 快速开始

### 环境要求

- Python 3.10 或更高
- 需要合成旁白/字幕时安装 FFmpeg(含 ffprobe)

在项目根目录执行:

```powershell
python -m pip install -e ".[web,narration]"
Copy-Item .env.example .env
notepad .env
```

启动网页:

```powershell
python -m guided_story_agent.web_app
```

浏览器打开:

```text
http://127.0.0.1:7860/
```

没有配置 API Key 时,文本生成使用本地离线方案,不发送网络请求。配置真实文本 API 后,
故事和剧本生成采用失败关闭策略:请求、JSON 解析或结构校验失败都会停止当前操作并显示
原始原因,不再把本地模板冒充成远程模型结果。

## API 配置

配置默认从命令执行目录下的 `.env` 读取。不要把包含 API Key 的 `.env`
上传到 GitHub 或发送给其他人。

### 文本模型

支持 OpenAI `chat/completions` 兼容接口(DeepSeek、Agnes 等第三方平台均可)。

```env
TEXT_PROVIDER=openai_compatible
TEXT_API_KEY=
TEXT_BASE_URL=
TEXT_MODEL=
TEXT_TIMEOUT=120
TEXT_JSON_MODE=auto
```

需要填写 `TEXT_API_KEY` 和 `TEXT_MODEL`;使用第三方平台时再填写其 `TEXT_BASE_URL`。

`TEXT_JSON_MODE` 取值:

- `auto`(默认):发送 `response_format`,服务端不支持时自动降级;解析失败自动发起
  一次结构化修复
- `required`:服务端必须支持 `response_format`,失败快速报错
- `disabled`:从不发送 `response_format`

原生非 OpenAI 协议的 Gemini、Claude 等平台需要单独实现文本 Provider 适配器。

### 视频模型

内置 Agnes 适配器,默认 `VIDEO_PROVIDER=agnes`。

```env
VIDEO_PROVIDER=agnes
VIDEO_API_KEY=
VIDEO_API_ROOT=https://apihub.agnes-ai.com
VIDEO_MODEL=agnes-video-v2.0
VIDEO_TIMEOUT=120
VIDEO_POLL_INTERVAL=5
VIDEO_MAX_POLL_SECONDS=900
VIDEO_NETWORK_RETRIES=2
VIDEO_RETRY_BACKOFF=1
```

视频 API 只会在故事、剧本和完整 VideoJob 全部确认,并再次确认费用后调用。
`VIDEO_NETWORK_RETRIES` 只作用于状态查询和文件下载;视频提交 POST 无论配置多少次
都只调用一次,响应丢失时必须先在 Provider 后台核对,再在网页"处理提交结果不确定的
任务"中登记真实任务 ID 或确认未受理。

### 接入其他视频 Provider

新增 Provider 只需三步:

1. 实现 `VideoProvider` 协议(`generate_shot` / `generate_video` /
   `capabilities` / `dimensions`)
2. 注册工厂:`register_video_provider("名字", factory)`
3. 使用方设置 `VIDEO_PROVIDER=名字`(或 Web 下拉框 / CLI `--video-provider`)

Provider 通过 `ProviderCapabilities` 声明能力:单镜时长边界(`min/max_duration_seconds`,
自动映射为 `ShotTimingProfile`,分镜规划无需手动传参)、画幅(`supported_aspect_ratios`)、
参考图(`supports_reference_images`)、图生视频(`supports_image_to_video`)等。

Agnes 适配器声明 3–15 秒能力边界;更长视频应接入声明 `supports_long_video` 的 Provider,
而不是修改脚本或核心渲染器。Agnes 参数依据其官方 Video V2.0 文档,项目没有添加文档中
不存在的私有字段。

## 测试运行

### 离线测试

运行完整的故事、剧本和分镜流程,不调用真实 API:

```powershell
python scripts/run_guided_story_selfplay.py --offline --output outputs/selfplay_auto
```

### 真实文本 API 测试

要求所有文本请求都通过真实接口完成;任一正式产物调用失败就停止,不生成替代成品:

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --output outputs/live_text --require-live-text
```

### 真实视频 API 测试

以下命令会调用付费视频接口:

```powershell
python scripts/run_guided_story_selfplay.py `
  --target-seconds 30 `
  --output outputs/live_video `
  --require-live-text `
  --render `
  --confirm-paid-video RENDER
```

测试结果保存在指定输出目录,包括故事、剧本、分镜、调用记录、会话状态和评测数据。
`quality_report.json` 是自动指标,`human_review.json` 是供测试者填写的人工评分表。
真实文本模型测试可增加 `--llm-judge`;离线模式不能伪装成 LLM 评审。

### 批量离线测试

默认使用安装包内置题目,不调用任何远端服务:

```powershell
guided-story-batch --offline --max-cases 2 --output outputs/offline_batch
```

`--resume` 只会复用输入、时长、文本模式、模型、视频设置以及代码和 Prompt 指纹均一致的
结果;配置或实现变化时会重新执行。批量视频仍需同时提供 `--render` 和
`--confirm-paid-video RENDER`。

## 命令行模式

除网页外,也可以使用命令行:

```powershell
guided-story-cli --offline
```

- 严格测试真实文本接口:`guided-story-cli --require-live-text`
- 允许调用视频接口:启动时增加 `--render`,真正调用前仍需输入 `RENDER` 确认
- 切换视频 Provider:`guided-story-cli --video-provider agnes`

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
  agent.py             文本模型调用、Prompt 加载、JSON 修复和离线兜底
  session.py           创作流程、状态和确认规则
  provider_config.py   Provider 配置校验与视频 Provider 注册表
  video_provider.py    视频 Provider 协议与 Agnes 适配器
  video_job.py         已确认剧本到完整 VideoJob 的转换
  timing.py            时长估算与 ShotTimingProfile
  storyboard.py        旧版分镜兼容层(非默认流程)
  continuity.py        参考资产解析、连续性校验和输入指纹
  rendering.py         VideoJob 单任务渲染与旧版分镜兼容渲染
  narration.py         逐句 TTS、字幕与精确旁白时间轴
  quality.py           故事到剧本语义覆盖、分镜质量指标和人工评测模板
  web_app.py           Gradio 网页(含视频 Provider 选择)
  prompts/             随安装包发布的创意、故事、剧本和审查 Prompt
  resources/           随安装包发布的默认批测题目

tests/                 自动测试
docs/                  测试指南、研究依据和版本说明
```

## 当前限制

- 仓库内置 Agnes 适配器;其他 Provider 通过 `register_video_provider` 接入
- Agnes 适配器目前只接受 3–15 秒单镜;这不是 VideoJob 或脚本层的固定分段规则
- Agnes 的本地图生视频需要调用方另外提供安全的公网图片托管和 URL 解析器
- 跨镜头链路能把真实末帧传入下一镜头,但最终一致性仍取决于视频模型本身
- 原生非 OpenAI 协议的文本平台尚未适配
- 不包含账号系统、数据库、多人协作和商业任务队列
- 当前不支持角色口型同步
- 离线测试不会验证真实 API 的鉴权、稳定性、费用和生成质量;这些仍需使用
  测试者自己的账号单独验收

研究与设计说明见 [研究与设计依据](docs/research-basis.md),代码来源和迁移边界见 [来源说明](docs/provenance.md)。

## 版本说明

- v1.2.0:移除 V2 离线执行运行时(`guided_story_agent/v2/` 及全部 v2 测试),
  正式交付链路只保留 v1 的 VideoProvider 管线。见 [v1.2.0 发布说明](docs/releases/v1.2.0.md)
- v1.1.0:Provider 中性视频管线(多 Provider 注册表、时长参数化、Web/CLI 切换)、
  旁白修复、JSON 修复器强化。见 [v1.1.0 发布说明](docs/releases/v1.1.0.md)
