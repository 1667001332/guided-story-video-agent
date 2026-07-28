# 师兄 API 测试指南

本指南用于 `v0.6.0` 的真实 API 验收。请使用测试者自己的 API Key，不要接收或复制项目作者的 `.env`。

## 1. 安装

```powershell
git clone https://github.com/1667001332/guided-story-video-agent.git
Set-Location guided-story-video-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[web,narration,dev]"
Copy-Item .env.example .env
notepad .env
```

视频拼接依赖 FFmpeg。开始视频测试前先确认：

```powershell
ffmpeg -version
```

如果 PowerShell 提示找不到 `ffmpeg`，先安装 FFmpeg 并重新打开终端；不要在缺少 FFmpeg 时开始付费视频批测。

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

Agnes 官方图生视频参数接收公网图片 URL，不接收本地文件路径。若测试者已经用
自己的静态服务器或 CDN 公开托管某个本地目录，可同时填写
`VIDEO_REFERENCE_ROOT` 和 `VIDEO_REFERENCE_BASE_URL`。项目会把当前运行需要的
单张首帧或真实末帧原子暂存到该目录的独立子目录，再生成对应 URL；
`VIDEO_OUTPUT_DIR` 可以放在公开目录外。项目不会公开整个仓库，也不会负责配置
静态服务器或 CDN。

Agnes 的 `image` 只使用明确标记的 `start_frame`，或明确连续动作所依赖的上一镜头
真实末帧；普通同场景切机位会重新构图，不会继承上一镜头画面，也不会把人物定妆照
自动当首帧。未满足公网映射条件时，连续动作链会显示“依赖失败”，
不会在后台偷偷改成文生视频；新场景文本回退会明确写入测试结果和 manifest。
分镜页可以上传图片、选择用途并绑定到具体镜头或视觉资产；确认、删除或替换参考图后
必须重新确认整套分镜。上传缓存默认保存在已被 Git 忽略的
`outputs/visual_inputs`，也可通过 `VISUAL_INPUT_DIR` 修改。复用旧任务的末帧时，
系统会按本次 `VIDEO_REFERENCE_BASE_URL` 重新生成 URL。

## 4. 建议测试内容

1. 普通剧情、悬疑、爱情和科幻各测试至少两次。
2. 检查人物动机、信息来源、证据链和相邻场景衔接。
3. 分别测试自动时长和15–300秒自定义时长。
4. 确认长故事可以生成超过12个镜头。
5. 记录输入文本、网页状态提示、完整错误和 `outputs/live_text` 结果。

反馈时不要提交 `.env`、API Key 或包含凭据的截图。

## 5. 批量测试

仓库提供了 12 条不同题材的示例用例：

```powershell
python scripts/run_batch_test.py `
  --output outputs/mentor_batch
```

省略 `--input` 时使用安装包内的示例题目，因此命令不依赖当前工作目录。也可以通过 `--input` 指定自己的 JSONL、JSON、CSV 或 TXT 文件。

批量测试默认是“严格文本、关闭视频”：每次文本调用都必须走真实 API，发生离线降级会直接记为失败，但不会调用视频 API。每条用例完成后都会立即保存结果，因此中断后可以继续：

```powershell
python scripts/run_batch_test.py `
  --output outputs/mentor_batch `
  --resume
```

`--resume` 只复用运行身份完全一致且已成功的结果。题目、时长、文本 Provider/模型、严格文本模式或视频设置发生变化时会重新执行；缺少运行身份的旧结果也不会被当作本次严格验收成功。

常用参数：

- `--repeat 3`：每条题目重复三次。
- `--max-cases 2`：先用前两条进行小规模试跑。
- `--target-seconds 90`：覆盖用例文件中的时长。
- `--retries 1 --delay-seconds 2`：失败后重试一次，并控制请求频率。
- `--offline`：完全不请求文本 API，仅用于快速检查脚本和输出目录。
- `--allow-fallback`：仍会优先请求真实 API，但允许失败后离线兜底，不用于严格验收。

建议先做无网络、无付费的离线冒烟：

```powershell
python scripts/run_batch_test.py `
  --offline `
  --max-cases 2 `
  --output outputs/offline_batch
```

`--offline` 不读取或请求真实文本 API；未添加 `--render` 时也绝不会调用视频 API。

结果目录包含：

- `summary.json`：总体成功率、失败类型、耗时分位数和平均结构指标。
- `results.csv`：Excel 可直接打开的逐条结果。
- `results.jsonl`：完整的机器可读逐条结果。
- `runs/<用例>/attempt-N/`：每次执行的诊断；失败目录也会真实存在。
- 首次成功生成的故事、剧本和分镜会保存在对应 attempt 目录；视频重试和
  `--resume` 会加载同一 Session、分镜、任务 ID 和视频目录，只继续失败或
  pending 的镜头。
- `render_manifest.json` 会记录每个镜头的固定参考图、首帧、上游镜头、
  连续性模式、输入指纹、seed、生成末帧、状态和错误；汇总可区分重新生成、
  复用、依赖失败和无参考文本回退。
- Resume 身份包含代码、Prompt 和本地规则内容指纹；实现发生变化后不会把旧成功
  误当成本轮结果。

批量视频会产生明显费用，只有下面这种带双重确认的命令才会调用视频 API：

```powershell
python scripts/run_batch_test.py `
  --output outputs/mentor_video_batch `
  --max-cases 1 `
  --render `
  --confirm-paid-video RENDER
```

首次付费测试务必保持 `--max-cases 1`。`succeeded_with_warnings` 会计入成功但在汇总中单独列出警告；`pending` 不计成功，会保留已有请求信息供续试。如果提交响应超时且无法取得远端任务 ID，状态会变为 `submission_uncertain`，系统不会自动重提；应先到 Provider 后台核对，避免重复计费。不要把离线或未渲染结果目录通过 `--resume` 冒充为严格文本或视频验收结果。
