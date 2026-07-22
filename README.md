# 引导式剧本到视频 Agent v0.2

这是一个“人主导、AI 辅助诊断”的短片共创 Agent。它从用户的一句话开始，每轮只处理当前最重要的故事缺口，并给出三条可以采用、忽略或改写的方向。故事圣经、大纲、剧本和分镜都可以编辑、局部重写、撤销、重做和确认。

```text
自由开头 → 动态故事教练 → 至少 5 轮且因果链完整
→ 可编辑五节点大纲 → 制作细节 → 可编辑定时剧本
→ 5–10 个动态分镜 → 质量检查 → 用户确认费用 → 视频生成
```

## v0.2 改变了什么

- 一次文本调用同时完成事实提取、冲突检测、完整度诊断、一个问题和三条建议，不再用固定字段轮询。
- 每项事实保留用户原句证据和置信度；建议只有被采用后才进入故事。
- 故事圣经是事实唯一来源，所有产物保存版本、父版本、反馈、来源轮次和确认状态。
- 五节点大纲、可拍摄剧本和逐镜头分镜都可编辑；确认后修改会产生新版本并重新进入审阅。
- 分镜数量按节奏动态规划：30 秒 5 镜头、45 秒 8 镜头、60 秒 10 镜头；每镜头 3–15 秒。
- 网页改为左侧共创对话、右侧工作区；新增真正的人工交互 CLI。
- 自演 Bench 新增问题重复率、事实保留率、冲突处理率、因果完整度、视觉锚点覆盖率和镜头多样性。

架构参考与取舍见 [研究与设计依据](docs/research-basis.md)；迁移来源见 [代码来源说明](docs/provenance.md)。

## 安装

```powershell
Set-Location "V:\term_3\科研实践附件\科研实践3\guided-story-video-agent"
python -m pip install -e ".[web,narration]"
Copy-Item .env.example .env
notepad .env
```

不配置密钥时，核心状态机、网页测试和自演使用确定性的本地 Agent，不发送网络请求。

## 人工 CLI

真实文本 API 严格模式：

```powershell
guided-story-cli --target-seconds 30 --require-live-text
```

常用命令：

```text
/suggest  获取三个方向       /use 2   采用第二条
/show     查看故事圣经       /outline 生成大纲候选
/edit     修改当前产物       /revise  按反馈局部重写
/review   质量检查           /undo    撤销
/redo     重做               /confirm 确认并推进
/render   付费生成入口       /quit    保存退出
```

`/render` 默认关闭。只有启动时增加 `--render`，且之后再次输入精确的 `RENDER`，才会调用视频 Provider。

## 网页工作台

```powershell
python -m guided_story_agent.web_app
```

浏览器打开 `http://127.0.0.1:7860/`。左侧完成对话、采用或忽略方向；右侧编辑故事地图、大纲、剧本和分镜。真实视频还需要勾选费用确认。

## 自演与真实文本验收

离线、自演、不生成视频：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 45 --max-turns 12 --output outputs/selfplay_45
```

要求每次文本调用都成功走真实 Agnes 接口，任何本地降级都判失败：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --max-turns 12 --output outputs/live_text --require-live-text
```

只有以下命令会进入真实视频链路：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --max-turns 12 --output outputs/live_video --require-live-text --render
```

输出包括 `transcript.json`、`story_bible.json`、`revisions.json`、`outline.json`、`script.json`、`storyboard.json`、`prompt_log.json`、`session.json` 和 `bench.json`。

## 安全门禁

- 少于 5 轮、缺少开头/目标/冲突/发展或转折/结局、存在未解决冲突时不能生成大纲。
- 用户必须主动生成大纲；模型不能自行越级。
- 大纲、剧本和分镜必须依次确认。
- 预览、拒绝建议、候选重写均不覆盖已确认版本。
- 自动测试和 CI 全部离线，绝不调用付费视频 API。
- `--require-live-text` 禁止悄悄使用本地 fallback。

## 测试

```powershell
python -m unittest discover -s tests -v
python -m compileall -q guided_story_agent scripts tests
python -m ruff check .
git diff --check
```

覆盖动态多事实提取、缺口选择、建议采用、冲突保护、三级编辑与版本回退、30/45/60 秒动态分镜、v0.1 只读迁移、人工 CLI、Gradio 事件和费用门禁。

## 项目边界

- 这是可复现的研究 MVP，不包含账号、数据库、多人协作和商业任务队列。
- 使用 Agnes 兼容文本接口和既有视频 Provider；真实接口可用性仍需用户显式运行在线测试确认。
- Edge TTS 旁白和字幕在视频请求前准备；当前不做角色口型同步。
- 本项目独立于已冻结的 `interactive-movie-agent v0.1.0`，不会修改旧仓库。
