# 一句话故事视频创意花园 v0.5.0

这是一个低压力的短片创作 Agent。你只需要给出一句模糊方向，AI 一次铺开 8 张完整创意卡；你可以挑选、换一批、找相似方向或混合 1–3 张，也可以什么都不补，先生成完整故事。

```text
一句方向 → 8 张完整创意卡 → 选择 / 相似 / 混合 / 可选故事零件
→ 完整故事 → 因果连续性审查 → 用户确认
→ 自动估算时长 / 15–300秒自定义 → 自适应场景剧本 → 连续性审查 → 用户确认
→ 视觉圣经 → 内容驱动的动态分镜 → 用户确认费用 → 视频生成
```

v0.5.0 取消了固定五节点大纲、固定五场剧本、固定 5–10 镜头模板以及30–60秒总时长限制。故事先作为独立产物生成，并经过一次因果连续性审查；确认后再按自动估算或15–300秒自定义时长改编剧本，剧本生成后逐场检查状态衔接。每个镜头仍保持3–15秒的 Provider 能力约束，并保存首帧、连续动作、结束帧和参考资产 ID。

研究取舍见 [研究与设计依据](docs/research-basis.md)，代码迁移边界见 [来源说明](docs/provenance.md)。

## 安装

```powershell
Set-Location "V:\term_3\科研实践附件\科研实践3\guided-story-video-agent"
python -m pip install -e ".[web,narration]"
Copy-Item .env.example .env
notepad .env
```

这是独立的新项目目录，旧项目 agent开发/.env 不会被 Python 自动读取。故事和剧本默认使用 DeepSeek 文本接口，视频 Provider 保持独立配置。请确认 API Key 填在当前目录的 .env 中；网页会明确显示当前结果来自真实文本模型、离线演示还是 API 失败后的离线兜底。

不配置密钥时，创意、故事、剧本、分镜和自动测试使用确定性的本地 Agent，不发送网络请求。

## 网页创作

```powershell
python -m guided_story_agent.web_app
```

浏览器打开 `http://127.0.0.1:7860/`。第一屏只需输入一个方向，然后直接点击 2×4 创意卡。聊天、故事零件和改写反馈都是可选的。故事确认后才能生成剧本；分镜页会展示视觉圣经、动态镜头目的、首帧、动作、结束帧和引用资产。视频按钮只有在剧本与分镜确认后可用，并要求再次勾选费用确认。

## 人工 CLI

```powershell
guided-story-cli --require-live-text
```

网页和 CLI 默认根据完整故事自动估算成片时长。需要固定时长时可增加
`--target-seconds 90`，允许范围为15–300秒。

主要命令：

```text
/pick 1 3       选择第1和第3张       /more 2       更多像第2张
/refresh        换一批               /mix         混合已选卡
/expand         展开可选故事零件     /choose ending 3
/auto           AI替我选             /story       生成完整故事
/revise-story   修改故事             /script      确认故事并生成剧本
/revise-script  修改剧本             /back        返回灵感区
/storyboard     接受剧本并生成分镜   /render      付费视频入口
/quit           保存并退出
```

旧 `/suggest`、`/use`、`/draft`、`/outline` 仍是兼容别名，但不会出现在主要引导中。`/render` 默认关闭；只有启动时增加 `--render`，并在终端再次输入精确的 `RENDER`，才会调用视频 Provider。

## 自演与真实链路验收

离线运行一句话完整链路，不生成视频：

```powershell
python scripts/run_guided_story_selfplay.py --output outputs/selfplay_auto
```

要求每次文本调用都成功走真实 DeepSeek 接口，任何本地降级都判失败：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --output outputs/live_text --require-live-text
```

只有以下命令会进入真实视频链路：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --output outputs/live_video --require-live-text --render
```

输出包括 `transcript.json`、`ideas.json`、`selection.json`、`story.json`、`script.json`、`storyboard.json`、`prompt_log.json`、`session.json` 和 `bench.json`。

Bench 记录创意多样性、重复率、选择保留率、AI 补全透明率、必要文字输入数、到达故事和剧本的点击数、动态场景数、时长、视觉锚点和镜头多样性。

## 安全边界

- 只要有一句初始方向就能生成完整故事，不再检查轮数或固定剧情节点。
- 默认时长由完整故事的正文、事件、人物和地点密度估算；自定义范围为15–300秒。
- 真实文本链路在故事和剧本生成后各增加一次低温度连续性审查；审查失败时保留第一遍产物并明确记录降级。
- 用户选中的卡片和故事零件在模型输出后再次由本地规则校验和写回。
- AI 补全字段保留来源标签；故事改写生成新版本，返回灵感区不会覆盖旧故事。
- 故事、剧本、分镜和费用分别设置确认边界，自动测试与 CI 不调用付费视频 API。
- 视频生成逐镜头保存进度；单镜头失败不会丢弃其他成功镜头，再次执行时只重试失败镜头。
- `--require-live-text` 禁止真实文本测试悄悄降级到本地规则。
- schema v4 可以迁移 v0.2/v0.3 会话，不覆盖旧文件。

## 测试

```powershell
python -m pytest -q
python -m compileall -q guided_story_agent scripts tests
python -m ruff check .
git diff --check
```

覆盖一句话 8 卡、选择与混合来源、可选故事零件、故事版本、故事与剧本连续性审查、自动/自定义时长、超过12镜头的长分镜、内容驱动分镜、视觉圣经、三段式镜头 Prompt、失败镜头续跑、schema 迁移、人工 CLI、Gradio `process_api` 和付费门禁。

## 项目边界

- 这是可复现的研究 MVP，不包含账号、数据库、多人协作和商业任务队列。
- 故事与剧本使用 DeepSeek 的 OpenAI 兼容接口；视频生成使用独立 Provider。真实接口是否可用，需要用户显式运行在线测试确认。
- Edge TTS 旁白和字幕在视频请求前准备；当前不做角色口型同步。
- 本项目独立于已冻结的 `interactive-movie-agent v0.1.0`。
