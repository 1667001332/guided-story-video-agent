# 引导式剧本到视频 Agent

这是一个面向 GitHub、简历和科研实践的轻量 Agent 项目。它不要求用户一开始就会写完整提示词，而是从一句故事开头开始，逐轮追问缺少的人物目标、冲突、发展、转折和结局；大纲确认后，再继续补齐人物外形、场景、道具、旁白和镜头承接，最终输出定时剧本、结构化分镜和可选的真实视频。

```text
用户给出开头
  -> AI 动态提问并提取剧情事实
  -> 至少 5 轮 + 明确结局 + 用户主动完成
  -> 大纲预览 / 确认
  -> 制作细节追问
  -> 30–60 秒定时剧本 / 确认
  -> 结构化分镜 / 确认
  -> Edge TTS 旁白 + SRT
  -> 逐镜头 Agnes 视频生成
  -> FFmpeg 拼接、音频与字幕封装
```

## 与上一版的关系

本项目是独立仓库，不包含旧版的 `WorldState`、导演命令解析、剧情分支和互动节点记忆。视频 Provider、结构化镜头数据、断点续跑与 FFmpeg 拼接思路迁移自同作者的 `interactive-movie-agent v0.1.0`，并在这里针对“引导创作”重新实现。详见 [docs/provenance.md](docs/provenance.md)。

## 安装

```powershell
python -m pip install -e ".[web,narration]"
Copy-Item .env.example .env
```

如需使用 Agnes 文本或视频模型，只在本地 `.env` 中填写
`AGNES_API_KEY` 的值，不要提交该文件。

没有密钥时，核心状态机、测试和自演流程使用确定性的本地 Agent，不会发送网络请求。

## 网页演示

```powershell
python -m guided_story_agent.web_app
```

网页只是核心状态机的薄界面。所有阶段门禁都在 `GuidedStorySession` 中执行，无法通过前端按钮绕过。

## LLM 自演测试

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 45 --max-turns 12
```

配置 Agnes 文本 API 后：

- 创作者角色由 LLM 生成开头并逐轮回答；
- 引导角色使用隔离提示词提取事实并生成下一条问题；
- 输出完整 transcript、outline、script、storyboard、session 和 bench JSON；
- 默认不会调用视频 API。

只有显式增加 `--render` 才允许逐镜头生成真实视频：

```powershell
python scripts/run_guided_story_selfplay.py --target-seconds 30 --max-turns 12 --render
```

## 确认和付费边界

- 少于 5 条有效创作输入不能生成大纲。
- 没有明确结局不能生成大纲。
- 大纲未确认不能进入剧本阶段。
- 制作细节未补齐不能生成剧本。
- 剧本未确认不能生成分镜。
- 分镜未确认时 `render_confirmed_plan()` 会在 Provider 调用前直接拒绝。
- 自动测试和默认自演永远不会调用付费视频 API。

## 核心接口

```python
session.submit_user_turn(text)
session.build_outline()
session.confirm_outline()
session.answer_detail_question(text)
session.build_script()
session.confirm_script()
session.build_storyboard()
session.confirm_storyboard()
session.render_confirmed_plan(renderer, output_dir)
```

状态机为：

```text
collecting
  -> outline_review
  -> detailing
  -> script_review
  -> storyboard_review
  -> render_ready
  -> completed
```

## 时长、旁白和视频

- 目标成片时长限制为 30–60 秒，默认 45 秒。
- 当前 MVP 使用五个叙事镜头，每镜头必须为 3–15 秒，总时长误差不超过 1 秒。
- Edge TTS 和 SRT 在任何视频请求前生成；TTS 失败时仍保留字幕和结构化分镜。
- 每个视频镜头独立生成和保存，成功镜头可以断点复用。
- FFmpeg 先拼接无声镜头，再封装旁白和字幕轨道。

## 测试

```powershell
python -m unittest discover -s tests -v
python -m compileall -q guided_story_agent scripts
```

测试不会读取真实 API Key，也不会创建视频任务。当前覆盖：

- 五轮和结局门禁；
- 空输入、重复输入；
- 大纲、剧本、分镜三级确认；
- 30、45、60 秒精确分配；
- 旁白先于视频；
- Provider 缺少密钥时零请求失败；
- 逐镜头失败 manifest；
- 自演默认不生成视频；
- Gradio 事件绑定和无浏览器 handler。

## 当前边界

- 项目是研究原型，不提供账号、数据库或多人协作。
- LLM 输出经过本地字段和阶段校验；远程失败时回退本地规则 Agent。
- 当前字幕以 MP4 字幕轨封装，不做角色口型同步。
- 当前镜头数量固定为五个叙事节拍；后续可在不改变状态机的情况下扩展为更细镜头。
