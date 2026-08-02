# guided-story-video-agent V2 架构审计与迁移方案

本文记录从当前 0.6 过渡架构迁移到 V2 的架构决策。V2 的目标不是把旧的
`Storyboard` 代码换一个名字，而是重新建立职责边界：

```text
CreativeBrief -> DirectorAgent -> MoviePlan -> FilmIR -> MovieIR -> VideoJob -> VideoProvider -> Artifacts
```

## 1. 当前仓库审计

### 1.1 仓库和验证基线

- 代码库分析器扫描到 1199 个文件，其中 873 个 Python 文件；绝大多数来自
  `outputs/`、缓存和历史实测产物。真正的运行包位于 `guided_story_agent/`，
  测试位于 `tests/`。
- 当前分支是 `codex/v0.5.1-provider-config`，工作树已经包含此前的 v0.6
  直出 `VideoJob` 改动。本次迁移不回滚、不覆盖这些用户已有改动。
- `tests/test_video_job.py`、`tests/test_timing_storyboard.py`、
  `tests/test_package_resources.py` 当前可独立通过，共 31 个测试。
- 从仓库根目录直接运行完整 `pytest` 会递归碰到受 ACL 保护的历史输出目录，
  在测试收集阶段产生 `PermissionError: outputs/_pytest_video_job`；后续每个
  迁移切片使用显式 `tests/...` 目标运行，并增加 V2 专门测试。

### 1.2 当前真实调用链

CLI 入口在 `guided_story_agent/cli.py`：

1. `main()` 选择 `RuleBasedStoryAgent` 或 `OpenAIStoryAgent.from_env()`。
2. `run_interactive()` 创建 `GuidedStorySession`，读取方向。
3. `/story` 调用 `session.generate_story()`。
4. `/script` 先确认故事，再调用 `session.generate_script()`。
5. `/video` 调用 `session.confirm_script()` 和 `session.build_video_job()`。
6. `/render` 创建 `VideoJobRenderer`，再调用 `session.render_confirmed_video()`。
7. `VideoJobRenderer` 读取 Provider capability，写入提交意图，调用
   `provider.generate_video(job, ...)`，最后保存 `RenderManifest`。

当前默认路径已经不再调用 `plan_storyboard()`；但旧路径仍由 Session、Web、Selfplay
和测试保留，因而“默认直出”和“旧导演兼容层”同时存在。

### 1.3 职责错误矩阵

| 当前模块 | 当前行为 | V2 判断 | 迁移方向 |
| --- | --- | --- | --- |
| `models.py` | brief、story、script、storyboard、artifact、render 状态混在一个文件 | 领域模型和兼容模型耦合 | 拆成 `domain` 模块；旧类型只做读取兼容 |
| `agent.py` | Idea、Story、Script、Storyboard、fallback、LLM 调用混在一起 | Agent 边界不清；旧 Agent 仍在做导演 | 新增 `DirectorAgent` port 和 LLM adapter |
| `timing.py` | `minimum=3`、加权分配、场景数量和可读时长推断 | Python 在替导演决定节奏 | 迁移为只读 Validator |
| `storyboard.py` | 推断镜头单元、镜头数、相机、过渡、连续性和 Prompt | 隐藏的第二导演 | V2 不进入主链；历史会话只读迁移 |
| `video_job.py` | `join()`、字符串拼接和默认负面 Prompt | Python 重新创作 Provider Prompt | 由 MovieIR Compiler 投影为 VideoJob |
| `rendering.py` | 完整 VideoJob renderer 与旧分镜渲染/旁白/FFmpeg 混在一起 | 执行层和旧导演层耦合 | 保留 `VideoJobRenderer`，隔离 `StoryRenderer` |
| `video_provider.py` | Agnes 将 VideoJob 包装为 `StoryboardShot` | Provider 仍知道旧 Storyboard | Agnes 直接实现 `generate_video(VideoJob)` |
| `session.py` | 同时是状态机、导演、校验器、迁移器和渲染编排器 | 生产数据库与编排没有分层 | 引入 ProductionSession facade |
| `web_app.py` | UI 仍暴露旧镜头修改和 retake | Retake 可能修改剧本事实 | 改为只提交 `RetakeRequest` |
| `selfplay.py`、`batch_test.py` | benchmark 仍强制构建 storyboard | 评测路径和产品路径不一致 | 先双写 MoviePlan，再删除旧 artifact |
| prompts | Story、Script、Storyboard 多套互相修补 | 内容生成分散 | 收敛为 Director MoviePlan schema Prompt |

### 1.4 必须从 V2 主链删除的本地导演规则

- `scene_count = max(2, round(...))`；
- `maximum_scenes = target_seconds // 3`；
- `_normalize_script_durations()`、`fit_scenes_to_duration()`；
- `plan_scene_durations()`、`allocate_weighted_durations()`；
- `storyboard.py` 中的 `CAMERA_BY_KIND`、`_plan_shot_units()` 和 Prompt 函数；
- `minimum=3`、`max=15` 的核心层判断；
- 把非法脚本“修复”为可接受脚本的 Session 逻辑。

V2 可以保留“验证”，不能保留“修复”：发现非法 MoviePlan 时返回结构化错误，
由 DirectorAgent 重新生成；Python 不改变场景数、时长、顺序、镜头或 Prompt。

## 2. V2 目标架构

### 2.1 分层

最终目录建议为：

```text
guided_story_agent/
  domain/       # brief, movie_plan, execution, review
  application/  # director orchestration, production service, validators
  ports/        # DirectorAgent, VideoProvider, Repository
  adapters/     # OpenAI Director, Agnes/Veo/Sora/Kling adapters, JSON repository
  compatibility/# legacy session/storyboard readers
```

本次第一阶段先以 `guided_story_agent/v2/` 建立不影响旧导入路径的契约层，后续按
验证结果迁移到最终目录。

### 2.2 CreativeBrief、DirectorAgent、MoviePlan

V2 Brief 只保存 `target_duration_seconds`、`video_type`、`visual_style`、`audience`、
`narration_requirement`、`output_format`。旧的 `duration_mode`、`resolved_target_seconds`、
`legacy_facts` 只在兼容读取时使用。

`DirectorAgent` 是唯一内容规划端口：

```python
create_movie_plan(brief, direction, feedback) -> MoviePlan
revise_movie_plan(brief, plan, feedback) -> MoviePlan
```

`MoviePlan` 是唯一可信规划，包含 Story、Script、ScenePlan、CameraPlan、TimingPlan、
ContinuityPlan、CharacterSheet、NarrationPlan、MusicPlan、TransitionPlan、VisualStyle、
ShotPlan、FilmBeats、EmotionCurve 和 ReviewCriteria。它不包含 Provider/API 字段；所有引用使用稳定
`scene_id`/`character_id`。执行提示由 MovieIR→VideoJob Compiler 生成，MoviePlan 不下沉到执行层。

Script 场景至少包含：`goal`、`emotion`、`importance`、`estimated_duration_weight`、
`minimum_duration`、`camera_language`、`motion_type`、`dialogue`、`narration`、
`characters`、`location`、`continuity_requirements`、`transition`、`timing_reason`。

### 2.3 Timing、VideoJob、Provider

Timing Validator 只检查：

1. 场景 ID、引用和必填字段完整；
2. TimingPlan 声明总时长与 Brief 目标一致；
3. 每个声明时长为正，且不小于 LLM 给出的 `minimum_duration`；
4. 选定 Provider 是否接受整个 VideoJob；
5. 是否超出预算。

它不增加、删除、合并场景，不分配时长，不归一化，也不生成 Prompt。失败时抛出
`DirectorOutputRejected`，把错误反馈送回 DirectorAgent，最多重试配置次数。

VideoJob 只含 Compiler 产出的 Provider Prompt、Negative Prompt、完整任务 duration、
reference/character reference、output format 和可追踪 metadata。Provider 只接受
`VideoJob`，不导入 Story、Script、Storyboard 或 MoviePlan。

FilmIR 位于 MoviePlan 与 MovieIR 之间：它承载导演决定的 cinematic beat、戏剧目的、
观众前后状态、视觉焦点、必需理解、人物情绪、连续性/转场/旁白/音乐意图和 beat-level
验收标准。它不包含 Provider/API 字段，也不生成 Prompt。

MovieIR 位于 FilmIR 与 VideoJob 之间：它把导演已声明的 shot-level visual beat、
timeline、continuity anchor、character anchor、narration/subtitle cue 和 acceptance
criteria 降级成 Provider-neutral 执行结构。它不直接读取 MoviePlan，也不包含 API payload、task id、URL 或计费
字段；Provider Adapter 留到后续阶段。

### 2.4 Production Session 和 Retake

Session 最终保存：`brief`、`movie_plan_revisions`、`confirmed_movie_plan`、
`video_job_revisions`、`provider_jobs`、`artifacts`、`reviews`、`retakes`、`history`。

Retake 只能追加摄影、灯光、表演、镜头语言要求，并基于同一 MoviePlan 新建 VideoJob；
不能修改 Story 或 Script。任何上游变更都产生新 revision，不覆盖已提交任务。

## 3. 分阶段 Migration Plan

### Phase 0：基线与边界（已完成）

文件：`docs/architecture-v2-migration.md`、`guided_story_agent/v2/*`、
`tests/test_v2_contracts.py`。动作：建立 V2 domain contract、DirectorAgent port、
只读 Validator、Provider-only VideoJob DTO；不改 CLI 默认路径，不删除旧模块。

合格条件：V2 契约测试通过；旧的 31 个定向测试仍通过；V2 没有任何 `minimum=3`、
`scene_count` 或 Prompt 拼接。

### Phase 1：导演主链双轨运行（本次完成）

文件：`guided_story_agent/v2/director.py`、`guided_story_agent/v2/openai_director.py`、
`guided_story_agent/v2/models.py`、`guided_story_agent/v2/validation.py`、`session.py`、
`cli.py`、`prompts/v2/director/movie_plan.md` 以及 V2 专项测试。
已增加 `--v2` feature flag、OpenAI Director adapter、一次性完整 MoviePlan 生成、
严格 schema 校验/拒绝重试、`confirmed_movie_plan` 持久化；legacy Story/Script 保留且
默认 CLI 路径不变。本阶段尚未接 Compiler 或 Provider，因此不会误称已经生成视频。

### Phase 2：MoviePlan → Compiler → VideoJob（兼容 facade）

文件：`guided_story_agent/v2/compiler.py`、`guided_story_agent/v2/execution.py`、
`guided_story_agent/v2/validation.py`、`session.py`、`cli.py`、
`tests/test_v2_compiler.py`。

`MoviePlanCompiler` 现在只是兼容 facade，内部明确经过 FilmIR 和 MovieIR；真正的
VideoJob 编译器只读取 MovieIR，校验结构和 ProviderCapabilities，再生成执行 Prompt、
duration、references 和 metadata。它不调用 Provider，不修改 MoviePlan，不使用旧
`timing.py` 分配时长，不使用旧 `storyboard.py` 推断镜头，也不自动拆分超长视频。
能力不满足时返回结构化 `CompileError`。

V2 Session 新增 `v2_video_job` 和 `VIDEO_JOB_COMPILED` 阶段；旧 `video_job.py` 仍是
legacy Story/Script 链路的兼容层。CLI 的 `/compile` 只保存 VideoJob，`/render` 明确
不执行 Provider。

### Phase 3A：MoviePlan → MovieIR → VideoJob（历史兼容切片）

文件：`guided_story_agent/v2/ir.py`、`guided_story_agent/v2/ir_builder.py`、
`guided_story_agent/v2/models.py`、`guided_story_agent/v2/compiler.py`、`session.py`、
`cli.py`、`tests/test_v2_movie_ir.py`。

MoviePlan 是导演意图；MovieIR 是 Provider-neutral 的电影中间表示，包含 timeline、
shot-level visible action、camera/motion、continuity anchors、character anchors、
references、narration/subtitle/music cues 和 acceptance criteria；VideoJob 只保存
可执行的执行参数，不包含 HTTP payload。MovieIR 不包含 Provider/API/task 字段。

历史 Session 可以只读加载旧的 MovieIR；新的 V2 构建不再让 MovieIRBuilder 直接读取
MoviePlan。

### Phase 3A.5：MoviePlan → FilmIR → MovieIR → VideoJob（本阶段）

文件：`guided_story_agent/v2/film_ir.py`、`guided_story_agent/v2/film_ir_builder.py`、
`guided_story_agent/v2/ir_builder.py`、`guided_story_agent/v2/models.py`、
`guided_story_agent/v2/openai_director.py`、`session.py`、`cli.py`、
`tests/test_v2_film_ir.py` 及相关 V2 测试。

新链路为：

```text
MoviePlan
    ↓ FilmIRBuilder（只校验/确定性投影）
FilmIR
    ↓ MovieIRBuilder（只接受 FilmIR）
MovieIR
    ↓ VideoJobCompiler（只接受 MovieIR）
VideoJob
```

Director Prompt 必须输出覆盖全部 shot 的 `film_beats`。缺少 beat、beat 无法追溯到
scene/shot、缺少 viewer state、required evidence 或 acceptance criteria 时，
`FilmIRBuilder` fail-closed，不自动补剧情、补镜头或重算节奏。Session 新增
`film_ir`、`FILM_IR_BUILT` 和对应 revision；CLI 顺序是 `/build-film-ir` → `/build-ir`
→ `/compile`。`VideoJob` 透传 `source_film_ir_id`，但仍不包含 HTTP payload。

本阶段不调用真实视频 API、不 submit/poll/download、不生成 MP4，也不写 Provider Runtime。

### Phase 3C: Validator and Pass Pipeline（本次完成）

本阶段把 IR 的“构建”和“验收/诊断”拆成独立职责。Builder 只负责把已经确认的
`MoviePlan` 或 `FilmIR` 降级为下一层的结构；它不修复场景数量、时长、镜头顺序或
Prompt。Validator 只报告结构化问题，发现硬错误即 fail-closed，由上游 DirectorAgent
重新生成，而不是由 Python 猜测导演意图。

新增链路为：

```text
MoviePlan
  -> FilmIRBuilder -> FilmIRValidator -> FilmIRPassPipeline
  -> MovieIRBuilder -> MovieIRValidator -> MovieIRPassPipeline
  -> VideoJobCompiler -> VideoJob
```

完整生产边界保持为：

```text
CreativeBrief -> DirectorAgent -> MoviePlan
  -> FilmIRBuilder -> FilmIRValidator -> FilmIRPassPipeline -> FilmIR
  -> MovieIRBuilder -> MovieIRValidator -> MovieIRPassPipeline -> MovieIR
  -> VideoJobCompiler -> VideoJob -> Provider Runtime（后续阶段）
```

`ValidationIssue` 统一携带 `code`、`message`、`path` 和 `severity`；Validator
覆盖 source id、beat/shot 引用、必填字段、时间线连续性、总时长、Provider/API/task
字段泄漏以及 Prompt stuffing。Pass 只做显式、可审计的 IR 变换或诊断：FilmIR 负责
beat 顺序和观众理解度，MovieIR 负责 shot timeline、continuity 和 prompt leakage。
硬错误不会被 Pass 吞掉，warning 会保留在 Session 中。

Session 现在保存每次构建的 validator issues 和 pass diagnostics，并在旧字段缺失时
使用空列表加载，保持历史 Session 兼容。CLI 仍使用 `/confirm-plan`、`/build-film-ir`、
`/build-ir`、`/compile`、`/render`；本阶段没有加入 Provider Runtime，不会 submit/poll/download，
也不会生成 MP4。

后续 Phase 3D 才把 `VideoJob` 接入 Provider Runtime（ProviderJob、轮询、下载和
Artifact 持久化）；Phase 4 再移除旧导演逻辑。两者都必须继续以本阶段的 IR 验证和
fail-closed 边界为前置条件。

### Phase 3C.5: Movie Compiler Creative Layer（本阶段完成）

本阶段在 3C 的结构化 Validator / Pass 之上增加三层能力：

```text
MoviePlan
  -> RuleBasedDirectorRevisionLoop（可插拔，离线不伪造内容）
  -> FilmIRBuilder -> FilmIRValidator
  -> CreativePassPipeline -> FilmIROptimizer
  -> MovieIRBuilder -> MovieIRValidator
  -> CompilerPassPipeline -> MovieIROptimizer
  -> VideoJobCompiler -> VideoJob
```

`CreativePassPipeline` 只理解电影语言：节奏、情绪交接、视觉母题、观众理解和冲突
清晰度；`CompilerPassPipeline`（原 MovieIR Pass Pipeline）只处理执行 IR 的时间线、
连续性和 Provider 字段泄漏。前者不生成镜头 Prompt，后者不重新解释剧情。

Validator 是硬约束检查器，发现非法结构就拒绝；Optimizer 是可追踪的建议器，返回
`before_ir`、`after_ir`、diagnostics 和 transformation candidates。本阶段默认不执行
合并/删除镜头，只记录 `code/message/path/before/after/reason/severity`，为后续策略化
优化保留接口。

`DirectorRevisionLoop` 是 DirectorAgent 的编排边界，不是本地导演。离线
`RuleBasedDirectorRevisionLoop` 遇到硬错误只生成 revision request 并 fail-closed；遇到
warning 只记录建议，不伪造 viewer state、beat、冲突或时长。真实 LLM revision 留到后续
阶段。

当前 FilmIR 仍保留 `camera/motion/lighting/composition`，因为 MovieIRBuilder 需要可追踪
的导演 lowering hint；这些字段不是 Provider payload，也不能被 Creative Pass 或 Optimizer
改写。未来若拆分为更薄的 FilmIR，应通过版本化 IR 迁移完成，而不是在本阶段破坏旧 Session。

Session 新增 `creative_pass_diagnostics`、`film_ir_optimizer_diagnostics`、
`movie_ir_optimizer_diagnostics`、`director_revision_history` 和
`director_revision_stop_reason`，旧 Session 缺失时默认空列表或 `None`。CLI 用户流程不变，
额外提供 `/diagnostics` 查看这些记录。本阶段仍不接 Provider Runtime，不 submit/poll/download，
不生成 MP4；`VideoJob` 仍是未来 Runtime 的唯一输入边界。

### Phase 3B：Production Database、Provider Runtime 和 Retake

文件：`session.py`、`domain/review.py`、`application/production_service.py`、`web_app.py`。
实现 append-only revision、ProviderJob、Artifact、Review、Retake history。Retake 后
Story/Script digest 不变，VideoJob revision 增加；pending/submission_uncertain 不重复提交。

### Phase 4：移除旧导演逻辑

文件：`storyboard.py`、`timing.py`、`continuity.py`、旧 prompts、旧 Web/selfplay artifact。
先标记 compatibility-only，再删除生产引用；保留一次性 legacy reader，不再支持旧模块
写入新状态。`rg` 必须找不到生产路径中的 `minimum=3`、`maximum_scenes`、
`fit_scenes_to_duration`、`refresh_shot_prompts`。

## 4. 风险与测试方案

| 风险 | 控制措施 |
| --- | --- |
| LLM 输出不完整/超时 | schema 校验、结构化错误、有限重试、保存原始响应和 provenance |
| Provider 只支持短视频 | capability adapter 明确 reject；业务层不切片 |
| 历史 Session 只有 Storyboard | 只读加载并标记 legacy，不伪装成 V2 MoviePlan |
| UI 仍假设镜头列表 | V2 UI 只展示 MoviePlan revision 和 Retake overlay |
| Prompt 不可审计 | 保存 provider key、prompt digest、model、raw response provenance |
| 双轨状态不一致 | `confirmed_movie_plan` 是 V2 真源，legacy 字段只读投影 |
| 历史 outputs 阻塞 pytest | 显式测试收集；修复 ACL 后再跑全量 |

每个阶段执行四层测试：domain contract、application orchestration、adapter contract、
legacy compatibility。任何“测试通过”都必须注明命令和结果，并区分代码缺陷、环境 ACL
和过期断言。

## 5. 本次切片完成定义

Phase 1、Phase 2、Phase 3A、Phase 3A.5、Phase 3C 和 Phase 3C.5 已分别验证：导演输出
MoviePlan 与 film beats；Builder 只转换已有导演决策，Validator 只报告结构化问题，
Creative / Compiler Pass 只执行显式 IR 变换或诊断，Optimizer 默认只记录建议；
VideoJobCompiler 只投影 MovieIR；Provider 仍未被调用。下一步才是
`VideoJob -> Provider Runtime -> ProviderJob -> Artifacts`。

### Phase 4A：StoryPlan / DirectorPlan 分层（本阶段）

本阶段把导演输出中的两类决策显式分层，但保留旧 `MoviePlan` 作为兼容聚合对象：

```text
CreativeBrief
  -> StoryPlan       # 发生什么：人物、事件、因果、冲突、 stakes、结局、故事节拍
  -> DirectorPlan    # 如何体验：节奏、悬念、观众信息、揭示、视觉母题、高潮与结尾语气
  -> MoviePlan       # 兼容聚合：保留旧 Story/Script/Camera/Timing 等字段
  -> FilmIRBuilder
```

`MoviePlan.story_plan` 和 `MoviePlan.director_plan` 是新的显式边界。旧 JSON
缺少这两个字段时，读取过程会从已有 `Story`、`Script`、`CharacterSheet` 和
`FilmBeatPlan` 做确定性投影；不会重新编写故事、分配时长或生成 Prompt。旧的
构造函数和字段仍然可用，因此旧 Session、旧测试和旧链路不需要一次性切换。

两个计划都只包含电影语义，不允许出现 Provider/API/payload/task/model/endpoint
字段，也不允许出现 `masterpiece`、`best quality`、`ultra realistic` 等 Prompt
stuffing。`FilmIRBuilder` 仍只做确定性 lowering，并把两个层的来源标记写入 IR
metadata；`VideoJob` 和 Provider Runtime 不在本阶段实现。

OpenAI Director JSON 现在可以显式返回 `story_plan` 与 `director_plan`；解析器对
嵌套字段执行严格白名单校验。离线 `RuleBasedDirectorAgent` 同样返回两层对象，
因此离线 Session 与真实 DirectorAgent 使用同一数据契约。
`VideoJob → Provider Runtime → ProviderJob → Artifacts`。
### Phase 4B：Creative Analysis Layer（本阶段）

Phase 4B 在 MoviePlan 和 FilmIR 之上增加只读创意分析，不改变任何输入：

```text
MoviePlan (StoryPlan + DirectorPlan)
        + FilmIR
        ↓
CreativeAnalysisPipeline
        ↓
CreativeAnalysisResult + CreativeGraph artifacts
```

`CreativeAnalysis` 只读取故事层、导演层和电影语义 IR，输出诊断、指标和图结构。
它不生成故事、不补导演意图、不修改 MoviePlan、FilmIR 或 MovieIR，也不访问
Provider。`CreativePass` 仍是 FilmIR 内部的诊断/变换边界；`Optimizer` 产生建议或
候选变换；`RevisionLoop` 负责未来把问题交回 DirectorAgent。三者都不会被 Analysis
替代。

本阶段提供五类分析：`EmotionFlowAnalysis`、`AudienceKnowledgeAnalysis`、
`ConflictProgressionAnalysis`、`CharacterArcAnalysis` 和
`PlanLayerConsistencyAnalysis`。后者专门比较 Phase 4A 新层与 legacy Story、
CharacterSheet、Script、FilmBeat、visual_style，发现漂移但不自动同步。

`CreativeGraph` 只允许电影语义节点和边，序列化前会拒绝 Provider/API/payload/
endpoint/model/task 字段及 Prompt stuffing。Session 新增
`creative_analysis_results`、`creative_analysis_diagnostics`、
`creative_analysis_artifacts`、`creative_analysis_metrics`；旧 Session 缺少这些
字段时默认空集合，MoviePlan、FilmIR、MovieIR 和 VideoJob 仍可独立加载。

CLI 新增 `/analysis`，允许只有 MoviePlan 时运行部分分析，也允许在 FilmIR 已构建后
运行完整分析；`/diagnostics` 会显示分析诊断、artifact 数量和 metrics。该阶段仍不
实现 Creative Optimizer、真实 Director Revision、Provider Runtime、submit/poll/
download 或 MP4 生成。VideoJob 仍是未来 Provider Runtime 的唯一输入边界。

### Phase 4C：Creative Optimizer 与 Revision Request Layer（本阶段）

Phase 4C 将只读创意分析的诊断转换为可追踪、不可自动执行的导演建议：

```text
CreativeAnalysisResult + CreativeGraph
        ↓
CreativeOptimizer
        ↓
OptimizationSuggestion + TransformationCandidate
        ↓
RevisionRequestBuilder
        ↓
CreativeRevisionRequest
        ↓
DirectorAgent（下一阶段接入）
```

`CreativeOptimizer` 与 `FilmIROptimizer` / `MovieIROptimizer` 分开。前者只处理
情绪峰值、观众信息、冲突升级、角色弧线和计划层一致性；后者只处理 IR 的确定性
执行风险。`OptimizationSuggestion` 是导演面对的建议，`TransformationCandidate`
保留 `before/after/reason/risk/confidence` 追踪字段，但本阶段 `executable` 永远为
false。优化器不会写回 MoviePlan、StoryPlan、DirectorPlan、FilmIR 或 MovieIR。

`RevisionRequestBuilder` 根据硬问题、警告和 deferred 策略将相关建议合并成
`CreativeRevisionRequest`。请求只描述 target、instruction、preserve、avoid 和
证据来源，`requires_director=true`、`auto_apply_allowed=false`；它不是导演输出，
也不包含 Provider/API/payload/task/model/endpoint 或 Prompt stuffing。硬 Validator
问题仍然 fail-closed；创意建议本身不改变编译结果。

`RuleBasedDirectorRevisionLoop` 现在可以记录这些请求并返回
`pending_director_revision`，但不会伪造新剧情、人物、反转、结局或任何 LLM 输出。
真正的 DirectorAgent 修订适配器留到后续阶段。Session 保存 optimizer 结果、建议、
候选、请求、请求历史和停止原因；重新生成 MoviePlan、FilmIR 或 Creative Analysis
时会清空旧优化结果，避免跨版本污染。CLI 的 `/optimize` 和 `/revision` 只运行/查看
离线机制；`/diagnostics` 展示计数与停止原因。

本阶段明确不实现 Creative Analysis 以外的新图算法、不实现 Creative Graph 扩展、
Provider Runtime、submit/poll/download、Agnes/Veo/Sora/Kling 接入或 MP4 生成。
`VideoJob` 仍是未来 Provider Runtime 的唯一输入边界。

### Phase 4D.1：Revision Candidate / Diff / Guard（本阶段）

Phase 4D.1 在真实 DirectorAgent Revision Adapter 之前增加安全包络：

```text
CreativeRevisionRequest[] + Original MoviePlan
        ↓
RevisionCandidate
        ↓
RevisionDiff
        ↓
RevisionGuard
        ↓
RevisionDecision
        ↓
accept / accept_with_warning / reject / rollback / pending_director
```

`RevisionCandidate` 是候选，不是最终 MoviePlan。它携带 source plan、请求来源、
候选类型、状态和可序列化的 revised plan，但不会覆盖 Session 当前的
`movie_plan`。本阶段的 `RevisionCandidateFactory` 只提供 deterministic noop、
policy-violation 和 pending-director fixture，用于测试 Diff/Guard；默认 CLI 不会
偷偷生成 fake revised plan。

`RevisionDiffBuilder` 对 StoryPlan、DirectorPlan 和有限 legacy 投影做结构化 diff，
记录 added/removed/modified/reordered 变化，检查主要人物、核心冲突、stakes、
resolution、story beats、preserve/avoid 约束、Provider 字段和 prompt stuffing，并
输出 changed field、violation、target response 和 leakage 指标。Diff 只读，不修改
任何输入对象。

`RevisionGuard` 根据 `RevisionGuardPolicy` fail-closed 决策：Provider leakage、
prompt stuffing、主要人物删除、preserve/avoid 违反、hard validation issue、未经
授权的核心冲突/结局修改都会 reject；没有候选或 revised plan 为空则是
`pending_director`；明确回应 target 且无违规才可 accept，否则可能
`accept_with_warning` 或 reject。rollback 只记录恢复到原 MoviePlan 的目标，仍不
自动写回 Session。

本阶段不直接接真实 LLM revision，是为了先验证 diff、guard、rollback 和历史序列化
边界，避免不可审计的自动剧情重写。Session 保存 candidates、diffs、decisions、
guard diagnostics 和 active/accepted/rollback 标识；重新生成 MoviePlan、Creative
Analysis、Optimizer 或 Revision Request 时清空 active candidate，但不强制旧 Session
重新 diff。CLI 新增 `/revision-guard`，无候选时安全显示 `pending_director`。

本阶段不调用 LLM、不调用 Provider、不 submit/poll/download、不生成 MP4，也不修改
MoviePlan。`VideoJob` 仍是未来 Provider Runtime 的唯一输入边界。

### Phase 4D.3：Explicit Apply / Rollback / Revalidation（本阶段）

Phase 4D.3 将 Guard 的许可与状态变更明确分开：

```text
RevisionDecision(accept / accept_with_warning)
        + RevisionCandidate
        + 当前 MoviePlan
        ↓
ApplyRevisionCommand
        ↓
RevisionApplyService
        ↓
MoviePlanVersionHistory + 新 Current MoviePlan
        ↓
validate_movie_plan
        ↓
Downstream Invalidation
        ↓
stage = movie_plan_revised
        ↓
手动重新构建 FilmIR / MovieIR / VideoJob
```

`accept` 不是 `apply`。`RevisionGuard` 只说明 candidate 满足安全策略；只有携带
`candidate_id`、来源 MoviePlan、原因和 `confirmed_by` 的 `ApplyRevisionCommand` 才能
触发 `RevisionApplyService`。Service 仍会检查 candidate、decision、来源 ID 和完整
`revised_movie_plan`，再次执行 `validate_movie_plan`；pending、reject、rollback 或
没有 revised plan 的 candidate 一律拒绝。

Apply 成功后，旧 MoviePlan 以 `MoviePlanVersionRecord` 写入
`movie_plan_version_history`，新计划成为 current，`previous_movie_plan_id` 指向旧
版本，并记录 `RevisionApplyRecord`。即使新计划沿用相同 `plan_id`，版本号仍然递增；
历史快照不保存 Python 对象或 Provider 字段。

MoviePlan 改变会使旧 FilmIR、MovieIR、VideoJob、Creative Analysis、Creative
Optimizer、RevisionRequest、Candidate/Diff/Decision 和 Guarded result 失效。统一的
`invalidate_downstream_after_movie_plan_change()` 清空 active 状态，保留 Apply/Rollback
历史；若宿主 Session 附带 ProviderJob、artifact 或旧 render 状态，只记录到
`stale_artifacts`，本阶段不删除外部文件。

`RollbackRevisionCommand` 必须明确指定历史 `movie_plan_id` 和确认人。Rollback 从版本
快照恢复计划，重新执行 `validate_movie_plan`，写入 `RevisionRollbackRecord`，同样使
所有下游 active 产物失效，并进入 `movie_plan_rolled_back`。Apply/Rollback 都不会自动
重建 IR 或 VideoJob；用户必须重新执行 `/build-film-ir`、`/build-ir`、`/compile`。

CLI 的 `/revision-apply` 要求输入 `APPLY <candidate_id>`，`/revision-rollback` 要求
输入 `ROLLBACK <movie_plan_id>`；空输入或错误确认不会改变状态。`/diagnostics` 展示
当前/上一 MoviePlan、版本历史、Apply/Rollback 记录和 stale 数量。本阶段仍不调用
Provider Runtime、不 submit/poll/download、不生成 MP4；`VideoJob` 仍是未来 Runtime
的唯一输入边界，Provider Runtime 推迟到 Phase 5。

### Phase 4D.2：DirectorAgent Revision Adapter（本阶段）

Phase 4D.2 将 DirectorAgent 接入既有安全包络，但仍坚持 candidate-only：

```text
CreativeRevisionRequest[] + Original MoviePlan + DirectorRevisionContext
        ↓
DirectorRevisionAdapter
        ↓
RevisionCandidate
        ↓
validate_movie_plan
        ↓
RevisionDiffBuilder
        ↓
RevisionGuard
        ↓
RevisionDecision
```

`DirectorRevisionAdapter` 只负责调用 DirectorAgent（或离线 fake）并把完整的
修订计划包装成 `RevisionCandidate`。它不写 Session、不覆盖当前 `MoviePlan`、不
决定 accept/reject，也不访问 Provider。即使 Guard 返回 `accept`，本阶段也不会
自动 apply；显式 Apply、Rollback 和重新 lowering 留到 Phase 4D.3。

`DirectorRevisionContext` 是安全的请求打包边界，保存原计划 ID、请求 ID、
preserve/avoid、allowed/forbidden change scope、最大尝试次数和 candidate-only 标志。
它不携带 API key、文件路径、Provider payload、endpoint、model、task 或 video_id。
`build_director_revision_prompt()` 将这些约束和最小必要的当前 MoviePlan JSON 传给
DirectorAgent，并明确要求只返回完整 MoviePlan JSON；LLM 输出仍必须通过
`validate_movie_plan`、`RevisionDiffBuilder` 和 `RevisionGuard`。

离线 `RuleBasedDirectorRevisionAdapter` 默认是 `pending_only`：只返回等待真实
导演的 pending candidate，不伪造剧情。`noop_candidate`、`safe_metadata_candidate`
和 `policy_violation_candidate` 仅用于显式测试。`DirectorAgentRevisionAdapter` 和
`OpenAIDirectorRevisionAdapter` 复用既有 `revise_movie_plan()` 端口，真实网络调用
由调用者选择；常规测试使用 fake Director，不依赖网络。

`run_director_revision_guarded()` 是一次性编排服务：生成 adapter result，验证候选，
生成 diff，调用 Guard，并返回可序列化的 `GuardedRevisionResult`。它不会把候选写
回 `movie_plan`，也不会产生 `provider_job`、artifact 或 MP4。Session 新增保存
`director_revision_adapter_results`、`director_revision_contexts`、
`guarded_revision_results`、`director_revision_attempt_count` 和
`director_revision_last_stop_reason`；旧 Session 缺失时使用空列表或零值。

CLI `/revise` 会执行这条 candidate-only 链路，`/revision-apply` 在本阶段明确拒绝，
`/diagnostics` 展示 adapter、context、guarded result 和停止原因。`VideoJob` 仍然是
未来 Provider Runtime 的唯一输入边界；本阶段不实现 Provider Runtime、submit/poll/
download、Agnes/Veo/Sora/Kling 接入或 MP4 生成。

### Phase 4E：Source Lineage & Stale Artifact Guard（本阶段）

Phase 4E 为每一级 V2 编译产物增加可验证的 Source Lineage。MoviePlan 改变后，旧
FilmIR、MovieIR 或 VideoJob 即使结构仍然有效，也不再代表当前导演决策，因而不能
继续作为下一步 lowering 或 Provider Runtime 的输入。

```text
current MoviePlan
        │ source_movie_plan_id
        ▼
FilmIR
  ├─ source_story_plan_id
  └─ source_director_plan_id
        │ source_film_ir_id + source_movie_plan_id
        ▼
MovieIR
        │ source_movie_ir_id + source_film_ir_id + source_movie_plan_id
        ▼
VideoJob
```

`SourceLineage` 是 provider-neutral 的来源值对象；`LineageCheckResult` 和
`StaleArtifactDiagnostic` 只描述来源是否仍然匹配，不读取 Provider payload，也不
调用 Provider。`SourceLineageGuard` 分别检查 FilmIR→MoviePlan、MovieIR→FilmIR /
MoviePlan、VideoJob→MovieIR / FilmIR / MoviePlan 的关系；对旧 Session 缺少来源字段
的对象标记为 `unknown_lineage`，而不是让 Session load 崩溃。

Session 新增 `current_film_ir_id`、`current_movie_ir_id`、`current_video_job_id`、
`source_lineage_diagnostics` 和 `stale_lineage_diagnostics`。旧 Session 如果已经有
artifact ID 会迁移出 current 指针；如果 source 字段缺失，保留对象并在下次命令前
报告 unknown/stale，必须重新构建才能继续 lowering。

`/build-ir` 只能接受来源匹配当前 MoviePlan 和 current FilmIR 的 FilmIR；不匹配时
拒绝并提示 `/build-film-ir`。`/compile` 只能接受来源匹配当前 MoviePlan 和 current
FilmIR 的 MovieIR；不匹配时拒绝并提示 `/build-ir`。`/render` 在离线阶段仍不执行
Provider，但会先确认 VideoJob 的三条来源链；不匹配时拒绝并提示 `/compile`。

`/diagnostics` 现在显示 current IDs、FilmIR/MovieIR/VideoJob 的 fresh/stale/unknown
状态、来源字段、stale artifact 数量以及下一步 Action。Phase 4D.3 的 Apply/Rollback
统一失效逻辑同时清空 current IR/VideoJob 指针和 lineage diagnostics，并把旧对象写
入 `stale_artifacts`，因此新 MoviePlan 不能复用旧下游产物。

本阶段没有实现 Provider Runtime、submit/poll/download、Agnes/Veo/Sora/Kling 接入或
MP4 生成。`VideoJob` 仍然是未来 Provider Runtime 的唯一输入边界；只有后续 Phase 5
才允许进入 Provider 执行。

### Phase 4F：Immutable MoviePlan Version & Content Fingerprint（本阶段）

Phase 4F 在 ID lineage 之上增加不可变的 MoviePlan 内容身份：

```text
MoviePlan
  ├─ movie_plan_version
  ├─ movie_plan_fingerprint (SHA-256)
  └─ movie_plan_lineage_token
        │
        ▼
FilmIR → MovieIR → VideoJob
```

`CanonicalSerializer` / `FingerprintBuilder` 只对创作内容做规范化 JSON 和 SHA-256。
序列化会排除 plan identity、revision/confirmed、版本和 token，以及 metadata、诊断、
cache、runtime、provider、artifact、task、endpoint、时间戳等会变化的状态；因此同一
创作内容不会因为 Session 保存时间或运行环境变化而得到不同 fingerprint。fingerprint
不是安全签名，也不包含 API key 或 Provider payload。

`MoviePlan` 初始版本为 1。显式 Apply 时先重新计算 candidate fingerprint：内容发生
变化才递增 version，并用新的 plan id、version、fingerprint 生成 lineage token；旧
快照连同 fingerprint/token 进入 `MoviePlanVersionRecord`。显式 Rollback 使用历史快照
中原有的 version/fingerprint/token，不把恢复动作伪装成新的创作版本；Apply/Rollback
都继续执行完整 MoviePlan revalidation，并清空下游 active IR/VideoJob。

FilmIR 保存 `source_movie_plan_version`、`source_movie_plan_fingerprint` 和
`source_movie_plan_lineage_token`；MovieIR 继续保存这三项并增加
`source_film_ir_fingerprint`；VideoJob 继续作为 Provider Runtime 的唯一输入边界，
同时携带 MoviePlan、FilmIR、MovieIR 的 provenance。旧 IR/VideoJob 缺字段时仍可反
序列化加载，但 `SourceLineageGuard` 会输出 `unknown_lineage`，不会静默复用旧产物。

旧 Session 迁移只为当前 MoviePlan 计算缺失 provenance，并默认填充三个 current
字段；不会重新调用 DirectorAgent、不会重写历史快照。`/diagnostics` 显示当前
MoviePlan 的 ID、version、fingerprint、lineage token，以及各级 artifact 的
fresh/stale/unknown 状态。

本阶段没有改变 Revision Guard 的决策语义，没有自动 Apply，没有 Creative Analysis /
Creative Graph / Creative Optimizer，也没有实现 Provider Runtime、submit/poll/download、
Agnes/Veo/Sora/Kling 接入或 MP4 生成。

### Phase 4G：Immutable ExecutionPlan Contract（本阶段）

Phase 4G 在 MovieIR 之后增加不可变的执行编译后端，但不改变 MoviePlan、FilmIR 或
MovieIR 的创作语义：

```text
MovieIR
  ↓
ExecutionPlanCompiler
  ↓
ExecutionBundle
  ├── ExecutionPlan
  └── VideoJobs[]
```

`ExecutionPlan` 描述静态执行单元、显式 DAG、ProviderAssignment、runtime policy、
reference-frame strategy、artifact policy 和 capability snapshot。每个 `ExecutionUnit`
同时保存 `video_job_id` 与 `video_job_fingerprint`；`DependencyEdge` 是唯一的显式依赖
图来源，`depends_on` 由同一 lowering 规则生成并由校验器强制一致。ExecutionPlan 不保存
Provider task ID、runtime status、retry count、last error、download path、API key、endpoint
或 HTTP payload。

`ExecutionBundle` 将一个 ExecutionPlan 与其引用的 VideoJob 集合绑定。校验器 fail-closed
检查 dangling/duplicate VideoJob、unit fingerprint、DAG 自引用与环、reference-frame 边、
capability snapshot、ExecutionPlan fingerprint 和 Bundle fingerprint。VideoJob 的稳定
fingerprint 覆盖 Provider 输入及来源 provenance，排除 job ID、创建时间和 runtime 状态；
Bundle fingerprint 对排序后的 `{video_job_id, video_job_fingerprint}` 计算。

Session 新增 `execution_plan`、`execution_bundle`、current fingerprint 指针、诊断和
`stale_execution_artifacts`。MoviePlan Apply/Rollback 或 MovieIR rebuild 会使旧
ExecutionPlan/Bundle 失效，但不会自动 Apply、rebuild 或运行。旧 Session 缺少这些字段时
仍可加载；新 Bundle 恢复时会执行完整结构和 fingerprint 校验。

CLI 新增 `/build-execution-plan`、`/show-execution-plan` 和 `/validate-execution-plan`。
旧 `/compile` 继续生成兼容的 V2 VideoJob，不被新入口替换；`/render` 在本阶段只做离线
边界提示。Phase 4G 没有调用 Agnes、Veo、Sora、Kling，没有实现 submit/poll/download，
没有实现真实 ExecutionRuntime 或 ProviderRuntime，没有生成 ProviderJob、Artifact 或 MP4。

### Phase 5A：Offline Durable Execution Runtime（本阶段）

Phase 5A 只接受已经校验的 `ExecutionBundle`，不重新编译 MovieIR、ExecutionPlan
或 VideoJob：

```text
ExecutionBundle
    ↓
ExecutionRuntime
    ├── Runtime State Store + append-only Event Store
    ├── Checkpoint Store
    ├── DependencyResolver / Lease / TransitionService
    ├── ProviderRuntimeRegistry
    │       └── FakeProviderRuntime
    └── FakeArtifactVerifier
```

`ExecutionPlan`、`ExecutionUnit`、`ExecutionBundle` 和 `VideoJob` 仍是不可变编译产物；
`ExecutionRun`、`ExecutionUnitState`、`ProviderJob`、`SubmissionIntent`、Retry、Lease、
Artifact 和 Event 都保存在独立 Runtime 状态中。所有状态变更必须经过
`RuntimeTransitionService`，并写入单调递增的 append-only Event；关键转换生成带
SHA-256 checksum 的不可覆盖 Checkpoint。Checkpoint、Run 和 Artifact 都绑定
ExecutionBundle / ExecutionPlan fingerprint 及 MoviePlan provenance。

提交顺序固定为：生成确定性 idempotency key → 持久化 `SubmissionIntent` → checkpoint →
`PREPARED → SUBMITTING` → ProviderRuntime submit → 保存 ProviderJob →
`SUBMITTED`。如果 submit 响应丢失，状态变为一等的 `SUBMISSION_UNCERTAIN`；它保留
intent，不增加普通 retry，也不自动重提，必须人工对账。Retry 按错误分类、静态
`RetryPolicy`、attempt 和 backoff 限制；poll/download retry 复用已有 ProviderJob，
不会重新 submit。

Scheduler 只读 ExecutionPlan 的显式 `dependency_graph`、Unit 状态和 RuntimePolicy。
它不会理解电影内容，也不会跳过 reference-frame 依赖；依赖失败的下游进入
`BLOCKED`，`max_parallel_units` 控制 Ready Unit 的并发租用。崩溃恢复先校验 Bundle
fingerprint，再从最新有效 Checkpoint 恢复 PREPARED、SUBMITTED/RUNNING、DOWNLOADING
或 VERIFIED 状态；已完成 Unit 不重复提交，过期 Lease 会清理。

Phase 5A 的 Provider Registry 只注册显式 `FakeProviderRuntime`。它模拟 success、
queue/running、可重试错误、不可重试错误、never-complete、download interruption、
corrupted artifact、cancel 和 submission uncertainty；所有 Fake ProviderJob 使用
`fake-task-*` 标识并遵守 idempotency。Fake Artifact 是普通 `fake-video.bin` 等离线
文件，不使用 FFmpeg、不生成 MP4；`FakeArtifactVerifier` 检查文件存在、非空、size、
SHA-256、artifact type 以及完整 MoviePlan/ExecutionPlan/VideoJob/ProviderJob provenance。

Session 只保存 Runtime 引用和摘要，详细运行状态放在独立 runtime 目录；旧 Session 缺少
这些字段时按空状态加载。CLI 增加 `/start-execution`、`/step-execution`、
`/run-execution`、`/resume-execution`、`/cancel-execution`、`/execution-status`、
`/execution-events` 和 `/execution-checkpoint`。旧 `/compile` 和 `/render` 兼容路径仍
保持原语义，`/render` 不会偷偷转入 Runtime。

本阶段明确没有调用 Agnes、Veo、Sora、Kling，没有读取真实 API Key，没有发送真实网络
Provider 请求，没有创建真实 Provider 任务，没有下载真实视频，没有生成 MP4，没有修改
MovieIR、ExecutionPlan 或 VideoJob，没有自动 Apply、rebuild 或处理
`SUBMISSION_UNCERTAIN`，也没有接入真实 Artifact Pipeline。

### Phase 5B-1：Provider Runtime Plugin Contract（当前阶段）

Phase 5B-1 将 Provider 接入边界收敛为统一插件契约：

```text
ExecutionRuntime
    ↓ ProviderRuntime Protocol
ProviderRuntimeRegistry
    ├── FakeProviderRuntime
    └── MockHttpProviderRuntime
```

`ProviderRuntime` 只接收 `VideoJob` 和不可变的 `ProviderRequestContext`，不接收
ExecutionBundle、Session、Story、Scene、MoviePlan、FilmIR 或 MovieIR。Adapter 只做
协议转换、状态映射、错误归一化和下载/Provider verify，不能修改 Runtime State、执行
Transition、重建 Prompt 或修改 Compiler IR。

统一契约包括 `ProviderCapabilities`、`ProviderJob`、`ProviderJobStatus`、
`ProviderSubmitResult`、`ProviderPollResult`、`ProviderCancelResult`、
`ProviderDownloadResult` 和 `ProviderVerificationResult`。Provider 特定的
`task_id`、`video_id`、`operation_id` 只在 Adapter 内部使用，持久化的通用句柄统一使用
`remote_job_id`。Provider 能力指纹只覆盖稳定能力语义，不覆盖 endpoint、API key、
队列、余额、健康状态或临时限流；与 ExecutionPlan snapshot 不匹配时 fail-closed，
不会自动重编译 Plan 或切换 Provider。

Provider 错误在 Adapter 边界归一化为 `ProviderErrorCategory` 和脱敏的
`ProviderRuntimeError`。`SUBMISSION_UNCERTAIN` 与已有 ProviderJob 的
`ProviderJobStatus.UNKNOWN` 保持分离。Event、Checkpoint、Session、ProviderJob 和
诊断只能保存递归脱敏后的响应；Authorization、token、cookie、secret、signature 和
signed URL query 均不得长期明文保存。

`MockHttpTransport` 是唯一的 HTTP 测试入口，不访问外网，可脚本化 202、429、503、断线、
超时、malformed JSON、签名 URL 和二进制下载。`MockHttpProviderRuntime` 验证 HTTP
字段到通用契约的映射；下载先写 `.part`，目标路径由 Runtime 提供并在完成后原子 rename。
Fake Adapter 只做纯内存/文件场景模拟。两者均不生成 MP4。

CLI 新增 `/providers`、`/provider-capabilities <provider_key>` 和
`/provider-contract-check <fake|mock-http>`；`/execution-status` 会显示 ProviderJob
通用状态、能力匹配和契约 schema，`/diagnostics` 会显示 Provider 缺失、能力漂移和
响应脱敏边界。

Phase 5B-1 仍不调用 Agnes、Veo、Kling 或 Sora，不读取真实 API Key，不访问真实
Provider endpoint，不创建真实 ProviderJob，不下载真实视频，不生成 MP4，不自动处理
`SUBMISSION_UNCERTAIN`，不自动切换 Provider，不自动 rebuild ExecutionPlan，也不接入
完整真实 Artifact Pipeline。Agnes Adapter 属于 Phase 5B-2，单镜头真实 API smoke 属于
Phase 5B-3。

### Phase P1：Low-Risk Architecture Pruning（已完成）

P1 只处理静态审计确认的低风险冗余，不改变运行时语义。删除项为：没有生产、测试、
文档或序列化引用的 `ReadinessReport`、V2 旧的 `execution.Artifact`、未使用的
`ProviderErrorMapper` 和 `legacy_capability_fingerprint`。它们没有参与 MoviePlan、IR、
VideoJob、ExecutionBundle、Session 或 Provider Runtime 的持久化/恢复。

`RetakeRequest`、`OpenAIDirectorRevisionAdapter` 和 `allocate_durations` 保留：前两者仍有
文档/合约承诺，后者仍属于 legacy timing 测试和生产链。三个历史兼容包装函数保留为薄
兼容入口并发出 `DeprecationWarning`，不再列入 V2 主 Facade 的声明式 `__all__`。

V2 Facade 从 292 个声明式名称收窄到当前核心边界；stores、checkpoint/event 实现、Fake/Mock
实现、HTTP test doubles、scheduler transitions、低层 provider result、creative graph 和
revision history records 不再作为主 Facade 导出，但其源模块和当前 CLI/Session 依赖仍保留。
这不是删除 Runtime durable state，也不是合并 legacy/V2 VideoJob；现有显式兼容导入暂时继续
可用，后续再按发布策略迁移。

P1 没有删除或修改 FilmIR、ExecutionPlan、ExecutionBundle、ExecutionRuntime、durable
state/checkpoint/submission intent、Session persistence schema、`/render`、legacy Agnes
chain，也没有调用真实 Provider、网络、API key 或生成 MP4。Fake/Mock 仍明确是 offline/test
support，下一阶段才讨论更深的 core merge 或真实 Provider Adapter。
