# 主实验错误分析与正确率提升建议

分析日期：2026-07-30

## 1. 结论摘要

当前 `data/reasoning/` 中的主实验结果为 **110/179 = 61.45%**。严格按绝对上限计算，距离 100% 还有 **38.55 个百分点**，而不只是约 30 个百分点；但其中也包含标注粒度、答案不完整和 LLM judge 严格性造成的损失，因此可实现的短期提升空间会小于理论上限。

最重要的结论是：**下一阶段应优先改进 memorization，但这里的 memorization 指“视频事实如何被抽取、结构化、更新和检索”，不是让模型背测试集。建议工程投入约 70% 放在 memory，30% 放在 reasoning/verifier。**

依据如下：

- 69 个错误中，精确空间位置/物体状态占 32 个，计数占 12 个，多物体或多细节提取占 14 个；三类合计 **58/69 = 84.1%**，都强依赖低层事实记忆。
- 按问题句式统计，位置题只有 **13/44 = 29.5%**，计数题只有 **9/21 = 42.9%**；但 Yes/No 题达到 **34/38 = 89.5%**。
- 数据集标签中，`General Knowledge Extraction` 为 **19/43 = 44.2%**，`Cross-Modal Reasoning` 为 **32/64 = 50.0%**，而 `Human Understanding` 为 **59/73 = 80.8%**。这说明高层语义和人物理解不是首要短板。
- abstraction 配置从 `30_10` 增加到 `100_60`，正确率仍只在 **60.80%–63.07%** 之间波动；扩大检索/抽象预算没有稳定增益。
- 四种配置共同覆盖的 176 题中，**40 题四次全部错误**，其中 21 题是位置题。它们更像稳定的表征、检索或工具错误，而不是一次采样偶然失误。

短期最值得做的两件事是：

1. 建立带时间版本的 object-state memory，专门回答“现在在哪里、从哪里拿、应该放到哪里、具体在哪一层/哪个容器”。
2. 用事件账本替换当前粗粒度 `get_frequency_stats` 计数逻辑，区分动作次数、物品数量、步骤数量和对话中提及数量。

## 2. 分析范围与方法

本报告分析以下本地结果：

- 主实验：`data/reasoning/*.json`，14 个视频，179 个问题。
- abstraction 对照：`data/ablation/reasoning_abs/{30_10,50_30,100_60}`。
- 原始 episodic memory：`data/memorization/*.json`。
- 推理日志：`data/logs/*_reason.log`。
- 三通道备份工具和提示：`reasoning_variants/three_route/tools.py`、`reasoning_variants/three_route/agent.py`、`utils/prompts.py`。

错误类型是根据 question、ground truth、final answer 和工具调用轨迹人工归为一个“主要错误原因”。一些问题同时含有空间、时间和多细节要求，因此该分类用于确定主要改进方向，不代表类别之间完全没有交叉。

当前结果只覆盖 14 个视频，所以结论适合作为下一轮迭代依据，但在扩展到全量 benchmark 后应重新统计一次。

## 3. 总体结果

### 3.1 主实验与 abstraction 配置

| 配置 | 正确/总数 | 正确率 |
|---|---:|---:|
| 主实验 baseline | 110/179 | 61.45% |
| `30_10` | 111/176 | 63.07% |
| `50_30` | 108/176 | 61.36% |
| `100_60` | 107/176 | 60.80% |

不同配置的总题数略有不同，不能只比较分子。对四种配置共同存在的 176 题进行逐题比较：

- 四次全部正确：81 题。
- 四次全部错误：40 题。
- 配置间结果不稳定：55 题。
- baseline 在共同题目中有 67 个错误，其中 27 个至少被某个配置答对，40 个始终没有被救回。

这表明系统同时存在两类问题：

- **系统性错误**：40 题始终错误，需要改 memory schema、抽取或专用工具。
- **检索/生成不稳定**：55 题随配置变化，需要 reranking、证据约束和 verifier。

### 3.2 数据集原始标签表现

以下标签是多标签，行之间不能相加：

| 数据集标签 | 正确/总数 | 正确率 |
|---|---:|---:|
| Multi-Hop Reasoning | 3/7 | 42.9% |
| General Knowledge Extraction | 19/43 | 44.2% |
| Cross-Modal Reasoning | 32/64 | 50.0% |
| Multi-Detail Reasoning | 62/109 | 56.9% |
| Human Understanding | 59/73 | 80.8% |

`Multi-Hop Reasoning` 虽然最低，但只有 7 题，样本很小。更稳定、更值得优先优化的信号是 General Knowledge、Cross-Modal 和 Multi-Detail 的低正确率。

### 3.3 按问题形式统计

| 问题形式 | 正确/总数 | 正确率 | 错误数 |
|---|---:|---:|---:|
| Location | 13/44 | 29.5% | 31 |
| Count | 9/21 | 42.9% | 12 |
| Who | 5/9 | 55.6% | 4 |
| Which | 4/7 | 57.1% | 3 |
| What | 31/45 | 68.9% | 14 |
| When | 3/4 | 75.0% | 1 |
| Yes/No | 34/38 | 89.5% | 4 |

位置题和计数题合计贡献 **43/69 = 62.3%** 的错误，是最明确的突破口。

## 4. 主要错误类型

### 4.1 精确位置与物体状态错误：32/69（46.4%）

这是最大的错误来源。常见模式包括：

- 只答到房间或家具，缺少容器/层级：`in the kitchen`，但 GT 要求 `on the table in the kitchen`。
- 把邻近位置当成精确位置：`on the desk`，但 GT 要求 `in the pen holder on the table`。
- 混淆物品的来源、临时放置位置和当前最终位置。
- 上层/下层、左/右、门边/桌边等视觉细节发生反转。
- 同一物体多次移动后，没有按问题时间点选择最新状态。

代表案例：

- `study_08_Q01`：GT 为 `In the pen holder`，预测为 `On the desk`，只保留了粗粒度区域。
- `study_18_Q07`：GT 为黑色储物架从上往下第二层，预测为最底层。
- `living_room_15_Q09`：空调遥控器的当前地点被回答成另一个遥控器/物体附近的位置。
- `study_03_Q09`：书的当前状态应为桌上，却回答为已经回到书架，属于时间版本选择错误。

工具轨迹也印证了问题：31 个 location-form 错误总共调用了 35 次 `general_search` 和 45 次 `search_temporal_context`，但只有 **2 题调用 `watch_video_clip`**。对于层级、左右、容器和“现在在哪里”等视觉状态问题，仅继续做文本图检索通常无法恢复抽取时丢失的细节。

### 4.2 计数与事件聚合错误：12/69（17.4%）

12 个错误覆盖：开门次数、礼物数量、制作步骤、使用次数、打印次数、整理桌面次数、提及的电影数量等。

值得注意的是，这 12 个错误已经合计调用了 **27 次 `get_frequency_stats`**。因此问题不是“忘记调用计数工具”，而是工具对计数语义建模不正确。

当前 `execute_get_frequency_stats` 的主要风险：

- 先取语义相似度 top 100，而不是找出满足严格条件的全部事件，可能漏召回或混入相似动作。
- 用 10 个 clip 的窗口进行去重；相距较近的两个真实动作可能被合并，同一真实动作也可能因谓词或目标写法不同而被拆开。
- `How many steps`、`How many items`、`How many movies are mentioned` 并不是同一种“动作频率”，却被同一个工具强制处理。
- 图中的一次连续动作可能被多个帧/多个 edge 重复描述；反过来，对话中的计划、提及和实际执行也可能被混为一谈。

这解释了为什么会同时出现 1→3、3→6、6→3、3→7 等双向偏差，而不是固定地只多算或只少算。

### 4.3 多物品、多细节与答案覆盖不足：14/69（20.3%）

这类问题通常找到了一部分相关证据，但最终答案漏项、选错版本或混入候选项。

代表案例：

- `bedroom_01_Q10`：最初菜单包含两种蛋糕，但 Lily 后续明确选择 mocha、strawberry cake、chips with ketchup；预测保留了早期候选 banana，遗漏最终选择的 ketchup。
- `living_room_15_Q03`：GT 为 apples and spaghetti，预测只答 apples。
- `living_room_22_Q05`：GT 为 liberal arts and sports，预测只答 sports。
- `living_room_22_Q07`：GT 为 tissue and towel，预测只答 tissue。
- `study_05_Q02`：从 toolbox 取出 scissors and glue，预测把 glue 混成 tape。

该类错误说明 conversation summary 或检索结果中虽然可能出现正确信息，但系统缺少“答案槽位是否完整”的最后检查，也没有明确区分 candidate/offered、requested、served、current 等语义角色。

### 4.4 人物属性、偏好、关系与比较错误：9/69（13.0%）

例如谁更爱整洁、谁棋下得最好、人物饮食偏好、恋爱/同事关系、两人品味是否相同。它们是真正更偏 reasoning 的错误，但总量明显少于低层 memory 错误。

这类问题常见原因：

- 依据单次行为做人物长期属性判断。
- 多个人物证据混在同一 conversation summary 中，发生主体错配。
- 比较题没有为每个候选人分别建立证据表。
- 旧的高层属性没有被后续行为修正，例如“现在是否还乱放东西”。

### 4.5 日期计算、未来意图等纯时间推理错误：2/69（2.9%）

例如把 “the third day after Labor Day” 原样输出，而没有计算为 May 4th；以及人物接电话后要继续做什么。这类问题应通过轻量 symbolic reasoning 和时间关系解析解决，但不是当前最大瓶颈。

### 4.6 评测粒度与近似答案问题（跨类别）

少量错误存在“语义接近但粒度不足”或 GT 本身偏抽象的情况，例如：

- `Mia's collections` 与列出盒内具体物品。
- `on the table/desk`、`black round table/black side table` 等可能指向同一实体，但文本名称不统一。
- 回答到正确房间或家具，但 GT 要求精确到抽屉、笔筒、层级或相对方位。

这部分不应直接当作模型能力提升空间。建议对 69 个错误抽取 20–30 个做人工复核，并在当前 179 题上补一次 cross-judge；仓库中已有的 NVR judge comparison 达到 96.61% pairwise agreement，但它不是这 179 题的完全同一结果集，不能直接替代本轮复核。

## 5. 为什么建议先做 memorization，而不是先堆 reasoning

这里建议的是 **memory-first, reasoning-second**：

1. 最大错误源是空间位置、物体状态、计数和多细节事实，不是开放式逻辑推导。
2. 错误题平均使用约 4.25 轮工具，正确题约 3.71 轮。错误题已经搜索得更多，继续增加 reasoning round 很可能只是反复消费同一批不准确事实。
3. abstraction 阈值扩大没有提升，说明“给更多相似文本”不是答案；需要改变 memory 的结构和语义。
4. Human Understanding 达到 80.8%，Yes/No 达到 89.5%，说明当前 reasoning backbone 在证据明确时通常能做出正确判断。
5. 40 个问题在四种配置中全部错误，证明其中相当一部分不是一次性采样问题。

推荐资源分配：

- **70%：memorization/representation/retrieval**，包括状态版本、事件去重、实体统一、空间层级和视觉复核。
- **30%：reasoning/verifier**，包括答案覆盖检查、时间运算、比较表和冲突证据处理。

## 6. 提升正确率的具体建议

### P0：建立 object-centric、带时间版本的状态记忆

不要只保存自然语言 edge，应为可移动物体维护结构化状态：

```text
object_id
canonical_name / aliases
location = room > furniture > container > shelf/level > relative_position
state_type = source | destination | current | intended
valid_from_clip / valid_to_clip
evidence_clip_ids
evidence_modality = dialogue | visual | action
confidence
```

关键规则：

- 每次 `pick up / move / put / store / retrieve` 都关闭旧状态并创建新状态。
- “Where is X now?” 只取问题时间点之前最新的 `current` 状态。
- “Where can robot get X?” 优先取当前可获取位置，而不是历史来源或曾经放置的位置。
- “Where should X be placed?” 区分规范目标位置和当前事实位置。
- location 输出必须通过层级 completeness check；若 GT 风格要求精确位置，答案不能只停在 room/furniture。

这一项直接覆盖约 32 个主要错误，是预期收益最高的改动。

### P0：把计数工具改成按“计数任务类型”路由的事件账本

先将 count 问题分类：

- `event_count`：开门、使用拖把、打印、整理桌面发生几次。
- `unique_object_count`：拿了几本书、盒子里有几件礼物。
- `step_count`：制作树干/树叶需要几步。
- `mention_count`：对话中提到几部汽车电影。
- `state_change_count`：物体被放到多少个不同位置。

事件账本建议字段：

```text
event_id, canonical_verb, actor_id, object_id,
start_clip, end_clip, completed,
source_location, destination_location,
evidence_edges, dialogue_or_action
```

去重应基于 actor/object/动作连续性和视觉 track，而不是固定 10-clip bucket。计数结果必须返回逐项 evidence ledger，reasoner 只能对 ledger 行数求和；若无法枚举全部事件，则触发定向视频回看，不能直接采用 embedding top-N 的 group count。

### P1：为精确视觉问题增加不确定性触发的 rewatch

当前位置错误几乎没有使用视频回看。建议在以下条件自动触发：

- 问题包含 left/right、upper/lower、first/second tier、behind/beside、drawer/holder/shelf。
- 检索得到两个冲突位置。
- 只有 room/furniture 级证据，但问题要求容器或层级。
- 当前状态来自低置信度 caption，或者物体在多个 clip 中移动过。

回看不应只给一个宽泛 focus；应让 MLLM 在候选位置之间做受约束比较，并返回结构化结果与置信度。例如：`candidate=A/B/unknown`、`supporting_frame_ids`、`container`、`relative_position`。

### P1：增加 final-answer evidence/coverage verifier

在 `complete_task` 前增加按题型检查：

- list/detail：GT 未知时也要从问题生成 slots，检查是否遗漏已检索出的并列项。
- count：必须显示逐事件/逐物品 tally，并验证 tally 数量等于答案。
- location：必须明确 room、support/container、level/relative position 中问题要求的字段。
- comparison：为每个候选人物分别列证据，禁止只找到一个人的证据就下结论。
- temporal：明确使用了 earliest/latest/current/before/after 中哪一种时间规则。
- contradiction：旧事实和新事实冲突时，优先时间上更接近问题时点、视觉证据更直接的记录。

这个 verifier 应检查“证据是否支持最终字符串”，而不只是提示模型再搜索一次。

### P1：让 conversation memory 保留事件角色和后续更新

当前自然语言 summary 容易把候选菜单、最终请求和实际送达物品混在一起。建议对对话命令显式保存：

```text
offered_options
requested_items
request_modifications
confirmed_items
served_items
requester / beneficiary
clip_id
```

同理，人物偏好要区分一次选择和长期 preference；关系/属性要保留支持证据与反证，并允许后续更新。

### P2：再做 reasoning 层专项改进

在 memory 改造之后，再处理剩余 reasoning 问题：

- 日期和相对时间交给确定性函数计算。
- 人物比较题构建 candidate-by-evidence 表。
- 多跳问题先生成最小 dependency chain，再逐节点检索。
- 对含 `now/still/just/originally/finally` 的问题强制解析 reference time。
- 对实体同义名做 canonicalization，例如 table/desk、side table/round table 是否为同一实例，应由 object ID 决定，而不是字符串相似度决定。

## 7. 建议的验证实验顺序

### 实验 A：Oracle evidence 上限测试

对全部 69 个错误，人工或脚本指定包含答案的 GT clip/subtitle，把它直接提供给当前 reasoner，不改模型。

- 若正确率大幅上升：主要是抽取/检索问题。
- 若证据明确仍答错：才是真正的 reasoning/verifier 问题。

这是判断 memorization 与 reasoning 责任比例最直接的实验。

### 实验 B：检索 Recall@K 与状态事实准确率

不要只记录最终 accuracy；为错误题标注正确 evidence clip，报告：

- correct clip Recall@5/10/20。
- correct object/location edge recall。
- latest-state selection accuracy。
- exact spatial field accuracy：room、container、level、relative position。

### 实验 C：三个独立小改动的 ablation

1. object-state ledger only。
2. event/count ledger only。
3. uncertainty-triggered rewatch only。

分别在当前 179 题上重跑，避免同时改多项后无法归因。

### 实验 D：按错误桶报告增益

除 overall accuracy 外，固定报告：

- Location accuracy。
- Count accuracy。
- Multi-item completeness。
- Human/relationship accuracy。
- 平均工具轮数、rewatch 次数和 token cost。

## 8. 预期收益与里程碑

以下是基于错误分布的工程目标，不是保证值：

- 第一阶段只修 object-state 和 count ledger，如果分别救回约 10–15 个位置错误、5–7 个计数错误，可带来约 **+8.4 到 +12.3 个百分点**。
- 再通过 list coverage、时间版本和比较 verifier 救回约 5–8 题，可增加约 **+2.8 到 +4.5 个百分点**。
- 因改动之间存在重叠，不能简单相加；较合理的短期目标是先从 61.45% 提升到 **70%–75%**，再根据 oracle evidence 结果决定是否冲击 78% 以上。

## 9. 最终建议

优先级排序如下：

1. **Memorization：object-state temporal ledger。**
2. **Memorization/Tool：typed event count ledger，重写 `get_frequency_stats`。**
3. **Retrieval：精确空间题的冲突检测和定向视频回看。**
4. **Reasoning：final-answer evidence/coverage verifier。**
5. **Reasoning：日期、比较、latest-state 等小型确定性算子。**
6. **Evaluation：当前 69 个错误的人工近似答案复核与同结果集 cross-judge。**

因此，不建议下一步首先换更强的通用 reasoning model 或继续增大 top-k。当前数据更支持先把“事实是否被正确记住、是否保存了最新版本、是否以可计数/可定位的结构被取出”解决好；证据质量提高后，现有 reasoner 很可能已经能吃到一大部分增益。
