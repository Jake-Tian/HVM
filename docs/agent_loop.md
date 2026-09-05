# CAM Agent Loop 设计与实现

本文档描述当前 CAM reasoning agent 的执行流程、问题路由、工具行为和 Location / Action Frequency 专项策略。对应实现位于：

- `reason.py`
- `reasoning_variants/three_route/agent.py`
- `reasoning_variants/three_route/tools.py`
- `utils/prompts.py`

## 1. 总体结构

```text
START -> planner -> executor -> route_after_executor
                          |              |
                          |              +-> final_answer -> END
                          |
                          +-> verifier -> planner
```

Agent 包含 4 个 node：

1. `planner`：判断下一步需要哪项证据并选择一个工具。
2. `executor`：执行 planner 产生的 tool call。
3. `verifier`：检查空结果、重复调用和遗漏的时序验证。
4. `final_answer`：基于完整消息历史生成一句最终答案。

流程由 budget 控制，默认值为 6。每次调用 `planner` 都会将 budget 减 1。当进入 planner 时 `budget <= 1`，agent 进入强制收尾轮，不再允许搜索或视频重看。

首轮固定调用 `general_search`，目的是先获得较宽的候选证据和 clip_id。后续轮次再选择 graph temporal context、路由专属工具或结束任务。

## 2. 输入数据流

### 2.1 Graph memory

`reason.py` 从已有 pickle 文件加载 `HeteroGraph`。Graph 中包含：

- high-level edges
- low-level edges
- appearance edges
- conversations
- scene 和 clip_id 等元数据

`general_search` 仍在 graph memory 上执行 weighted-triple 检索，现有 memorization 和 graph construction 均未改变。

### 2.2 原始 episodic memory

Action Frequency 的 `search_action_evidence` 会按需从以下路径读取 triple 转换前的 episodic memory：

```text
data/memorization/<video_name>.json
```

工具只读取其中的 `episodic_memory` 字段：

```text
reason
  -> build_agent
     -> get_tools
        -> search_action_evidence
```

这条数据流只改变 reasoning 阶段的上下文恢复方式，不重新运行 memorization，也不改变已有 graph。

`search_temporal_context` 仍只返回 graph actions 和 graph conversations。General 和 Location 不会读取原始 episodic behavior。

## 3. AgentState

```python
class AgentState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], operator.add]
    findings: Annotated[list[str], operator.add]
    clip_history: Annotated[list[int], operator.add]
    tool_call_history: Annotated[list[dict], operator.add]
    location_candidate_clips: Annotated[list[int], operator.add]
    location_object_name: str
    location_intent: str
    action_frequency_memory: dict
    budget: int
    total_tokens: Annotated[int, operator.add]
    token_details: Annotated[dict, merge_stage_usage]
```

各字段作用如下：

| 字段 | 作用 |
|---|---|
| `question` | 原始问题，也是问题路由的输入 |
| `messages` | Human、AI、Tool 和 System 消息的累积历史 |
| `findings` | 兼容保留字段，当前专项 findings 实际保存在 planner 消息文本中 |
| `clip_history` | 已重看的 clip_id，用于阻止重复视频调用 |
| `tool_call_history` | 工具名和参数历史，用于首轮判断和重复调用检测 |
| `location_candidate_clips` | object timeline 给出的合法视频重看 clips |
| `location_object_name` | 第一次成功 object-event 调用锁定的准确对象 |
| `location_intent` | 锁定对象对应的 source、destination、current 或 history intent |
| `action_frequency_memory` | counting unit、事件状态、occurrence count 和 confirmed total |
| `budget` | 剩余 planner 轮数 |
| `total_tokens` | 累计总 token 数 |
| `token_details` | 按 planner、final answer 和具体工具记录的总 token 数 |

Token 统计不再区分 input、output、cached 或 reasoning token，也不再按模型建立嵌套统计。写入结果文件的 `token_summaries` 是扁平结构：

```json
{
  "planner": 1200,
  "tool:general_search": 400,
  "tool:watch_video_clip": 800,
  "final_answer": 200,
  "total": 2600
}
```

## 4. 轻量问题路由

问题路由由 `classify_question_route()` 完成，不调用 LLM。它只识别明确的 Location 和 Action Frequency 表达，其余问题全部进入 General。

路由顺序如下：

1. Action Frequency
2. Location
3. General

### 4.1 Action Frequency

只有明确询问动作发生次数的表达会进入 Action Frequency：

```text
how many times
how often
number of times
how frequently
frequency of
```

普通物品、人数和其他数量问题不会进入专项流程。例如 `How many cups are on the table?` 和 `How many drawers did the robot open?` 都进入 General。

### 4.2 Location

以下表达会进入 Location：

```text
where
whereabouts
location of
should X be placed/put/stored/hung/laced
questions that explicitly contrast alternative spatial relations
```

Location 内部继续以纯规则区分两种 mode。Object-location 使用 object timeline，event-location 使用 general search 和 temporal context。`Where did Alice do her homework?` 属于 event-location，不会调用 object tools。

### 4.3 General

没有匹配上述规则的问题沿用原来的通用 planner strategy。专项提示只注入对应路由，不会全局施加在其他题型上。

路由是确定性规则，不增加模型调用和 token 开销。它也不是一个新的 memorization 分类器。

## 5. 当前工具集

工具注册表包含以下 6 个工具。实际暴露给 planner 的集合由 route 决定：

| 工具 | 是否调用模型 | 用途 |
|---|---:|---|
| `general_search` | 是 | 使用 LLM 将自然语言 query 解析为 weighted triples，再检索 graph memory |
| `search_temporal_context` | 否 | 读取目标 clip 前后各一个 clip 的 graph actions 和 graph conversations |
| `search_action_evidence` | 否 | Frequency-only，读取候选 clip 前后原始 pre-triple behaviors 和 dialogue |
| `search_object_events` | 否 | Location-only，确定性重建单一物体的事件与位置时间线 |
| `watch_video_clip` | 是 | Location-only，调用 MLLM 验证候选事件 clip 中未解决的空间细节 |
| `complete_task` | 否 | 告知 workflow 已具备足够证据，可以进入最终答案 |

`executor` 本身只是工具调度逻辑，不直接调用 LLM。但 `general_search` 和 `watch_video_clip` 的内部实现会调用模型。因此不能将整个 executor node 视为零模型开销。

General 只看到通用工具。Location 看不到 `search_action_evidence`。Action Frequency 看不到 `search_object_events` 或 `watch_video_clip`。

`get_frequency_stats` 的实现暂时保留用于对照，但已从 `get_tools()` 的返回列表中移除，planner 无法再看到或调用它。原因是 graph-edge frequency 可能把同一事件的多个 triple 或跨 clip 描述误当成多次物理事件。

## 6. 工具行为

### 6.1 `general_search`

```python
general_search(
    query: str,
    k_low_level: int,
    k_conversations: int,
    k_high_level: int,
    k_appearance: int,
)
```

首轮必须调用该工具，总检索预算约为 50。Planner 根据题型分配各模态的 k 值。

搜索逻辑仍是 weighted-triple matching。`prompt_parse_query` 针对 Location 问题增加了以下约束：

- 具体目标物体在 source 或 target 位置的权重应为 `0.9-1.0`
- 未知 actor、location 或 relation 不应比目标物体权重更高
- 可以同时构造 object-state triple 和 movement-transition triple

这使检索更关注问题中的具体 object，但不改变 graph schema 或检索返回结构。

### 6.2 `search_temporal_context`

```python
search_temporal_context(clip_id: int)
```

固定读取窗口为 `clip_id ± 1`。对每个 clip 返回：

- graph low-level actions
- graph conversations

需要注意：

- 多个句子可能描述同一次连续行为
- 一个句子也可能包含多个事件
- 原始描述不是自动去重后的可计数事件
- 相邻 clip 可能是同一行为的延续

因此该工具提供的是判断依据，而不是直接的动作次数。

### 6.3 `search_action_evidence`

```python
search_action_evidence(clip_id: int)
```

该工具只暴露给 Action Frequency。它读取候选 clip 前后各一个 clip 的 `characters_behavior` 和 `conversation`，不调用 LLM，也不观看视频。用途是恢复 graph triples 没有保留的 clip 内重复次数和明确数量。

### 6.4 `watch_video_clip`

```python
watch_video_clip(clip_id: int, focus: str)
```

该工具只暴露给 object-location，并且只能在 object timeline 已建立、目标 clip 出现在 `Suggested rewatch clips`、且 temporal context 已验证同一候选之后使用。它用于解决文本证据中仍未确定的单一空间细节，例如：

- container、shelf、drawer、level、side 或 supporting surface
- source、destination 和 current state 的区分

Planner 应只选择 object timeline 建议的候选事件 clip。`focus` 必须包含准确物体、目标状态、已知候选位置和需要区分的一个空间问题，不能请求普通 clip 总结。`clip_history` 会阻止重复观看同一 clip。

### 6.5 `complete_task`

Planner 判断证据充分后调用：

```python
complete_task(ready=True)
```

Executor 执行后，`route_after_executor` 检测最近一条 AIMessage 的 tool call。Route-specific 工具是可选证据源，不再作为结束任务的硬性前置条件。Planner 在现有证据足够支持答案时可以结束，预算耗尽时仍会强制收尾。

## 7. Planner prompt 组合

Planner 每轮的输入结构为：

```python
[planner_system_prompt] + state["messages"] + [route_strategy]
```

### 7.1 System prompt

System prompt 包含：

- video、text、graph 三层结构说明
- clip_id、character 和 object 的格式说明
- 动态生成的 graph stats
- 用一行 `Findings so far` 保留已确认事实的要求

Graph stats 包含角色、对象数、边数、clip 数、conversation 数和 scene 列表。

`Findings so far` 存在于 AIMessage 文本中并随消息历史保留，而不是由代码写入 `state["findings"]`。当前 prompt 要求它保持为一行，只记录与问题有关的确认事实，避免复杂模板消耗 reasoning。

### 7.2 首轮

首轮通过 `tool_call_history` 是否为空判断。首轮提示要求：

- 必须先调用 `general_search`
- 总 k 约为 50
- appearance 不相关时将预算分配给 low-level 或 conversations
- 同时注入当前问题路由对应的 strategy

### 7.3 正常轮

正常轮只注入当前路由的 strategy：

- `prompt_planner_strategy`
- `prompt_planner_strategy_location`
- `prompt_planner_strategy_event_location`
- `prompt_planner_strategy_action_frequency`

Planner 每轮必须调用恰好一个工具。

General strategy 还要求：

- 保留问题中的实体、限定词、时间条件和所需细节
- 最多调用两次 `general_search`，不能只做同义改写
- 只在候选事件需要上下文时使用 graph temporal context
- 结束前检查答案是否完整、直接受证据支持且没有证据冲突

### 7.4 强制收尾轮

当 `budget <= 1` 时，planner 不再 bind tools。它基于已有 evidence 生成最终 findings，代码随后补入一个 synthetic `complete_task` tool call，使 workflow 正常进入 `final_answer`。

Location 和 Action Frequency 会分别追加专项的 final-round 约束。

## 8. Location 专项流程

Location 路由首先区分 object-location 和 event-location。前者追踪物体状态，后者定位动作发生的 scene。

Object-location 执行流程：

1. 保留问题中的完整 target phrase 和 temporal condition。
2. 用该 target 和问题要求的状态做 object-first `general_search`。
3. 只有在有序 object timeline 能补充搜索结果时才使用 `search_object_events`。首次成功调用后对象和 intent 仍会锁定，避免切换实体。
4. 只有在候选事件需要相邻证据时才使用 graph temporal context。
5. 只有候选 clip 合法且存在视觉空间歧义时才能 rewatch。
6. 最终保留定位目标所需的全部受支持空间细节。

Event-location 只使用 general search 和 temporal context，不调用 object events 或视频重看。

时间语义的主要解释：

| 问法 | 目标状态 |
|---|---|
| `now/current/latest` | 最后一个 release 后的稳定位置，held state 不是最终位置 |
| `was/originally/previously` | 问题指定的历史状态 |
| `get/retrieve/from` | source location |
| `put/place/should` | observed 或 intended destination |

最终答案不应丢弃证据中已经确认、且定位目标所必需的空间层级或相对关系。

## 9. Action Frequency 专项流程

该流程只负责计算一个动作发生了多少次。物品数量和人数等问题使用 General。

Planner 先通过一次 exact-action `general_search` 获取候选 clips。若结果为零或明显过少，只允许再做一次 object-only search，以找到物体被操作的候选 clips。检索结果数或 triple 数不能直接当作动作次数。

```text
Confirmed:
- E1 [clip] evidence
- E2 [clip] evidence
```

核心规则：

- 同一事件的多个 triples 只计一次
- 同一事件的多个句子只计一次
- 跨相邻 clips 的连续事件只计一次
- graph 无法区分 clip 内重复次数时，可以使用 `search_action_evidence` 读取原始 behavior 和 dialogue
- 每个 confirmed event 包含 `occurrence_count`，一个 episode 可以贡献多次 occurrence
- 检索没有返回匹配事件时不能直接得出零次

Intermediate memory 是工作摘要而不是权威结果。最终数字应结合明确证据核对 confirmed events 的 `occurrence_count`，不能因为 ledger 为空就直接输出零。

## 10. Verifier

Verifier 是纯 Python 规则，不调用 LLM。它可能向下一轮 planner 注入以下 SystemMessage：

1. 空结果或工具失败，建议更换关键词、k 分配或工具。
2. 重复观看同一 clip，要求使用已有证据或更换 clip。
3. 连续两次工具名和参数完全相同，要求改变搜索策略。
4. 连续两次调用 `general_search`，要求停止同义搜索并转向最强候选 clip 的 temporal context 或视频。
5. 时序问题已经使用 `general_search`，但尚未使用 `search_temporal_context`，提醒恢复目标 clip 附近的原始 context。

Verifier 不再提供 frequency tool 相关提醒。

## 11. Final answer

`final_answer` node 将以下内容作为输入：

```python
[final_answer_system_prompt] + state["messages"]
```

它不 bind tools，并要求输出一句直接答案。

专项约束如下：

- Location：选择与问题时间状态一致的位置，输出最具体且有证据支持的位置层级。
- Action Frequency：将 intermediate memory 作为工作摘要，与明确证据核对后输出完成动作的数量。
- 多答案问题：保留所有 distinct supported answers，不只选择最后一个。
- Yes/No：以 `Yes` 或 `No` 开头。

## 12. 模型调用边界

一次 agent run 中可能发生的模型调用包括：

| 位置 | 类型 |
|---|---|
| `planner` | text LLM |
| `general_search` query parsing | text LLM |
| `watch_video_clip` | multimodal LLM |
| `final_answer` | text LLM |

以下部分不调用模型：

- 问题路由
- `search_temporal_context`
- `complete_task`
- verifier
- workflow routing

具体模型、API key 和 base URL 由 `utils/llm_qwen.py` 的当前配置提供。缺少有效 API key 时，`build_agent()` 会在运行前报错。

## 13. 兼容性与当前边界

- Memorization prompt、triple output structure 和 graph schema 均未修改。
- High-level memory 和 conversation 的存储结构未修改。
- General 问题继续使用原有通用策略。
- 原始 episodic memory 只用于 `search_temporal_context`。
- `get_frequency_stats` 未删除，只是对 agent 隐藏。
- 当前路由依赖英文显式关键词，不是学习式分类器。
- 真实 accuracy 变化需要在固定评测集上运行实验后确认。

## 14. 关键 Prompt 原文

本节记录三通道备份 agent 使用的核心 prompt。Prompt 常量位于 `utils/prompts.py`，首轮拼接逻辑位于 `reasoning_variants/three_route/agent.py`，视频重看 prompt 位于 `reasoning_variants/three_route/tools.py`。

### 14.1 Planner system prompt

对应 `prompt_planner_system`。运行时会将 `{budget}` 和 `{graph_stats}` 替换为当前剩余 budget 和 graph 统计。

```text
You are a strategic Planner answering questions about a long entity-centric video. You have access to tools that search the video's heterogeneous memory graph. Raw clip rewatch is available only for location questions. You have {budget} turns left to gather evidence. Think step-by-step: analyze what you know, what is missing, and which tool is best to fill the gap. Then, call EXACTLY ONE tool to gather missing information. Once you have sufficient evidence to confidently answer the user's question, call complete_task.

The system processes video information in three layers:
1. Video: Videos are split into 30-second segments, each assigned a unique clip_id.
2. Text: Each segment's text descriptions (behaviors, conversations, scenes) are stored by clip_id.
3. Graph: Text is converted into graph edges with different types:
   - High-level: Abstract character attributes and relationships (clip_id=0).
   - Appearance: Character physical looks, hair, clothing.
   - Low-level: Specific actions/states with temporal and spatial information (clip_id>0).
   - Conversations: Dialogue transcripts as [speaker, text] pairs.

Input format in the evidence:
- Parentheses (X): Confidence scores in high-level information.
- Square brackets [X]: Clip IDs. Each clip is 30 seconds.
- Angle brackets <X>: Character nodes. Objects are plain text.

## Evidence Memory
End each response with one short `Findings so far:` line. Keep only confirmed facts relevant to the question, include clip IDs, and carry earlier confirmed facts forward.

{graph_stats}
```

### 14.2 首轮附加约束

首轮会在 route strategy 前拼接：

```text
**FIRST ROUND**: You MUST use `general_search` FIRST and allocate the FULL budget (total k=50) to get a comprehensive view of the video. (If appearance is irrelevant, set k_appearance to 0 and redistribute its budget to k_low_level or k_conversations.)
```

因此首轮实际输入是：

```text
FIRST ROUND constraint
+ General / Location / Action Frequency route strategy
```

### 14.3 General strategy

对应 `prompt_planner_strategy`：

```text
## Strategy
1. Preserve every entity, qualifier, temporal condition, and requested detail in the question.
2. Start with one broad `general_search`. Never use more than two `general_search` calls or repeat a query with minor wording changes.
3. Use `search_temporal_context` only when a candidate event needs nearby actions or dialogue.
4. Before completing, check that the proposed answer fully addresses the question, is directly supported, and is not contradicted by collected evidence. Stop when it is supported.

Call EXACTLY ONE tool.
```

### 14.4 Location strategy

对应 `prompt_planner_strategy_location`：

```text
## Location Workflow
Goal: reconstruct the requested state of one exact object.
1. Preserve the complete target phrase and infer the requested temporal state from the question. Do not substitute a related object or the latest mention.
2. Use `general_search` for that target and state. Call `search_object_events` only when an ordered object timeline would add information beyond the search results.
3. Use `search_temporal_context` only when the strongest event needs surrounding actions or dialogue.
4. If an object timeline identifies a visual ambiguity, use `watch_video_clip` only on a validated candidate clip with a focus that names the target and unresolved distinction.
5. Stop when one state-consistent location is supported and answer with the necessary spatial hierarchy.

Keep one short location timeline in `Findings so far:` with `[clip] location | source/destination/current`.
Call EXACTLY ONE tool.
```

Event-location 使用独立的紧凑 strategy，只调用 `general_search` 和 `search_temporal_context`。

### 14.5 Action Frequency strategy

对应 `prompt_planner_strategy_action_frequency`。普通物品数量不会看到这段 prompt。

```text
## Action Frequency Workflow
An intermediate memory is a working summary of the counting unit, candidate events, merged duplicates, and current total. It is not authoritative when it conflicts with explicit evidence.
1. Define one completed occurrence of the target action as the `counting_unit`.
2. Search the exact action once. If needed, use one object-only `general_search`. Never use a third general search.
3. Use `search_temporal_context` on candidate clips to separate completed episodes from setup steps and merge repeated descriptions across adjacent clips.
4. If graph evidence does not show repetitions inside a candidate episode, use `search_action_evidence` when its raw descriptions or dialogue can resolve the count.
5. Set each confirmed event's `occurrence_count` to the number of completed counting units inside it. One event may contribute more than one occurrence.
6. Classify ledger events as `confirmed`, `rejected`, `merged`, or unresolved `candidate` according to the available evidence.
7. Count a new episode only after a reset, clear stop and restart, new actor, or later independent episode. Preserve explicit multiplicity within an episode.
8. Never infer zero solely because retrieval returned no matching event. Complete when the collected evidence supports the best available count, reconciling the ledger with explicit evidence.

`watch_video_clip` is not exposed to this route.
Call EXACTLY ONE tool.
```

### 14.6 强制收尾 prompt

当进入 planner 时 `budget <= 1`，不再 bind tools，使用 `prompt_answer_with_search_results_final`：

```text
## FINAL ROUND
You have exhausted your search budget. You MUST NOT call any more tools. Based on ALL the evidence collected in the conversation history, reason about the question and provide your best answer now. If the evidence is incomplete, make the most reasonable guess based on what you have.

Before answering, update the short `Findings so far:` line. Preserve every distinct supported answer when the question requires multiple details. Then call `complete_task`.
```

Location 额外附加：

```text
In the final findings, choose the location matching the requested time/state. Do not merge alternative object nodes. Preserve every supported level of spatial detail needed to identify the location.
```

Action Frequency 额外附加：

```text
Use the action-frequency memory as a working summary and reconcile it with explicit evidence. Count completed occurrences, merge duplicate descriptions and continuous actions, and never infer zero only from missing retrieval results.
```

### 14.7 Final Answer prompt

对应 `prompt_final_answer`：

```text
You are the Final Answer synthesizer. Using all the collected evidence in the conversation history, provide a concise, direct answer to the original question. Respond in exactly ONE SENTENCE. Do NOT include explanations, meta-commentary, or justifications.

IMPORTANT:
- Answers like "I don't know", "The information is not sufficient", or "It is unclear" are STRICTLY FORBIDDEN.
- If you are uncertain, you MUST make the most reasonable guess based on the available evidence in the history.
- Reuse exact terms from the question and search results.
- If multiple distinct answers are supported, include all of them rather than selecting only one.
- For yes/no questions, start with Yes or No.
```

Location 额外附加：

```text
For this location question, answer with the location matching the requested time/state and preserve the most specific supported spatial hierarchy.
```

Action Frequency 额外附加：

```text
Use the action-frequency memory as a working summary, but reconcile it with explicit evidence before giving the final count. Missing retrieval results alone do not support an answer of zero.
```

### 14.8 Video rewatch prompt

`watch_video_clip` 会将 planner 提供的 `focus` 放在最前面，然后拼接以下固定文本：

```text
Focus: {focus}

Watch this 30-second video clip (sequential frames in chronological order) and describe what happens related to the focus. Be concise and factual. Only report what you can actually see in the frames.
Report every supported level of spatial detail and distinguish source, destination, and current state.
```

### 14.9 `general_search` query parser 的关键规则

完整 `prompt_parse_query` 较长，包含 graph schema、allocation 和多个示例。与当前 reasoning 调整最相关的是：

```text
Special Rules for Location Queries:
- Object-first weighting is mandatory: If the question centers on a concrete object, give that object's source or target weight 0.9-1.0. Unknown actors, unknown locations, and generic relation words must not outweigh the object.
- Cover both states and transitions: When useful, emit one state triple and one transition triple.
- Preserve hierarchical locations as single entities.
- "where is X now?": prioritize the most recent state edges.
- "where should X be placed?": search placement instructions.
- "where can robot get X?": search source locations.
- Prioritize low-level edges for location queries.
```

这里仍然只是 query parsing 和 weighted-triple retrieval 规则。它不会直接决定最终答案，也不会替代 planner 对 clip、时间状态和视频证据的判断。
