# Reasoning Accuracy Rerun Analysis

分析对象：`data/ablation/reasoning_accuracy_test/`

该目录只包含此前 13-video baseline 中的 67 道错题。此次重跑复用了已有 graph、episodic memory 和 frames，没有重跑 memorization。

## 1. 结果

- 新结果：19/67 = 28.36%
- 仍然错误：48/67
- 原 13-video baseline：109/176
- 若原来正确的 109 题保持不变，投影结果为 128/176 = 72.73%

这个 72.73% 是基于错题子集的投影，不是对全部 176 题重新评测得到的无偏结果。

按当前 route：

| Route | Correct | Wrong | Accuracy on old errors |
|---|---:|---:|---:|
| Location | 6 | 23 | 20.69% |
| Action Frequency | 0 | 4 | 0.00% |
| General | 13 | 21 | 38.24% |

## 2. 工具行为

48 道仍然错误的题共调用：

| Tool | Calls |
|---|---:|
| `general_search` | 135 |
| `search_temporal_context` | 58 |
| `watch_video_clip` | 8 |
| `complete_task` | 48 |

其中 9 道错题连续调用了 5 次 `general_search`，随后因 budget 用尽而直接结束。主要问题不是搜索次数不足，而是重复搜索挤占了 temporal verification 和 video rewatch 的预算。

Location 的对比也支持这一点：

- 答对的 6 题中有 4 题使用视频重看。
- 仍错的 23 题中只有 6 题使用视频重看。

4 道 Action Frequency 题全部错误，并且没有调用一次视频重看。

## 3. 主要错误原因

### 3.1 重复搜索，没有进入 clip 验证

Agent 经常将同一查询换词后继续调用 `general_search`。当 graph 中没有直接出现目标 relation 时，它会重复搜索到最后一轮，而不是选择 object-interaction clips，通过原始 episodic context 或视频恢复信息。

代表问题：

- `bedroom_06_Q03`：连续搜索 wardrobe/open/closet 五次，最终回答 0。
- `study_18_Q03`：多次搜索 folder 的同义表达，始终没有进入候选 clip。

### 3.2 Location 找到相关对象，但选错状态或精确层级

常见错误包括：

- 混淆 source、destination 和 current location。
- 混淆相似物体，如普通 remote 与 air-conditioning remote。
- 只回答 table、shelf 等宽泛位置，遗漏 drawer、tier、side 或 container。
- 文本与视觉冲突时没有重看视频。
- 重看了错误 clip，或问题需要的状态发生在其他 clip。

代表问题包括遥控器当前位置、书本最终位置、冰箱层级、书架层级和 pen holder。

### 3.3 Action Frequency 依赖不完整的文本事件

Graph 和 episodic descriptions 不一定显式记录每一次重复动作：

- wardrobe opened：预测 0，GT 4
- poster printed：预测 3，GT 6
- desktop organized：预测 2，GT 3
- mop used：预测 1，GT 2

这四题都是漏计。对于 clip 内重复动作，仅做 text retrieval 无法可靠得到次数。

### 3.4 多答案问题丢失已找到的证据

Agent 有时已经找到多个答案，之后却只保留一个：

- Bob 使用了 tissue 和 towel，最终只回答 tissue。
- Bob 擅长 liberal arts 和 sports，最终只回答 sports。
- Felix 和 Daniel 都整理了 office，最终只回答 Felix。

这属于 reasoning/final synthesis 的 completeness 问题，不一定需要修改 memory。

### 3.5 Memory 上限与 judge 粒度

部分 GT 信息在当前搜索结果和原始描述中均不明显，例如 folder 中的 USB/task list、wardrobe opening 和 box 中 gift count。这类问题仅调整 prompt 可能无法解决。

另有少量近义答案受到 judge 粒度影响，例如 `icy drinks` 与 `iced water`。它们应单独人工复核，不应通过数据集专用 prompt 修补。

## 4. 本轮 Reasoning 修改

本轮只修改通用 reasoning 行为：

1. `general_search` 最多两次，禁止只做同义改写。
2. 连续两次 general search 后，verifier 要求转向 temporal context 或视频。
3. Location 使用简短四步流程，并强化精确位置的视频验证。
4. Action Frequency 在零结果时使用 object-only fallback，文本无法分离重复动作时重看候选 clip。
5. 多答案问题必须保留所有 distinct supported answers。
6. `Findings so far` 压缩为一行，减少 prompt 和 reasoning 负担。

这些修改不改变 memorization、graph schema 或工具集合。
