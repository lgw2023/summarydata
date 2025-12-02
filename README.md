````markdown
# TASK.md — 数据合成与 LLM 评估项目

> 目标：从零搭建一套“给定上下文，生成高质量回复 + LLM-as-judge 打分 + 正负样本构建”的完整代码工程。

---

## 0. 项目整体目标与产出

### 0.1 项目目标

1. **从 Excel 表格构建上下文**（prompt context）。
2. **对每个样本生成三类回复**：
   - 实验模型回复（internal / experimental model）。
   - 开源 SOTA 模型回复（huggingface / 本地推理）。
   - 闭源 SOTA 模型回复（各家 API）。
   - （可选）已有参考回复（reference answer）。
3. **将上述回复统一整理成对比样本集**。
4. **用 LLM-as-judge 进行盲评打分**：
   - 使用 *ground 提示词*（ground prompt）。
   - 使用 *structure 提示词*（structure prompt）。
   - 分别打分并解析结果。
5. **统计分析三类回复的表现**，包括均值、方差、分布。
6. **对所有回复进行排序并人工抽检排序合理性**。
7. **在排序可信的前提下，构建训练数据**：
   - 排序高的作为正样本（positive）。
   - 排序低的作为负样本（negative）。
   - 存储为后续训练/对齐可直接使用的数据格式（如 JSONL）。

### 0.2 预期产出

1. 完整代码工程（Python 为主，后续可扩展）。
2. 可复现的 **数据生成 & 评估 pipeline**：
   - 命令行入口，例如：
     - `python scripts/run_pipeline.py --config configs/default.yaml`
3. 结构化输出数据：
   - `data/processed/samples.jsonl`：包含上下文 + 各模型回复。
   - `data/processed/judge_results.jsonl`：包含 LLM-as-judge 打分。
   - `data/processed/ranked_pairs.jsonl`：清洗后的正负样本。
4. 统计与可视化报告（可选）：
   - `reports/score_stats.xlsx`
   - `reports/score_plots/*.png`

---

## 1. 技术栈 & 目录结构约定

### 1.1 技术栈假设（可按需调整）

- 语言：**Python 3.10+**
- 依赖：
  - `pandas` / `openpyxl`：读取 Excel。
  - `pyyaml`：配置管理。
  - `httpx` / `requests`：调用闭源模型 API。
  - `transformers` / `vllm`（可选）：开源模型推理。
  - `numpy`, `scipy`：统计分析。
  - `loguru` 或 `logging`：日志。
  - `tqdm`：进度条。
- 版本管理：`poetry` 或 `pip + requirements.txt`。

### 1.2 初始目录结构

```text
project_root/
  configs/
    default.yaml                 # 全局配置（路径、模型、API 等）
  data/
    raw/                         # 原始 Excel 等
    intermediate/                # 中间产物（构建上下文后的样本）
    processed/                   # 评估结果、排序结果等
  prompts/
    ground_prompt.txt            # ground 提示词
    structure_prompt.txt         # structure 提示词
  src/
    __init__.py
    config/                      # 配置加载 & 校验
    data_loader/                 # Excel -> 样本 & 上下文构建
    generators/                  # 各类模型调用封装
    judges/                      # LLM-as-judge 封装
    scoring/                     # 打分解析 & 聚合
    ranking/                     # 排序 & 正负样本抽取
    analysis/                    # 统计分析 & 可视化
    utils/                       # 通用工具
  scripts/
    run_pipeline.py              # 一键跑全流程
    generate_responses.py        # 只跑回复生成
    run_judge.py                 # 只跑 LLM-as-judge
    analyze_scores.py            # 只跑分析
  reports/
    .gitignore
  TASK.md                        # 本文档
  requirements.txt / pyproject.toml
````

---

## 2. 数据与样本格式规范

### 2.1 本项目 `data.csv`、`score_prompt.py` 与 `response_prompt.py` 说明

当前项目已经有一份示例数据表 `data.csv`，以及两类提示词定义文件 `score_prompt.py`（用于 **LLM-as-judge 打分**）和 `response_prompt.py`（用于 **根据上下文采样生成回复**）。

#### 2.1.1 `score_prompt.py`（评估提示词）

`score_prompt.py` 主要用于“评估阶段”，与 `data.csv` 的映射关系如下：

- **输入 query（input_data 基础）**：来自 `data.csv` 的 `典型query` 列；如果 `last_answer_phone` 不为空，则会把上一轮助手回复与当前提问按照对话模板拼接成多轮对话，组成 `input_data`。
- **模块数据（modules_block）**：来自 `data`、`suggest`、`rag` 三列，这三列会被格式化为一个统一的模块数据文本块（带简单分区/标题），作为评估时的标准依据。
- **已有参考回复（reference_answer）**：分别取自 `a_answer` 与 `b_answer` 两列；对于同一行样本，在固定 `input_data` 和 `modules_block` 的前提下，分别把 `a_answer` 和 `b_answer` 代入 `answer` 占位符，形成两份完整的评估提示词，供 LLM-as-judge 独立打分。
- **标签（winner）**：表示在人工或规则判断下更优的一侧（如 `a` / `b`），可用于提供额外可视化信息，辅助人工抽检排序合理性。

#### 2.1.2 `response_prompt.py`（回复生成提示词）

`response_prompt.py` 用于“生成阶段”，定义了面向终端用户问答场景的系统提示词模板 `SYSTEMTs_PROMPT_PHONE_GENERAL`，其核心作用是：

- **统一人设与回复策略**：将“小艺”健康管家的人设、语言风格、回答范围与安全边界等一次性固化在系统提示词中，保证不同模型、不同样本下回复风格一致。
- **约定输入结构**：要求真实调用时按照以下模块化格式组织上下文：
  - `[个人数据]`（对应 `data.csv` 中与用户画像/状态相关的字段）；
  - `[专家建议]`（对应规则或专家给出的结构化建议，如 `suggest` 列）；
  - `[知识库知识]`（对应检索到的 RAG 相关内容，如 `rag` 列，按编号+title+content 组织）；
  - `[课程库]`（候选课程列表及简要描述）；
  - `[对话历史]`（上一轮 user/assistant 对话）；
  - `[用户提问]`（当前轮用户 query）。
- **用于采样候选回复**：在实际生成候选答案时，可以将 `SYSTEMTs_PROMPT_PHONE_GENERAL` 与按上述结构拼接好的上下文文本合并，作为生成模型的完整输入 prompt，对不同模型（实验模型/开源模型/闭源模型）进行采样，得到多样化的候选回复。

简而言之：`response_prompt.py` 负责“如何喂给模型信息并让它说人话”，`score_prompt.py` 负责“在统一上下文下如何评价这些回复好坏”。下文的 Excel/JSONL 规范可以理解为对这套 `data.csv` + `score_prompt.py` + `response_prompt.py` 方案的工程化抽象。

### 2.2 Excel 输入格式（示例约定）

> **任务：** 根据现有 Excel 格式，定义统一的中间样本字段。

假设 Excel 每行对应一个样本，包含字段（示例）：

* `id`：样本 ID（如果没有，代码中自动生成）。
* `question` / `query`：用户提问。
* `context_col1`, `context_col2`, ...：构建上下文所需字段。
* `reference_answer`（可选）：已有参考回复。

**任务：** 在 `src/data_loader/schema.py` 中定义数据结构（可用 `dataclasses`）：

```python
@dataclass
class RawSample:
    sample_id: str
    question: str
    context_fields: Dict[str, Any]
    reference_answer: Optional[str] = None
```

### 2.2 构建后的上下文样本格式

中间产物（写入 `data/intermediate/context_samples.jsonl`）：

```json
{
  "sample_id": "xxx",
  "context": "...根据多个 Excel 列拼成的上下文字符串...",
  "question": "原始问题",
  "meta": {
    "source_row_index": 12,
    "extra": {}
  },
  "reference_answer": "..."  // 如果有
}
```

### 2.3 模型回复统一格式

每个样本最终会有多条候选回复：

```json
{
  "sample_id": "xxx",
  "context": "...",
  "question": "...",
  "candidates": [
    {
      "candidate_id": "exp_model_0",
      "model_type": "experimental",   // experimental | open_source | closed_source | reference
      "model_name": "exp_v1",
      "response": "模型输出文本",
      "gen_config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512
      }
    },
    {
      "candidate_id": "open_llama_0",
      "model_type": "open_source",
      "model_name": "Llama-3-8B",
      "response": "模型输出文本",
      "gen_config": { ... }
    }
  ]
}
```

文件：`data/processed/generated_responses.jsonl`

### 2.4 LLM-as-judge 结果格式

对每个 `candidate`，用两个提示词分别打分：

```json
{
  "sample_id": "xxx",
  "candidate_id": "exp_model_0",
  "model_type": "experimental",
  "model_name": "exp_v1",
  "scores": {
    "ground": {
      "score": 4.5,
      "max_score": 5,
      "raw_judge_output": "原始 judge 模型回复文本"
    },
    "structure": {
      "score": 3.8,
      "max_score": 5,
      "raw_judge_output": "原始 judge 模型回复文本"
    }
  },
  "aggregate_score": 4.15,         // 例如简单平均，可在 scoring 模块配置
  "judge_meta": {
    "judge_model": "gpt-4.1-mini",
    "timestamp": "2025-12-02T10:00:00Z"
  }
}
```

文件：`data/processed/judge_results.jsonl`

### 2.5 排序结果 & 正负样本格式

对于每个 `sample_id`，得到排序后列表，并抽取正负样本对：

```json
{
  "sample_id": "xxx",
  "ranking": [
    {
      "candidate_id": "exp_model_0",
      "aggregate_score": 4.5,
      "rank": 1
    },
    {
      "candidate_id": "open_llama_0",
      "aggregate_score": 3.8,
      "rank": 2
    }
  ],
  "pairs": [
    {
      "sample_id": "xxx",
      "pos_candidate_id": "exp_model_0",
      "neg_candidate_id": "open_llama_0",
      "pos_response": "....",
      "neg_response": "....",
      "context": "...",
      "question": "...",
      "pos_score": 4.5,
      "neg_score": 3.8
    }
  ]
}
```

文件：`data/processed/ranked_pairs.jsonl`

---

## 3. 模块划分与开发任务

### 3.1 配置与公共工具（`src/config`, `src/utils`）

**目标：** 提供统一的配置和公共工具函数。

**子任务：**

* [ ] `config_loader.py`：从 `configs/*.yaml` 读取配置，解析为 `Config` 对象。
* [ ] `logging_utils.py`：初始化日志格式和日志等级。
* [ ] `io_utils.py`：封装读取/写入 JSONL、Excel、CSV 的工具函数。
* [ ] `id_utils.py`：为样本和候选生成稳定可复现的 ID（如 hash）。

---

### 3.2 数据加载与上下文构建（`src/data_loader`）

**目标：** 从 Excel 读取原始数据，构建上游标准化样本与上下文。

**子任务：**

1. **Excel 读取**

   * [ ] `excel_loader.py`：

     * 输入：Excel 路径，sheet 名，列名配置。
     * 输出：`List[RawSample]`。
2. **上下文构建逻辑**

   * [ ] `context_builder.py`：

     * 从多个列组合出 `context` 字符串（例如模板：`"用户信息：...\n历史记录：...\n附加说明：..."`）。
     * 支持通过配置控制拼接规则。
3. **样本序列化**

   * [ ] 将 `RawSample` + `context` 转为 JSONL，写入 `data/intermediate/context_samples.jsonl`。
4. **边界处理**

   * [ ] 处理缺失值、无效行（例如 question 为空）。

---

### 3.3 回复生成模块（`src/generators`）

**目标：** 调用不同类型模型，针对每个样本生成多种回复输出。

#### 3.3.1 实验模型（`experimental_generator.py`）

* [ ] 封装内部实验模型接口：

  * 例如本地服务 HTTP API / gRPC 接口。
* [ ] 实现 `generate_responses_for_sample(sample)`：

  * 输入：`context` + `question`。
  * 输出：一条或多条候选回复，并附带采样参数。

#### 3.3.2 开源模型（`open_source_generator.py`）

* [ ] 集成 `transformers` 或本地推理框架（如 `vllm`）：

  * 加载模型：如 `Llama-3-8B` / `Qwen` 等（由配置决定）。
* [ ] 支持批量推理，减少开销。
* [ ] 同样实现 `generate_responses_for_sample`.

#### 3.3.3 闭源模型（`closed_source_generator.py`）

* [ ] 集成多个供应商（如 OpenAI、Anthropic、月之暗面等，名称仅作为示例）：

  * 使用统一的抽象接口 `LLMClient`。
* [ ] 配置中指定使用哪些模型、并行度、重试策略、rate limit 控制。
* [ ] 注意：**对调用成本进行控制**（批量、节流、日志）。

#### 3.3.4 回复整合（`response_aggregator.py`）

* [ ] 接收上述三类模型输出 + reference answer（如果有）。
* [ ] 为每个 `candidate` 生成统一的结构（见 2.3）。
* [ ] 对 `model_type`、`model_name` 做统一枚举管理。

---

### 3.4 回复混合与样本规范化（`src/generators/merger.py`）

**目标：** 将实验模型、开源模型、闭源模型和参考回复统一混合成候选集，以便后续 LLM-as-judge。

**子任务：**

* [ ] 定义混合策略：

  * 每个样本至少包含：`[实验模型, 开源模型, 闭源模型]`，如果 reference 存在则附加。
* [ ] 确保每个样本的候选数目在可控范围（如 3–5 个）。
* [ ] 写入 `data/processed/generated_responses.jsonl`。

---

### 3.5 LLM-as-judge 评估模块（`src/judges`）

**目标：** 使用 ground 提示词 & structure 提示词对候选进行盲评打分。

#### 3.5.1 提示词管理（`score_prompt.py` / `prompt_loader.py`）

当前 Demo 中，ground / structure 两类评估提示词统一定义在根目录的 `score_prompt.py` 中：

- `GROUND_PROMPT_TPL`：Grounding/Consistency 评估提示词模板。
- `STRUCT_PROMPT_TPL`：Structure/Policy 评估提示词模板。
- 二者均通过 `{input_data}`、`{modules_block}`、`{answer}` 三个占位符接收实际内容：
  - `{input_data}`：由 `data.csv` 中的 `典型query` 和可选的 `last_answer_phone` 按对话模板拼接而成。
  - `{modules_block}`：由 `data`、`suggest`、`rag` 三列格式化组合而成。
  - `{answer}`：传入 `answer`，形成待评估答案。

后续若工程化拆分，可在 `src/judges/prompt_loader.py` 中增加封装：

* [ ] 从 `prompts/ground_prompt.txt` 与 `prompts/structure_prompt.txt` 或其他配置源读取内容。
* [ ] 支持不同语言、不同任务版本（可通过配置切换）。

#### 3.5.2 盲评策略（`judge_runner.py`）

* [ ] **盲评要求**：

  * Judge 只看到上下文 & 问题 & 候选回复文本。
  * **不能暴露**候选来源信息（不告诉是哪个模型）。
* [ ] 对于每个 `candidate`：

  * 使用 `ground_prompt` 调用 judge 模型一次。
  * 使用 `structure_prompt` 再调用一次。
* [ ] 需要约定 judge 模型输出格式，例如让 judge 输出 JSON：

  * 如：`{"score": 4.5, "max_score": 5, "reason": "..."}`。
* [ ] 实现重试、超时、错误重跑机制。

---

### 3.6 打分解析与统计分析（`src/scoring`, `src/analysis`）

#### 3.6.1 打分解析（`parser.py`）

* [ ] 从 judge 模型原始文本解析出：

  * 分数（`score`，浮点数）。
  * 最大分数（`max_score`）。
  * 解释/理由（可选）。
* [ ] 对异常格式进行 robust 处理（回退策略、日志警告）。

#### 3.6.2 分数聚合（`aggregator.py`）

* [ ] 将 ground & structure 两个分数合成 `aggregate_score`：

  * 简单方案：`(ground_score + structure_score) / 2`。
  * 保留原始分数以便后续分析。
* [ ] 写入 `data/processed/judge_results.jsonl`。

#### 3.6.3 统计分析（`stats.py`）

* [ ] 按 `model_type` / `model_name` 统计：

  * 样本数量。
  * 平均分、标准差、中位数。
  * 分数分布（直方图）。
* [ ] 输出：

  * `reports/score_stats.xlsx`
  * `reports/score_stats.json`
* [ ] 可选：绘制分布图 `reports/score_plots/`。

---

### 3.7 排序与正负样本抽取（`src/ranking`）

**目标：** 对每个样本的候选按 `aggregate_score` 排序，并在排序合理时生成正负样本。

#### 3.7.1 排序实现（`ranker.py`）

* [ ] 对每个 `sample_id`，收集所有 `candidate` 的 `aggregate_score`。
* [ ] 降序排序，赋予 `rank`（从 1 开始）。

#### 3.7.2 排序合理性检查（`sanity_check.py`）

> “可视化输出，人工观察排序合理性，当排序合理时，取排序低的作为负样本，排序高的作为正样本。”

* [ ] 定义可配置规则，用于自动筛选“排序可信”的样本：

  * 例如：

    * 最高分与最低分差值 >= 某阈值（如 0.5）。
    * ground 与 structure 的排序一致率 > 某阈值。
* [ ] 提供辅助 CLI：

  * `python scripts/inspect_rankings.py --sample-id xxx`
  * 打印某个样本的上下文、问题、候选、分数 & 排序，供人工 spot check。
* [ ] 对通过规则的样本标记 `is_rank_trustworthy = True`。

#### 3.7.3 正负样本构建（`pair_builder.py`）

* [ ] 对于 `is_rank_trustworthy` 的样本：

  * 选取 **排序最高** 和 **排序最低** 的候选，构建一对 `(pos, neg)`。
  * 可扩展为多对，例如 top-k vs bottom-k。
* [ ] 输出 `data/processed/ranked_pairs.jsonl`，格式见 2.5。
* [ ] 在配置中支持：

  * 最小分差阈值。
  * whether 使用 top-1 vs bottom-1，或 top-2 vs bottom-2 交叉配对。

---

### 3.8 脚本与一键流程（`scripts/`）

**目标：** 提供命令行入口，支持分步骤和一键跑全流程。

#### 3.8.1 一键全流程脚本（`run_pipeline.py`）

* [ ] 执行顺序：

  1. 读取配置。
  2. 加载 Excel 并构建上下文样本。
  3. 调用各模型生成回复并合并。
  4. 调用 judge 进行打分。
  5. 解析打分，生成统计报告。
  6. 排序并生成正负样本。
* [ ] 支持参数：

  * `--config`：配置文件路径。
  * `--stage`：可选，只跑 `generate`, `judge`, `rank` 等特定阶段。

#### 3.8.2 分阶段脚本

* [ ] `generate_responses.py`
* [ ] `run_judge.py`
* [ ] `analyze_scores.py`
* [ ] `build_pairs.py`

---

## 4. 质量控制与测试

### 4.1 单元测试与集成测试

* [ ] 为关键模块编写单元测试：

  * `data_loader`：Excel -> context 是否符合预期。
  * `judges.parser`：对典型 judge 输出的解析。
  * `ranking`：排序与 pair 构建逻辑。
* [ ] 小规模端到端测试：

  * 用 5–10 条 Excel 样本，跑全流程，确保所有文件成功生成。

### 4.2 日志与监控

* [ ] 对外部 API 调用进行详细日志记录（不记录敏感数据）。
* [ ] 对解析异常和失败重试进行告警（日志级别 WARNING / ERROR）。
* [ ] 对耗时较长阶段打印进度条和时间统计。

---

## 5. 后续扩展方向（非必须）

* [ ] 引入多 judge 模型 ensemble，提升评估稳健性。
* [ ] 引入更多维度评分（安全性、事实性等），构建多标签训练数据。
* [ ] 将正负样本导出为特定训练格式（例如 RLHF / DPO / ORPO 等）。
* [ ] 将整个 pipeline 做成 Airflow / Prefect 工作流，实现定期自动跑数。

---

## 6. 开发优先级建议

1. **MVP 流程**（优先完成）：

   * Excel → 上下文 → 实验模型 + 1 个开放模型 + 1 个闭源模型 → LLM-as-judge（1 个 judge 模型，ground + structure）→ 排序 → 输出正负样本。
2. **增强 &优化**：

   * 更多模型接入、更多统计指标、可视化。
   * 更细致的排序合理性自动检测。
3. **工程化**：

   * 配置化、参数化、日志、错误恢复、测试覆盖率。

---

> 完成以上 TASK.md 后，后续可以直接按照模块与子任务逐项实现、打勾，确保整个数据合成与评估 pipeline 按预期落地。

```
```
