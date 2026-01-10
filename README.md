## summarydata

面向“运动健康问答/总结”场景的数据流水线：从 CSV/Excel 生成多模型候选回复 → 用 LLM-as-judge 做 KTO 打分（支持多轮重复、断点续跑）→ 可视化人工打分 → 导出训练用 `messages.jsonl`（SFT/KTO）。

### 能做什么

- **个人数据样式抽取（独立的数据分类、清洗、解析、格式化重构输出模块）**：`scripts/class_personal_data.py`
  - 对 Excel 的 `data` 列做“行级样式 JSON”抽取与统计分析  
  - `src/data_clean/` 提供对“个人数据文本”的解析、归一化、表格化与按时间桶聚合


- **候选生成**：`scripts/run_pipeline.py`  
  - 读取 CSV/Excel，构造标准化上下文 `context`  
  - 自动拆分为 **phone/watch 两个样本变体**（`<sample_id>::phone`、`<sample_id>::watch`）  
  - 支持多模型（实验/开源/闭源/OpenAI 兼容接口）生成候选 + 参考答案（reference）候选  
  - 支持 **断点续跑**（跳过已完整生成的样本）与 **reference 自动刷新**（不重跑大模型）

- **KTO 打分（双裁判 + 程序规则）**：`scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py`  
  - Ground / Structure 两个 judge（可配置不同 base_url/api_key/model）  
  - 结合确定性校验规则，输出 **0–20 total_score**，并映射到 **0–5 aggregate_score**  
  - 支持 `--num_repeat` 多次重复评估，并生成聚合结果

- **增量合并 + 补打分**：`scripts/merge_and_score_addtional_response.py`  
  - 把 `generated_responses_*.jsonl` 的新候选合并到基线数据集  
  - 只对“新增 candidate”补打分，产出 `generated_responses_merged.jsonl` / `judge_results_kto_merger.jsonl`

- **可视化与人工打分**：`scripts/visualize_data_app.py`（Streamlit）  
  - 浏览 `data/**.jsonl`  
  - 在 `judge_results_kto*.jsonl` 上逐样本给候选打 **0–5 分**  
  - 自动写出 `*_rank_manually.jsonl`

- **训练数据导出**：`scripts/run_trainningdata.py`  
  - 合并 `generated_responses.jsonl` + `*_rank_manually.jsonl`  
  - 导出 `messages.jsonl`，并额外导出 `messages_sft.jsonl` / `messages_kto.jsonl`

---

## 环境准备

### Python 依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### `.env`（推荐）

项目入口脚本会调用 `src/utils/env.py:load_env()`：若安装了 `python-dotenv`，会自动加载项目根目录下的 `.env`。

你通常需要配置以下几类环境变量（示例仅展示字段形状，按你实际供应商填写）：

```env
# ===== generators（run_pipeline 使用）=====
LLM_MODEL_QWEN235_URL=
LLM_MODEL_QWEN235_API_KEY=
LLM_MODEL_QWEN80_URL=
LLM_MODEL_QWEN80_API_KEY=
LLM_MODEL_QWEN32_URL=
LLM_MODEL_QWEN32_API_KEY=
LLM_MODEL_DSCHAT_URL=
LLM_MODEL_DSCHAT_API_KEY=
LLM_MODEL_DSREASON_URL=
LLM_MODEL_DSREASON_API_KEY=
LLM_MODEL_CLAUDE_URL=
LLM_MODEL_CLAUDE_API_KEY=
LLM_MODEL_QWENMAX_URL=
LLM_MODEL_QWENMAX_API_KEY=
LLM_MODEL_GEMINI25_URL=
LLM_MODEL_GEMINI25_API_KEY=

# ===== judge（KTO 打分脚本使用；GROUND/STRUCT 可分开）=====
LLM_MODEL_GROUND_URL=
LLM_MODEL_GROUND_API_KEY=
LLM_MODEL_GROUND_NAME=gpt-5-mini-2025-08-07

LLM_MODEL_STRUCT_URL=
LLM_MODEL_STRUCT_API_KEY=
LLM_MODEL_STRUCT_NAME=gpt-5-mini-2025-08-07
```

可选并发/批量参数：

```env
# run_pipeline generate 阶段：按批写盘的 batch size
GENERATE_BATCH_SIZE=16

# src/generators/base.py：同一模型内部对样本的并发数（默认 1，>1 会加大对服务的压力）
GENERATOR_SAMPLE_WORKERS_PER_MODEL=1
```

---
## 示例：用仓库内置 xlsx 快速跑通


### 个人数据 data 列的数据分类、清洗、解析、格式化重构输出

默认以summary_eval_diff.xlsx数据为例，展示对原始数据文件中，个人数据 data 列的数据分类、清洗、解析、格式化重构输出的 api。  
独立使用时需要参考该脚本的代码，适配到自己的代码中（思远和杰克，会需要用到这部分代码）  
```bash
python scripts/classify_personal_data.py
```

### 模型采样回复
针对不同数据源文件的采样生成，即完成“加载原始数据（目前默认不做数据清洗和格式化重构）->区分手机手表->构造系统提示词->构造上下文提示词->遍历各个模型进行结果采样->采样结果见 data/${file} 目录”，此处仅用“--max-rows 3”进行测试。  
（嘉诚会需要用到这部分，但不需要用到代码，仅需要用到采样的jsonl 结果，以及参考最新版手机手表的系统提示词prompts/system_prompt_v5_yixuan.py去写打分 prompt）  
```bash
# rl数据（优先使用该数据做 demo算法开发）
export file='summary_train_data_grpo_v6' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate --max-rows 3
# sft数据
export file='summary_train_v36' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate --max-rows 3
# 评测数据
export file='summary_eval_diff' && python scripts/run_pipeline.py --config configs/$file.yaml --raw-data $file.xlsx --stage generate --max-rows 3
```
### LLM-as-judge
对以上采样回复结果的打分代码，已废弃，后续这部分使用王博的代码。  
```bash
export file='summary_train_data_grpo_v6' && python scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py --workers 2 --inner_workers 12 --num_repeat 3 --raw-data $file.xlsx
```

---


## 快速开始：从原始 CSV/Excel 到训练数据

## 输入数据格式（CSV/Excel）

### 必需列

- `query`：用户问题（空行会被跳过）

### 常用可选列（用于构造上下文）

- `data`：个人数据（为空时会走 general prompt；非空会走 personal prompt）
- `suggest`：专家建议
- `rag`：知识库/检索内容
- `service`：课程库（watch 变体会自动不拼接该块）
- `last_query`：上一轮用户问题（可选）
- `last_answer_phone` / `last_answer_watch`：上一轮助手回复（按 device 变体选择）
- `domain`：领域（用于 personal prompt 的领域说明注入；详见 `prompts/domain.jsonl` 与 `prompts/system_prompt_v5_yixuan.py`）
- `sample_id`：样本 ID（缺失时按行号从 1 开始生成）

### 参考答案列（reference）

reference 列名**不写死**在代码里，而是由配置文件 `configs/*.yaml` 的 `pipeline.generators[*].answer_key` 决定。  
如果输入文件里不存在该列，对应 reference 生成器会被自动跳过。

---

## 配置文件（configs/*.yaml）

配置核心是 `pipeline` 段（`paths` 段可省略）：

- `pipeline.generators`：生成器列表  
  - `name`: `experimental` / `open_source` / `closed_source` / `reference_phone` / `reference_watch`  
  - `model_name`: 写入产物中的模型名  
  - `base_url_env` / `api_key_env`: 从环境变量取 OpenAI 兼容 endpoint 与 key  
  - `answer_key`: reference 生成器使用的“参考答案列名”
- `pipeline.judges`：保留字段（当前 `run_pipeline.py` 只负责 generate；judge 由 KTO 脚本负责）

---


下面假设你的原始数据文件为 `./my_dataset.xlsx`（也支持 `.xlsx/.xlsm`；输出目录始终按输入文件名隔离）。

### 1) 生成样本与多模型候选（generate）

`scripts/run_pipeline.py` 当前只负责 **generate**（`--stage all` 也等价于 `generate`）。

```bash
python scripts/run_pipeline.py \
  --config configs/summary_train_v36.yaml \
  --raw-data my_dataset.xlsx \
  --stage generate \
  --max-rows 0
```

说明：

- `--max-rows 0` 表示全量；用 `--max-rows 3` 可快速冒烟测试。
- 支持断点续跑：重复执行会自动跳过已完成样本；并会在结束时按 YAML 的 `answer_key` 刷新 reference 候选并压缩输出（避免同一 `sample_id` 多条旧行）。

默认输出目录会根据输入文件名自动切分为：

- `data/my_dataset/intermediate/context_samples.jsonl`
- `data/my_dataset/processed/samples.jsonl`
- `data/my_dataset/processed/generated_responses.jsonl`

### 2) 自动打分（产出 judge_results*.jsonl）

这一步会生成可视化/人工打分所需的 `judge_results*.jsonl`。

> 注意：仓库内置的 KTO 打分脚本为**历史版本/已废弃**（后续会替换为新的 judge 代码）；这里保留命令形状用于回溯与对齐产物格式。

```bash
python scripts/kto_binary_label_pipeline_dual_multi_judge_patched_v2_batch_repeats.py \
  --raw-data my_dataset.xlsx \
  --workers 16 \
  --num_repeat 3
```

默认会写入：

- `data/my_dataset/processed/judge_results_kto.jsonl`
- 多轮评估：`data/my_dataset/processed/judge_results_kto_repeats/`

### 3) 可视化与人工打分（写出 *_rank_manually.jsonl）

```bash
streamlit run scripts/visualize_data_app.py
```

在侧边栏选择一个 `judge_results*.jsonl` 文件后，为每个候选选择 0–5 分并保存，文件会写到与该 `judge_results` 同目录下，形如：

- `data/my_dataset/processed/judge_results_kto_rank_manually.jsonl`

### 4) 导出训练用 messages 数据集

默认会读取：

- `generated_responses.jsonl`
- `judge_results_kto_rank_manually.jsonl`（若你用的不是 KTO 文件名，请通过 `--manual` 显式指定对应的 `*_rank_manually.jsonl`）

```bash
python scripts/run_trainningdata.py --raw-data my_dataset.xlsx
```

默认产出：

- `data/my_dataset/processed/messages.jsonl`（含 `manual_score/model_name/label/token_len` 等）
- `data/my_dataset/processed/messages_sft.jsonl`（只保留 `manual_score=5`）
- `data/my_dataset/processed/messages_kto.jsonl`（只保留 `label=true/false`）

## 增量工作流：新增模型候选后补打分

典型场景：你用本地 vLLM 或 LoRA 模型对同一批样本生成了新候选，想合并进基线并补打分。

### 1) 生成“额外候选文件”

`scripts/run_model_infer.py` 会读取 `generated_responses.jsonl` 并输出一个仅包含“新模型候选”的文件：

```bash
python scripts/run_model_infer.py \
  --test-context data/my_dataset/processed/generated_responses.jsonl \
  --vllm-base-url http://localhost:8000/v1 \
  --vllm-model-name qwen32_sft
```

输出类似：

- `data/my_dataset/processed/generated_responses_qwen32_sft.jsonl`

### 2) 合并并只对新候选补打分

```bash
python scripts/merge_and_score_addtional_response.py \
  --raw-data my_dataset.csv \
  --workers 8 \
  --num_repeat 3
```

默认会自动发现 `data/my_dataset/processed/` 下所有 `generated_responses_*.jsonl`（排除基础与 merged 文件），并写出：

- `data/my_dataset/processed/generated_responses_merged.jsonl`
- `data/my_dataset/processed/judge_results_kto_merger.jsonl`

---

## 个人数据样式抽取（可选，仅自测用）

对 `data` 列做“行级样式 JSON”抽取，输出到：

- `data/<csv_stem>/perlsonal_datapatterns.jsonl`（注意文件名里是历史拼写：`perlsonal`）

运行：

```bash
python scripts/predict_personal_data.py --raw-data my_dataset.csv
```

只做统计分析（不调用大模型）：

```bash
python scripts/predict_personal_data.py --raw-data my_dataset.csv --analyze-patterns
```

---

## 代码结构速览

```text
requirements.txt          # Python 依赖
.env                      # 环境变量（如 API KEY；注意不要提交）
configs/                 # 生成器/流水线配置（YAML）
  summary_*.yaml         # 不同实验配置（train/eval/diff 等）
prompts/                 # system prompt / judge prompt / domain 映射
  system_prompt_*.py     # system prompt 版本
  response_prompt*.py    # 生成 prompt 版本
  score_prompt*.py       # judge/打分 prompt 版本
  domain.jsonl           # domain 映射/元信息
scripts/
  run_pipeline.py        # 生成 samples/context/candidates（断点续跑 + reference 刷新）
  run_model_infer.py     # 单次推理/生成（通常被 pipeline 调用或用于自测）
  kto_*.py               # KTO 打分（双 judge + repeat + resume）
  merge_and_score_*.py   # 合并增量候选并补打分
  visualize_data_app.py  # Streamlit 可视化 + 人工打分
  run_trainningdata.py   # 生成训练 messages 数据集
  predict_personal_data.py # 个人数据样式抽取（可选）
  classify_personal_data.py # 个人数据分类（可选/自测）
  test_personal_data.py  # 个人数据链路的简单测试脚本（自测）
src/
  config/                # YAML -> PipelineConfig
  data_loader/           # CSV/Excel 读取 + context 构建
  generators/            # OpenAI 兼容接口生成器 + candidate_id
  analysis/              # 简单统计（目前仅提供基础函数）
  data_clean/            # 个人数据解析/聚合（从旧版脚本拆分，含少量 test_*.py）
  utils/                 # env/io/logging 等工具

# 下面这些更多是“运行产物/实验结果”，不直接参与代码 import
data_stage1/             # 数据流水线运行产物（每个数据集一个子目录）
  <dataset_name>/
    intermediate/        # 中间产物（如 context_samples.jsonl）
    processed/           # 最终产物（samples/generated_responses/judge_results/messages 等）
train_stage1/            # 训练过程可视化/日志产物（如 loss/kl 曲线图）
summary_*.xlsx           # 人工评审/对比用的表格（示例/导出）
```

