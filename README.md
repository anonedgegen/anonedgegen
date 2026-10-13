# EDGEGEN: Improving Tool-Calling Agents Beyond Happy Paths with Synthetic Edge Case Generation

This repository is the code and dataset release for **EDGEGEN**, a framework for improving
tool-calling LLM agents on the hard cases that generic synthetic data misses.

## Introduction

Tool-calling LLM agents deployed in enterprise settings must do more than invoke the right
tool — they must respect domain-specific **business rules** (eligibility constraints, policy
conditions, state-dependent restrictions). Evaluating and optimizing such agents requires
high-quality, diverse task data, which is hard to obtain because production logs are private
and hand-authored tasks are expensive. Existing synthetic generators tend to produce generic,
state-agnostic "happy-path" tasks that rarely probe real failure modes.

**EDGEGEN** is a violation-driven synthetic task-generation framework. It:

1. **Extracts compliance rules** from an agent's policy specification.
2. **Enumerates combinations** of rule violations to produce diverse edge-case scenarios.
3. **Grounds each scenario in real database state** using a scenario-aware sampler that
   selects values which make the task feasible.
4. **Verifies every generated task** before admission, rejecting infeasible ones and fixing
   hallucinated values.

Paired with a synthetic database generator, EDGEGEN forms a **fully automated closed loop**
that requires no human annotation and no production data, yet still improves performance on
human-curated test sets — through both **supervised fine-tuning** and **harness
optimization**. The method is evaluated on **τ²-bench** (airline and retail domains) and
**ToolSandbox**, across multiple model families (GPT, Gemma, Qwen).

The closed loop implemented in this repo:

```
Data_generation (EdgeGen tasks)
        │
        ▼
  agent harness server  ──►  Evaluation-agent-inspect  ──►  traces + metrics
        ▲                                                        │
        └───────────  Finetuning_code / harness optimization  ◄──┘
```

## Repository structure

Every main folder is summarized below.

| Folder | What it is |
|---|---|
| [`Data_generation/`](Data_generation/README.md) | The EDGEGEN testcase synthesis pipeline — a 4-step flow (graph → violations → sample → generate) that produces database-grounded edge-case tasks. |
| [`Evaluation-agent-inspect/`](Evaluation-agent-inspect/README.md) | Runs agent evaluations against a running harness server using the public open source package `agent-quality-inspect` with some overridden classes meant for this experiment. |
| [`Finetuning_code/`](Finetuning_code/README.md) | Advantage-weighted LoRA fine-tuning on agent interaction traces — upweights high-performing trajectories relative to the group. |
| [`helper_script/`](helper_script/README.md) | Post-processing utilities: splitting metrics (Original vs. Policy-based vs. Union test cases), computing bootstrap confidence intervals for baseline vs. optimized runs and script to host Gemma models. |
| [`huggingface/`](huggingface/README.md) | Scripts to upload/download the public EdgeGen experiment results and finetuning model checkpoints on HuggingFace. |
| [`Datasets/`](Datasets/) | Benchmark and EdgeGen sample splits consumed by the evaluator via `--samples-file` (tau2 airline/retail, ToolSandbox). |
| `Results_logs/` | Evaluation, meta-harness and fine-tuning run logs (see `meta_harness_logs/ and finetuning_logs/`). Hosted on HuggingFace — see below. |
| `Finetuning_checkpoints/` | Model snapshot checkpoints. Hosted on HuggingFace — see below. |

### Results logs and Fine Tuning Checkpoints

`Results_logs/` and `Finetuning_checkpoints/` is not included in this repository. Download it from the [anonedgegen/anonedgegen HuggingFace dataset](https://huggingface.co/datasets/anonedgegen/anonedgegen/tree/main) using the helper scripts in [`huggingface/`](huggingface/README.md).

### Agent harnesses

Each harness is a FastAPI server wrapping an agent under test. They come in **base**
(original) and **opt** (meta-harness-optimized) variants, backed either by an Azure OpenAI
model or a separately deployed open-source **Gemma** endpoint (`_gemma` variants). Point the
evaluator's `SERVER_ENDPOINT` at whichever one you start.

| Harness | Benchmark | Model | Default port |
|---|---|---|---|
| [`Tau2airline_base_harness/`](Tau2airline_base_harness/README.md) | τ²-airline | Azure GPT | 5500 |
| [`Tau2airline_base_harness_gemma/`](Tau2airline_base_harness_gemma/README.md) | τ²-airline | Gemma | 5500 |
| [`Tau2airline_opt_harness_gpt5.4/`](Tau2airline_opt_harness_gpt5.4/README.md) | τ²-airline | Azure GPT-5.4 | 5500 |
| [`Tau2airline_opt_harness_gemma/`](Tau2airline_opt_harness_gemma/README.md) | τ²-airline | Gemma | 5500 |
| [`Tau2retail_base_harness/`](Tau2retail_base_harness/README.md) | τ²-retail | Azure GPT | 5500 |
| [`Toolsandbox_base_harness/`](Toolsandbox_base_harness/README.md) | ToolSandbox | Azure GPT | 8000 |
| [`Toolsandbox_base_harness_gemma/`](Toolsandbox_base_harness_gemma/README.md) | ToolSandbox | Gemma | 8000 |
| [`Toolsandbox_opt_harness_gpt4omini/`](Toolsandbox_opt_harness_gpt4omini/README.md) | ToolSandbox | Azure GPT-4o-mini | 5501 |
| [`Toolsandbox_opt_harness_gemma/`](Toolsandbox_opt_harness_gemma/README.md) | ToolSandbox | Gemma | 5501 |

## Quick start

The core workflow runs **data generation first**, then **evaluation** against a harness.
Each step links to its folder README for the full set of options.

### 0. Prerequisites

- Python (3.10–3.12 depending on the component), with a virtual environment per folder.
- Azure OpenAI credentials exported in your environment:
  ```bash
  export AZURE_API_KEY=...
  export AZURE_API_BASE=...      # Azure Studio endpoint (not ending in /models)
  export AZURE_API_VERSION=...
  ```
- The `claude` CLI installed and on your `PATH` (used by the data-generation DB sampler and verifier).

### 1. Generate edge-case tasks — [`Data_generation/`](Data_generation/README.md)

```bash
cd Data_generation
pip install -r requirements.txt

python run_pipeline.py \
    --data-dir example_data/ \
    --db-path example_data/db.json \
    --out example_data/output/test_cases.json \
    --n 5
```

`example_data/` contains a worked example (`tools.json`, `online_subgoals.json`, `PRD.md`,
`db.json`) you can run as-is. For your own agent, supply a data directory with those four
inputs. Output is a `test_cases.json` of grounded, verified edge-case tasks.

### 2. Start an agent harness

Pick the harness matching your benchmark and model, and start its server. For example, the
τ²-airline base harness (see [its README](Tau2airline_base_harness/README.md) for setup):

```bash
cd Tau2airline_base_harness/tau2-bench
# ... set up venv and .env per the harness README ...
uvicorn fastapi_server.main:app --reload --port 5500
```

### 3. Evaluate — [`Evaluation-agent-inspect/`](Evaluation-agent-inspect/README.md)

Point `SERVER_ENDPOINT` at the running harness and launch the runner against your generated
tasks (or a split from `Datasets/`):

```bash
cd Evaluation-agent-inspect

SERVER_ENDPOINT=http://127.0.0.1:5500 \
python -m paper_experiments.simplified_runner \
  --agent tau2bench \
  --agent-model "azure/gpt-5.4" \
  --samples-file ../Data_generation/example_data/output/test_cases.json \
  --max-turns 15 --n-trials 1 --max-workers 1 \
  --output-dir experiment_logs
```

This writes per-trial results, aggregate metrics, and error analysis to the output directory.

### 4. (Optional) Close the loop

- Fine-tune a model on the evaluation traces with [`Finetuning_code/`](Finetuning_code/README.md).
- Compare baseline vs. optimized runs (metric splits + bootstrap CIs) with [`helper_script/`](helper_script/README.md).

## Viewing results without running experiments

If you do not want to run the experiments yourself, all evaluation logs and finetuning logs are available in `Results_logs/` on the [anonedgegen/anonedgegen HuggingFace dataset](https://huggingface.co/datasets/anonedgegen/anonedgegen/tree/main). Download them using the helper scripts in [`huggingface/`](huggingface/README.md).
