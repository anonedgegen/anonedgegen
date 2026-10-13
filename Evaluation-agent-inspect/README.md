# Evaluation — Agent Quality Inspect

The Evaluation-agent-inspect folder in this repo is the extension of the open-source TED evaluation framework https://arxiv.org/pdf/2603.15483

Run agent evaluations against a running agent server using the [agent-quality-inspect](https://github.com/SAP/agent-quality-inspect) package.

## Prerequisites

The target agent server must already be running and reachable (see the
`Tau2airline_base_harness` / `Toolsandbox_base_harness` READMEs for how to start one).
By default the runner expects:

| Agent          | Default endpoint          |
|----------------|---------------------------|
| `tau2bench`    | `http://127.0.0.1:5500`   |
| `toolsandbox`  | `http://localhost:8000`   |

Override it with the `SERVER_ENDPOINT` environment variable.

The server can be any variant of the agent harness, not just the base one — e.g.
the **optimized (opt) harness** or a harness running an **open-source LLM model**.
Whichever you start, point `SERVER_ENDPOINT` at it; the evaluation runs the same way.

## 1. Install

```bash
git clone https://github.com/SAP/agent-quality-inspect
cd agent-quality-inspect
pip install -e .
```

or 

```bash
pip install agent-quality-inspect
```

## 2. Configure Azure OpenAI

Create a `.env` file in the **project root** with:

```bash
AZURE_API_VERSION=your_api_version      # e.g. 2024-02-15-preview
AZURE_API_BASE=your_azure_endpoint      # e.g. https://your-resource.openai.azure.com/
AZURE_API_KEY=your_api_key              # your Azure OpenAI API key
```

Azure OpenAI powers the user-proxy agent (`--user-proxy-model`), the LLM-based
metric judge (`--judge-model`), and error analysis.

## 3. Run an evaluation

Point `SERVER_ENDPOINT` at the running agent and launch the simplified runner.

**tau2bench:**

```bash
SERVER_ENDPOINT=http://127.0.0.1:5500 \
python -m paper_experiments.simplified_runner \
  --agent tau2bench \
  --agent-model "azure/gpt-5.4" \
  --samples-file <path-to>/Datasets/tau2airline/test.json \
  --max-turns 15 \
  --n-trials 1 \
  --max-workers 1 \
  --output-dir experiment_logs
```

**toolsandbox:**

```bash
SERVER_ENDPOINT=http://127.0.0.1:8000 \
python -m paper_experiments.simplified_runner \
  --agent toolsandbox \
  --agent-model "GPT_4_o_mini" \
  --samples-file <path-to>/Datasets/toolsandbox_splits_w_domain/toolsandbox_test.json \
  --max-turns 8 \
  --n-trials 1 \
  --max-workers 1 \
  --output-dir experiment_logs
```

Notes:
- `--samples-file` is a **JSON file** of samples (e.g. `Datasets/tau2airline/airline_edge_test.json`).
- For `toolsandbox`, `--agent-model` is the agent type (e.g. `GPT_4_o_mini`, `Gemma_4_E4B`),
  not a LiteLLM model string.

## Running with the Gemma (open-source) model

To evaluate against a deployed open-source **Gemma** model, start the matching
`*_harness_gemma` server first (it talks to your Gemma endpoint, served
OpenAI-compatibly, e.g. via vLLM). The two benchmarks wire the endpoint differently:

**tau2bench** — the endpoint is passed *through* the runner into each
`/start_conversation` request. Set `GEMMA_API_BASE` (or `CUSTOM_API_BASE`, which takes
precedence) in the runner's environment and use the LiteLLM model string as `--agent-model`:

```bash
SERVER_ENDPOINT=http://127.0.0.1:5500 \
GEMMA_API_BASE=http://localhost:8001/v1 \
python -m paper_experiments.simplified_runner \
  --agent tau2bench \
  --agent-model "openai/gemma-4-e4b" \
  --samples-file Datasets/tau2airline/test.json \
  --n-trials 1 --max-workers 1 \
  --output-dir experiment_logs
```

**toolsandbox** — the endpoint lives on the **harness server**, not the runner. Start
the server with `GEMMA_API_BASE` set, then select the Gemma agent via `--agent-model`
(which maps to `agent_type`); the runner sends no `api_base`:

```bash
# server (toolsandbox gemma harness): GEMMA_API_BASE=http://localhost:8001/v1 OPENAI_API_KEY=dummy uvicorn ... --port 8000
SERVER_ENDPOINT=http://127.0.0.1:8000 \
python -m paper_experiments.simplified_runner \
  --agent toolsandbox \
  --agent-model "Gemma_4_E4B" \
  --samples-file Datasets/toolsandbox_splits_w_domain/toolsandbox_test.json \
  --n-trials 1 --max-workers 1 \
  --output-dir experiment_logs
```

| Benchmark     | `--agent-model`      | Gemma endpoint set via                          |
|---------------|----------------------|-------------------------------------------------|
| `tau2bench`   | `openai/gemma-4-e4b` | `GEMMA_API_BASE`/`CUSTOM_API_BASE` on the runner (forwarded as `api_base`) |
| `toolsandbox` | `Gemma_4_E4B`        | `GEMMA_API_BASE` on the harness server          |

## Command-line arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--agent` | Agent type (`tau2bench` or `toolsandbox`) | `tau2bench` |
| `--agent-model` | Agent model under test (e.g. `azure/gpt-4.1`) | `azure/gpt-4.1` |
| `--max-turns` | Maximum conversation turns | `15` (tau2bench) / `8` (toolsandbox) |
| `--n-trials` | Trials per sample | `20` (tau2bench) / `8` (toolsandbox) |
| `--max-workers` | Parallel samples to run | `3` (tau2bench) / `1` (toolsandbox) |
| `--user-proxy-model` | User-proxy model | `gpt-4.1` |
| `--user-proxy-persona` | User-proxy persona (`expert` or `non-expert`) | `expert` |
| `--judge-model` | LLM-as-a-judge model for scoring | `gpt-5.4` |
| `--terminating-msg` | Message that ends a conversation | `END_CONVERSATION` |

## Output

The output directory contains:

| File | Contents |
|------|----------|
| `trial_<n>_results.json` | Per-trial results, one file per trial (incl. trajectories) |
| `aggregate_metrics_results.json` | Summary + aggregate metrics across all trials |
| `evaluation_results.pkl` | Full evaluation results (pickled) |
| `error_analysis.pkl` | Automated error analysis (pickled) |
| `evaluation.log` | Run log |

The `trial_*_results.json` and `aggregate_metrics_results.json` files feed the
[`helper_script/`](../helper_script/) post-processing (split metrics + confidence intervals).
