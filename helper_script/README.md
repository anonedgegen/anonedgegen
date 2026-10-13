# Helper Scripts

Two post-processing scripts for calculating the metric scores under different split and then plotting the confidence interval (run them in order), plus a serving script for hosting the Gemma model.

**Requirements:** Python 3, plus `pip install -r requirements.txt`.

## 1. Split metrics — `recalc_aggregate_split.py`

Splits test cases into **Original**, **Policy-based**, and **Union** based on their source, then computes aggregate metrics for each split.
Writes the results to `aggregate_metrics_split_results.json` in the job directory.

```bash
python3 recalc_aggregate_split.py <job_dir> --benchmark tau2|toolsandbox
```

Run it for every run directory (the `base` run and each optimized candidate run).

## 2. Confidence intervals — `bootstrap_ci_split.py`

Bootstraps 95% confidence intervals comparing `base` vs each candidate run, and writes
a JSON, a table log, and a comparison plot into `<model_folder>/bootstrap_ci/`.

```bash
python3 bootstrap_ci_split.py <model_folder>
```

`<model_folder>` must contain a `base/` run and one or more candidate subfolders,
each with an `aggregate_metrics_split_results.json` (from step 1):

```
<model_folder>/
├── base/        └── aggregate_metrics_split_results.json
└── <candidate>/ └── aggregate_metrics_split_results.json
```

## 3. Gemma inference server — `gemma_server.py`

An OpenAI-compatible FastAPI server for hosting the Gemma model — both the base
model and its fine-tuned (LoRA) version. It loads the model once at startup,
serves `/v1/chat/completions` with Gemma-specific tool call parsing, and returns
OpenAI-format responses that `litellm` (and other OpenAI clients) can consume
directly.

Host the base model:

```bash
CUDA_VISIBLE_DEVICES=0 python3 gemma_server.py \
    --model google/gemma-4-E4B-it \
    --port 8001
```

Host a fine-tuned version by merging a LoRA adapter at load time:

```bash
CUDA_VISIBLE_DEVICES=0 python3 gemma_server.py \
    --model google/gemma-4-E4B-it \
    --adapter /path/to/lora/checkpoint \
    --port 8001
```

Key options: `--model` (HuggingFace model ID or local path), `--adapter`
(optional LoRA checkpoint to merge), `--host` (default `0.0.0.0`), `--port`
(default `8001`), `--device` (default `cuda:0`), and `--max-tokens` (default `2048`).
