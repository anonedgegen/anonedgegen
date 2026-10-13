# Advantage-Weighted RL Fine-Tuning for Agentic LLMs

Fine-tune language models on agent interaction traces using advantage-weighted
supervised learning with LoRA.

## Method

Each training trace is weighted by how much better (or worse) it performed
relative to other traces on the same task:

```
advantage_i = (score_i - mean(group)) / std(group)
weight_i    = exp(beta * advantage_i) / mean(exp(beta * advantage))
```

- `beta=0`: Uniform weighting (standard behavioral cloning)
- `beta>0`: High-advantage traces upweighted, low-advantage downweighted

## Files

```
├── README.md           # This file
├── train.py            # Main training script
├── prepare_data.py     # Convert evaluation traces → training JSONL
└── merge_adapter.py    # Merge LoRA adapter with base model
```

## Requirements

```
torch>=2.13.0
transformers>=5.8.1
trl>=1.4.0
peft>=0.21.0
datasets
numpy
```

## Quick Start

### 1. Prepare Data

Convert evaluation traces to training format:

```bash
python prepare_data.py \
    --results-dir path/to/experiment_outputs/ \
    --system-prompt path/to/system_prompt.txt \
    --tool-schemas path/to/tool_schemas.json \
    --output-dir data/
```

Produces `data/rl_train.jsonl` and `data/rl_val.jsonl`.

### 2. Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model Qwen/Qwen3.5-4B \
    --train-data data/rl_train.jsonl \
    --val-data data/rl_val.jsonl \
    --output-dir outputs/run1 \
    --beta 0.5
```

Adapter checkpoints are saved every `--save-every` steps:
```
outputs/run1/
├── step_5/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── base_model_path.txt
├── step_10/
└── ...
```

### 3. Merge Adapter

Merge the LoRA adapter into a standalone model:

```bash
python merge_adapter.py \
    --adapter outputs/run1/step_10 \
    --output outputs/run1/step_10_merged
```

## Training Data Format

Each line in the training JSONL:

```json
{
  "messages": [
    {"role": "system", "content": "You are a customer service agent..."},
    {"role": "user", "content": "I need to cancel my flight"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"type": "function", "id": "call_1", "function": {
        "name": "get_reservation_details",
        "arguments": {"confirmation_number": "ABC123"}
      }}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"status\": \"confirmed\"}"},
    {"role": "assistant", "content": "{\"message\": \"I found your reservation...\"}"}
  ],
  "tools": [{"type": "function", "function": {"name": "get_reservation_details", ...}}],
  "reward": 0.85,
  "advantage": 1.23
}
```

## train.py Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | *required* | Base model path or HuggingFace ID |
| `--train-data` | *required* | Training JSONL path |
| `--val-data` | *required* | Validation JSONL path |
| `--output-dir` | *required* | Directory for adapter checkpoints |
| `--beta` | 0.5 | Advantage sharpness (0 = uniform) |
| `--lr` | 2e-5 | Learning rate |
| `--lora-r` | 8 | LoRA rank |
| `--lora-alpha` | 2×r | LoRA alpha |
| `--epochs` | 2 | Training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation steps |
| `--max-length` | 16384 | Maximum sequence length |
| `--save-every` | 5 | Checkpoint interval (steps) |
| `--seed` | 42 | Random seed |
