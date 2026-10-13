#!/usr/bin/env python3
"""
Advantage-weighted SFT training with LoRA for agentic LLMs.

Trains a causal LM on conversational agent traces with per-example reward
weighting. High-advantage traces (better-than-group-average outcomes) are
upweighted; low-advantage traces are downweighted. When beta=0, all weights
are uniform (equivalent to standard SFT).

Supports:
  - Qwen 2.5/3.5 family (assistant-only loss, per-example reward weighting)
  - Gemma 4 family (full-sequence loss, ClippableLinear unwrapping)

Key design decisions (see README.md § Known Issues for rationale):
  1. Weights are pre-computed at the *dataset* level, not per-microbatch.
  2. Weights are passed through a custom collator — no silent fallback.
  3. model_accepts_loss_kwargs = False (correct gradient accumulation scaling).
  4. Gemma models use standard SFT loss (NaN with per-example + grad ckpt).

Usage:
    # Qwen3.5-4B with reward weighting (beta=0.5):
    CUDA_VISIBLE_DEVICES=0 python train.py \\
        --model Qwen/Qwen3.5-4B \\
        --train-data data/rl_train.jsonl \\
        --val-data data/rl_val.jsonl \\
        --output-dir outputs/qwen35_b0.5 \\
        --beta 0.5

    # Gemma-4-E4B pure SFT (beta=0):
    CUDA_VISIBLE_DEVICES=0 python train.py \\
        --model google/gemma-4-E4B-it \\
        --train-data data/rl_train.jsonl \\
        --val-data data/rl_val.jsonl \\
        --output-dir outputs/gemma_e4b_sft \\
        --beta 0 --batch-size 1 --max-length 8192
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Gemma 4 LoRA preparation
# ---------------------------------------------------------------------------

def prepare_gemma4_for_lora(model):
    """Unwrap Gemma4ClippableLinear wrappers to expose nn.Linear for PEFT.

    Gemma 4 wraps its linear layers in Gemma4ClippableLinear, which PEFT
    cannot target for LoRA injection. This function replaces each wrapper
    with the underlying nn.Linear so that LoRA works normally.

    Returns the model (modified in-place).
    """
    import transformers

    clippable_cls = getattr(
        transformers.models.gemma4.modeling_gemma4, "Gemma4ClippableLinear", None
    )
    if clippable_cls is None:
        return model

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, clippable_cls):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.linear)
            count += 1

    print(f"  Unwrapped {count} Gemma4ClippableLinear → nn.Linear for PEFT")
    return model


def _is_gemma_model(model_path: str) -> bool:
    """Check if the model is a Gemma model based on path or name."""
    return "gemma" in model_path.lower()


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_fixed_weights(advantages: List[float], beta: float) -> List[float]:
    """Compute dataset-level fixed weights using log-sum-exp for stability.

    w_i = exp(beta * adv_i) / mean(exp(beta * adv))

    Normalized so mean(w) = 1.0. When beta=0, all weights are exactly 1.0.

    Args:
        advantages: Per-example advantage scores (group-relative).
        beta: Sharpness parameter. 0 = uniform (SFT), higher = more contrast.

    Returns:
        List of weights with mean ≈ 1.0.
    """
    if beta == 0:
        return [1.0] * len(advantages)

    log_weights = [beta * a for a in advantages]
    # log-sum-exp for numerical stability
    max_lw = max(log_weights)
    log_mean = max_lw + math.log(
        sum(math.exp(lw - max_lw) for lw in log_weights) / len(log_weights)
    )

    weights = [math.exp(lw - log_mean) for lw in log_weights]
    return weights


# ---------------------------------------------------------------------------
# Custom collator
# ---------------------------------------------------------------------------

class RewardWeightedCollator:
    """Custom data collator that passes pre-computed example_weight to the loss.

    Pops ``example_weight`` from each feature before padding, then returns it
    as a float32 tensor in the batch dict. The trainer's ``compute_loss``
    retrieves it explicitly — no ``_indices`` lookup, no silent fallback.

    Raises RuntimeError if any example is missing its weight (fail-loud over
    silent degradation to uniform weighting).
    """

    def __init__(self, tokenizer, max_length: int = 16384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract weights before collating
        weights = [f.pop("example_weight", None) for f in features]
        if any(w is None for w in weights):
            raise RuntimeError(
                "example_weight missing from dataset features. "
                "Did you forget to add the weight column to the dataset?"
            )

        # Standard padding to max length in batch
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = f["input_ids"][:max_len]
            lab = f["labels"][:max_len] if "labels" in f else ids.copy()
            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "example_weights": torch.tensor(weights, dtype=torch.float32),
        }
        return batch


# ---------------------------------------------------------------------------
# Trainer with correct reward-weighted loss
# ---------------------------------------------------------------------------

class AdvantageWeightedTrainer(SFTTrainer):
    """SFTTrainer with correctly implemented advantage-weighted loss.

    When ``use_per_example_loss=True`` (default for Qwen), computes:
        loss = mean(w_i * CE_i)
    where w_i are pre-computed dataset-level weights (mean=1) and CE_i is
    the per-example cross-entropy averaged over valid tokens.

    When ``use_per_example_loss=False`` (Gemma — NaN with gradient
    checkpointing), falls back to standard SFT loss ignoring weights.

    Critical: ``model_accepts_loss_kwargs`` is forced to ``False`` so that
    HuggingFace Trainer correctly divides the loss by
    ``gradient_accumulation_steps``. Without this, gradient accumulation > 1
    silently multiplies the effective learning rate.
    """

    def __init__(self, *args, use_per_example_loss: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_per_example_loss = use_per_example_loss
        # FIX: Prevent Trainer from skipping grad-accumulation loss scaling.
        # See README § Known Issues for details.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # During eval or when per-example loss is disabled, use standard loss
        if not model.training or not self.use_per_example_loss:
            inputs.pop("example_weights", None)
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )

        # Extract weights (passed by our custom collator)
        weights = inputs.pop("example_weights", None)
        if weights is None:
            raise RuntimeError(
                "example_weights not found in batch. This means the "
                "RewardWeightedCollator is not being used or the weight "
                "column was not added to the dataset."
            )

        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits

        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size = shift_labels.size(0)
        mask = (shift_labels != -100).float()

        # Per-example loss (mean over valid tokens per example)
        per_example_losses = []
        for i in range(batch_size):
            n_valid = mask[i].sum()
            if n_valid == 0:
                per_example_losses.append(
                    torch.tensor(0.0, device=logits.device)
                )
                continue
            token_loss = F.cross_entropy(
                shift_logits[i],
                shift_labels[i],
                reduction="none",
                ignore_index=-100,
            )
            example_loss = (token_loss * mask[i]).sum() / n_valid
            per_example_losses.append(example_loss)

        per_example_loss = torch.stack(per_example_losses)
        weights = weights.to(per_example_loss.device)

        # Weighted loss: mean(w_i * loss_i).
        # Dataset-level weights have mean=1, so this equals the intended
        # objective. Gradient accumulation normalization handled by Trainer
        # since model_accepts_loss_kwargs = False.
        loss = (weights * per_example_loss).mean()

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Checkpoint callback
# ---------------------------------------------------------------------------

class AdapterCheckpointCallback(TrainerCallback):
    """Save LoRA adapter checkpoint every N steps.

    Saves only the adapter weights (not the full model) and a
    ``base_model_path.txt`` marker file. The adapter can be merged later
    using ``merge_adapter.py``. This avoids the expensive CPU merge during
    training.
    """

    def __init__(self, save_every: int, output_dir: str, tokenizer, base_model_path: str):
        self.save_every = save_every
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.base_model_path = base_model_path

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.save_every == 0 and state.global_step > 0:
            step = state.global_step
            save_path = self.output_dir / f"step_{step}"
            save_path.mkdir(parents=True, exist_ok=True)

            print(f"\n  [Checkpoint] Saving adapter step {step} to {save_path}")

            # Save LoRA adapter weights
            model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))

            # Save marker with base model path for merge_adapter.py
            marker = save_path / "base_model_path.txt"
            marker.write_text(str(self.base_model_path))

            print(f"  [Checkpoint] Saved adapter ({save_path.name})")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_weighting(dataset, beta: float):
    """Verify that reward weighting is correctly applied before training.

    Checks:
      1. Weight mean ≈ 1.0
      2. beta=0 → uniform weights; beta>0 → non-uniform weights
      3. High-advantage examples have higher weights than low-advantage ones
      4. Reports effective sample size

    Raises AssertionError if any check fails.
    """
    print("\n=== VERIFICATION: Reward weighting ===")

    all_weights = [dataset[i]["example_weight"] for i in range(len(dataset))]
    all_advantages = [dataset[i]["advantage"] for i in range(len(dataset))]
    all_rewards = [dataset[i]["reward"] for i in range(len(dataset))]

    # 1. Check weights have correct mean
    w_mean = sum(all_weights) / len(all_weights)
    print(f"  Weight mean: {w_mean:.6f} (should be ~1.0)")
    assert abs(w_mean - 1.0) < 0.01, f"Weight mean {w_mean} is not ~1.0!"

    # 2. Check beta=0 gives uniform, beta>0 gives non-uniform
    if beta == 0:
        assert all(
            abs(w - 1.0) < 1e-6 for w in all_weights
        ), "beta=0 should give uniform weights!"
        print("  ✓ beta=0: all weights = 1.0 (uniform / plain SFT)")
    else:
        assert not all(
            abs(w - 1.0) < 1e-6 for w in all_weights
        ), "beta>0 should give non-uniform weights!"
        print(
            f"  ✓ beta={beta}: weights vary "
            f"[{min(all_weights):.4f}, {max(all_weights):.4f}]"
        )

    # 3. Check that high-advantage examples get high weights
    if beta > 0:
        paired = sorted(zip(all_advantages, all_weights))
        n_check = min(20, len(paired) // 4)
        if n_check > 0:
            low_adv_weight = sum(w for _, w in paired[:n_check]) / n_check
            high_adv_weight = sum(w for _, w in paired[-n_check:]) / n_check
            print(f"  Avg weight of bottom-{n_check} advantage: {low_adv_weight:.4f}")
            print(f"  Avg weight of top-{n_check} advantage:    {high_adv_weight:.4f}")
            assert (
                high_adv_weight > low_adv_weight
            ), "High-advantage examples should have higher weights!"
            print(
                f"  ✓ High-advantage traces are upweighted "
                f"({high_adv_weight / low_adv_weight:.1f}x)"
            )

    # 4. Effective sample size
    w_arr = np.array(all_weights)
    eff_n = w_arr.sum() ** 2 / (w_arr**2).sum()
    print(
        f"  Effective sample size: {eff_n:.1f}/{len(all_weights)} "
        f"({100 * eff_n / len(all_weights):.1f}%)"
    )

    # 5. Show a few examples
    paired = sorted(
        zip(all_advantages, all_rewards, all_weights), key=lambda x: x[2]
    )
    print(f"\n  Lowest-weight examples:")
    for adv, rew, w in paired[:3]:
        print(f"    adv={adv:+.3f}, reward={rew:.3f}, weight={w:.4f}")
    print(f"  Highest-weight examples:")
    for adv, rew, w in paired[-3:]:
        print(f"    adv={adv:+.3f}, reward={rew:.3f}, weight={w:.4f}")

    print("\n  ✓ All verification checks passed\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Advantage-weighted SFT training with LoRA"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training JSONL (messages + tools + reward + advantage)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to validation JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for adapter checkpoints (step_5/, step_10/, ...)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Advantage sharpness. 0=uniform/SFT, higher=more contrast",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: 2 * lora_r)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save adapter checkpoint every N steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size (use 1 for Gemma / large-vocab models)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="Maximum sequence length (use 8192 for Gemma to avoid OOM)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = args.model
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r * 2
    output_dir = Path(args.output_dir)

    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(
        f"Config: beta={args.beta}, lr={args.lr}, r={args.lora_r}, "
        f"alpha={lora_alpha}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, grad_accum={args.grad_accum}, "
        f"max_length={args.max_length}"
    )
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset ----
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_data, "val": args.val_data},
    )
    print(f"\nTrain: {len(dataset['train'])} examples")
    print(f"Val: {len(dataset['val'])} examples")

    # ---- Pre-compute fixed weights at dataset level ----
    train_advantages = dataset["train"]["advantage"]
    weights = compute_fixed_weights(train_advantages, args.beta)
    dataset["train"] = dataset["train"].add_column("example_weight", weights)

    # Uniform weights for val (no reward weighting during eval)
    val_weights = [1.0] * len(dataset["val"])
    dataset["val"] = dataset["val"].add_column("example_weight", val_weights)

    # Weight stats
    w_arr = np.array(weights)
    eff_n = w_arr.sum() ** 2 / (w_arr**2).sum()
    print(
        f"\nWeights: min={w_arr.min():.4f}, max={w_arr.max():.4f}, "
        f"mean={w_arr.mean():.4f}, std={w_arr.std():.4f}"
    )
    print(
        f"Effective sample size: {eff_n:.1f}/{len(weights)} "
        f"({100 * eff_n / len(weights):.1f}%)"
    )

    # ---- Verify weighting ----
    verify_weighting(dataset["train"], args.beta)

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Gemma 4 needs ClippableLinear unwrapping before LoRA can be applied
    is_gemma = _is_gemma_model(model_path)
    if is_gemma:
        model = prepare_gemma4_for_lora(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    # ---- Training config ----
    training_args = SFTConfig(
        output_dir=str(output_dir / "hf_checkpoints"),
        max_length=args.max_length,
        packing=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Gemma 4 chat templates can't be auto-patched by TRL for
        # assistant_only_loss; fall back to full-sequence loss.
        assistant_only_loss=not is_gemma,
        eval_strategy="no",
        save_strategy="no",  # We save via AdapterCheckpointCallback
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Keep example_weight for collator
    )

    # ---- Custom collator ----
    collator = RewardWeightedCollator(tokenizer, max_length=args.max_length)

    # ---- Adapter checkpoint callback ----
    adapter_callback = AdapterCheckpointCallback(
        save_every=args.save_every,
        output_dir=str(output_dir),
        tokenizer=tokenizer,
        base_model_path=model_path,
    )

    # ---- Trainer ----
    # Gemma 4 produces NaN gradients with per-example loss under gradient
    # checkpointing. Fall back to standard SFT loss for Gemma.
    use_per_example_loss = not is_gemma
    if not use_per_example_loss:
        print(
            "  Note: Using standard SFT loss (no per-example reward weighting) "
            "for Gemma — see README § Known Issues"
        )

    trainer = AdvantageWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        peft_config=lora_config,
        data_collator=collator,
        callbacks=[adapter_callback],
        use_per_example_loss=use_per_example_loss,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nTrainable: {trainable / 1e6:.1f}M / {total_params / 1e9:.2f}B "
        f"({100 * trainable / total_params:.2f}%)"
    )

    n_steps = len(dataset["train"]) // (args.batch_size * args.grad_accum)
    n_total_steps = n_steps * args.epochs
    n_checkpoints = n_total_steps // args.save_every
    print(
        f"Expected: {n_steps} steps/epoch × {args.epochs} epochs = "
        f"{n_total_steps} steps, {n_checkpoints} checkpoints"
    )

    # ---- Safety check ----
    assert not trainer.model_accepts_loss_kwargs, (
        "model_accepts_loss_kwargs should be False! "
        "Otherwise grad accumulation scaling is broken."
    )
    print("✓ model_accepts_loss_kwargs = False (grad accumulation scaling correct)")

    # ---- Train ----
    print(
        f"\nStarting training: "
        f"beta={args.beta} → "
        f"{'uniform weighting (SFT baseline)' if args.beta == 0 else 'advantage-weighted'}"
    )
    train_result = trainer.train()

    print(f"\nTraining complete! Loss: {train_result.training_loss:.4f}")
    for k, v in sorted(train_result.metrics.items()):
        print(f"  {k}: {v}")

    # List all saved checkpoints
    checkpoints = sorted(output_dir.glob("step_*"))
    print(f"\nSaved {len(checkpoints)} adapter checkpoints:")
    for cp in checkpoints:
        print(f"  {cp.name}")


if __name__ == "__main__":
    main()
