#!/usr/bin/env python3
"""
Merge a LoRA adapter with its base model and save as a standalone model.

Supports two modes:
  1. Auto-detect: read base model path from ``base_model_path.txt`` marker
     (written by train.py's AdapterCheckpointCallback).
  2. Explicit: pass both ``--base`` and ``--adapter``.

Handles Gemma 4 ClippableLinear unwrapping automatically.

Usage:
    # Auto-detect base model from marker file:
    python merge_adapter.py \\
        --adapter outputs/qwen35_b0.5/step_10 \\
        --output outputs/qwen35_b0.5/step_10_merged

    # Explicit base + adapter:
    python merge_adapter.py \\
        --base Qwen/Qwen3.5-4B \\
        --adapter outputs/qwen35_b0.5/step_10 \\
        --output outputs/qwen35_b0.5/step_10_merged
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model"
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Base model path (auto-detected from base_model_path.txt if omitted)",
    )
    parser.add_argument(
        "--adapter", type=str, required=True, help="LoRA adapter directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for merged model"
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    output_path = Path(args.output)

    # ---- Resolve base model ----
    if args.base:
        base_model_path = args.base
    else:
        marker = adapter_path / "base_model_path.txt"
        if marker.exists():
            base_model_path = marker.read_text().strip()
            print(f"Auto-detected base model from marker: {base_model_path}")
        else:
            raise ValueError(
                f"No --base specified and no base_model_path.txt found in "
                f"{adapter_path}. Either pass --base or use an adapter saved "
                f"with AdapterCheckpointCallback."
            )

    print(f"Base model: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")

    # ---- Load base model on CPU to save GPU memory ----
    print("Loading base model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # ---- Gemma 4 ClippableLinear unwrapping ----
    if "gemma" in base_model_path.lower():
        import transformers

        clippable_cls = getattr(
            transformers.models.gemma4.modeling_gemma4,
            "Gemma4ClippableLinear",
            None,
        )
        if clippable_cls is not None:
            count = 0
            for name, module in model.named_modules():
                if isinstance(module, clippable_cls):
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = (
                        model.get_submodule(parent_name) if parent_name else model
                    )
                    setattr(parent, child_name, module.linear)
                    count += 1
            print(f"  Unwrapped {count} Gemma4ClippableLinear → nn.Linear")

    # ---- Apply and merge adapter ----
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("Merging weights...")
    model = model.merge_and_unload()

    # ---- Save ----
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(str(output_path))

    # Save tokenizer from adapter (it may have saved special tokens)
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True
    )
    tokenizer.save_pretrained(str(output_path))

    # Reset generation_config to neutral defaults
    gen_config_path = output_path / "generation_config.json"
    if gen_config_path.exists():
        gen_config = json.loads(gen_config_path.read_text())
        gen_config.update({
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
        })
        gen_config_path.write_text(json.dumps(gen_config, indent=2))

    print(f"Done! Merged model saved to {output_path}")


if __name__ == "__main__":
    main()
