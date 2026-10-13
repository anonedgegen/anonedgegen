#!/usr/bin/env python3
"""
Convert tau2bench evaluation traces to advantage-weighted training JSONL.

Uses ALL traces (not just high-scoring ones) and assigns each a
group-relative advantage score:

    advantage_i = (score_i - mean(group)) / std(group)

where "group" = all trials for the same sample. High-scoring traces get
positive advantage (upweighted during training), low-scoring get negative
(downweighted). The training script (train.py) converts these advantages
to exponential weights via the beta parameter.

Tau2bench trajectory format:
    A list of turns, each turn a list of messages with roles:
    "user", "agent"/"assistant" (with optional tool_calls), "tool".

Output: JSONL files with ``messages``, ``tools``, ``reward``, and
``advantage`` fields, ready for ``train.py``.

Usage:
    python prepare_data.py \\
        --results-dir experiment_outputs/ \\
        --system-prompt system_prompt.txt \\
        --tool-schemas tool_schemas.json \\
        --output-dir data/
"""

import argparse
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Trajectory → messages conversion
# ---------------------------------------------------------------------------

def convert_tool_calls(tool_calls: list) -> list:
    """Convert tau2bench tool_calls format to OpenAI/TRL format.

    tau2bench: {"id": "call_xxx", "name": "func", "arguments": {...}}
    OpenAI:    {"type": "function", "id": "call_xxx",
                "function": {"name": "func", "arguments": {...}}}

    Note: ``arguments`` is kept as dict (not JSON string) because Qwen's
    chat template calls ``.items()`` on it during Jinja rendering.
    """
    if not tool_calls:
        return None

    converted = []
    for tc in tool_calls:
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        converted.append({
            "type": "function",
            "id": tc["id"],
            "function": {
                "name": tc["name"],
                "arguments": args,
            },
        })
    return converted


def convert_trajectory(trajectory: list, system_prompt: str) -> list:
    """Convert a tau2bench trajectory to TRL messages format.

    Flattens the nested turn structure into a single messages list with
    roles: system, user, assistant, tool.

    The original content format from traces is preserved verbatim — tool-only
    turns stay tool-only, text-only stays text-only. The model's chat template
    handles rendering.
    """
    messages = [{"role": "system", "content": system_prompt}]

    for turn in trajectory:
        for msg in turn:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                messages.append({"role": "user", "content": content})

            elif role in ("agent", "assistant"):
                tool_calls = msg.get("tool_calls")

                if tool_calls:
                    openai_tcs = convert_tool_calls(tool_calls)
                    messages.append({
                        "role": "assistant",
                        "content": content if content is not None else "",
                        "tool_calls": openai_tcs,
                    })
                else:
                    # Plain text response — preserve as-is (including JSON)
                    if content is not None:
                        messages.append({
                            "role": "assistant",
                            "content": content,
                        })

            elif role == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_id", ""),
                    "content": content or "",
                })

    return messages


def validate_messages(messages: list) -> bool:
    """Validate that a converted message sequence is structurally sound.

    Checks:
      - At least 3 messages (system + user + assistant minimum)
      - Starts with a system message
      - Has at least one assistant message
      - Every tool message is preceded by an assistant message with tool_calls
    """
    if len(messages) < 3:
        return False

    if messages[0]["role"] != "system":
        return False

    has_assistant = any(m["role"] == "assistant" for m in messages)
    if not has_assistant:
        return False

    for i, msg in enumerate(messages):
        if msg["role"] == "tool" and i > 0:
            found = False
            for j in range(i - 1, -1, -1):
                if (
                    messages[j]["role"] == "assistant"
                    and messages[j].get("tool_calls")
                ):
                    found = True
                    break
                if messages[j]["role"] == "user":
                    break
            if not found:
                return False

    return True


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def compute_advantages(traces_by_sample: dict) -> dict:
    """Compute group-relative advantages for all traces.

    For each sample group:
        advantage = (score - mean) / std

    If std is 0 (all traces have the same score), use absolute score
    centered at 0.5 as a fallback signal:
        advantage = (score - 0.5) * 2.0

    Returns dict mapping (sample_id, trial_id) → {advantage, reward, ...}.
    """
    advantages = {}

    for sample_id, traces in traces_by_sample.items():
        scores = [t["score"] for t in traces]
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0

        for t in traces:
            if std > 0:
                adv = (t["score"] - mean) / std
            else:
                adv = (t["score"] - 0.5) * 2.0

            advantages[(sample_id, t["trial_id"])] = {
                "advantage": adv,
                "reward": t["score"],
                "group_mean": mean,
                "group_std": std,
            }

    return advantages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert tau2bench traces to advantage-weighted training JSONL"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory with trial_*_results.json files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--system-prompt",
        required=True,
        help="Path to system_prompt.txt",
    )
    parser.add_argument(
        "--tool-schemas",
        required=True,
        help="Path to tool_schemas.json",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of *samples* (not traces) held out for validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system_prompt = Path(args.system_prompt).read_text()
    tool_schemas = json.loads(Path(args.tool_schemas).read_text())

    trial_files = sorted(glob.glob(f"{args.results_dir}/trial_*_results.json"))
    if not trial_files:
        raise FileNotFoundError(f"No trial files in {args.results_dir}")

    # ---- Collect ALL traces grouped by sample_id ----
    traces_by_sample = defaultdict(list)
    total = 0

    for trial_file in trial_files:
        with open(trial_file) as fh:
            data = json.load(fh)
        trial_id = data["trial_id"]
        for sample in data["samples"]:
            progress_rates = sample.get("metrics", {}).get("progress_rates", [])
            score = progress_rates[-1] if progress_rates else 0.0
            total += 1

            traces_by_sample[sample["sample_id"]].append({
                "sample_id": sample["sample_id"],
                "trial_id": trial_id,
                "score": score,
                "trajectory": sample.get("trajectory", []),
            })

    print(
        f"Loaded {total} traces from {len(trial_files)} trials, "
        f"{len(traces_by_sample)} unique samples"
    )

    # ---- Compute group-relative advantages ----
    advantages = compute_advantages(traces_by_sample)

    all_advs = [v["advantage"] for v in advantages.values()]
    all_rewards = [v["reward"] for v in advantages.values()]
    print(f"Advantage range: [{min(all_advs):.3f}, {max(all_advs):.3f}]")
    print(f"Reward range: [{min(all_rewards):.3f}, {max(all_rewards):.3f}]")
    positive = sum(1 for a in all_advs if a > 0)
    negative = sum(1 for a in all_advs if a < 0)
    zero = sum(1 for a in all_advs if a == 0)
    print(f"Positive advantage: {positive}, Negative: {negative}, Zero: {zero}")

    # ---- Split by sample_id (prevents leakage) ----
    sample_ids = sorted(traces_by_sample.keys())
    random.seed(args.seed)
    random.shuffle(sample_ids)
    n_val = max(1, int(len(sample_ids) * args.val_ratio))
    val_ids = set(sample_ids[:n_val])
    train_ids = set(sample_ids[n_val:])
    print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")

    # ---- Convert and write ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "skipped": 0}

    for split, split_ids in [("train", train_ids), ("val", val_ids)]:
        output_file = output_dir / f"rl_{split}.jsonl"
        with open(output_file, "w") as f:
            for sample_id in sorted(split_ids):
                for trace in traces_by_sample[sample_id]:
                    messages = convert_trajectory(
                        trace["trajectory"], system_prompt
                    )
                    if not validate_messages(messages):
                        stats["skipped"] += 1
                        continue

                    adv_info = advantages[(sample_id, trace["trial_id"])]
                    example = {
                        "messages": messages,
                        "tools": tool_schemas,
                        "reward": adv_info["reward"],
                        "advantage": adv_info["advantage"],
                    }
                    f.write(json.dumps(example) + "\n")
                    stats[split] += 1

        print(f"Wrote {stats[split]} examples to {output_file}")

    print(
        f"\nSummary: Train={stats['train']}, Val={stats['val']}, "
        f"Skipped={stats['skipped']}"
    )

    # Print per-sample breakdown for train
    print(f"\nTrain set sample breakdown:")
    for sid in sorted(train_ids):
        traces = traces_by_sample[sid]
        scores = [t["score"] for t in traces]
        advs = [
            advantages[(sid, t["trial_id"])]["advantage"] for t in traces
        ]
        print(
            f"  {sid}: scores={[round(s, 2) for s in scores]}  "
            f"advs=[{min(advs):.2f}..{max(advs):.2f}]"
        )


if __name__ == "__main__":
    main()
