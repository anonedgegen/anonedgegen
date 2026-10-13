#!/usr/bin/python3
"""Bootstrap confidence intervals: base vs candidate, per model folder.

Run with system python3 (needs numpy + matplotlib)::

    python3 scripts/bootstrap_ci_split.py <model_folder>

Pass one model folder containing a ``base/`` run and one or more candidate
subfolders.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


SPLIT_FILE = "aggregate_metrics_split_results.json"
BASE_DIRNAME = "base"

# Only these metrics, in this order. Value = the per-sample field name inside the
# metric block of the split-results JSON.
METRICS: List[Tuple[str, str]] = [
    ("mean_progress_rate", "per_sample"),
    ("max_progress_rate", "max_per_sample"),
    ("tool_call_count", "avg_per_sample"),
    ("total_tokens", "avg_per_sample"),
]
METRIC_ORDER = [m for m, _ in METRICS]
PERSAMPLE_FIELD = dict(METRICS)
# For the reported-avg sanity check: field holding the file's own mean.
AVG_FIELD = {
    "mean_progress_rate": "avg",
    "max_progress_rate": "avg_max",
    "tool_call_count": "avg",
    "total_tokens": "avg",
}

# 0-1 style metrics (display + y-axis cap only).
RATE_METRICS = {"mean_progress_rate", "max_progress_rate"}

# Pretty metric names for subplot titles.
METRIC_DISPLAY = {
    "mean_progress_rate": "Mean Progress Rate",
    "max_progress_rate": "Max Progress Rate",
    "tool_call_count": "Tool Call Count",
    "total_tokens": "Total Tokens",
}

# Group display order and the raw-key -> display-label mapping per benchmark.
GROUP_DISPLAY = ("Original", "Policy-based", "Union")
# Friendlier labels for the plot x-axis (internal keys stay as GROUP_DISPLAY).
GROUP_PLOT_LABEL = {
    "Original": "Original",
    "Policy-based": "Policy-based",
    "Union": "Original\n+ Policy-based",
}
# benchmark -> {display_label: raw location}. "Union" is the sibling top-level
# block; the other two live under ["groups"][raw_key].
BENCH_GROUPS = {
    "tau2": {"Original": "multi", "Policy-based": "tc", "Union": "__overall__"},
    "toolsandbox": {"Original": "non_tc", "Policy-based": "tc", "Union": "__overall__"},
}

# Paper-ready display names for figure titles (dir name -> pretty label).
DATASET_DISPLAY = {
    "tau2airline": r"$\tau^2$-bench Airline",
    "toolsandbox": "ToolSandbox",
}
MODEL_DISPLAY = {
    "gpt5.4": "gpt-5.4",
    "gpt4o-mini": "gpt-4o-mini",
    "gemma4_e4b": "Gemma-4-e4b",
}


def bootstrap_ci(
    values: List[float],
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Bootstrap CI around the mean of ``values`` (identical to reference)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    point = float(arr.mean())

    if n == 1:
        return {
            "mean": point,
            "ci_low": point,
            "ci_high": point,
            "n": int(n),
            "degenerate": True,
        }

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "mean": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "degenerate": False,
    }


def infer_benchmark(data: Dict[str, Any]) -> Optional[str]:
    """Benchmark from the explicit key, else from the group names."""
    b = data.get("benchmark")
    if b in BENCH_GROUPS:
        return b
    groups = data.get("groups", {})
    if "multi" in groups:
        return "tau2"
    if "non_tc" in groups:
        return "toolsandbox"
    return None


def get_block(data: Dict[str, Any], raw_key: str) -> Optional[Dict[str, Any]]:
    """Return the group/overall block for a raw group location."""
    if raw_key == "__overall__":
        return data.get("overall")
    return data.get("groups", {}).get(raw_key)


def load_split(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def values_for(block: Dict[str, Any], metric: str) -> List[float]:
    """Per-sample scalar values for a metric, Nones dropped."""
    mblock = block.get("metrics", {}).get(metric)
    if not mblock:
        return []
    per_sample = mblock.get(PERSAMPLE_FIELD[metric])
    if not per_sample:
        return []
    return [v for v in per_sample.values() if v is not None]


def compute_system(
    data: Dict[str, Any],
    benchmark: str,
    system_label: str,
    results: Dict[str, Dict[str, Dict[str, Any]]],
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
    tag: str,
) -> None:
    """Fill ``results[metric][system_label][group]`` for one run's file."""
    gmap = BENCH_GROUPS[benchmark]
    for metric in METRIC_ORDER:
        for disp in GROUP_DISPLAY:
            block = get_block(data, gmap[disp])
            if block is None:
                continue
            vals = values_for(block, metric)
            if not vals:
                continue
            ci = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, rng=rng)
            # Sanity check vs the file's reported avg.
            mblock = block.get("metrics", {}).get(metric, {})
            reported = mblock.get(AVG_FIELD[metric])
            if reported is not None and not math.isclose(
                ci["mean"], reported, rel_tol=1e-6, abs_tol=1e-6
            ):
                ci["reported_avg"] = reported
                print(
                    f"  NOTE: {tag}: {metric} {system_label} {disp}: computed mean "
                    f"{ci['mean']:.6g} != file avg {reported:.6g}"
                )
            results.setdefault(metric, {}).setdefault(system_label, {})[disp] = ci


def fmt_value(metric: str, v: float) -> str:
    if metric in RATE_METRICS:
        return f"{v:.4f}"
    if metric == "total_tokens":
        return f"{v:,.0f}"
    if metric == "tool_call_count":
        return f"{v:,.2f}"
    return f"{v:.4f}"


def fmt_cell(metric: str, entry: Optional[Dict[str, Any]]) -> str:
    if entry is None:
        return "-"
    mean = fmt_value(metric, entry["mean"])
    lo = fmt_value(metric, entry["ci_low"])
    hi = fmt_value(metric, entry["ci_high"])
    return f"{mean} [{lo}, {hi}]"


def build_quadrant_log(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    systems: List[str],
    params: Dict[str, Any],
    header_lines: List[str],
) -> str:
    lines: List[str] = []
    lines.append("Bootstrap 95% confidence intervals (mean [ci_low, ci_high])")
    lines.append(
        "params: n_boot={n_boot}, alpha={alpha}, seed={seed}".format(**params)
    )
    lines.extend(header_lines)
    lines.append("Each cell: point estimate = mean(per_sample); CI via bootstrap.")
    lines.append("")

    groups = list(GROUP_DISPLAY)
    for metric in METRIC_ORDER:
        by_system = results.get(metric)
        if not by_system:
            continue

        col_cells: Dict[str, List[str]] = {g: [] for g in groups}
        for system in systems:
            for g in groups:
                entry = by_system.get(system, {}).get(g)
                col_cells[g].append(fmt_cell(metric, entry))

        row_label_w = max([len(s) for s in systems] + [4]) + 2
        col_w = {}
        for g in groups:
            widest = max([len(g)] + [len(c) for c in col_cells[g]])
            col_w[g] = widest + 2

        def hline(left: str, mid: str, right: str) -> str:
            segs = ["─" * row_label_w] + ["─" * col_w[g] for g in groups]
            return "  " + left + mid.join(segs) + right

        def row(label: str, cells: List[str]) -> str:
            parts = [f" {label} ".ljust(row_label_w)]
            for g, c in zip(groups, cells):
                parts.append(c.center(col_w[g]))
            return "  │" + "│".join(parts) + "│"

        lines.append(f"  {metric}")
        lines.append(hline("┌", "┬", "┐"))
        lines.append(row("", list(groups)))
        lines.append(hline("├", "┼", "┤"))
        for system in systems:
            cells = [fmt_cell(metric, by_system.get(system, {}).get(g)) for g in groups]
            lines.append(row(system, cells))
        lines.append(hline("└", "┴", "┘"))
        lines.append("")

    return "\n".join(lines) + "\n"


def build_plot(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    systems: List[str],
    out_path: Path,
    title: str,
    legend_labels: Optional[Dict[str, str]] = None,
) -> None:
    metrics = [m for m in METRIC_ORDER if results.get(m)]
    if not metrics:
        return

    groups = list(GROUP_DISPLAY)
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0 * ncols, 7.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(len(groups))
    n_sys = len(systems)
    total_w = 0.5
    bar_w = total_w / max(n_sys, 1)
    # base first, then candidate(s). Blue for base, orange+ for candidates.
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    colors = {s: palette[i % len(palette)] for i, s in enumerate(systems)}

    for ax, metric in zip(axes, metrics):
        by_system = results[metric]
        for si, system in enumerate(systems):
            heights, lo_err, hi_err, xs = [], [], [], []
            offset = (si - (n_sys - 1) / 2.0) * bar_w
            for gi, g in enumerate(groups):
                entry = by_system.get(system, {}).get(g)
                if entry is None:
                    continue
                heights.append(entry["mean"])
                lo_err.append(entry["mean"] - entry["ci_low"])
                hi_err.append(entry["ci_high"] - entry["mean"])
                xs.append(gi + offset)
            if not heights:
                continue
            ax.bar(
                xs,
                heights,
                width=bar_w,
                color=colors[system],
                label=system,
                yerr=[lo_err, hi_err],
                capsize=5,
                error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
            )
        ax.set_title(METRIC_DISPLAY.get(metric, metric), fontsize=28, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_PLOT_LABEL[g] for g in groups], fontsize=22)
        ax.tick_params(axis="y", labelsize=20)
        ax.grid(True, axis="y", alpha=0.3)
        if metric in RATE_METRICS:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=30, fontweight="bold", y=0.99)
    legend_handles = [
        Patch(facecolor=colors[s], label=(legend_labels or {}).get(s, s))
        for s in systems
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(systems),
        fontsize=24,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def sanitize(name: str) -> str:
    """Filesystem-safe token for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def discover_candidates(model_dir: Path) -> List[Path]:
    """Non-base subfolders of model_dir that contain the split-results file."""
    cands = []
    for sub in sorted(model_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name == BASE_DIRNAME or sub.name == "bootstrap_ci":
            continue
        if (sub / SPLIT_FILE).is_file():
            cands.append(sub)
    return cands


def process_model_dir(
    model_dir: Path,
    n_boot: int,
    alpha: float,
    seed: int,
) -> int:
    """Process one model folder: base vs each candidate. Returns #pairs written."""
    base_file = model_dir / BASE_DIRNAME / SPLIT_FILE
    base_data = load_split(base_file)
    benchmark = infer_benchmark(base_data)
    if benchmark is None:
        print(f" SKIP {model_dir}: cannot determine benchmark", file=sys.stderr)
        return 0

    candidates = discover_candidates(model_dir)
    if not candidates:
        print(f"SKIP {model_dir}: no candidates", file=sys.stderr)
        return 0

    # dataset = model_dir.parent.name, model = model_dir.name
    dataset = model_dir.parent.name
    model = model_dir.name
    single = len(candidates) == 1

    params = {"n_boot": n_boot, "alpha": alpha, "seed": seed}
    out_root = model_dir / "bootstrap_ci"
    written = 0

    for cand_dir in candidates:
        cand_name = cand_dir.name
        cand_label = "optimized" if single else cand_name
        cand_data = load_split(cand_dir / SPLIT_FILE)
        cand_bench = infer_benchmark(cand_data) or benchmark

        # Fresh RNG per pair so base numbers are identical across pairs.
        rng = np.random.default_rng(seed)
        systems = ["base", cand_label]
        results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        compute_system(
            base_data, benchmark, "base", results, n_boot, alpha, rng,
            tag=f"{dataset}/{model}/base",
        )
        compute_system(
            cand_data, cand_bench, cand_label, results, n_boot, alpha, rng,
            tag=f"{dataset}/{model}/{cand_name}",
        )

        out_dir = out_root / cand_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out_dir / "bootstrap_ci_results.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "params": params,
                    "benchmark": benchmark,
                    "dataset": dataset,
                    "model": model,
                    "base_dir": str(base_file.parent.relative_to(model_dir.parent)),
                    "candidate_dir": str(cand_dir.relative_to(model_dir.parent)),
                    "candidate_label": cand_label,
                    "systems": systems,
                    "groups": list(GROUP_DISPLAY),
                    "metrics": results,
                },
                fh,
                indent=2,
            )

        # Log (paths shown from the model folder onwards)
        base_rel = base_file.parent.relative_to(model_dir.parent)
        cand_rel = cand_dir.relative_to(model_dir.parent)
        header_lines = [
            f"dataset={dataset}  model={model}  benchmark={benchmark}",
            f"base={base_rel}",
            f"candidate={cand_rel}  (label: {cand_label})",
            "groups: Original / Policy-based / Union",
        ]
        log_path = out_dir / "bootstrap_ci_quadrant.log"
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(build_quadrant_log(results, systems, params, header_lines))

        # Plot
        png_name = f"{sanitize(dataset)}_{sanitize(model)}_{sanitize(cand_name)}_bootstrap_ci.png"
        png_path = out_dir / png_name
        dataset_disp = DATASET_DISPLAY.get(dataset, dataset)
        model_disp = MODEL_DISPLAY.get(model, model)
        title = f"{dataset_disp} — {model_disp}: Base vs. Optimized"
        legend_labels = {"base": "base harness", cand_label: f"optimized harness ({cand_name})"}
        build_plot(results, systems, png_path, title, legend_labels)

        print(f"  [{dataset}/{model}] base vs {cand_label} ({cand_name}) -> {png_path}")
        written += 1

    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs: base vs each candidate for ONE model folder "
        "(split results)."
    )
    parser.add_argument(
        "model_dir",
        help="A single model folder containing a 'base/' run and one or more "
        "candidate run subfolders (each with aggregate_metrics_split_results.json).",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: not a directory: {model_dir}", file=sys.stderr)
        return 1

    base_file = model_dir / BASE_DIRNAME / SPLIT_FILE
    if not base_file.is_file():
        print(
            f"ERROR: no base run found at {base_file}. Point this at a model folder "
            f"that contains a '{BASE_DIRNAME}/{SPLIT_FILE}'.",
            file=sys.stderr,
        )
        return 1
    if not discover_candidates(model_dir):
        print(
            f"ERROR: no candidate subfolders with {SPLIT_FILE} under {model_dir}.",
            file=sys.stderr,
        )
        return 1

    n_pairs = process_model_dir(model_dir, args.n_boot, args.alpha, args.seed)
    print(f"Done. Wrote {n_pairs} base-vs-candidate comparison(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
