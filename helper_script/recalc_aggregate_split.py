#!/usr/bin/env python3
"""Recompute aggregate_metrics_results.json, splitting samples into groups by
sample-id name prefix. The grouping depends on the benchmark:

  tau2bench    : "multi" (multiturnmessage_*), "tc" (tc_*), "other" (rest)
  toolsandbox  : "tc" (tc_*), "non_tc" (everything else)

Toolsandbox non-tc samples carry descriptive prefixes (search_*, find_*,
cellular_*, turn_*, add_*, send_*, update_*, remove_*, wifi_*, modify_*,
get_*, ...) rather than a single multiturn prefix, so they all collapse into
one "non_tc" group.

Reads the raw trial_*_results.json files in the job dir and reproduces the
same aggregation the original pipeline used:

  - max_progress_rate : per sample, max over its runs of max(progress_rates);
                        reported as avg over samples (avg_max).
  - mean_progress_rate: per sample, mean over runs of mean(progress_rates);
                        reported as avg over samples (avg).
  - total_tokens      : per sample, mean over successful runs; avg over samples.
  - tool_call_count   : per sample, mean over successful runs; avg over samples.
  - pass@k            : per sample, 1.0 if any run fully solved (pr==1.0) else 0.0;
                        null if the sample had any failed run (incomplete k).
  - pass^k            : per sample, 1.0 if all runs fully solved else 0.0;
                        null if the sample had any failed run.

Usage:  python3 scripts/recalc_aggregate_split.py [job_dir] --benchmark tau2|toolsandbox
"""
import argparse
import json
import glob
import os
import statistics


# Per-benchmark grouping. Each entry maps a benchmark name to (group_of, order)
# where group_of(sample_id) -> group label and order is the group iteration
# order used for the output and console summary.
def _group_tau2(sample_id):
    if sample_id.startswith("multiturnmessage"):
        return "multi"
    if sample_id.startswith("tc"):
        return "tc"
    return "other"


def _group_toolsandbox(sample_id):
    if sample_id.startswith("tc_") or sample_id == "tc":
        return "tc"
    return "non_tc"


BENCHMARKS = {
    "tau2": {"group_of": _group_tau2, "order": ("multi", "tc", "other")},
    "toolsandbox": {"group_of": _group_toolsandbox, "order": ("tc", "non_tc")},
}


def collect_runs(job_dir):
    """Return {sample_id: [run_dict, ...]} across all trial files."""
    runs = {}
    for tf in sorted(glob.glob(os.path.join(job_dir, "trial_*_results.json"))):
        d = json.load(open(tf))
        for e in d.get("samples", []):
            runs.setdefault(e["sample_id"], []).append(e)
    return runs


def mean(xs):
    return statistics.fmean(xs) if xs else None


def aggregate(sample_ids, runs):
    """Compute the aggregate metric block for a given set of sample ids."""
    max_per_sample = {}
    mean_per_sample = {}
    tokens_per_sample = {}
    toolcalls_per_sample = {}
    passk_per_sample = {}
    passhat_per_sample = {}

    for sid in sample_ids:
        sample_runs = runs[sid]
        # progress rates from successful runs that have a list
        succ = [r for r in sample_runs if r.get("status") == "success"]
        had_failure = any(r.get("status") != "success" for r in sample_runs)

        # per-run max and per-run mean of progress_rates
        run_maxes = []
        run_means = []
        toks = []
        tcs = []
        for r in succ:
            m = r.get("metrics") or {}
            pr = m.get("progress_rates")
            if isinstance(pr, list) and pr:
                run_maxes.append(max(pr))
                run_means.append(mean(pr))
            if m.get("total_tokens") is not None:
                toks.append(m["total_tokens"])
            if m.get("tool_call_count") is not None:
                tcs.append(m["tool_call_count"])

        if run_maxes:
            max_per_sample[sid] = max(run_maxes)
            mean_per_sample[sid] = mean(run_means)
        if toks:
            tokens_per_sample[sid] = mean(toks)
        if tcs:
            toolcalls_per_sample[sid] = mean(tcs)

        # pass metrics: null if incomplete k (any failed run)
        if had_failure:
            passk_per_sample[sid] = None
            passhat_per_sample[sid] = None
        else:
            solved = [1.0 if (max(r["metrics"]["progress_rates"]) >= 1.0) else 0.0
                      for r in succ]
            passk_per_sample[sid] = 1.0 if any(v == 1.0 for v in solved) else 0.0
            passhat_per_sample[sid] = 1.0 if all(v == 1.0 for v in solved) else 0.0

    def avg_valid(d):
        vals = [v for v in d.values() if v is not None]
        return (mean(vals) if vals else None, len(vals))

    avg_max, cnt_max = avg_valid(max_per_sample)
    avg_mean, cnt_mean = avg_valid(mean_per_sample)
    avg_tok, cnt_tok = avg_valid(tokens_per_sample)
    avg_tc, cnt_tc = avg_valid(toolcalls_per_sample)
    avg_pk, cnt_pk = avg_valid(passk_per_sample)
    avg_ph, cnt_ph = avg_valid(passhat_per_sample)

    return {
        "n_samples": len(sample_ids),
        "metrics": {
            "max_progress_rate": {
                "max_per_sample": max_per_sample,
                "avg_max": avg_max,
                "count": cnt_max,
            },
            "mean_progress_rate": {
                "per_sample": mean_per_sample,
                "avg": avg_mean,
                "count": cnt_mean,
            },
            "total_tokens": {
                "avg_per_sample": tokens_per_sample,
                "avg": avg_tok,
                "count": cnt_tok,
            },
            "tool_call_count": {
                "avg_per_sample": toolcalls_per_sample,
                "avg": avg_tc,
                "count": cnt_tc,
            },
            "pass@k": {
                "per_sample": passk_per_sample,
                "avg": avg_pk,
                "count": cnt_pk,
            },
            "pass^k": {
                "per_sample": passhat_per_sample,
                "avg": avg_ph,
                "count": cnt_ph,
            },
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", nargs="?",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="job dir containing trial_*_results.json")
    ap.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True,
                    help="grouping scheme")
    args = ap.parse_args()

    job_dir = args.job_dir
    benchmark = args.benchmark
    group_of = BENCHMARKS[benchmark]["group_of"]
    order = BENCHMARKS[benchmark]["order"]

    runs = collect_runs(job_dir)

    groups = {g: [] for g in order}
    for sid in runs:
        groups[group_of(sid)].append(sid)
    for g in groups:
        groups[g].sort()

    # carry over run-level config from the original aggregate if present
    orig_path = os.path.join(job_dir, "aggregate_metrics_results.json")
    header = {}
    if os.path.exists(orig_path):
        o = json.load(open(orig_path))
        header = {k: o.get(k) for k in [
            "timestamp", "agent_model", "agent", "max_turns",
            "user_proxy_model", "user_proxy_persona", "judge_model", "n_trials",
        ]}

    out = dict(header)
    out["benchmark"] = benchmark
    out["groups"] = {}
    for g in order:
        if groups[g]:
            out["groups"][g] = aggregate(groups[g], runs)
    # overall (all samples) for reference
    out["overall"] = aggregate([s for g in groups.values() for s in g], runs)

    out_path = os.path.join(job_dir, "aggregate_metrics_split_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # console summary
    print(f"job_dir: {job_dir}")
    print(f"benchmark: {benchmark}")
    for g in tuple(order) + ("overall",):
        block = out["groups"].get(g) if g != "overall" else out["overall"]
        if not block:
            continue
        m = block["metrics"]
        print(f"\n[{g}]  n_samples={block['n_samples']}")
        print(f"  avg_max_progress_rate : {m['max_progress_rate']['avg_max']:.6f} (n={m['max_progress_rate']['count']})")
        print(f"  mean_progress_rate    : {m['mean_progress_rate']['avg']:.6f} (n={m['mean_progress_rate']['count']})")
        print(f"  avg_total_tokens      : {m['total_tokens']['avg']:.2f} (n={m['total_tokens']['count']})")
        print(f"  avg_tool_call_count   : {m['tool_call_count']['avg']:.4f} (n={m['tool_call_count']['count']})")
        pk = m['pass@k']['avg']; ph = m['pass^k']['avg']
        print(f"  pass@k                : {pk:.6f} (n={m['pass@k']['count']})" if pk is not None else "  pass@k                : n/a")
        print(f"  pass^k                : {ph:.6f} (n={m['pass^k']['count']})" if ph is not None else "  pass^k                : n/a")
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
