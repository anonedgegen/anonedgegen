#!/usr/bin/env python3
"""
Testcase synthesis pipeline — data-dir controlled.

Reads tools.json, online_subgoals.json, and PRD.md from a data directory,
plus a JSON database, to generate grounded test cases with violation scenarios.

Usage:
    python testcase_synthesis/run_pipeline.py \
        --data-dir tau2bench_new/tau2-bench/sdg-data \
        --db-path tau2bench_new/tau2-bench/data/tau2/domains/airline/db.json \
        --out output/test_cases.json \
        --n 5 \
        --sampler-model anthropic--claude-4.6-sonnet \
        --verifier-model anthropic--claude-4.6-opus \
        --from-step graph \
        --log-dir logs/
"""
import argparse, asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph import build_graph, make_client
from sampler import sample_chains, load_graph
from violations import parse_subgoals, generate_violation_combos, build_scenario
from db_sampler import ClaudeDBSampler
from testcase_gen import generate_testcase
from verifier import ConsistencyVerifier


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_tools(data_dir: Path) -> dict:
    """Load tools.json, return dict of tool_name -> tool info."""
    with open(data_dir / "tools.json") as f:
        data = json.load(f)
    tools = {t["name"]: t for t in data["tools"]}
    return tools


def load_subgoals(data_dir: Path) -> list[dict]:
    """Load online_subgoals.json."""
    with open(data_dir / "online_subgoals.json") as f:
        data = json.load(f)
    return data["subgoals"]


def load_prd(data_dir: Path) -> str:
    """Load PRD.md."""
    with open(data_dir / "PRD.md") as f:
        return f.read()


# ── Pipeline steps ───────────────────────────────────────────────────────────

def step_graph(data_dir: Path, out_dir: Path, client) -> None:
    """Step 1: Build capability graph from tools.json."""
    print("\n--- Step 1: Build capability graph ---")
    graph = build_graph(data_dir / "tools.json", client=client)

    graph_file = out_dir / "graph.json"
    with open(graph_file, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)

    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  -> {graph_file}")


def step_sample(out_dir: Path, n: int) -> None:
    """Step 2: Sample tool chains from the graph."""
    print(f"\n--- Step 2: Sample chains (n={n}) ---")
    graph_file = out_dir / "graph.json"
    graph = load_graph(graph_file)

    all_chains = sample_chains(graph, n=n * 3)
    chains = all_chains[:n]
    if len(chains) < n:
        print(f"  WARNING: only {len(chains)} chains available (requested {n})")

    print(f"  Sampled {len(chains)} chains:")
    for i, c in enumerate(chains):
        tools = c.get("tools", c.get("capabilities", []))
        label = " -> ".join(tools[:3])
        if len(tools) > 3:
            label += "..."
        print(f"    [{i+1}] {label}")

    chains_file = out_dir / "sampled_chains.json"
    with open(chains_file, "w") as f:
        json.dump(chains, f, indent=2)
    print(f"  -> {chains_file}")


def step_violations(out_dir: Path, data_dir: Path, n: int, client) -> None:
    """Step 3: Parse subgoals and generate violation combos."""
    print(f"\n--- Step 3: Generate violation combos (cap={n}) ---")
    tools = load_tools(data_dir)
    raw_subgoals = load_subgoals(data_dir)
    prd_context = load_prd(data_dir)
    tool_names = list(tools.keys())

    print(f"  Tools: {len(tool_names)}")
    print(f"  Raw subgoals: {len(raw_subgoals)}")

    subgoals = parse_subgoals(raw_subgoals, tool_names, prd_context, client)
    print(f"  Parsed subgoals: {len(subgoals)}")

    combos = generate_violation_combos(subgoals, client)
    combos = combos[:n]
    print(f"  Violation combos: {len(combos)}")

    combos_file = out_dir / "violation_combos.json"
    with open(combos_file, "w") as f:
        json.dump(combos, f, indent=2)
    print(f"  -> {combos_file}")


async def step_generate(
    out_dir: Path,
    out_file: Path,
    log_dir: Path,
    data_dir: Path,
    db_path: Path,
    n: int,
    sampler_model: str,
    verifier_model: str,
    client,
) -> list[dict]:
    """Step 4: For each combo -- build scenario, sample DB, generate, verify."""
    print(f"\n--- Step 4: Generate {n} test cases ---")

    with open(out_dir / "sampled_chains.json") as f:
        chains = json.load(f)
    with open(out_dir / "graph.json") as f:
        graph = json.load(f)
    with open(out_dir / "violation_combos.json") as f:
        combos = json.load(f)[:n]

    graph_nodes = graph["nodes"]
    tools = load_tools(data_dir)
    prd_context = load_prd(data_dir)
    tool_descriptions = {name: t.get("description", "") for name, t in tools.items()}

    sampler = ClaudeDBSampler(db_json_path=db_path, model=sampler_model, log_dir=log_dir)
    verifier = ConsistencyVerifier(db_json_path=db_path, policy_content=prd_context, model=verifier_model, log_dir=log_dir)
    gen_client = make_client()

    test_cases = []
    stats = {"valid": 0, "fixed": 0, "rejected": 0}

    for i, combo in enumerate(combos):
        chain = chains[i % len(chains)] if chains else {}
        chain_tools = chain.get("tools", [])
        violated = combo.get("subgoal_map", {})
        combo_label = ", ".join(violated.keys()) if violated else "(happy path)"
        print(f"\n  [{i+1}/{len(combos)}] {' → '.join(chain_tools[:3])}{'...' if len(chain_tools)>3 else ''} | {combo_label}", flush=True)

        # Build scenario
        print("    building scenario...", flush=True)
        scenario = build_scenario(combo, graph_nodes, chain_tools, prd_context, client)

        # DB sampling
        print("    sampling DB...", flush=True)
        db_snapshot = await sampler.sample(scenario, chain_tools, tool_descriptions)
        if not db_snapshot:
            print("    x No DB snapshot -- skipping")
            continue
        tables = {k: len(v) for k, v in db_snapshot.items()
                  if not k.startswith("_") and isinstance(v, (list, dict))}
        print(f"    snapshot: {tables}")

        # Generate test case
        print("    generating...", flush=True)
        tc = generate_testcase(chain, graph_nodes, db_snapshot, scenario, prd_context, gen_client)
        if not tc:
            print("    x Generation failed")
            continue

        # Verification
        print("    verifying...", flush=True)
        status, fixed_tc, issues = await verifier.verify_and_fix(tc)
        if status == "valid":
            print("    ok Consistent")
            tc["_verification_status"] = "valid"
            tc["_verification_issues"] = []
            stats["valid"] += 1
        elif status == "fixed":
            print(f"    ok Fixed {len(issues)} issue(s):")
            for issue in issues:
                print(f"      - {issue[:100]}")
            tc = fixed_tc
            stats["fixed"] += 1
        elif status == "rejected":
            print(f"    x Rejected:")
            for issue in issues:
                print(f"      - {issue[:100]}")
            tc["_verification_status"] = "rejected"
            tc["_verification_issues"] = issues
            stats["rejected"] += 1

        test_cases.append(tc)
        print(f"    task: {tc.get('task_summary', '')[:80]}")
        print(f"    subgoals: {len(tc.get('subgoals', []))}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    print(f"\n  {len(test_cases)} test cases -> {out_file}")
    print(f"  Verification: valid={stats['valid']} fixed={stats['fixed']} rejected={stats['rejected']}")
    return test_cases


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Testcase synthesis pipeline")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with tools.json, online_subgoals.json, PRD.md")
    parser.add_argument("--db-path", required=True,
                        help="Path to JSON database")
    parser.add_argument("--out", required=True,
                        help="Output path for test_cases.json")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of violation combos / test cases (default: 5)")
    parser.add_argument("--sampler-model", default="anthropic--claude-4.6-sonnet",
                        help="Model for Claude Code DB sampler")
    parser.add_argument("--verifier-model", default="anthropic--claude-4.6-opus",
                        help="Model for Claude Code verifier (default: opus)")
    parser.add_argument("--from-step",
                        choices=["graph", "sample", "violations", "generate"],
                        default="graph",
                        help="Resume from this step (default: graph = run all)")
    parser.add_argument("--log-dir", default=None,
                        help="Log directory (default: auto-created next to --out)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    db_path = Path(args.db_path)
    out_file = Path(args.out)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    client = make_client()
    steps = ["graph", "sample", "violations", "generate"]
    start = steps.index(args.from_step)

    print("=== Testcase synthesis pipeline ===")
    print(f"  Data dir:     {data_dir}")
    print(f"  DB path:      {db_path}")
    print(f"  Output:       {out_file}")
    print(f"  n:            {args.n}")
    print(f"  From step:    {args.from_step}")
    print(f"  Log dir:      {log_dir}")

    if start <= 0:
        step_graph(data_dir, out_dir, client)
    if start <= 1:
        step_sample(out_dir, args.n)
    if start <= 2:
        step_violations(out_dir, data_dir, args.n, client)
    if start <= 3:
        await step_generate(
            out_dir, out_file, log_dir, data_dir, db_path,
            args.n, args.sampler_model, args.verifier_model, client,
        )

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    asyncio.run(main())
