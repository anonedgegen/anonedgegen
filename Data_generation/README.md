# Testcase Synthesis Pipeline

Generates grounded test cases for AI agent evaluation (e.g., tau2bench airline, ToolSandbox). Each test case chains tools from the agent's API, with arguments and expected outputs grounded in real database values, and scenarios driven by business rule violation combinations.

## How It Works

The pipeline runs in four steps:

1. **Graph** — Builds a directed tool dependency graph from `tools.json`. Resource edges come from output-type → input-type matching; additional workflow edges are discovered by LLM (GPT-5.4 via Azure).
2. **Violations** — Parses business rules from `online_subgoals.json`, enriches them via LLM, then generates 2^n violation combinations (powerset). Conflicting combos are filtered by LLM.
3. **Sample** — Randomly walks the graph to produce tool chains (2–5 steps).
4. **Generate** — For each violation combo + tool chain:
   - `violations.build_scenario()` calls GPT-5.4 to design a scenario constrained to the tool chain.
   - `ClaudeDBSampler` reads `db.json` via Claude Code subprocess to find real, self-consistent entities.
   - `testcase_gen.py` calls GPT-5.4 to produce a structured test case grounded in the DB snapshot.
   - `ConsistencyVerifier` spawns Claude Code to cross-check all arguments/outputs against db.json + policy, fixing hallucinations or rejecting infeasible scenarios.

## Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Orchestrates all four steps; main entry point |
| `graph.py` | Builds `ToolGraph` from `tools.json`; resource edges + LLM enrichment |
| `graph_enricher.py` | LLM discovers additional workflow edges beyond type matching |
| `sampler.py` | Random-walks the graph to produce tool chains |
| `violations.py` | Subgoal parsing, 2^n violation combinator, scenario builder |
| `db_sampler.py` | `ClaudeDBSampler` — reads `db.json` via Claude Code subprocess |
| `testcase_gen.py` | Calls GPT-5.4 to generate a test case from chain + DB snapshot + scenario |
| `verifier.py` | `ConsistencyVerifier` — checks/fixes values against db.json + policy via Claude Code |

## Prerequisites

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Set Azure OpenAI environment variables:

```bash
export AZURE_API_KEY=...
export AZURE_API_VERSION=...
export AZURE_API_BASE=...
```

Also requires the `claude` CLI installed and available in `PATH`.

Input data directory must contain:
- `tools.json` — tool schemas with input/output types and parameters
- `online_subgoals.json` — business rules to test
- `PRD.md` — domain policy document

A separate database file is also required:
- `db.json` — the agent's backing database as a flat JSON object where each top-level key is a collection name and its value is a list of entity dicts (or a single entity dict). Used by `ClaudeDBSampler` to ground test case arguments and expected outputs in real, self-consistent data values.

## Usage

### Run the full pipeline

```bash
python run_pipeline.py \
    --data-dir <path-to-data-dir> \
    --db-path <path-to-db.json> \
    --out output/test_cases.json \
    --n 5
```

### Resume from a later step

```bash
# Graph already built — re-sample and regenerate
python run_pipeline.py ... --from-step sample

# Violations already generated — regenerate only
python run_pipeline.py ... --from-step generate
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | (required) | Directory with tools.json, online_subgoals.json, PRD.md |
| `--db-path` | (required) | Path to JSON database (e.g., db.json) |
| `--out` | (required) | Output path for test_cases.json |
| `--n` | 5 | Number of violation combos to process |
| `--sampler-model` | `claude-4.6-sonnet` | Claude model for DB sampling + verification |
| `--from-step` | `graph` | Resume from: graph / sample / violations / generate |

## Output

```
<output_dir>/test_cases.json        # All test cases with metadata
<output_dir>/logs/                  # Claude Code subprocess logs
<output_dir>/graph.json             # Tool dependency graph
<output_dir>/violation_combos.json  # Generated violation combinations
<output_dir>/sampled_chains.json    # Sampled tool chains
```

Each test case contains:
- `task_summary` — user-facing task prompt
- `expected_tools` — ordered list of tools with input arguments and expected outputs, grounded in real DB values
- `subgoals` — verifiable assertions about agent behavior
- `violated_subgoals` — which business rules this test case targets
- `_db_snapshot`, `_verification_status` — internal metadata
