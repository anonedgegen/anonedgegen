"""Claude Code-based DB sampler for JSON database files.

Given a sampled tool chain, scenario context, and path to a JSON database,
spawns Claude Code to read the DB and return a self-consistent snapshot of
real entities that make the chain executable.

Claude reads db.json with Python — no SQLite, no writes.

Usage (standalone):
    python db_sampler.py \\
        --db-path  tau2bench_new/tau2-bench/data/tau2/domains/airline/db.json \\
        --chain    "get_user_details,get_reservation_details,cancel_reservation" \\
        --out      /tmp/sample_out.json
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import backoff

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

SAMPLER_PROMPT = """\
You are sampling database state for a test case scenario.

## Scenario Context
{scenario_description}

## Required Tool Chain
{tool_chain}

## Tool Descriptions
{tool_descriptions}

## Database Structure
{db_structure}

## Database Path
{db_path}

## CRITICAL: READ-ONLY ACCESS
The database is a JSON file. You MUST NOT modify it. Read it with Python:
    python3 -c "import json; db = json.load(open('{db_path}')); ..."

## Task
Using the structure above, query the database with Python to find entities that match this scenario.
Skip structure exploration — you already have it. Go directly to finding the right data.

You must:
1. Find a real entity that fits the scenario requirements for the given tool chain
2. Retrieve all related data by following references across collections
3. Verify that the scenario is FEASIBLE with this data:
   - If the tool chain involves searching or creating records: confirm the prerequisites exist
   - If the workflow is multi-step: confirm each step's output can feed the next step's input
   - If the scenario involves denial/rejection: confirm the data state actually triggers that denial
4. If no suitable entity exists for a feasible scenario, output {{"error": "no_suitable_entity", "reason": "..."}}

## CRITICAL: Real DB entities only

Every user_id, reservation_id, flight_number, and payment_id you include in the output
MUST be read directly from the database file using python3. Do NOT invent, fabricate, or
carry forward any IDs from the scenario context description above — the scenario context
may contain placeholder names like "U100", "fixture_user", "RES-4812", "card_204", or
similar synthetic values that do NOT exist in the real database. Ignore those completely.

If you cannot find a real entity in the DB that fits the scenario requirements, output:
{{"error": "no_suitable_entity", "reason": "<why no real entity fits>"}}

Do NOT synthesize or construct a fake snapshot to satisfy the scenario description.

## Output Format

Write the JSON to `{output_path}`.

The output MUST include:
1. All relevant entity data as `"collection_name": [entity_dicts]` or `"collection_name": {{entity_dict}}`
   sourced exclusively from the database file — no invented values.
2. A special `"_feasibility"` key that summarizes what operations are actually achievable with this data

The `_feasibility` summary should contain:
- What entities were found and their key properties
- What operations the tool chain can successfully perform given this data
- Any constraints or limitations
- A plain-language summary of what a coherent scenario looks like with this data

Include ONLY data relevant to this specific scenario. Do not dump the entire database.
"""


# ── DB structure loader ───────────────────────────────────────────────────────

def _load_db_structure(db_path: Path) -> str:
    """Load db.json and produce a concise structure overview.

    Shows top-level keys, entity counts, and a sample of the first entity
    in each collection (truncated) so Claude knows what to query without
    spending turns exploring the 200K+ line file.
    """
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    parts = []
    for key, value in db.items():
        if isinstance(value, dict):
            count = len(value)
            parts.append(f"- `{key}`: dict with {count} entries")
            # Show first entry (truncated)
            if value:
                first_key = next(iter(value))
                first_val = value[first_key]
                sample = json.dumps({first_key: first_val}, indent=2, default=str)
                if len(sample) > 600:
                    sample = sample[:600] + "\n  ... (truncated)"
                parts.append(f"  Sample entry:\n```json\n{sample}\n```")
        elif isinstance(value, list):
            count = len(value)
            parts.append(f"- `{key}`: list with {count} entries")
            if value:
                sample = json.dumps(value[0], indent=2, default=str)
                if len(sample) > 600:
                    sample = sample[:600] + "\n  ... (truncated)"
                parts.append(f"  Sample entry:\n```json\n{sample}\n```")
        else:
            parts.append(f"- `{key}`: {type(value).__name__} = {str(value)[:200]}")

    return "\n".join(parts)


# ── Main sampler ──────────────────────────────────────────────────────────────

class ClaudeDBSampler:
    """Invokes Claude Code to sample real entities from a JSON database.

    Args:
        db_json_path: Path to the JSON database file (e.g., db.json)
        model: Claude model identifier
        max_turns: Max Claude Code turns
        log_dir: Optional directory to save Claude Code logs
    """

    def __init__(
        self,
        db_json_path: str | Path,
        model: str = "anthropic--claude-4.6-sonnet",
        max_turns: int = 35,
        log_dir: str | Path | None = None,
    ):
        self.db_path = Path(db_json_path)
        self.model = model
        self.max_turns = max_turns
        self.log_dir = Path(log_dir) if log_dir else None
        self._call_count = 0
        self._db_structure = _load_db_structure(self.db_path)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=600,
                          giveup=lambda e: "No such file" in str(e))
    async def sample(
        self,
        scenario: dict,
        sampled_chain: list[str],
        tool_descriptions: dict[str, str],
    ) -> dict:
        """Sample database state for the given scenario using Claude Code.

        Args:
            scenario: dict with violated_subgoals, context_constraints, expected_behaviors
                      (from violations.build_scenario)
            sampled_chain: ordered list of tool names
            tool_descriptions: dict of tool_name → description

        Returns:
            Dict with entity data + _feasibility. Empty dict on failure.
        """
        # Write to the db's parent directory so Claude's sandbox allows the write
        tmp_dir = self.db_path.parent
        tmp_fd, output_path = tempfile.mkstemp(suffix=".json", prefix="db_sample_", dir=tmp_dir)
        os.close(tmp_fd)

        # Format scenario
        scenario_desc = ""
        if scenario:
            violated = scenario.get("violated_subgoals", {})
            if violated:
                scenario_desc += "Violated subgoals:\n"
                for sg_id, details in violated.items():
                    scenario_desc += f"  {sg_id}: {details}\n"
            constraints = scenario.get("context_constraints", {})
            if constraints:
                scenario_desc += f"\nContext constraints: {json.dumps(constraints)}\n"
            behaviors = scenario.get("expected_behaviors", [])
            if behaviors:
                scenario_desc += f"Expected behaviors: {json.dumps(behaviors)}\n"

        tool_chain_str = " → ".join(sampled_chain)
        tool_desc_str = "\n".join(f"- {name}: {desc}" for name, desc in tool_descriptions.items())

        prompt = SAMPLER_PROMPT.format(
            scenario_description=scenario_desc or "General test case (no specific violation scenario)",
            tool_chain=tool_chain_str,
            tool_descriptions=tool_desc_str,
            db_structure=self._db_structure,
            db_path=str(self.db_path),
            output_path=output_path,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "--print",
                "--permission-mode", "bypassPermissions",
                "--model", self.model,
                "--max-turns", str(self.max_turns),
                "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.db_path.parent),
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.warning("Claude DB sampler timed out")
                return {}

            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""

            self._call_count += 1
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                log_file = self.log_dir / f"sampler_{self._call_count:03d}.log"
                with open(log_file, "w") as lf:
                    lf.write(f"=== Chain: {tool_chain_str} ===\n\n")
                    lf.write(f"=== Prompt ===\n{prompt}\n\n")
                    lf.write(f"=== RC: {proc.returncode} ===\n\n")
                    lf.write(f"=== Stdout ===\n{stdout_str}\n\n")
                    lf.write(f"=== Stderr ===\n{stderr_str}\n")

            if proc.returncode != 0:
                logger.warning(f"Claude DB sampler failed (rc={proc.returncode}): {stderr_str[:300]}")
                return {}

            out_file = Path(output_path)
            if out_file.exists():
                with open(out_file) as f:
                    data = json.load(f)
                if "error" in data:
                    logger.info(f"No suitable entity: {data.get('reason', '')}")
                    return {}
                return data
            else:
                logger.warning("Claude DB sampler produced no output file")
                return {}

        except Exception as e:
            logger.error(f"Claude DB sampler error: {e}")
            return {}
        finally:
            Path(output_path).unlink(missing_ok=True)


# ── Standalone runner ─────────────────────────────────────────────────────────

async def _run_standalone(args):
    db_path = Path(args.db_path)
    chain = [t.strip() for t in args.chain.split(",")]

    print(f"Chain: {' → '.join(chain)}")
    print(f"DB: {db_path}\n")

    sampler = ClaudeDBSampler(
        db_json_path=db_path,
        model=args.model,
        max_turns=args.max_turns,
        log_dir=Path(args.log_dir) if args.log_dir else None,
    )

    # No scenario for standalone run
    tool_descs = {t: "" for t in chain}
    result = await sampler.sample({}, chain, tool_descs)

    if not result:
        print("No result — no suitable entity found or sampler failed.")
        return

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Snapshot written to {args.out}")
    else:
        print(json.dumps(result, indent=2))

    print(f"\n_feasibility: {result.get('_feasibility', '')}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude Code DB sampler for JSON databases")
    parser.add_argument("--db-path", required=True, help="Path to db.json")
    parser.add_argument("--chain", required=True, help="Comma-separated tool names")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--model", default="anthropic--claude-4.6-sonnet")
    parser.add_argument("--max-turns", type=int, default=35)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()
    asyncio.run(_run_standalone(args))


if __name__ == "__main__":
    main()
