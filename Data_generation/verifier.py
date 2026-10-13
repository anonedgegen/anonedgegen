"""Test case verifier — checks generated test cases against JSON DB + policy.

Spawns Claude Code to cross-check arguments, expected outputs, subgoals,
and grading notes against db.json and the policy document (PRD).
Returns the original (valid), a corrected version (fixed), or rejects
if the scenario is fundamentally infeasible.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import backoff

logger = logging.getLogger(__name__)


VERIFY_PROMPT = """\
You are an expert verifier for AI-agent test cases. Your job is to reason \
carefully about whether a generated test case is logically consistent, \
then judge and correct it.

## Test Case
```json
{test_case_json}
```

## Sampled Snapshot (source of truth)
```json
{snapshot_json}
```

## Policy Document
{policy_section}

## Environment Access
Database: {db_path}

## DB Access Instructions
To look up additional data not in the snapshot, read the JSON database with Python:
```
python3 -c "import json; db = json.load(open('{db_path}')); print(json.dumps(db['<collection>'], indent=2))"
```
Do NOT use SQLite. The database is a JSON file. Do NOT modify it.

## Step 1: Reasoning (MANDATORY — do this BEFORE judging)

Think through each of the following checks. For each one, write your reasoning \
explicitly. Do not skip any.

### A. Task-summary ↔ subgoal alignment
For EACH subgoal, verify it traces back to something in the task_summary. Flag:
- Subgoals that describe actions NOT requested in the task_summary
- Subgoals that reference IDs, values, or entities absent from the task_summary
- Missing subgoals for actions that ARE in the task_summary

### B. Internal contradictions
Look for pairs of statements that cannot both be true:
- Does the task_summary ask the user to both cancel AND update the same resource?
- Does a subgoal say a flight is already flown while another subgoal plans to keep/use it?
- Do two subgoals set contradictory expectations for the same field?

### C. Information leakage in task_summary
The task_summary is what a USER PROXY sees — it should describe what the user WANTS, \
not what the correct agent ANSWER is. Flag if the task_summary:
- Reveals policy conclusions the user would not know (e.g. "because it is basic_economy", \
  "because the flight has already landed")
- Contains internal references like "SG1", "SG3", "subgoal 2" — the user has no \
  knowledge of subgoals
- Includes prescriptive agent instructions ("explicit confirmations", "supported \
  tool chain") that a real user would never say

### D. Violated-subgoal coherence
The keys in violated_subgoals (SG1, SG2, …) refer to COMPLIANCE RULE IDs from the \
policy — they are NOT positional indices into this test case's subgoals list. \
Do NOT flag a mismatch between an SG key and the subgoal's position in the list. \
If violated_subgoals is EMPTY, this is a **happy-path** test case — the agent is \
expected to complete all actions successfully. Do NOT invent violations or flag the \
absence of violated_subgoals as an issue. \
For each entry in violated_subgoals (if any):
- Is the described violation actually triggered by the DB state and task scenario?
- Is the corresponding subgoal written as the CORRECT agent behavior (refuse/deny), \
  NOT as the violation itself?

### E. Data grounding
For every user_id, reservation_id, flight_number, and payment_id in expected_tools[].arguments
and expected_output, you MUST look it up directly in the DB file using python3 — do NOT rely
solely on the snapshot, which may itself contain fabricated values. Run:
    python3 -c "import json; db = json.load(open('{db_path}')); print('user' in db['users'])"
Replace 'user' with the actual ID to verify. If any ID from the test case is absent from the
real DB (not just the snapshot), flag it as a data fabrication error and set status to
'rejected'. Also confirm payment IDs referenced in subgoals match those in the task_summary
and DB.

### F. Scenario feasibility & tool chain logic
- Is the full workflow executable end-to-end given the current DB state?
- Are tool calls logically ordered and non-contradictory?
- Is every tool call justified by the task (no superfluous calls)?

### G. Grading note / policy correctness
- Is each grading note factually correct given DB data and policy?
- Do policy claims match the actual policy document?

## Step 2: Judgment

Based on your reasoning above, decide the status and write JSON to `{output_path}`.

### Output format:

If the test case is CORRECT (no issues from any check):
{{
  "status": "valid",
  "reasoning_summary": "<2-3 sentence summary of your reasoning>",
  "issues_found": [],
  "fixed_test_case": null
}}

If issues found and fixable:
{{
  "status": "fixed",
  "reasoning_summary": "<2-3 sentence summary of what was wrong and why>",
  "issues_found": ["<description of each issue>"],
  "fixed_test_case": {{
    "id": "<same id>",
    "task_summary": "<corrected — remove leaked answers, internal refs, prescriptive language>",
    "expected_tools": [
      {{
        "name": "<tool name>",
        "input_schema": {{...unchanged...}},
        "output_schema": {{...unchanged...}},
        "arguments": {{<corrected using snapshot/DB values only>}},
        "expected_output": {{<corrected using snapshot/DB values>}}
      }}
    ],
    "subgoals": [{{"description": "<corrected — each must trace to task_summary>"}}],
    "violated_subgoals": {{<corrected if needed>}}
  }}
}}

If fundamentally infeasible (contradictions that cannot be resolved, or DB has no \
suitable data):
{{
  "status": "rejected",
  "reasoning_summary": "<why this cannot be salvaged>",
  "issues_found": ["<reason>"],
  "fixed_test_case": null
}}

IMPORTANT: Use ONLY values from the snapshot or DB when fixing. Never invent IDs or values.
"""


class ConsistencyVerifier:
    """Verify and fix test cases using Claude Code against a JSON database + policy."""

    def __init__(
        self,
        db_json_path: str | Path,
        policy_content: str = "",
        model: str = "anthropic--claude-4.6-opus",
        max_turns: int = 35,
        log_dir: Path | None = None,
    ):
        self.db_json_path = Path(db_json_path)
        self.policy_content = policy_content
        self.model = model or "anthropic--claude-4.6-opus"
        self.max_turns = max_turns
        self.log_dir = Path(log_dir) if log_dir else None
        self._call_count = 0

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=600,
                          giveup=lambda e: "No such file" in str(e))
    async def verify_and_fix(
        self, tc: dict
    ) -> tuple[str, dict | None, list[str]]:
        """Check test case consistency with its _db_snapshot and policy.

        Returns:
            (status, fixed_tc_or_None, issues)
            status: "valid" | "fixed" | "rejected"
        """
        snapshot = tc.get("_db_snapshot", {})
        if not snapshot:
            logger.warning("No snapshot in test case — cannot verify data grounding; rejecting")
            return "rejected", None, ["No _db_snapshot: cannot verify data grounding against DB"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="cv_out_"
        ) as f:
            output_path = f.name

        policy_section = self.policy_content if self.policy_content else "(no policy document provided)"

        prompt = VERIFY_PROMPT.format(
            test_case_json=json.dumps(
                {k: v for k, v in tc.items() if not k.startswith("_")},
                indent=2, default=str,
            ),
            snapshot_json=json.dumps(snapshot, indent=2, default=str),
            policy_section=policy_section,
            db_path=str(self.db_json_path),
            output_path=output_path,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "--print", "--permission-mode", "bypassPermissions",
                "--model", self.model,
                "--max-turns", str(self.max_turns),
                "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.db_json_path.parent),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=600
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.warning("Consistency verifier timed out")
                return "valid", None, []

            self._call_count += 1
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                log_file = self.log_dir / f"verifier_{self._call_count:03d}.log"
                with open(log_file, "w") as lf:
                    lf.write(f"=== TC id={tc.get('id')} ===\n\n")
                    lf.write(f"=== RC: {proc.returncode} ===\n\n")
                    lf.write(f"=== Stdout ===\n{stdout.decode()}\n\n")
                    lf.write(f"=== Stderr ===\n{stderr.decode()}\n")

            if proc.returncode != 0:
                logger.warning(f"Consistency verifier failed rc={proc.returncode}")
                return "valid", None, []

            out_file = Path(output_path)
            if out_file.exists():
                data = json.load(open(out_file))
                out_file.unlink()
                return self._parse(data, tc)
            else:
                logger.warning("Consistency verifier produced no output file")
                return "valid", None, []

        except Exception as e:
            logger.error(f"Consistency verifier error: {e}")
            return "valid", None, []
        finally:
            Path(output_path).unlink(missing_ok=True)

    def _parse(
        self, data: dict, original: dict
    ) -> tuple[str, dict | None, list[str]]:
        status = data.get("status", "valid")
        issues = data.get("issues_found", [])

        if status == "valid":
            return "valid", None, []

        if status == "rejected":
            return "rejected", None, issues

        if status == "fixed":
            fixed = data.get("fixed_test_case")
            if not fixed:
                return "valid", None, []
            # Merge corrected fields back, preserving underscore metadata keys
            merged = {**original, **fixed}
            merged["_verification_status"] = "fixed"
            merged["_verification_issues"] = issues
            merged["_verification_reasoning"] = data.get("reasoning_summary", "")
            return "fixed", merged, issues

        return "valid", None, []
