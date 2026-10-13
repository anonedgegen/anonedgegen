"""Test case generator — produces grounded test cases from tool chains + JSON DB snapshots."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import backoff
from openai import AzureOpenAI

DEPLOYMENT = os.environ.get("LLM_DEPLOYMENT", "gpt-5.4")


def make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
    )


@backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=120)
def _chat(client: AzureOpenAI, prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _parse_json(raw: str) -> dict | list | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    for start in ["{", "["]:
        idx = raw.find(start)
        if idx != -1:
            raw = raw[idx:]
            break
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ── Prompts ──────────────────────────────────────────────────────────────────

# ── Prompts: VIOLATION path ─────────────────────────────────────────────────

VIOLATION_SYSTEM = """\
You are an expert at creating test cases for AI agent evaluation. \
These test cases involve policy VIOLATIONS — the user requests something the agent \
must partially or fully refuse based on business rules.

CRITICAL RULES:
- Use ONLY entity values (IDs, names, amounts, dates) from the DATABASE SNAPSHOT
- Do NOT invent any IDs, names, or values
- arguments keys must EXACTLY match the parameter names in the tool's input schema
- arguments values must be exact values from the DB snapshot
- expected_output keys must match the tool's output schema fields
- EXCEPTION: for tools that CREATE new entities (e.g. book_reservation), use a \
placeholder phrase like "newly generated reservation ID" for the new entity ID
- subgoals are short factual statements of what the Agent concretely does or communicates
- Each subgoal must include exact values from the DB snapshot (IDs, amounts, dates, names)
- Write subgoals in the style: "Agent cancels reservation XYZ.", "Agent communicates that \
total cost is $450.", "Agent refuses to change cabin on reservation XYZ because HAT176 \
on 2024-05-12 has already landed."
- Do NOT write about agent reasoning or verification process (e.g., NOT "Agent verifies \
eligibility before issuing compensation")
- Do NOT write vague behavioral guidance (e.g., NOT "Agent should inform the user about \
the policy")
- Every subgoal must be checkable from the agent's actions and final response alone

TASK_SUMMARY RULES — the task_summary is what a USER PROXY sees. It must read as a \
natural user request:
- Start with "You are [real name] (user ID [id])."
- Describe ONLY what the user WANTS to do, not what the agent should do
- Do NOT include policy conclusions the user would not know (e.g., NOT "the agent must \
refuse because it is basic_economy", NOT "compensation should be denied because you \
are a regular member")
- Do NOT include prescriptive agent instructions (e.g., NOT "the agent must refuse", \
NOT "with explicit confirmations before each write action", NOT "the agent should \
explain that...")
- Do NOT reference internal subgoal IDs (e.g., NOT "SG1", "SG3", "subgoal 2")
- Do NOT reveal expected agent behavior or answers — the user is asking, not instructing
- If the user's request will be denied by policy, the task_summary should still present \
it as a genuine user request (e.g., "You want to change your flight to HAT014" NOT \
"You want to change your flight but the agent must refuse because basic_economy")
- Each distinct user objective should map to at least one subgoal

Output valid JSON matching the requested schema."""

VIOLATION_PROMPT = """\
Generate a test case for evaluating an AI agent executing this tool chain. \
This scenario involves policy violations — see BUSINESS RULES TO TEST below.

## Tool Chain
{chain_summary}

## Database Snapshot (ONLY source of real values)
{db_snapshot}

## Feasibility Note
{feasibility}
{scenario_section}{domain_context}
---

Generate a JSON object with this exact structure:

{{
  "task_summary": "You are [real name from DB] (user ID [id]). \
You want to [natural user request that requires the tool chain]. \
[2-3 sentences describing what the user wants, using real values from DB. \
Do NOT reveal policy outcomes, agent decisions, or prescriptive instructions. \
Write as a genuine user who does not know the policy rules.]",

  "expected_tools": [
    {{
      "name": "<tool name>",
      "input_schema": {{<param_name>: {{"type": "<type>", "description": "<desc>", "required": true|false}}}},
      "output_schema": {{<field_name>: {{"type": "<type>", "description": "<desc>"}}}},
      "arguments": {{<param_name>: <real value from DB>}},
      "expected_output": {{<realistic output using real DB values>}}
    }}
  ],

  "subgoals": [
    {{"description": "Agent [concrete action or communication with exact DB values]"}},
    ...
  ]
}}

Rules:
1. expected_tools must include ALL {n_caps} tools in the chain, in order
2. arguments must use ONLY values that appear in the DB snapshot
3. Argument keys must EXACTLY match the parameter names in the tool's input schema
4. expected_output keys must match the tool's output schema fields
5. 3-5 subgoals, each a short factual statement of what the Agent does or communicates (with exact values)
6. Subgoals must include specific IDs, names, amounts, or dates from the DB snapshot
7. Every subgoal must trace back to something in the task_summary — do NOT add subgoals \
for actions the user never requested
8. The task_summary must present the user's request naturally WITHOUT mentioning the \
expected refusal or policy reason
9. Do NOT include contradictory actions (e.g., cancelling and updating the same resource)

JSON Output:"""


# ── Prompts: HAPPY PATH ────────────────────────────────────────────────────

HAPPY_PATH_SYSTEM = """\
You are an expert at creating test cases for AI agent evaluation. \
These test cases are HAPPY PATH scenarios — the user's request is fully feasible and \
the agent should complete all actions successfully following all policies.

CRITICAL RULES:
- Use ONLY entity values (IDs, names, amounts, dates) from the DATABASE SNAPSHOT
- Do NOT invent any IDs, names, or values
- arguments keys must EXACTLY match the parameter names in the tool's input schema
- arguments values must be exact values from the DB snapshot
- expected_output keys must match the tool's output schema fields
- EXCEPTION: for tools that CREATE new entities (e.g. book_reservation), use a \
placeholder phrase like "newly generated reservation ID" for the new entity ID
- subgoals are short factual statements of what the Agent concretely does or communicates
- Each subgoal must include exact values from the DB snapshot (IDs, amounts, dates, names)
- Write subgoals in the style: "Agent cancels reservation XYZ.", "Agent communicates that \
total cost is $450.", "Agent books flight HAT271 on May 26 in economy."
- Do NOT write about agent reasoning or verification process
- Do NOT write vague behavioral guidance
- Every subgoal must be checkable from the agent's actions and final response alone

TASK_SUMMARY RULES — the task_summary is what a USER PROXY sees:
- Start with "You are [real name] (user ID [id])."
- Describe what the user wants to do — all requested actions should be feasible
- Do NOT include actions that would be denied by policy
- Do NOT include prescriptive agent instructions or internal references
- Each distinct user objective should map to at least one subgoal

HAPPY PATH RULES:
- Every tool call must succeed — no refusals, no denials
- The user's request must be fully satisfiable given the DB state and policy
- Subgoals describe successful completions: "Agent books...", "Agent updates...", \
"Agent communicates that..."
- If a tool call would be denied by policy given the DB state (e.g., cancelling a \
basic_economy reservation), do NOT include that action — pick a different scenario \
that works

Output valid JSON matching the requested schema."""

HAPPY_PATH_PROMPT = """\
Generate a HAPPY PATH test case for evaluating an AI agent executing this tool chain. \
All actions should succeed — no policy violations, no refusals.

## Tool Chain
{chain_summary}

## Database Snapshot (ONLY source of real values)
{db_snapshot}

## Feasibility Note
{feasibility}
{scenario_section}{domain_context}
---

Generate a JSON object with this exact structure:

{{
  "task_summary": "You are [real name from DB] (user ID [id]). \
You want to [natural user request that requires the tool chain]. \
[2-3 sentences describing what the user wants, using real values from DB. \
All requested actions should be feasible and policy-compliant.]",

  "expected_tools": [
    {{
      "name": "<tool name>",
      "input_schema": {{<param_name>: {{"type": "<type>", "description": "<desc>", "required": true|false}}}},
      "output_schema": {{<field_name>: {{"type": "<type>", "description": "<desc>"}}}},
      "arguments": {{<param_name>: <real value from DB>}},
      "expected_output": {{<realistic output using real DB values>}}
    }}
  ],

  "subgoals": [
    {{"description": "Agent [successful action or communication with exact DB values]"}},
    ...
  ]
}}

Rules:
1. expected_tools must include ALL {n_caps} tools in the chain, in order
2. arguments must use ONLY values that appear in the DB snapshot
3. Argument keys must EXACTLY match the parameter names in the tool's input schema
4. expected_output keys must match the tool's output schema fields
5. 3-5 subgoals, each a short factual statement of what the Agent does or communicates (with exact values)
6. Subgoals must include specific IDs, names, amounts, or dates from the DB snapshot
7. Every subgoal must trace back to something in the task_summary
8. Every tool call must SUCCEED — do not include actions the agent would refuse
9. Do NOT include contradictory actions (e.g., cancelling and updating the same resource)

JSON Output:"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_chain_summary(tools: list[str], tool_graph_nodes: dict[str, dict]) -> str:
    """Build chain summary section from tool graph nodes."""
    lines: list[str] = []
    for i, name in enumerate(tools):
        node = tool_graph_nodes.get(name, {})
        lines.append(f"  {i+1}. **{name}**: {node.get('description', '')}")
        for pname, pspec in node.get("input", {}).items():
            p = f'      "{pname}": {{"type": "{pspec.get("type","string")}", "description": "{pspec.get("description","")}"'
            if pspec.get("enum"):
                p += f', "enum": {json.dumps(pspec["enum"])}'
            lines.append(f"     {p}}}")
        out_desc = node.get("output", {}).get("description", "")
        if out_desc:
            lines.append(f"     Returns: {out_desc}")
        lines.append("")
    return "\n".join(lines)


def _build_scenario_section(scenario: dict | None) -> str:
    """Build violation scenario section for the prompt (empty if no violations)."""
    if not scenario or not scenario.get("violated_subgoals"):
        # Happy-path: no violations — guide the generator toward correct execution
        return """
## SCENARIO TYPE: HAPPY PATH
This is a happy-path test case — the agent should complete ALL requested actions \
successfully, following all policies correctly. There are no policy violations to test.

Focus on:
- The agent correctly executes each tool in the chain in order
- The agent communicates accurate results (prices, dates, statuses, names) from the DB
- The agent obtains confirmation before write actions (bookings, updates, cancellations)
- Each subgoal describes a concrete action or communication with exact DB values
- The task_summary describes a natural, feasible user request that the tools CAN handle
- Do NOT include actions that would be denied by policy — this is a success scenario
"""
    violated = scenario["violated_subgoals"]
    behaviors = scenario.get("expected_behaviors", [])
    constraints = scenario.get("context_constraints", {})
    sg_lines = []
    for i, sg_id in enumerate(violated):
        sg_lines.append(f"- {sg_id}: {behaviors[i] if i < len(behaviors) else violated[sg_id]}")
    return f"""
## BUSINESS RULES TO TEST
{chr(10).join(sg_lines)}

## SCENARIO CONSTRAINTS
{json.dumps(constraints, indent=2)}

IMPORTANT: These business rules describe what the AGENT must enforce. The user in the \
task_summary does NOT know about these rules. Write the task_summary as a natural user \
request — the user simply wants something done, and the agent must apply these rules \
in its response. The subgoals should describe the correct agent behavior (including \
refusals/denials where required by the rules above).
"""


# ── Generator ────────────────────────────────────────────────────────────────

def generate_testcase(
    chain: dict,
    tool_graph_nodes: dict,
    db_snapshot: dict,
    scenario: dict | None,
    prd_context: str,
    client: AzureOpenAI,
) -> dict | None:
    """Return a NAACL-format test case dict, or None on failure."""
    tools = chain["tools"]
    chain_summary = _build_chain_summary(tools, tool_graph_nodes)
    scenario_section = _build_scenario_section(scenario)
    domain_ctx = f"\n## Domain Context\n{prd_context[:2000]}\n" if prd_context else ""

    # Pick prompts based on happy-path vs violation
    is_happy_path = not scenario or not scenario.get("violated_subgoals")
    system_prompt = HAPPY_PATH_SYSTEM if is_happy_path else VIOLATION_SYSTEM
    gen_prompt_template = HAPPY_PATH_PROMPT if is_happy_path else VIOLATION_PROMPT

    # Clean snapshot (remove internal keys)
    clean_snapshot = {k: v for k, v in db_snapshot.items() if not k.startswith("_")}
    feasibility = db_snapshot.get("_feasibility", "")

    prompt = gen_prompt_template.format(
        chain_summary=chain_summary,
        db_snapshot=json.dumps(clean_snapshot, indent=2, default=str),
        feasibility=feasibility,
        scenario_section=scenario_section,
        domain_context=domain_ctx,
        n_caps=len(tools),
    )

    # Retry up to 3 times if LLM returns unparseable response
    result = None
    for attempt in range(3):
        raw = _chat(client, prompt, system=system_prompt, max_tokens=8000)
        result = _parse_json(raw)
        if isinstance(result, dict):
            break
        print(f"    [gen] parse failed (attempt {attempt+1}/3), raw[:200]={raw[:200]!r}")
        import time; time.sleep(2 ** attempt)

    if not isinstance(result, dict):
        print("    [gen] failed to parse LLM response after 3 attempts")
        return None

    # Build expected_tools from LLM output + tool graph nodes
    llm_caps = result.get("expected_tools", [])
    merged_caps = []
    for i, tool_name in enumerate(tools):
        node = tool_graph_nodes.get(tool_name, {})
        llm_cap = llm_caps[i] if i < len(llm_caps) else {}

        input_schema = {}
        for pname, pspec in node.get("input", {}).items():
            entry = {"type": pspec.get("type", "string"), "description": pspec.get("description", ""),
                     "required": pspec.get("required", True)}
            if "enum" in pspec:
                entry["enum"] = pspec["enum"]
            input_schema[pname] = entry

        output_schema = {
            fname: {"type": fspec.get("type", "string"), "description": fspec.get("description", "")}
            for fname, fspec in node.get("output", {}).get("properties", {}).items()
        }

        merged_caps.append({
            "name": tool_name,
            "description": node.get("description", ""),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "arguments": llm_cap.get("arguments", {}),
            "expected_output": llm_cap.get("expected_output", {}),
        })

    subgoals = result.get("subgoals", [])

    # Enforce minimum 3 subgoals — retry if needed
    if len(subgoals) < 3:
        retry_prompt = (
            f"The test case below has only {len(subgoals)} subgoal(s). "
            f"Generate at least 3 subgoals.\n\n"
            f"Task: {result.get('task_summary', '')}\n"
            f"Tools: {' → '.join(tools)}\n"
            f"DB snapshot (abbreviated): "
            f"{json.dumps({k: (v[:1] if isinstance(v, list) else v) for k, v in clean_snapshot.items()}, default=str)}\n\n"
            f"Return ONLY a JSON array of at least 3 subgoal objects:\n"
            f'[{{"description": "Agent [action/communication with exact DB values, e.g. cancels reservation XYZ, communicates total cost is $450]"}}, ...]'
        )
        retry_raw = _chat(client, retry_prompt, system=system_prompt, max_tokens=800)
        retry_parsed = _parse_json(retry_raw)
        if isinstance(retry_parsed, list) and len(retry_parsed) >= 3:
            subgoals = retry_parsed

    tc = {
        "id": "tc_" + uuid.uuid4().hex[:8],
        "task_summary": result.get("task_summary", ""),
        "expected_tools": merged_caps,
        "subgoals": subgoals,
        # Metadata (needed by verifier for data grounding checks)
        "_chain": chain["tools"],
        "_db_snapshot": clean_snapshot,
        "_feasibility": feasibility,
        "violated_subgoals": scenario["violated_subgoals"] if scenario else {},
    }

    return tc


# ── Standalone runner ────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate test case from tool chain + DB snapshot")
    p.add_argument("--chain", required=True, help="JSON file: {chain, tools}")
    p.add_argument("--tool-graph", required=True, help="JSON file: tool graph nodes")
    p.add_argument("--db-snapshot", required=True, help="JSON file: DB snapshot")
    p.add_argument("--scenario", default=None, help="JSON file: violation scenario")
    p.add_argument("--prd", default="", help="PRD markdown file")
    p.add_argument("--out", default=None, help="Output JSON path")
    args = p.parse_args()

    chain = json.loads(Path(args.chain).read_text())
    tool_graph_nodes = json.loads(Path(args.tool_graph).read_text())
    db_snapshot = json.loads(Path(args.db_snapshot).read_text())
    scenario = json.loads(Path(args.scenario).read_text()) if args.scenario else None
    prd_context = Path(args.prd).read_text() if args.prd else ""

    client = make_client()
    tc = generate_testcase(chain, tool_graph_nodes, db_snapshot, scenario, prd_context, client)
    if tc is None:
        print("Failed to generate test case.")
        return
    out_path = args.out or f"tc_{tc['id']}.json"
    Path(out_path).write_text(json.dumps(tc, indent=2))
    print(f"Wrote test case to {out_path}")


if __name__ == "__main__":
    main()
