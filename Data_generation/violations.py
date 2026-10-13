"""Violation combination system — subgoal parsing, 2^n combos, scenario building.

Given online_subgoals.json (business rules), generates all valid subgoal
combinations and builds LLM-constrained scenarios for each.
"""

from __future__ import annotations

import json

import backoff
from openai import AzureOpenAI

import os

DEPLOYMENT = os.environ.get("LLM_DEPLOYMENT", "gpt-5.4")


@backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=120)
def _chat(client: AzureOpenAI, prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages, max_completion_tokens=max_tokens,
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


# ── 1. Subgoal enrichment ────────────────────────────────────────────────────

def parse_subgoals(
    raw_subgoals: list[dict],
    tool_names: list[str],
    prd_context: str,
    client: AzureOpenAI,
) -> list[dict]:
    """Assign IDs to subgoals and LLM-extract required_context + expected_behavior."""
    subgoals = []
    for i, sg in enumerate(raw_subgoals):
        subgoals.append({
            "id": f"SG{i + 1}",
            "details": sg["details"],
            "category": sg.get("category", "compliance_rule"),
            "required_context": {},
            "expected_behavior": "",
        })

    sg_list = "\n".join(f"{s['id']}: {s['details']}" for s in subgoals)
    prompt = f"""Analyze these business rules and extract the required context and expected behavior for each.

## Business Rules
{sg_list}

## Available Tools
{', '.join(tool_names) if tool_names else 'Not specified'}

## Domain Context
{prd_context[:2000] if prd_context else 'Not specified'}

For each rule, extract:
1. required_context: dict of context variables needed to create a test scenario
2. expected_behavior: single sentence describing what the agent should do

Output JSON array:
[{{"id": "SG1", "required_context": {{"key": "description"}}, "expected_behavior": "What agent should do"}}, ...]

JSON Output:"""

    raw = _chat(client, prompt,
                system="You analyze business rules for AI agent evaluation. Output valid JSON.",
                max_tokens=3000)
    extracted = _parse_json(raw)
    if isinstance(extracted, list):
        by_id = {item["id"]: item for item in extracted if "id" in item}
        for sg in subgoals:
            if sg["id"] in by_id:
                sg["required_context"] = by_id[sg["id"]].get("required_context", {})
                sg["expected_behavior"] = by_id[sg["id"]].get("expected_behavior", "")

    return subgoals


# ── 2. Violation combinations ────────────────────────────────────────────────

def generate_violation_combos(
    subgoals: list[dict],
    client: AzureOpenAI,
    max_subgoals: int | None = None,
) -> list[dict]:
    """Generate 2^n powerset of subgoals, filtering conflicting combos via LLM."""
    n = len(subgoals)
    conflict_cache: dict[frozenset, bool] = {}
    combos = []

    for i in range(2 ** n):
        mask = format(i, f"0{n}b")
        selected = [subgoals[j] for j, bit in enumerate(mask) if bit == "1"]

        if max_subgoals is not None and len(selected) > max_subgoals:
            continue

        # Check conflicts for combos with >1 subgoal
        if len(selected) > 1:
            cache_key = frozenset(s["details"] for s in selected)
            if cache_key not in conflict_cache:
                conflict_cache[cache_key] = _check_conflict(selected, client)
            if conflict_cache[cache_key]:
                continue  # conflicting — skip

        subgoal_map = {}
        for j, bit in enumerate(mask):
            if bit == "1":
                subgoal_map[subgoals[j]["id"]] = subgoals[j]["details"]

        combos.append({
            "index": i,
            "binary_mask": mask,
            "subgoals": selected,
            "subgoal_map": subgoal_map,
        })

    return combos


def _check_conflict(selected: list[dict], client: AzureOpenAI) -> bool:
    """Return True if the subgoals conflict and cannot be tested together."""
    descriptions = "\n".join(f"- {s['id']}: {s['details']}" for s in selected)
    prompt = f"""Can these business rules be tested together in a single test scenario?

{descriptions}

Consider if their required contexts conflict (e.g., one needs premium user, one needs basic),
if they apply to incompatible situations, or if testing one prevents testing another.

Output JSON:
{{"can_test_together": true/false, "reason": "explanation or null"}}

JSON Output:"""

    raw = _chat(client, prompt,
                system="You analyze business rules for testing compatibility.",
                max_tokens=500)
    result = _parse_json(raw)
    if isinstance(result, dict):
        return not result.get("can_test_together", True)
    return False  # default to compatible on parse failure


# ── 3. Scenario building ─────────────────────────────────────────────────────

def build_scenario(
    combo: dict,
    tool_graph_nodes: dict[str, dict],
    sampled_chain: list[str],
    prd_context: str,
    client: AzureOpenAI,
) -> dict:
    """Build a ScenarioContext dict for a violation combo + tool chain."""
    if not combo["subgoals"]:
        return {
            "violated_subgoals": {},
            "context_constraints": {},
            "expected_behaviors": [
                "Agent should handle the request successfully following all policies."
            ],
        }

    sg_descriptions = "\n".join(
        f"- {s['id']}: {s['details']}" for s in combo["subgoals"]
    )

    tool_lines = []
    for i, name in enumerate(sampled_chain):
        desc = tool_graph_nodes.get(name, {}).get("description", "")
        tool_lines.append(f"  {i + 1}. {name} — {desc}")
    tool_section = "\n".join(tool_lines)

    prompt = f"""Design a test scenario using ONLY the tools below.

## Agent's Available Tools (in execution order)
{tool_section}

These are the ONLY tools the agent has. The agent CANNOT call any tool not listed above.

## Business Rules to Test
{sg_descriptions}

## Domain Context
{prd_context[:1500] if prd_context else 'Not specified'}

KEY PRINCIPLE: If a tool for action X is not in the chain, the scenario CANNOT involve
performing action X. Design the request around what the tools CAN do.

Create a user request where:
- The agent calls each tool in the chain (in order) to handle it
- The business rules are tested through the agent's RESPONSE, not through extra tool calls
- The user does NOT ask for anything that requires a tool not in the chain

CRITICAL — DO NOT INVENT ENTITY IDs:
- Do NOT include specific user IDs, reservation IDs, flight numbers, or payment IDs in
  context_constraints or expected_behaviors. Real entity IDs will be supplied by the
  database sampler at the next step.
- Use only abstract placeholders like "the user's economy reservation", "the user's saved
  credit card", "the flight on the requested date". Never write values like "U100",
  "RES-4812", "fixture_user", "card_204", "AX101", or any other concrete identifier.

Output JSON:
{{"context_constraints": {{"key": "value"}}, "expected_behaviors": ["What agent should do"], "scenario_description": "Brief description"}}

JSON Output:"""

    raw = _chat(client, prompt,
                system="You create test scenarios strictly constrained to specific tool chains.",
                max_tokens=1500)
    result = _parse_json(raw)

    if isinstance(result, dict):
        return {
            "violated_subgoals": combo["subgoal_map"],
            "context_constraints": result.get("context_constraints", {}),
            "expected_behaviors": result.get("expected_behaviors", []),
        }

    # Fallback on parse failure
    return {
        "violated_subgoals": combo["subgoal_map"],
        "context_constraints": {},
        "expected_behaviors": [],
    }
