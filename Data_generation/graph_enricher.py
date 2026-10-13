"""LLM-based graph enrichment — discover additional tool edges from descriptions.

Single function that asks the LLM to find workflow relationships between tools
that go beyond simple output-type → input-type matching.
"""

import json
import os

import backoff
from openai import AzureOpenAI

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


SYSTEM_PROMPT = """You are an expert at analyzing tool APIs to identify workflow dependencies.

An edge from Tool A to Tool B means: "After using Tool A, a user might logically use Tool B next."

Consider these relationship patterns:
1. Sequential workflows: multi-step processes where one action leads to another
2. Data flow: Tool A retrieves/produces information that Tool B needs
3. Verification flows: getting information before/after modifications
4. Prerequisite relationships: actions that must logically precede others
5. Escalation paths: when automated handling isn't sufficient

Be thorough. Include all plausible workflow edges, not just the most obvious ones.
Do NOT include edges already in the existing edges list."""


def find_additional_edges(
    tools: dict[str, dict],
    existing_edges: list[tuple[str, str]],
    client: AzureOpenAI,
) -> list[tuple[str, str]]:
    """Discover workflow edges between tools via LLM analysis.

    Args:
        tools: tool_name → tool dict (with "description", "input", "output", etc.)
        existing_edges: list of (from_tool, to_tool) already in the graph
        client: Azure OpenAI client

    Returns:
        List of new (from_tool, to_tool) edges not in existing_edges.
    """
    if not tools:
        return []

    # Format tools for prompt
    tool_lines = []
    for name, tool in tools.items():
        desc = tool.get("description", "")
        params = list(tool.get("input", {}).keys())
        line = f"- **{name}**: {desc}"
        if params:
            line += f"\n  Parameters: {', '.join(params)}"
        tool_lines.append(line)
    tools_section = "\n".join(tool_lines)

    # Format existing edges
    if existing_edges:
        edges_section = "\n".join(f"- {a} -> {b}" for a, b in existing_edges)
    else:
        edges_section = "(none)"

    prompt = f"""Analyze these tools and identify workflow edges between them.

## Tools Available:
{tools_section}

## Existing Edges (DO NOT include these):
{edges_section}

Identify edges that represent realistic workflow relationships NOT in the existing list.
Return ONLY a valid JSON array:
[{{"from": "tool_a", "to": "tool_b"}}, ...]

Return [] if no additional meaningful edges exist."""

    raw = _chat(client, prompt, system=SYSTEM_PROMPT, max_tokens=2000)

    # Parse response
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        raw = "\n".join(json_lines)

    try:
        edges_data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(edges_data, list):
        return []

    # Validate and deduplicate
    existing_set = set(existing_edges)
    tool_names = set(tools.keys())
    new_edges = []

    for edge in edges_data:
        if not isinstance(edge, dict):
            continue
        src = edge.get("from")
        tgt = edge.get("to")
        if not src or not tgt:
            continue
        if src not in tool_names or tgt not in tool_names:
            continue
        if src == tgt:
            continue
        if (src, tgt) in existing_set:
            continue
        new_edges.append((src, tgt))
        existing_set.add((src, tgt))

    print(f"  LLM graph enricher found {len(new_edges)} additional edges")
    return new_edges
