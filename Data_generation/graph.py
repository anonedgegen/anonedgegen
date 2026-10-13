"""Tool Graph — builds a directed graph of tool dependencies from tools.json"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from openai import AzureOpenAI

DEPLOYMENT = "gpt-5.4"


def make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
    )


@dataclass
class ToolGraph:
    """Directed graph where nodes are tools and edges represent dependencies."""

    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    outgoing: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    incoming: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node_id: str, info: dict) -> None:
        self.nodes[node_id] = info

    def add_edge(self, source: str, target: str) -> None:
        if source in self.nodes and target in self.nodes and source != target:
            if (source, target) not in set(self.edges):
                self.edges.append((source, target))
                self.outgoing[source].append(target)
                self.incoming[target].append(source)

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [{"source": s, "target": t} for s, t in self.edges],
        }


def build_graph(tools_json_path: Path, client: AzureOpenAI | None = None) -> ToolGraph:
    """Build tool graph with resource-dependency edges + optional LLM enrichment.

    Args:
        tools_json_path: path to tools.json
        client: Azure OpenAI client for LLM graph enrichment (optional)
    """
    with open(tools_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)["tools"]

    graph = ToolGraph()

    # --- nodes ---
    for tool in tools:
        # Support both key conventions: input-type/output-type and input_types/output_types
        in_types = tool.get("input-type") or tool.get("input_types", [])
        out_types = tool.get("output-type") or tool.get("output_types", [])

        # Normalize params: "input" is dict, "arguments" may be list of {name, type, ...}
        params = tool.get("input")
        if not params:
            args_raw = tool.get("arguments", {})
            if isinstance(args_raw, list):
                params = {p["name"]: {k: v for k, v in p.items() if k != "name"} for p in args_raw}
            else:
                params = args_raw

        # Normalize output: "output" is dict, "returns" may be dict or other
        output = tool.get("output") or tool.get("returns", {})

        graph.add_node(tool["name"], {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input-type": in_types,
            "output-type": out_types,
            "input": params,
            "output": output,
        })

    # --- resource-dependency edges: A → B if output-type(A) ∩ input-type(B) ---
    for a in tools:
        out_types = {t.lower() for t in (a.get("output-type") or a.get("output_types", []))}
        if not out_types:
            continue
        for b in tools:
            if a["name"] == b["name"]:
                continue
            in_types = {t.lower() for t in (b.get("input-type") or b.get("input_types", []))}
            if out_types & in_types:
                graph.add_edge(a["name"], b["name"])

    print(f"  Resource edges: {len(graph.edges)}")

    # --- LLM enrichment ---
    if client:
        from graph_enricher import find_additional_edges

        existing = list(graph.edges)
        new_edges = find_additional_edges(graph.nodes, existing, client)
        for src, tgt in new_edges:
            graph.add_edge(src, tgt)
        print(f"  LLM-enriched edges: {len(new_edges)}")

    return graph


def main():
    if len(sys.argv) < 2:
        print("Usage: python graph.py tools.json [--no-enrich]")
        sys.exit(1)

    tools_path = Path(sys.argv[1]).resolve()
    no_enrich = "--no-enrich" in sys.argv

    print(f"Loading tools from {tools_path}")
    client = None if no_enrich else make_client()
    graph = build_graph(tools_path, client)

    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Total edges: {len(graph.edges)}")

    out_path = tools_path.parent / "graph.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
