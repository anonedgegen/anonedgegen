---
name: create-synthesizer
description: Creates a synthesizer for an agent using the generic BaseSynthesizer framework, with tool registrations, intents, samplers, and postconditions
---

<command-name>create-synthesizer-agent</command-name>

# Create Synthesizer Agent

Creates a complete synthesizer for an agent domain, following the pattern established in `full_airline_synthesizer.py`.

## What This Skill Does

1. **Copies `base.py`** - Duplicates the BaseSynthesizer framework to the target synthesizer directory
2. **Scans the mock system** - Reads all APIs from `mock_system/apis/` and the database schema from `mock_system/db/schema.sql`
3. **Generates `synthesizer_agent.py`** - Creates a complete synthesizer with:
   - All tools registered from the APIs (CREATE/WRITE operations only for intents, but READ tools included for discovery)
   - CREATE intents for each table in the schema
   - Samplers that generate realistic test data
   - Postconditions that verify successful operations
4. **Updates `__init__.py`** - Exports `SynthesizerAgent` and `ALL_TOOLS` from the synthesizer module

## Usage

```
/create-synthesizer-agent <agent_path>
```

Where `<agent_path>` is the path to the agent domain folder (e.g., `src/tau2/domains/airline`).

## Required Directory Structure

The agent domain must have this structure:
```
<agent_path>/
├── mock_system/
│   ├── apis/           # Python files with API functions
│   │   ├── __init__.py
│   │   ├── users.py
│   │   ├── flights.py
│   │   └── ...
│   └── db/
│       ├── schema.sql  # SQLite schema defining tables
│       ├── db.py       # Database connection (get_conn)
│       └── models.py   # Dataclass models
└── synthesizer/        # Output directory (created if needed)
```

## Instructions

<instructions>
You are creating a synthesizer agent for a domain. Follow these steps exactly:

### Step 1: Validate Inputs and Structure

1. Parse the agent path from the user's command arguments
2. If no path provided, ask: "Please provide the path to the agent domain folder (e.g., src/tau2/domains/airline)"
3. Verify the following exist:
   - `<agent_path>/mock_system/apis/` directory with Python files
   - `<agent_path>/mock_system/db/schema.sql` file
   - `<agent_path>/mock_system/db/db.py` file
4. Create `<agent_path>/synthesizer/` directory if it doesn't exist

### Step 2: Verify agent-synth dependency

1. Confirm that `agent-synth` is available as a dependency (it provides `BaseSynthesizer`, `IntentDef`, etc.)
2. No need to copy `base.py` — it is provided by the `agent_synth` package
3. The synthesizer_agent.py will import from `agent_synth.synthesizer.base` directly

### Step 3: Analyze the Mock System

**3a. Parse the Database Schema**

Read `<agent_path>/mock_system/db/schema.sql` and extract:
- All table names
- Column names and types for each table
- Primary keys (fields with PRIMARY KEY constraint)
- Foreign keys (fields with REFERENCES constraint)
- NOT NULL constraints
- UNIQUE constraints
- CHECK constraints (for enums like "membership IN ('gold', 'silver', 'regular')")

**3b. Scan All API Files**

For each Python file in `<agent_path>/mock_system/apis/` (except `__init__.py`):
1. Read the file content
2. Extract all function definitions with their:
   - Function name
   - Parameters (name, type hint, default value)
   - Docstring (for description)
   - Return type hint
3. Categorize functions:
   - **CREATE functions**: Names starting with `create_`, `add_`, `insert_`, `book_`, `register_`
   - **UPDATE functions**: Names starting with `update_`, `modify_`, `change_`, `set_`
   - **DELETE functions**: Names starting with `delete_`, `remove_`, `cancel_`
   - **READ functions**: Names starting with `get_`, `list_`, `search_`, `find_` (include these as helper tools)

**3c. Parse Data Models**

Read `<agent_path>/mock_system/db/models.py` to understand:
- Dataclass definitions
- Field names and types
- Which models correspond to which tables

### Step 4: Generate synthesizer_agent.py

Create the file `<agent_path>/synthesizer/synthesizer_agent.py` following this exact structure:

```python
"""
Synthesizer Agent for [Domain Name].

This synthesizer has CREATE intents for ALL tables in the schema.
Generated automatically by create-synthesizer-agent skill.
"""

from __future__ import annotations

import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_MOCK_SYSTEM_DIR = _THIS_DIR.parent / "mock_system"
sys.path.insert(0, str(_MOCK_SYSTEM_DIR))

from agent_synth.synthesizer.base import (
    BaseSynthesizer,
    IntentDef,
    Step,
    SynthesizerConfig,
    find_in_steps,
)

# Import all API modules
from apis import [module1], [module2], ...
from db.db import get_conn


class SynthesizerAgent(BaseSynthesizer):
    """
    Synthesizer with CREATE intents for all [domain] tables.
    """

    name = "[domain]_synthesizer"
    description = "[Domain] synthesizer with all CREATE intents"

    def setup_database(self) -> None:
        from setup import setup_database
        setup_database(reset=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TOOL REGISTRATION
    # ═══════════════════════════════════════════════════════════════════════

    def setup_tools(self) -> None:
        """Register ALL tools from APIs."""
        
        # For each API file, register all public functions as tools
        # Include both READ tools (for discovery) and WRITE tools (for actions)
        
        # Example pattern for each tool:
        self.register_tool(
            name="function_name",
            fn=lambda **kw: module.function_name(**kw),
            description="Description from docstring",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "..."},
                    "param2": {"type": "integer"},
                },
                "required": ["param1"],
            },
        )

    # ═══════════════════════════════════════════════════════════════════════
    # INTENT REGISTRATION
    # ═══════════════════════════════════════════════════════════════════════

    def setup_intents(self) -> None:
        """
        Register CREATE intents for all tables.
        Only CREATE/WRITE intents - no READ intents.
        """

        # For each table in the schema that has a create function, create an intent:
        self.register_intent(IntentDef(
            name="CREATE_[TABLE_NAME]",
            brief="Create a new [entity] in the system.",
            informed_rules="",  # Left empty for learning
            tools="give it a list which has all the tools. No need to alter that list for intents. Whatever you register above in the previous step, store all in 1 list and give that list here",
            sampler=self._sample_create_[entity],
            task_prompt=lambda p: (
                f"Create a new [entity]:\n"
                f"- Field1: {p['field1']}\n"
                f"- Field2: {p['field2']}\n"
            ),
            postconditions=self._postcond_create_[entity],
        ))

    # ═══════════════════════════════════════════════════════════════════════
    # SAMPLERS (CRITICAL: Must NEVER return None - always return dict)
    # ═══════════════════════════════════════════════════════════════════════

    # For each CREATE intent, create a sampler method
    # Return type is `dict`, NOT `Optional[dict]`
    
    def _sample_create_[entity](self) -> dict:
        # Generate realistic random values for required fields
        # Respect constraints from schema (enums, ranges, etc.)
        # Sometimes generate invalid values (10-15%) to test error handling
        # Use existing data from DB for foreign key references
        
        field_values = ["value1", "value2", "value3"]
        
        # For foreign keys, query existing records:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT id, name FROM related_table ORDER BY RANDOM() LIMIT 1"
            ).fetchone()
        
        # ALWAYS provide fallback - never return None!
        if row:
            related_id = row["id"]
            related_name = row["name"]
        else:
            # Generate fake ID to trigger meaningful error for learning
            related_id = f"fake_id_{random.randint(1000, 9999)}"
            related_name = "Nonexistent"
        
        # Always return a dict, never None
        return {
            "field1": random.choice(field_values),
            "field2": random.randint(1, 100),
            "related_id": related_id,
            "related_name": related_name,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # POSTCONDITIONS
    # ═══════════════════════════════════════════════════════════════════════

    # For each CREATE intent, create postconditions
    
    def _postcond_create_[entity](self, params: dict, steps: List[Step]) -> Dict[str, bool]:
        result = find_in_steps(steps, "create_[entity]")
        if result is None:
            return {"create_[entity]_called": False}
        return {
            "create_[entity]_called": True,
            "[entity]_created": result.get("[primary_key]") is not None,
        }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run Synthesizer Agent")
    parser.add_argument("--intent", help="Intent to run")
    parser.add_argument(
        "--mode", choices=["blind", "informed"], default="informed",
        help="Agent mode (default: informed)",
    )
    parser.add_argument("--list-intents", action="store_true", help="List available intents")
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    parser.add_argument("--reset-db", action="store_true", help="Reset database with seed data")
    parser.add_argument(
        "--empty-db", action="store_true",
        help="Start with empty database (schema only)",
    )
    parser.add_argument("--batch", type=int, metavar="N", help="Run N iterations")
    parser.add_argument("--verbose", action="store_true", help="Show detailed tool calls")
    args = parser.parse_args()

    config = SynthesizerConfig.from_env()
    synth = SynthesizerAgent(config)

    if args.list_intents:
        print("Available intents:")
        for name in synth.list_intents():
            print(f"  - {name}: {synth.intents[name].brief}")
        return

    if args.list_tools:
        print("Available tools:")
        for name in synth.list_tools():
            print(f"  - {name}")
        return

    # Database setup
    from db.db import reset_db as _reset_empty
    db_path = _MOCK_SYSTEM_DIR / "db" / "[domain]_mock.db"

    if args.empty_db:
        print("Starting with empty database...")
        _reset_empty()
    elif args.reset_db or not db_path.exists():
        synth.setup_database()
    else:
        print(f"Using existing database: {db_path}")

    # Run
    if args.batch:
        results = synth.run_batch(args.batch, intent=args.intent, mode=args.mode)
        successes = sum(1 for r in results for t in r.trajectories if t["success"])
        print(f"\nBatch: {successes}/{len(results)} succeeded")
    else:
        if not args.intent:
            print("Error: --intent is required (or use --list-intents)")
            sys.exit(1)

        params = synth.sample_params(intent=args.intent, mode=args.mode)
        if not params:
            print(f"Error: No eligible data for intent {args.intent}")
            sys.exit(1)

        print(f"Running: {params['intent']} ({params['mode']} mode)")
        print(f"Params: {json.dumps(params['run_params'], indent=2)}")
        print("-" * 60)

        result = synth.run(params)

        for traj in result.trajectories:
            print(f"\nSuccess: {traj['success']}")
            print(
                f"Steps: {traj['total_steps']}, "
                f"Errors: {traj['error_count']}, "
                f"Recoveries: {traj['recovery_count']}"
            )
            print(f"Postconditions: {json.dumps(traj['postconditions'], indent=2)}")

            if args.verbose:
                print("\nTool calls:")
                for i, step in enumerate(traj["steps"], 1):
                    status = "ERROR" if step["is_error"] else "OK"
                    args_str = json.dumps(step["args"])
                    if len(args_str) > 80:
                        args_str = args_str[:77] + "..."
                    print(f"  {i}. {step['tool']}({args_str}) -> {status}")


if __name__ == "__main__":
    main()
```

### Step 5: Fill In the Generated Code

When generating the actual code:

**For Tool Registration (CRITICAL - Register ALL API functions):**

For EVERY Python file in `mock_system/apis/` (except `__init__.py`):
1. Import the module: `from apis import modulename`
2. For EVERY public function (not starting with `_`), register it as a tool
3. Map Python types to JSON schema types:
   - `str` → `"string"`
   - `int` → `"integer"`
   - `float` → `"number"`
   - `bool` → `"boolean"`
   - `list` → `"array"`
   - `dict` → `"object"`
   - `Optional[X]` → type X but not in "required"
4. Extract descriptions from docstrings
5. Mark parameters without defaults as "required"

**For Intent Names:**
- Use SCREAMING_SNAKE_CASE: `CREATE_USER`, `CREATE_FLIGHT`, `CREATE_RESERVATION`
- Only create intents for tables that have corresponding create/write API functions
- Do NOT create READ intents (get_, list_, search_ functions are tools only, not intents)

**For Samplers (CRITICAL - Samplers must NEVER return None):**

Samplers must ALWAYS return a `dict`, never `None` or `Optional[dict]`. This is critical because:
- Returning `None` causes the intent to be skipped entirely
- Skipped intents mean no learning/evolution happens
- The agent needs to encounter errors to learn from them

Pattern for samplers that depend on database data:

```python
def _sample_create_[entity](self) -> dict:  # Return type is dict, NOT Optional[dict]
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, name FROM related_table ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
    
    # ALWAYS provide fallback values when DB is empty
    if row:
        related_id = row["id"]
        related_name = row["name"]
    else:
        # Generate fake ID to trigger error - agent learns from this!
        related_id = f"fake_id_{random.randint(1000, 9999)}"
        related_name = "Nonexistent"
    
    # Always return a dict, never None
    return {
        "related_id": related_id,
        "related_name": related_name,
        "field1": random.choice(["value1", "value2"]),
    }
```

Additional sampler guidelines:
- Generate realistic sample data based on field types and names:
  - Names: Use lists like `["Alex", "Jordan", "Morgan", "Taylor"]`
  - Emails: Generate from name pattern `f"{first.lower()}.{last.lower()}.{random.randint(100,999)}@email.com"`
  - Dates: Use `date.today() + timedelta(days=random.randint(1, 60))`
  - IDs: For foreign keys, query the related table with `ORDER BY RANDOM() LIMIT 1`
  - Enums: Extract valid values from CHECK constraints in schema
- Add 10-15% chance of invalid values for learning
- When DB query returns no results, generate fake/placeholder data that will cause meaningful errors

**For Postconditions:**
- Check that the main create function was called using `find_in_steps`
- Verify the primary key field is present in the result

### Step 6: Update __init__.py

Update or create `<agent_path>/synthesizer/__init__.py` to export the new synthesizer:

```python
"""Synthesizer module for [Domain] domain."""

from agent_synth.synthesizer.base import (
    BaseSynthesizer,
    SynthesizerConfig,
    SynthesisResult,
    Trajectory,
    Step,
    IntentDef,
    find_in_steps,
)
from .synthesizer_agent import SynthesizerAgent, ALL_TOOLS

__all__ = [
    "BaseSynthesizer",
    "SynthesizerConfig",
    "SynthesisResult",
    "Trajectory",
    "Step",
    "IntentDef",
    "find_in_steps",
    "SynthesizerAgent",
    "ALL_TOOLS",
]
```

If the `__init__.py` already exists, add the imports for `SynthesizerAgent` and `ALL_TOOLS` from `.synthesizer_agent` and include them in `__all__`.

### Step 7: Report Results

After generation, print:
```
✓ Created synthesizer at: <agent_path>/synthesizer/synthesizer_agent.py
✓ Using agent_synth.synthesizer.base (from agent-synth package)
✓ Updated __init__.py to export SynthesizerAgent and ALL_TOOLS

Registered:
  - Tools: [count] tools from [count] API modules
  - Intents: [list of CREATE_* intent names]

To run:
  cd <project_root>
  python -m <module.path>.synthesizer.synthesizer_agent --list-intents
  python -m <module.path>.synthesizer.synthesizer_agent --intent CREATE_[TABLE] --mode blind
```
</instructions>

## Example

For the airline domain:
```
/create-synthesizer-agent src/tau2/domains/airline
```

This would:
1. Verify agent-synth package is available (provides BaseSynthesizer framework)
2. Scan `mock_system/apis/` (users.py, flights.py, payment_methods.py, reservations.py)
3. Parse `mock_system/db/schema.sql` for tables
4. Generate `synthesizer_agent.py` with:
   - CREATE_USER, CREATE_PAYMENT_METHOD, CREATE_FLIGHT, CREATE_FLIGHT_DATE, CREATE_RESERVATION intents
   - All API functions registered as tools (list_users, get_user, create_user, etc.)
   - Samplers with realistic airline data
   - Postconditions checking database state

## Reference Implementation

See `/Users/redacted_num/tau2bench-1/tau2-bench/src/tau2/domains/airline/synthesizer/full_airline_synthesizer.py` for the reference implementation that this skill replicates.
