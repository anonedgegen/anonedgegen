---
name: generate-mock-system
description: Scans an agent's codebase to generate SQLite database schema, mock API implementations, and seed data for synthesis testing
---

<command-name>generate-mock-system</command-name>

You are generating a mock system for agent synthesis testing. This involves analyzing an agent's tools/APIs and creating:
1. SQLite database schema
2. Mock API implementations with business logic
3. Seed data for testing

## CRITICAL: The Synthesizer's Purpose

The synthesizer's main job is to **populate the database from scratch**. This means:

1. **Every entity MUST have create/write APIs** - even if the original agent only has read APIs
2. The agent must be able to bootstrap all data it needs through API calls
3. Seed data is optional - the synthesizer should work with an empty database

**Example**: If the original agent has `get_user_details(user_id)` but no `create_user()`, you MUST create a `create_user()` API. The synthesizer agent needs to create users before it can book reservations for them.

**The rule**: For every table in the schema, generate:
- `create_{entity}()` - Create new records
- `get_{entity}()` - Get by ID  
- `list_{entities}()` - List with filters
- `update_{entity}()` - Update fields (if applicable)
- `delete_{entity}()` - Delete records (if applicable)

## CRITICAL: Reference/Lookup Tables Need APIs Too

**This is the most commonly missed requirement!**

Reference tables (also called lookup tables) are tables that store static/semi-static data that other tables reference via foreign keys. Examples:
- `countries` - country codes, names, currencies
- `departments` - department codes and names
- `status_types` - valid status values
- `categories` - product/item categories
- `currencies` - currency codes
- `legal_entities` - business units
- `wage_types` - payroll earning/deduction types

**These tables MUST have create APIs!** Without them, the synthesizer cannot:
1. Create the reference data needed for foreign key constraints
2. Bootstrap an empty database
3. Run intents that depend on these references existing

**Example failure scenario:**
```
Schema has: countries(country_code, country_name, currency_code)
Schema has: assignments(assignment_id, user_id, country_code REFERENCES countries)

If you DON'T create a create_country() API:
- Agent tries to create_assignment(country_code="USA")
- Fails with "Country 'USA' not found" because countries table is empty
- All dependent intents (CREATE_ASSIGNMENT, CREATE_PAYSLIP) fail
```

**The fix:**
```python
# apis/reference_data.py - ALWAYS CREATE THIS FILE

def create_country(country_code: str, country_name: str, currency_code: str) -> dict:
    """Create a country reference record."""
    ...

def list_countries() -> List[dict]:
    """List all countries."""
    ...

# Same pattern for ALL reference tables
```                

## Step 1: Analyze the Agent's Tools

First, find and analyze the agent's tool definitions. Look for:
- `@tool` decorated functions (LangChain style)
- Tool registrations in agent configuration
- API client calls that need mocking

Search for tools:
```
Grep for: @tool|def.*tool|tools\s*=|TOOL_REGISTRY
In: {agent_directory}
```

For each tool found, extract:
- **Name**: The tool/function name
- **Parameters**: Input parameters with types
- **Return type**: What it returns
- **Description**: From docstring
- **External calls**: What external APIs/services it calls

## Step 2: Identify Entities and Relationships

From the tools, identify:
- **Entities**: Users, Orders, Timesheets, etc.
- **Relationships**: User has many Timesheets, Order belongs to Customer
- **Status flows**: DRAFT → SUBMITTED → APPROVED
- **Business rules**: Validation, constraints

Create an entity map:
```
Entity: User
  Fields: user_id (PK), name, email, department, manager_id (FK→User), country
  Relationships: has_many Timesheets, has_many Orders

Entity: Timesheet  
  Fields: timesheet_id (PK), user_id (FK), week_start, status, hours
  Status flow: DRAFT → SUBMITTED → APPROVED/REJECTED
  Business rules: week_start must be Monday, hours ≤ 24/day
```

## Step 3: Generate Database Schema

Create `{output_dir}/db/schema.sql`:

```sql
-- {Agent Name} Mock Database Schema
-- Generated for synthesis testing

-- Core entities
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    -- ... fields from entity analysis
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for common queries
CREATE INDEX idx_users_country ON users(country);

-- Add foreign key relationships
-- ... 
```

**Schema guidelines:**
- Use TEXT for IDs (UUIDs)
- Use TEXT for dates (ISO format YYYY-MM-DD)
- Use REAL for decimals
- Use INTEGER for booleans (0/1)
- Add status fields with TEXT type
- Add created_at/updated_at timestamps
- Create indexes for foreign keys and filter columns

## Step 4: Generate Database Connection Module

Create `{output_dir}/db/db.py`:

```python
"""Database connection for {Agent Name} mock system."""

from pathlib import Path

from agent_synth.mock_system.db import get_conn as _get_conn, init_db as _init_db, reset_db as _reset_db

DB_PATH = Path(__file__).parent / "{agent_name}_mock.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def dict_factory(cursor, row):
    """Convert rows to dictionaries."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_conn():
    """Get database connection with dict rows."""
    return _get_conn(DB_PATH)


def init_db():
    """Initialize database from schema."""
    _init_db(DB_PATH, SCHEMA_PATH)


def reset_db():
    """Drop and recreate database."""
    _reset_db(DB_PATH, SCHEMA_PATH)
```

## Step 5: Generate Data Models

Create `{output_dir}/db/models.py`:

```python
"""Data models for {Agent Name} mock system."""

from dataclasses import dataclass
from typing import Optional


def row_to_dict(row) -> dict:
    """Convert sqlite3.Row to dictionary."""
    return dict(row) if row else {}


@dataclass
class User:
    """User entity."""
    user_id: str
    name: str
    email: Optional[str]
    department: Optional[str]
    country: str
    manager_id: Optional[str]
    # ... other fields


@dataclass  
class Timesheet:
    """Timesheet entity."""
    timesheet_id: str
    user_id: str
    week_start: str
    week_end: str
    status: str  # DRAFT, SUBMITTED, APPROVED, REJECTED
    total_hours: float
    # ... other fields
```

## Step 6: Generate Mock APIs

**IMPORTANT**: Generate CRUD APIs for ALL entities, even if the original agent only has read APIs.

The synthesizer needs to populate the database from scratch. If there's a `users` table, there MUST be a `create_user()` function. If there's a `flights` table, there MUST be a `create_flight()` function.

**Checklist for each entity:**
- [ ] `create_{entity}()` - REQUIRED for synthesizer to populate data
- [ ] `get_{entity}()` - Get single record by ID
- [ ] `list_{entities}()` - List/search with filters  
- [ ] `update_{entity}()` - Update existing record
- [ ] `delete_{entity}()` - Delete record (optional)

For each entity, create `{output_dir}/apis/{entity}.py`:

```python
"""
{Entity} API for {Agent Name} mock system.
Simulates: {original API endpoint if known}
"""

import uuid
from datetime import date
from typing import List, Optional

from db.db import get_conn
from db.models import {Entity}, row_to_dict


def _row_to_{entity}(row) -> {Entity}:
    """Convert database row to {Entity} dataclass."""
    d = row_to_dict(row)
    return {Entity}(
        # ... field mappings
    )


# ── Read Operations ─────────────────────────────────────────────────────────

def get_{entity}({id_field}: str) -> Optional[{Entity}]:
    """Get {entity} by ID."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM {table} WHERE {id_field} = ?",
            ({id_field},)
        ).fetchone()
    
    if row is None:
        return None
    return _row_to_{entity}(row)


def list_{entities}(
    # filter parameters
    limit: int = 50,
    offset: int = 0,
) -> List[{Entity}]:
    """List {entities} with optional filters."""
    query = "SELECT * FROM {table} WHERE 1=1"
    params = []
    
    # Add filter conditions
    
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    
    return [_row_to_{entity}(r) for r in rows]


# ── Write Operations ────────────────────────────────────────────────────────

def create_{entity}(
    # required parameters
    # optional parameters with defaults
) -> {Entity}:
    """
    Create a new {entity}.
    
    Business rules:
    - {rule 1}
    - {rule 2}
    """
    # Generate ID
    {id_field} = str(uuid.uuid4())
    
    with get_conn() as conn:
        # Validate business rules
        # Example: Check for duplicates
        if conn.execute(
            "SELECT 1 FROM {table} WHERE {unique_field} = ?",
            ({unique_value},)
        ).fetchone():
            raise ValueError(f"{Entity} with {unique_field}='{unique_value}' already exists")
        
        # Insert record
        conn.execute(
            """
            INSERT INTO {table} ({fields})
            VALUES ({placeholders})
            """,
            ({values})
        )
        
        row = conn.execute(
            "SELECT * FROM {table} WHERE {id_field} = ?",
            ({id_field},)
        ).fetchone()
    
    return _row_to_{entity}(row)


def update_{entity}(
    {id_field}: str,
    # optional update fields
) -> {Entity}:
    """Update {entity}."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM {table} WHERE {id_field} = ?",
            ({id_field},)
        ).fetchone()
        
        if row is None:
            raise LookupError(f"{Entity} '{id_field}' not found")
        
        # Build update query dynamically
        updates = {}
        # ... add non-None fields to updates
        
        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [{id_field}]
            conn.execute(
                f"UPDATE {table} SET {set_clause} WHERE {id_field} = ?",
                values
            )
        
        row = conn.execute(
            "SELECT * FROM {table} WHERE {id_field} = ?",
            ({id_field},)
        ).fetchone()
    
    return _row_to_{entity}(row)


def delete_{entity}({id_field}: str) -> bool:
    """Delete {entity}."""
    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM {table} WHERE {id_field} = ?",
            ({id_field},)
        )
        return result.rowcount > 0
```

## Step 7: Implement Business Logic

For entities with status workflows, implement state machine logic:

```python
# Status constants
VALID_STATUSES = ["DRAFT", "SUBMITTED", "APPROVED", "REJECTED"]
STATUS_TRANSITIONS = {
    "DRAFT": ["SUBMITTED"],
    "SUBMITTED": ["APPROVED", "REJECTED"],
    "REJECTED": ["DRAFT", "SUBMITTED"],  # Can resubmit
    "APPROVED": [],  # Terminal state
}


def update_status({id_field}: str, new_status: str, **kwargs) -> {Entity}:
    """
    Update {entity} status with validation.
    
    Business rules:
    - Can only transition to allowed statuses
    - APPROVED requires approver_id
    - REJECTED requires rejection_reason
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {new_status}")
    
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM {table} WHERE {id_field} = ?",
            ({id_field},)
        ).fetchone()
        
        if row is None:
            raise LookupError(f"{Entity} '{id_field}' not found")
        
        current_status = row["status"]
        allowed = STATUS_TRANSITIONS.get(current_status, [])
        
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition from {current_status} to {new_status}. "
                f"Allowed: {allowed}"
            )
        
        # Status-specific validation
        if new_status == "APPROVED" and not kwargs.get("approver_id"):
            raise ValueError("approver_id required for approval")
        
        if new_status == "REJECTED" and not kwargs.get("rejection_reason"):
            raise ValueError("rejection_reason required for rejection")
        
        # Update
        conn.execute(
            "UPDATE {table} SET status = ?, updated_at = ? WHERE {id_field} = ?",
            (new_status, date.today().isoformat(), {id_field})
        )
        
        row = conn.execute(
            "SELECT * FROM {table} WHERE {id_field} = ?",
            ({id_field},)
        ).fetchone()
    
    return _row_to_{entity}(row)
```

## Step 8: Generate Seed Data

Create `{output_dir}/setup.py`:

```python
"""Setup script for {Agent Name} mock database."""

from db.db import init_db, reset_db, get_conn


def setup_database(reset: bool = True):
    """Initialize database with sample data."""
    print("Setting up {Agent Name} mock database...")
    
    if reset:
        print("  Resetting database...")
        reset_db()
    else:
        init_db()
    
    print("  Creating sample data...")
    _create_reference_data()
    _create_users()
    _create_{entities}()
    # ... other seed functions
    
    print("  Done!")


def _create_reference_data():
    """Create lookup/reference data."""
    # Countries, statuses, categories, etc.
    pass


def _create_users():
    """Create sample users."""
    users = [
        ("USER001", "uuid-1", "Alice Smith", "Engineering", "USA", None),
        ("USER002", "uuid-2", "Bob Jones", "Sales", "GBR", "USER001"),
        # ... more users with realistic data
    ]
    
    with get_conn() as conn:
        for user in users:
            conn.execute(
                "INSERT INTO users (user_id, ...) VALUES (?, ...)",
                user
            )
    
    print(f"    - {len(users)} users created")


def _create_{entities}():
    """Create sample {entities}."""
    # Create realistic test data
    # Include various statuses for testing different intents
    # Example: some SUBMITTED timesheets for APPROVE_TIMESHEET intent
    pass


if __name__ == "__main__":
    setup_database()
```

**Seed data guidelines:**
- Create enough data for all intents to find eligible records
- Include records in various statuses (DRAFT, SUBMITTED, APPROVED)
- Create relationships (users with managers, orders with items)
- Use realistic but fake data (names, dates, amounts)
- Include edge cases (empty lists, max values)

## Step 9: Create Package Structure

Create the following directory structure:

```
{output_dir}/
├── __init__.py
├── setup.py
├── db/
│   ├── __init__.py
│   ├── db.py
│   ├── models.py
│   └── schema.sql
└── apis/
    ├── __init__.py
    ├── reference_data.py    <-- CRITICAL: Don't forget this! 
    ├── users.py
    ├── {entity1}.py
    ├── {entity2}.py
    └── ...
```

Create `{output_dir}/__init__.py`:

```python
"""
{Agent Name} Mock System.

Usage:
    from {package} import setup_database
    setup_database()
    
    from {package}.apis import users, {entities}
"""

from .setup import setup_database
from .db import init_db, reset_db, get_conn

__all__ = ["setup_database", "init_db", "reset_db", "get_conn"]
```

## Step 10: Validate the Mock System

Run validation:

```python
# Test database setup
from {package} import setup_database
setup_database()

# Test each API
from {package}.apis import users

# Read operations
user = users.get_user("USER001")
assert user is not None

all_users = users.list_users()
assert len(all_users) > 0

# Write operations
new_user = users.create_user(
    user_id="TEST001",
    name="Test User",
    # ...
)
assert new_user.user_id == "TEST001"

# Business rule validation
try:
    users.create_user(user_id="TEST001", ...)  # Duplicate
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "already exists" in str(e)

print("All validations passed!")
```

## Output Checklist

After running this skill, verify:

- [ ] `db/schema.sql` - Complete schema with all tables, indexes, foreign keys
- [ ] `db/db.py` - Connection management with dict_factory
- [ ] `db/models.py` - Dataclasses for all entities
- [ ] `apis/reference_data.py` - **CRITICAL: CRUD for ALL reference/lookup tables** 
- [ ] `apis/*.py` - CRUD operations for each entity with business logic
- [ ] `setup.py` - Seed data covering all test scenarios (OPTIONAL - synthesizer should work without it)
- [ ] `__init__.py` files - Proper package structure
- [ ] All APIs handle errors properly (LookupError, ValueError)
- [ ] Status workflows enforce valid transitions
- [ ] Seed data includes records in various states

**CRITICAL - Synthesizer Compatibility Check:**

- [ ] **Every table has a `create_*()` function** - The synthesizer populates the database, so it needs write APIs
- [ ] **Reference/lookup tables have APIs** - countries, departments, categories, etc.       
- [ ] Create APIs exist for: users, flights, reservations, payment_methods, etc.
- [ ] The agent can bootstrap ALL required data through API calls alone
- [ ] Test: Can you run intents with an EMPTY database? If not, add missing create APIs.

**Empty Database Test (MUST PASS):**

```python
# Reset to empty DB (schema only, no data)
reset_db()

# Try to run each intent - if any step fails with "X not found", you're missing a create API!
```

Example - if the original agent only has these tools:
```
get_user_details(user_id)
search_direct_flight(origin, destination, date)
book_reservation(...)
```

The mock system MUST also include:
```
create_user(first_name, last_name, email, ...) -> User
create_flight(flight_number, origin, destination, ...) -> Flight
create_payment_method(user_id, source, ...) -> PaymentMethod
```

Without these, the synthesizer cannot populate data and will fail when samplers return None or when foreign key constraints are violated.                                                                                                                  