# Generate PRD and Tools

Generate PRD (Product Requirements Document) and tools.json schema files for an AI agent by analyzing its codebase.

## Usage

```
/generate_prd_tools
```

Or simply ask: "generate PRD and tools"

The skill automatically looks for `README.md` in the current working directory.

## Important Guidelines

**DO NOT hallucinate or generate information that is not present in the codebase.**

- Only include sections in the PRD where information can be directly extracted or clearly inferred from the README and code
- Omit any sections where the information is not available
- Do not make up user personas, business needs, or success criteria unless explicitly documented
- If a field cannot be determined, leave it out entirely rather than guessing

## Process

When this skill is invoked, follow these steps:

### Step 1: Read and Analyze the Agent README

Read the `README.md` file from the current directory. Extract ONLY what is explicitly stated:
- Agent name and title
- Agent purpose (if described)
- Key features and capabilities (if listed)
- Architecture overview (if documented)
- Example use cases / questions (if provided)

### Step 2: Discover the Tools Directory

From the README, identify the tools folder location (typically `app/tools/` or `src/tools/`). List all Python files in that directory.

### Step 3: Extract Tool Schemas

For each tool file found, parse the Python code to extract:
- Function name (the tool name)
- Function docstring (description)
- Function parameters with their types and descriptions
- Return type and description

Build a schema for each tool following this JSON structure:
```json
{
  "name": "tool_name",
  "description": "Tool description from docstring",
  "input-type": ["semantic_type"],
  "output-type": ["semantic_type"],
  "input": {
    "param_name": {
      "type": "string|integer|array|object|boolean",
      "description": "Parameter description"
    }
  },
  "output": {
    "type": "object|array|string",
    "description": "Output description",
    "properties": {}
  }
}
```

### Step 4: Generate tools.json

Combine all tool schemas into a single JSON file:
```json
{
  "tools": [
    { /* tool 1 schema */ },
    { /* tool 2 schema */ }
  ]
}
```

Save to `synthetic_data/tools.json`.

### Step 5: Generate PRD Document

Using ONLY the information gathered from the codebase, generate a PRD document. **Omit any sections where information is not available.**

Available sections (include only if information exists):

```markdown
# Product Requirements Document (PRD) - AI Extension

**Title:** <Agent Title from README>
**Date:** <Current Date>
**Owner:** <Only if specified in README>
**Solution Category:** AI Agent

---

## Product Purpose & Value Proposition

**Elevator Pitch:**
<Only if described in README>

**Business Need:**
<Only if described in README>

**Expected Value:**
<Only if described in README>

**Product Objectives (Prioritized):**
<Only if described in README>

---

## User Profiles & Personas
<OMIT THIS SECTION if no personas are documented>

---

## User Goals & Tasks
<OMIT THIS SECTION if not documented - do not invent user goals>

---

## Product Principles
<OMIT THIS SECTION if not documented>

---

## Business Context & AI Rationale

**Current State:**
<Only if described>

**Why AI?:**
<Only if explained in README>

**Success Criteria:**
<Only if documented>

---

## Goals and Non-Goals

### Goals (In Scope)
<Derive from available tools and README features>

### Non-Goals (Out of Scope)
<Only if explicitly documented>

---

## AI Agent Design

### Agent Role & Capabilities

**Primary Role:** <From README description>

**Core Capabilities:**
<Extract from README features section>

**Autonomous Actions:**
<Derive from available tools>

**Human-in-the-Loop Actions:**
<Only if documented>

### Process Integration
<OMIT if not documented>

---

## Requirements

### Must-Have Requirements

<Generate requirements ONLY for tools that exist in the codebase>

**REQ-001**: <Tool Name> Data Retrieval
- **Problem to Solve**: <From tool docstring>
- **User Story**: As a user, I need to <action based on tool function>
- **Acceptance Criteria**:
  - <Derive from tool parameters and return type>
- **Priority Rank**: <Based on order in code>

---

## AI System Design

### Model & Orchestration

**Model Selection:**
<Only if specified in README or config>

**Orchestration Approach:**
<Only if documented - e.g., LangGraph, sequential, etc.>

---

## AI Guardrails & Safety
<OMIT THIS SECTION if not documented>

---

## Release Criteria
<OMIT THIS SECTION if not documented>

---

## Risks, Assumptions, and Dependencies
<OMIT THIS SECTION if not documented>
```

Save to `synthetic_data/PRD.md`.

### Step 6: Report Results

After generating both files, report:
1. Path to generated tools.json
2. Path to generated PRD.md
3. Summary of tools discovered
4. List of sections omitted due to missing information

## Output Directory

Create and save generated files to:
- `synthetic_data/tools.json`
- `synthetic_data/PRD.md`

## Example

Running `/generate_prd_tools` in an agent directory produces:
- `synthetic_data/tools.json` - Tool schemas extracted from code
- `synthetic_data/PRD.md` - PRD with only documented information (missing sections omitted)
