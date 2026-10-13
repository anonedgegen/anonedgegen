# ToolSandbox harness

This directory combines `src/tool_sandbox/` with the supplied base runtime:
`fastapi_server/`, `scenarios/`, dependency files, tests, and `sdg-data/`.
The source implementation depends on the checked-out Git branch.

See the [repository setup instructions](../README.md) and the
[original ToolSandbox documentation](README_TOOLSANDBOX.md).

Start the API from this directory after installing its dependencies:

```bash
python -m uvicorn fastapi_server.main:app --host 127.0.0.1 --port 5501
```

`data/` contains generated session trajectories and is created during execution.
Existing session outputs are not required runtime inputs and are not imported.
