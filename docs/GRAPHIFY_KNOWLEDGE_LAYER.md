# Graphify Knowledge Layer

Graphify is integrated as an optional **read-only structural knowledge layer**.
It is not a trading signal engine, not a source of market facts, and does not
replace Ahmed Research Engine or PanWatch belief state.

## Responsibility boundary

```text
Ahmed Intelligence
├─ Ahmed Research Engine  -> evidence, provenance, verification
├─ Graphify               -> code/docs structure and relationships
├─ PanWatch               -> persistent state, monitoring, reasoning
└─ Gen1                   -> gold decision/execution-facing analysis
```

PanWatch exposes Graphify to the Assistant through deferred Tool Research. The
model sees Graphify tools only when a question needs codebase or architecture
knowledge.

The registered surface is a strict read-only allowlist:

- `query_graph`
- `get_node`
- `get_neighbors`
- `get_community`
- `god_nodes`
- `graph_stats`
- `shortest_path`
- `list_prs`
- `get_pr_impact`
- `triage_prs`

Unknown future Graphify tools are ignored.

## Isolation

Do **not** install Graphify into PanWatch's main virtualenv. Graphify's MCP HTTP
stack has its own MCP/Starlette dependency range. The repository therefore
creates `.graphify-venv`; PanWatch talks to it only over MCP HTTP.

## Setup

```bash
make knowledge-install
make knowledge-build
make knowledge-serve
```

The default build uses `--code-only`: AST extraction is local and does not need
an LLM key.

For semantic extraction of docs/media:

```bash
python scripts/graphify_knowledge.py build --semantic --backend openai --force
```

Then configure PanWatch:

```dotenv
GRAPHIFY_MCP_URL=http://127.0.0.1:8766/mcp
GRAPHIFY_MCP_TOKEN=
GRAPHIFY_MCP_TIMEOUT_SECONDS=5
```

If Graphify is bound beyond loopback, set `GRAPHIFY_API_KEY` on the Graphify
service and copy that value into `GRAPHIFY_MCP_TOKEN`.

## Updating

```bash
make knowledge-update
```

Optional post-commit synchronization:

```bash
make knowledge-hook
```

Generated `graphify-out/` is runtime/index state and is ignored by Git.

Inside PanWatch, Graphify tools use the `kg_` namespace, for example
`kg_query_graph` and `kg_shortest_path`. If Graphify is unavailable, the
Assistant fails soft and continues with local tools and Ahmed ToolBox.
