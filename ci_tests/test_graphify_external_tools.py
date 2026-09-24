from __future__ import annotations

import json

import httpx
from pan_agent import ToolRegistry

from src.platform.external_tools import GraphifyClient, register_graphify_tools


def test_graphify_client_follows_streamable_http_session_lifecycle():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        method = payload["method"]
        seen.append((method, request.headers.get("mcp-session-id", "")))

        if method == "initialize":
            assert request.headers["accept"] == "application/json, text/event-stream"
            assert request.headers["authorization"] == "Bearer secret"
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "session-1"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "serverInfo": {"name": "graphify", "version": "test"},
                        "capabilities": {},
                    },
                },
            )

        assert request.headers["mcp-session-id"] == "session-1"
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "query_graph",
                                "description": "Query the graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                    },
                                },
                            }
                        ]
                    },
                },
            )
        if method == "tools/call":
            assert payload["params"]["name"] == "query_graph"
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "content": [{"type": "text", "text": "NODE A -> NODE B"}]
                    },
                },
            )
        raise AssertionError(f"unexpected MCP method: {method}")

    client = GraphifyClient(
        "http://graphify.test/mcp",
        token="secret",
        transport=httpx.MockTransport(handler),
    )

    tools = client.list_tools()
    assert [tool["name"] for tool in tools] == ["query_graph"]
    result = client.call_tool("query_graph", {"question": "what connects A to B?"})
    assert result["content"][0]["text"] == "NODE A -> NODE B"
    assert [method for method, _ in seen] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    client.close()


def test_graphify_registry_exposes_only_allowlisted_read_tools():
    class FakeGraphify:
        def list_tools(self):
            schema = {"type": "object", "properties": {}}
            return [
                {
                    "name": "query_graph",
                    "description": "Query project relationships",
                    "inputSchema": schema,
                },
                {
                    "name": "shortest_path",
                    "description": "Find a graph path",
                    "inputSchema": schema,
                },
                {
                    "name": "delete_everything",
                    "description": "Must never be exposed",
                    "inputSchema": schema,
                },
            ]

        def call_tool(self, name, arguments):
            return {
                "content": [{"type": "text", "text": f"{name}: ok"}]
            }

    registry = ToolRegistry()
    descriptors = register_graphify_tools(registry, FakeGraphify())
    names = {tool.name for tool in registry.registered_tools()}

    assert "kg_query_graph" in names
    assert "kg_shortest_path" in names
    assert "kg_delete_everything" not in names
    assert {item.tool_name for item in descriptors} == {
        "kg_query_graph",
        "kg_shortest_path",
    }
