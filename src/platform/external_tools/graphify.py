"""Streamable HTTP MCP client for Graphify knowledge graphs.

Graphify's HTTP transport follows the MCP 2025-03-26 lifecycle: initialize,
capture the returned session id, send notifications/initialized, then issue
tools/list and tools/call requests. PanWatch deliberately uses the JSON response
mode so external knowledge lookup stays a small, deterministic read-only surface.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import httpx

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class GraphifyError(RuntimeError):
    """Raised when the Graphify MCP service cannot satisfy a request."""


class GraphifyClient:
    def __init__(
        self,
        url: str,
        *,
        token: str = "",
        timeout_seconds: float = 5.0,
        protocol_version: str = "2025-03-26",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.protocol_version = protocol_version
        self._lock = threading.RLock()
        self._initialized = False
        self._session_id = ""
        self._next_id = 1
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def _id(self) -> int:
        with self._lock:
            value = self._next_id
            self._next_id += 1
            return value

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _post(
        self,
        payload: dict[str, Any],
        *,
        expect_json: bool = True,
    ) -> dict[str, Any]:
        try:
            response = self._client.post(self.url, headers=self._headers(), json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GraphifyError(f"Graphify MCP request failed: {exc}") from exc

        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

        if not expect_json:
            return {}

        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise GraphifyError("Graphify MCP response exceeded size limit")

        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            raise GraphifyError(
                "Graphify MCP returned SSE; start it with --json-response"
            )
        if not response.content:
            return {}

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise GraphifyError("Graphify MCP returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise GraphifyError("Graphify MCP returned a non-object response")
        if body.get("error"):
            raise GraphifyError(f"Graphify MCP error: {body['error']}")
        return body

    def _rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            {
                "jsonrpc": "2.0",
                "id": self._id(),
                "method": method,
                "params": params or {},
            }
        )

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._rpc(
                "initialize",
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "PanWatch",
                        "version": "graphify-knowledge-0.1",
                    },
                },
            )
            self._post(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                },
                expect_json=False,
            )
            self._initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        self.ensure_initialized()
        result = self._rpc("tools/list").get("result") or {}
        tools = result.get("tools") or []
        if not isinstance(tools, list):
            raise GraphifyError("Graphify MCP tools/list payload is malformed")
        return [item for item in tools if isinstance(item, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.ensure_initialized()
        result = (
            self._rpc(
                "tools/call",
                {"name": name, "arguments": arguments},
            ).get("result")
            or {}
        )
        if not isinstance(result, dict):
            raise GraphifyError("Graphify MCP tools/call payload is malformed")
        return result

    def close(self) -> None:
        self._client.close()
