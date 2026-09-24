from __future__ import annotations

import json

import httpx

from src.platform.external_tools import ahmed_toolbox as toolbox_module


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode()

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://toolbox.test/mcp")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                "test response",
                request=request,
                response=response,
            )


class _Backend:
    def __init__(self):
        self.refreshes = 0
        self.fail_transport_once = False

    def post(self, url, headers, payload):
        if url.endswith("/auth/token"):
            self.refreshes += 1
            return _Response(
                200,
                {"access_token": f"short-{self.refreshes}", "expires_in": 10},
            )

        if self.fail_transport_once:
            self.fail_transport_once = False
            raise httpx.ConnectError(
                "simulated restart",
                request=httpx.Request("POST", url),
            )

        expected = f"Bearer short-{self.refreshes}"
        if headers.get("Authorization") != expected:
            return _Response(401, {"error": "unauthorized"})

        method = payload["method"]
        if method == "initialize":
            result = {}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "runtime_status",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        else:
            result = {
                "content": [
                    {"type": "text", "text": json.dumps({"status": "ok"})}
                ],
                "isError": False,
            }
        return _Response(
            200,
            {"jsonrpc": "2.0", "id": payload["id"], "result": result},
        )


class _FakeClient:
    def __init__(self, backend: _Backend):
        self.backend = backend

    def post(self, url, *, headers, json):
        return self.backend.post(url, headers, json)

    def close(self):
        return None


def test_toolbox_client_75_calls_refresh_and_restart_recovery(monkeypatch):
    backend = _Backend()
    clock = [1000.0]
    monkeypatch.setattr(toolbox_module.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        toolbox_module.httpx,
        "Client",
        lambda **_kwargs: _FakeClient(backend),
    )

    client = toolbox_module.AhmedToolboxClient(
        "https://toolbox.test/mcp",
        token="compat-token",
        refresh_token="refresh-credential",
        access_ttl_seconds=10,
    )

    assert client.list_tools()[0]["name"] == "runtime_status"

    for _ in range(40):
        result = client.call_tool("runtime_status", {})
        assert json.loads(result["content"][0]["text"])["status"] == "ok"

    clock[0] += 11

    for _ in range(35):
        result = client.call_tool("runtime_status", {})
        assert json.loads(result["content"][0]["text"])["status"] == "ok"

    assert backend.refreshes >= 2

    backend.fail_transport_once = True
    result = client.call_tool("runtime_status", {})
    assert json.loads(result["content"][0]["text"])["status"] == "ok"
