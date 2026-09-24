"""Synchronous HTTP client for the Ahmed ToolBox MCP gateway.

Authentication uses a durable refresh credential from the deployment secret
store and disposable short-lived access tokens. Authentication continuity does
not depend on server-side or client-side RAM session state.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class AhmedToolboxError(RuntimeError):
    pass


class AhmedToolboxClient:
    def __init__(
        self,
        url: str,
        *,
        token: str = "",
        refresh_token: str = "",
        access_ttl_seconds: int = 300,
        timeout_seconds: float = 5.0,
        protocol_version: str = "2024-11-05",
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token.strip()
        self.refresh_token = refresh_token.strip()
        self.access_ttl_seconds = max(1, int(access_ttl_seconds))
        self.timeout_seconds = timeout_seconds
        self.protocol_version = protocol_version
        self._lock = threading.RLock()
        self._initialized = False
        self._next_id = 1
        self._access_token = ""
        self._access_expires_at = 0.0
        self._client = self._new_client()

    def _new_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout_seconds)

    def _replace_client(self) -> None:
        with self._lock:
            try:
                self._client.close()
            finally:
                self._client = self._new_client()

    def _id(self) -> int:
        with self._lock:
            value = self._next_id
            self._next_id += 1
            return value

    def _base_url(self) -> str:
        return self.url[:-4] if self.url.endswith("/mcp") else self.url

    def _post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        try:
            return self._client.post(url, headers=headers, json=payload)
        except httpx.TransportError:
            self._replace_client()
            return self._client.post(url, headers=headers, json=payload)

    def _refresh_access_token_locked(self) -> None:
        if not self.refresh_token:
            raise AhmedToolboxError("Ahmed ToolBox refresh token is not configured")
        response = self._post(
            self._base_url() + "/auth/token",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.refresh_token}",
            },
            payload={"ttl_seconds": self.access_ttl_seconds},
        )
        if response.status_code != 200:
            raise AhmedToolboxError(
                f"Ahmed ToolBox token refresh failed: HTTP {response.status_code}"
            )
        try:
            body = response.json()
            self._access_token = str(body["access_token"])
            expires_in = max(1, int(body["expires_in"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AhmedToolboxError("Ahmed ToolBox token refresh payload is invalid") from exc
        self._access_expires_at = time.time() + expires_in

    def _authorization_token(self) -> str:
        if not self.refresh_token:
            return self.token
        with self._lock:
            if not self._access_token or self._access_expires_at - time.time() <= 5.0:
                self._refresh_access_token_locked()
            return self._access_token

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.protocol_version,
        }
        token = self._authorization_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _invalidate_access_token(self) -> None:
        with self._lock:
            self._access_token = ""
            self._access_expires_at = 0.0

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = self._id()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        try:
            response = self._post(
                self.url,
                headers=self._headers(),
                payload=payload,
            )
            if response.status_code == 401 and self.refresh_token:
                self._invalidate_access_token()
                response = self._post(
                    self.url,
                    headers=self._headers(),
                    payload=payload,
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AhmedToolboxError(f"Ahmed ToolBox request failed: {exc}") from exc

        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise AhmedToolboxError("Ahmed ToolBox response exceeded size limit")
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise AhmedToolboxError("Ahmed ToolBox returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise AhmedToolboxError("Ahmed ToolBox returned a non-object response")
        if body.get("error"):
            raise AhmedToolboxError(f"Ahmed ToolBox MCP error: {body['error']}")
        return body

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
                    "clientInfo": {"name": "PanWatch", "version": "external-tools-0.2"},
                },
            )
            self._initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        self.ensure_initialized()
        result = self._rpc("tools/list").get("result") or {}
        tools = result.get("tools") or []
        if not isinstance(tools, list):
            raise AhmedToolboxError("Ahmed ToolBox tools/list payload is malformed")
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
            raise AhmedToolboxError("Ahmed ToolBox tools/call payload is malformed")
        return result

    def close(self) -> None:
        self._client.close()
