"""Register Ahmed ToolBox MCP tools in PanAgent as deferred read tools."""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from pan_agent import (
    RunRequest,
    ToolExposure,
    ToolRegistry,
    ToolResult,
    ToolRisk,
    ToolSpec,
)
from pan_agent_tool_research import (
    ToolDataFreshness,
    ToolDescriptor,
)

from .ahmed_toolbox import AhmedToolboxClient, AhmedToolboxError

_NAME_RE = re.compile(r"[^a-z0-9_]+")
_TEXT_LIMIT = 18_000


def _pan_tool_name(remote_name: str, namespace: str = "ext") -> str:
    slug = _NAME_RE.sub("_", remote_name.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug or not slug[0].isalpha():
        slug = f"tool_{slug}"
    candidate = f"{namespace}_{slug}"
    if len(candidate) <= 64:
        return candidate
    digest = hashlib.sha1(remote_name.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[:55]}_{digest}"


def _tool_title(remote_name: str) -> str:
    return remote_name.replace("__", " / ").replace("_", " ").strip()[:120] or "External tool"


def _descriptor_metadata(remote_name: str) -> tuple[str, list[str], list[str]]:
    if remote_name == "reach_doctor":
        return (
            "external_tools",
            ["tool health", "source availability", "doctor"],
            ["tool_health", "source_health"],
        )
    if remote_name == "reach_web_search":
        return (
            "web_research",
            ["search web", "discover sources", "latest news", "research topic"],
            ["web_search", "source_discovery", "research"],
        )
    if remote_name == "reach_read_url":
        return (
            "web_research",
            ["read webpage", "source page", "article", "URL"],
            ["web_read", "research"],
        )
    if remote_name.startswith("scrapling__"):
        return (
            "web_research",
            ["scrape", "dynamic webpage", "anti-bot", "browser fetch"],
            ["web_fetch", "web_read", "research"],
        )
    return (
        "external_research",
        ["external research", "source lookup"],
        ["external_read", "research"],
    )


def _extract_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in result.get("content") or []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = str(block.get("text") or "").strip()
        if text:
            parts.append(text)
    if not parts:
        return "External tool completed without text output."
    text = "\n\n".join(parts)
    return text[:_TEXT_LIMIT]


def register_ahmed_toolbox_tools(
    registry: ToolRegistry,
    client: AhmedToolboxClient,
    *,
    namespace: str = "ext",
) -> list[ToolDescriptor]:
    """Discover gateway tools once, register deferred executors, return descriptors.

    The Ahmed ToolBox gateway is the safety boundary for remote tools: only
    allowlisted read-only remote tools are returned by its tools/list endpoint.
    PanWatch therefore registers the discovered surface as READ + DEFERRED.
    """

    descriptors: list[ToolDescriptor] = []
    seen: set[str] = set()

    for remote in client.list_tools():
        remote_name = str(remote.get("name") or "").strip()
        if not remote_name:
            continue
        local_name = _pan_tool_name(remote_name, namespace)
        if local_name in seen:
            raise ValueError(f"duplicate external PanAgent tool name: {local_name}")
        seen.add(local_name)

        description = str(remote.get("description") or f"External tool {remote_name}").strip()
        description = description[:2_000] or f"External tool {remote_name}"
        input_schema = remote.get("inputSchema")
        if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
            input_schema = {"type": "object", "properties": {}}

        async def _execute(
            _request: RunRequest,
            arguments: dict[str, Any],
            *,
            _remote_name: str = remote_name,
        ) -> ToolResult:
            try:
                result = await asyncio.to_thread(
                    client.call_tool,
                    _remote_name,
                    arguments,
                )
            except AhmedToolboxError as exc:
                return ToolResult.failure(
                    summary=f"External tool unavailable: {exc}",
                    error_code="external_tool_unavailable",
                )

            text = _extract_text(result)
            if result.get("isError"):
                return ToolResult.failure(
                    summary=text,
                    error_code="external_tool_failed",
                )
            return ToolResult.success(
                summary=text,
                data={
                    "remote_tool": _remote_name,
                    "result": result,
                },
                sources=[{"name": f"Ahmed ToolBox / {_remote_name}"}],
                observed_at=datetime.now(timezone.utc),
            )

        title = _tool_title(remote_name)
        registry.register(
            ToolSpec(
                name=local_name,
                title=title,
                description=description,
                risk=ToolRisk.READ,
                confirmation_required=False,
                exposure=ToolExposure.DEFERRED,
                input_schema=input_schema,
            ),
            _execute,
        )

        domain, keywords, capabilities = _descriptor_metadata(remote_name)
        descriptors.append(
            ToolDescriptor(
                tool_name=local_name,
                title=title,
                summary=description[:500],
                use_cases=keywords,
                keywords=keywords,
                aliases=[remote_name],
                domain=domain,
                capabilities=capabilities,
                data_freshness=ToolDataFreshness.NEAR_REAL_TIME,
                estimated_latency_ms=3_000,
                output_summary="External read-only research result.",
                risk=ToolRisk.READ,
                confirmation_required=False,
                implementation_version="ahmed-toolbox-0.1",
            )
        )

    return descriptors


_GRAPHIFY_ALLOWED_TOOLS = frozenset(
    {
        "query_graph",
        "get_node",
        "get_neighbors",
        "get_community",
        "god_nodes",
        "graph_stats",
        "shortest_path",
        "list_prs",
        "get_pr_impact",
        "triage_prs",
    }
)


def _graphify_descriptor_metadata(
    remote_name: str,
) -> tuple[list[str], list[str]]:
    mapping = {
        "query_graph": (
            ["codebase question", "architecture lookup", "project relationships"],
            ["knowledge_query", "architecture", "code_navigation"],
        ),
        "get_node": (
            ["explain concept", "find symbol", "inspect component"],
            ["knowledge_lookup", "code_navigation"],
        ),
        "get_neighbors": (
            ["dependencies", "callers", "related components"],
            ["dependency_graph", "code_navigation"],
        ),
        "get_community": (
            ["subsystem", "module cluster", "architecture community"],
            ["architecture", "community_detection"],
        ),
        "god_nodes": (
            ["central concepts", "architecture hubs"],
            ["architecture", "centrality"],
        ),
        "graph_stats": (
            ["knowledge graph health", "graph size"],
            ["knowledge_graph", "diagnostics"],
        ),
        "shortest_path": (
            ["how components connect", "dependency path", "relationship path"],
            ["dependency_graph", "architecture"],
        ),
        "list_prs": (
            ["pull requests", "active code changes"],
            ["pull_requests", "code_change"],
        ),
        "get_pr_impact": (
            ["pull request impact", "affected components"],
            ["pull_requests", "change_impact"],
        ),
        "triage_prs": (
            ["pull request triage", "change conflicts"],
            ["pull_requests", "change_impact"],
        ),
    }
    return mapping.get(remote_name, (["project knowledge"], ["knowledge_graph"]))


def register_graphify_tools(
    registry: ToolRegistry,
    client: GraphifyClient,
    *,
    namespace: str = "kg",
) -> list[ToolDescriptor]:
    """Register a strict read-only allowlist from a Graphify MCP server."""

    descriptors: list[ToolDescriptor] = []
    seen: set[str] = set()

    for remote in client.list_tools():
        remote_name = str(remote.get("name") or "").strip()
        if remote_name not in _GRAPHIFY_ALLOWED_TOOLS:
            continue

        local_name = _pan_tool_name(remote_name, namespace)
        if local_name in seen:
            raise ValueError(f"duplicate Graphify PanAgent tool name: {local_name}")
        seen.add(local_name)

        description = str(
            remote.get("description") or f"Graphify knowledge tool {remote_name}"
        ).strip()
        description = description[:2_000] or f"Graphify knowledge tool {remote_name}"
        input_schema = remote.get("inputSchema")
        if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
            input_schema = {"type": "object", "properties": {}}

        async def _execute(
            _request: RunRequest,
            arguments: dict[str, Any],
            *,
            _remote_name: str = remote_name,
        ) -> ToolResult:
            try:
                result = await asyncio.to_thread(
                    client.call_tool,
                    _remote_name,
                    arguments,
                )
            except GraphifyError as exc:
                return ToolResult.failure(
                    summary=f"Graphify knowledge lookup unavailable: {exc}",
                    error_code="graphify_unavailable",
                )

            text = _extract_text(result)
            if result.get("isError"):
                return ToolResult.failure(
                    summary=text,
                    error_code="graphify_tool_failed",
                )
            return ToolResult.success(
                summary=text,
                data={
                    "remote_tool": _remote_name,
                    "result": result,
                },
                sources=[{"name": f"Graphify / {_remote_name}"}],
                observed_at=datetime.now(timezone.utc),
            )

        title = f"Knowledge / {_tool_title(remote_name)}"
        registry.register(
            ToolSpec(
                name=local_name,
                title=title,
                description=description,
                risk=ToolRisk.READ,
                confirmation_required=False,
                exposure=ToolExposure.DEFERRED,
                input_schema=input_schema,
            ),
            _execute,
        )

        keywords, capabilities = _graphify_descriptor_metadata(remote_name)
        descriptors.append(
            ToolDescriptor(
                tool_name=local_name,
                title=title,
                summary=description[:500],
                use_cases=keywords,
                keywords=keywords,
                aliases=[remote_name, f"graphify {remote_name}"],
                domain="knowledge_graph",
                capabilities=capabilities,
                data_freshness=ToolDataFreshness.NEAR_REAL_TIME,
                estimated_latency_ms=750,
                output_summary="Scoped read-only project knowledge graph result.",
                risk=ToolRisk.READ,
                confirmation_required=False,
                implementation_version="graphify-mcp-0.1",
            )
        )

    return descriptors
