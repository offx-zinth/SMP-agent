"""SMP-specific tools that give the agent access to structural memory.

These tools expose SMP's graph queries to the LLM tool-calling loop,
so the model can proactively explore the codebase graph.
"""

from __future__ import annotations

from typing import Any
import json

from nemotron.memory.smp_client import SMPClient, SMPError
from nemotron.tools.base import ToolParam, ToolResult, ToolSpec


class SMPNavigateTool:
    """Find an entity in the codebase graph and its relationships."""

    def __init__(self, smp: SMPClient) -> None:
        self._smp = smp

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="smp_navigate",
            description=(
                "Find a code entity (function, class, file) in the structural memory graph "
                "and get its relationships (calls, imports, defines, etc.)."
            ),
            parameters=[
                ToolParam("query", "string", "Entity identifier, e.g. 'src/auth.py::Function::login::10' or function name"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._smp.navigate(kwargs["query"])
            return ToolResult(output=json.dumps(result, indent=2))
        except SMPError as e:
            return ToolResult(error=str(e))


class SMPTraceTool:
    """Trace call graph from an entity."""

    def __init__(self, smp: SMPClient) -> None:
        self._smp = smp

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="smp_trace",
            description="Trace the call graph from an entity to see what it calls or what calls it.",
            parameters=[
                ToolParam("start", "string", "Entity ID to start tracing from"),
                ToolParam("depth", "integer", "How many levels deep to trace (default: 3)", required=False),
                ToolParam("direction", "string", "Trace direction: 'outgoing' or 'incoming'", required=False,
                          enum=["outgoing", "incoming"]),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._smp.trace(
                kwargs["start"],
                depth=int(kwargs.get("depth", 3)),
                direction=kwargs.get("direction", "outgoing"),
            )
            return ToolResult(output=json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result))
        except SMPError as e:
            return ToolResult(error=str(e))


class SMPContextTool:
    """Get rich structural context for a file — the programmer's mental model."""

    def __init__(self, smp: SMPClient) -> None:
        self._smp = smp

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="smp_context",
            description=(
                "Get structural context for a file: what it defines, imports, "
                "who depends on it, related patterns, and risk assessment. "
                "Use this before editing a file to understand its role."
            ),
            parameters=[
                ToolParam("file_path", "string", "File path relative to workspace"),
                ToolParam("scope", "string", "Context scope", required=False,
                          enum=["edit", "create", "debug", "review"]),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._smp.get_context(
                kwargs["file_path"],
                scope=kwargs.get("scope", "edit"),
            )
            return ToolResult(output=json.dumps(result, indent=2))
        except SMPError as e:
            return ToolResult(error=str(e))


class SMPImpactTool:
    """Assess the impact of changing a code entity."""

    def __init__(self, smp: SMPClient) -> None:
        self._smp = smp

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="smp_impact",
            description=(
                "Assess the blast radius of changing or deleting a code entity. "
                "Returns affected files, severity, and recommendations."
            ),
            parameters=[
                ToolParam("entity", "string", "Entity ID to assess impact for"),
                ToolParam("change_type", "string", "Type of change", required=False,
                          enum=["signature_change", "delete", "move"]),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._smp.assess_impact(
                kwargs["entity"],
                change_type=kwargs.get("change_type", "signature_change"),
            )
            return ToolResult(output=json.dumps(result, indent=2))
        except SMPError as e:
            return ToolResult(error=str(e))


class SMPLocateTool:
    """Semantic code search — find code by description using graph RAG."""

    def __init__(self, smp: SMPClient) -> None:
        self._smp = smp

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="smp_locate",
            description=(
                "Find code by semantic description. Uses SMP's SeedWalkEngine "
                "for community-routed graph RAG. Example: 'authentication logic' or "
                "'database connection pooling'."
            ),
            parameters=[
                ToolParam("description", "string", "Natural language description of the code to find"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._smp.locate(kwargs["description"])
            return ToolResult(output=json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result))
        except SMPError as e:
            return ToolResult(error=str(e))


class SMPSearchTool:
    """Full-text BM25 search across the codebase graph."""

    def __init__(self, smp: SMPClient) -> None:
        self._smp = smp

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="smp_search",
            description="Full-text search (BM25) across the codebase graph nodes for exact term matching.",
            parameters=[
                ToolParam("query", "string", "Search query"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._smp.search(kwargs["query"])
            return ToolResult(output=json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result))
        except SMPError as e:
            return ToolResult(error=str(e))


class SMPFlowTool:
    """Trace data or execution flow between two points in the codebase."""

    def __init__(self, smp: SMPClient) -> None:
        self._smp = smp

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="smp_flow",
            description="Trace data/execution flow between two code entities to understand how data moves.",
            parameters=[
                ToolParam("start", "string", "Starting entity ID"),
                ToolParam("end", "string", "Ending entity ID", required=False),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._smp.flow(kwargs["start"], kwargs.get("end"))
            return ToolResult(output=json.dumps(result, indent=2))
        except SMPError as e:
            return ToolResult(error=str(e))
