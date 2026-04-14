"""Tool registry — collects all tools and dispatches calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemotron.memory.smp_client import SMPClient
from nemotron.tools.base import ToolResult, ToolSpec
from nemotron.tools.file_ops import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from nemotron.tools.shell import ShellTool
from nemotron.tools.smp_tools import (
    SMPContextTool,
    SMPFlowTool,
    SMPImpactTool,
    SMPLocateTool,
    SMPNavigateTool,
    SMPSearchTool,
    SMPTraceTool,
)


class ToolRegistry:
    """Holds every tool the agent can use and dispatches calls by name."""

    def __init__(self, workspace: Path, smp: SMPClient, on_file_write: Any = None) -> None:
        self._tools: dict[str, Any] = {}

        # File tools
        self._register(ReadFileTool(workspace))
        self._register(WriteFileTool(workspace, on_write=on_file_write))
        self._register(EditFileTool(workspace, on_write=on_file_write))
        self._register(ListDirTool(workspace))
        self._register(GlobTool(workspace))
        self._register(GrepTool(workspace))

        # Shell
        self._register(ShellTool(workspace))

        # SMP structural memory tools
        if smp.is_connected:
            self._register(SMPNavigateTool(smp))
            self._register(SMPTraceTool(smp))
            self._register(SMPContextTool(smp))
            self._register(SMPImpactTool(smp))
            self._register(SMPLocateTool(smp))
            self._register(SMPSearchTool(smp))
            self._register(SMPFlowTool(smp))

    def _register(self, tool: Any) -> None:
        self._tools[tool.spec.name] = tool

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        return [t.spec.to_openai_schema() for t in self._tools.values()]

    def get_anthropic_schemas(self) -> list[dict[str, Any]]:
        return [t.spec.to_anthropic_schema() for t in self._tools.values()]

    def get_gemini_tools(self) -> list[Any]:
        """Return a list of google.genai Tool objects wrapping all registered tools."""
        from google.genai import types
        declarations = [t.spec.to_gemini_declaration() for t in self._tools.values()]
        return [types.Tool(function_declarations=declarations)]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(error=f"Unknown tool: {name}")
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return ToolResult(error=f"Tool {name} failed: {e}")
