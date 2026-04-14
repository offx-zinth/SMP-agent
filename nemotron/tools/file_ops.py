"""File operation tools — read, write, search, list, glob."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from nemotron.tools.base import ToolParam, ToolResult, ToolSpec


class ReadFileTool:
    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_file",
            description="Read the contents of a file. Returns numbered lines.",
            parameters=[
                ToolParam("path", "string", "File path relative to workspace"),
                ToolParam("offset", "integer", "Start line (1-indexed)", required=False),
                ToolParam("limit", "integer", "Number of lines to read", required=False),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = self._ws / kwargs["path"]
        if not path.exists():
            return ToolResult(error=f"File not found: {kwargs['path']}")
        if not path.is_file():
            return ToolResult(error=f"Not a file: {kwargs['path']}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult(error=str(e))

        lines = text.splitlines()
        offset = int(kwargs.get("offset", 1)) - 1
        limit = kwargs.get("limit")

        if limit is not None:
            lines = lines[offset : offset + int(limit)]
        elif offset > 0:
            lines = lines[offset:]

        start = offset + 1
        numbered = [f"{start + i:>6}|{line}" for i, line in enumerate(lines)]
        return ToolResult(output="\n".join(numbered), metadata={"total_lines": len(text.splitlines())})


class WriteFileTool:
    def __init__(self, workspace: Path, on_write: Any = None) -> None:
        self._ws = workspace
        self._on_write = on_write  # callback(rel_path, content)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write_file",
            description="Write content to a file, creating directories as needed. Overwrites existing content.",
            parameters=[
                ToolParam("path", "string", "File path relative to workspace"),
                ToolParam("content", "string", "Full file content to write"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = self._ws / kwargs["path"]
        content = kwargs["content"]

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(error=str(e))

        if self._on_write:
            await self._on_write(kwargs["path"], content)

        return ToolResult(output=f"Wrote {len(content)} bytes to {kwargs['path']}")


class EditFileTool:
    def __init__(self, workspace: Path, on_write: Any = None) -> None:
        self._ws = workspace
        self._on_write = on_write

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="edit_file",
            description=(
                "Replace an exact string in a file with new content. "
                "The old_string must match exactly (including whitespace). "
                "Use for surgical edits instead of rewriting entire files."
            ),
            parameters=[
                ToolParam("path", "string", "File path relative to workspace"),
                ToolParam("old_string", "string", "Exact text to find and replace"),
                ToolParam("new_string", "string", "Replacement text"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = self._ws / kwargs["path"]
        if not path.exists():
            return ToolResult(error=f"File not found: {kwargs['path']}")

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult(error=str(e))

        old = kwargs["old_string"]
        new = kwargs["new_string"]

        count = content.count(old)
        if count == 0:
            return ToolResult(error="old_string not found in file")
        if count > 1:
            return ToolResult(error=f"old_string found {count} times — provide more context to make it unique")

        updated = content.replace(old, new, 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as e:
            return ToolResult(error=str(e))

        if self._on_write:
            await self._on_write(kwargs["path"], updated)

        return ToolResult(output=f"Applied edit to {kwargs['path']}")


class ListDirTool:
    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_dir",
            description="List files and directories in a path.",
            parameters=[
                ToolParam("path", "string", "Directory path relative to workspace (default: '.')", required=False),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        rel = kwargs.get("path", ".")
        path = self._ws / rel
        if not path.exists():
            return ToolResult(error=f"Directory not found: {rel}")
        if not path.is_dir():
            return ToolResult(error=f"Not a directory: {rel}")

        entries: list[str] = []
        try:
            for item in sorted(path.iterdir()):
                name = item.name
                if name.startswith(".") and name not in (".env.example",):
                    continue
                suffix = "/" if item.is_dir() else ""
                entries.append(f"{name}{suffix}")
        except OSError as e:
            return ToolResult(error=str(e))

        return ToolResult(output="\n".join(entries) if entries else "(empty directory)")


class GlobTool:
    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="glob",
            description="Find files matching a glob pattern recursively.",
            parameters=[
                ToolParam("pattern", "string", "Glob pattern like '**/*.py' or 'src/**/*.ts'"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern = kwargs["pattern"]
        matches: list[str] = []
        try:
            for p in self._ws.glob(pattern):
                if p.is_file():
                    matches.append(str(p.relative_to(self._ws)))
                if len(matches) >= 200:
                    break
        except (OSError, ValueError) as e:
            return ToolResult(error=str(e))

        if not matches:
            return ToolResult(output="No files matched.")
        return ToolResult(output="\n".join(sorted(matches)), metadata={"count": len(matches)})


class GrepTool:
    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="grep",
            description="Search for a regex pattern across files. Returns matching lines with file paths and line numbers.",
            parameters=[
                ToolParam("pattern", "string", "Regex pattern to search for"),
                ToolParam("path", "string", "Directory or file to search in (default: '.')", required=False),
                ToolParam("include", "string", "Glob to filter files, e.g. '*.py'", required=False),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        pattern_str = kwargs["pattern"]
        search_path = self._ws / kwargs.get("path", ".")
        include = kwargs.get("include", "")

        try:
            regex = re.compile(pattern_str)
        except re.error as e:
            return ToolResult(error=f"Invalid regex: {e}")

        results: list[str] = []
        max_results = 100

        def _search(p: Path) -> None:
            if p.is_file():
                if include and not fnmatch.fnmatch(p.name, include):
                    return
                try:
                    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if regex.search(line):
                            rel = str(p.relative_to(self._ws))
                            results.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(results) >= max_results:
                                return
                except (OSError, UnicodeDecodeError):
                    pass
            elif p.is_dir():
                if p.name in {"__pycache__", "node_modules", ".git", ".venv", "venv"}:
                    return
                for child in sorted(p.iterdir()):
                    _search(child)
                    if len(results) >= max_results:
                        return

        _search(search_path)

        if not results:
            return ToolResult(output="No matches found.")
        output = "\n".join(results)
        if len(results) >= max_results:
            output += f"\n... (truncated at {max_results} matches)"
        return ToolResult(output=output, metadata={"count": len(results)})
