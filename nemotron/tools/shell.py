"""Shell command execution tool."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from nemotron.tools.base import ToolParam, ToolResult, ToolSpec


class ShellTool:
    def __init__(self, workspace: Path) -> None:
        self._ws = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell",
            description=(
                "Execute a shell command in the workspace directory. "
                "Use for git, build tools, test runners, package managers, etc. "
                "Commands time out after 120 seconds."
            ),
            parameters=[
                ToolParam("command", "string", "The shell command to run"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs["command"]
        timeout = 120

        # Block dangerous commands
        dangerous = ["rm -rf /", "mkfs", ":(){ :|:& };:", "dd if=/dev"]
        for d in dangerous:
            if d in command:
                return ToolResult(error=f"Blocked dangerous command: {command}")

        try:
            shell = "powershell" if os.name == "nt" else "/bin/bash"
            flag = "-Command" if os.name == "nt" else "-c"

            proc = await asyncio.create_subprocess_exec(
                shell, flag, command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._ws),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(error=f"Command timed out after {timeout}s: {command}")
        except OSError as e:
            return ToolResult(error=f"Failed to execute: {e}")

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()

        # Truncate very long output
        max_len = 30_000
        if len(out) > max_len:
            out = out[:max_len] + f"\n... (truncated, {len(stdout)} bytes total)"

        if proc.returncode != 0:
            combined = f"Exit code: {proc.returncode}"
            if out:
                combined += f"\n\nSTDOUT:\n{out}"
            if err:
                combined += f"\n\nSTDERR:\n{err}"
            return ToolResult(error=combined)

        result = out
        if err:
            result += f"\n\nSTDERR:\n{err}"
        return ToolResult(output=result, metadata={"exit_code": proc.returncode})
