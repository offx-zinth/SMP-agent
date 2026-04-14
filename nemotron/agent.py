"""Core agent loop — the brain of Nemotron.

Implements a ReAct-style loop: the LLM reasons about the task, picks tools,
executes them, feeds results back, and repeats until the task is done.

SMP integration happens at two levels:
  1. **Proactive** — before the LLM sees a task, the ContextManager enriches
     the system prompt with structural intelligence from the codebase graph.
  2. **Reactive** — the LLM can invoke SMP tools directly (navigate, trace,
     impact, locate, etc.) during its reasoning loop.
  3. **Post-action** — after every file write, the agent pushes the change
     into SMP so the graph stays current.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from nemotron.config import AgentConfig
from nemotron.llm.provider import (
    LLMProvider,
    LLMResponse,
    TextChunk,
    ToolCallChunk,
    create_provider,
)
from nemotron.memory.context import ContextManager
from nemotron.memory.smp_client import SMPClient
from nemotron.tools.registry import ToolRegistry

SYSTEM_PROMPT = """\
You are **Nemotron**, an expert AI coding agent. You help users understand, \
navigate, modify, and build codebases.

## Capabilities
- Read, write, and edit files in the user's workspace.
- Run shell commands (git, build tools, tests, etc.).
- Search across files with grep and glob.
- **Structural Memory (SMP)**: You have access to a graph-based memory of the \
codebase. Use SMP tools to:
  - `smp_navigate` — look up any entity and its relationships.
  - `smp_trace` — follow call chains to understand control flow.
  - `smp_context` — get the full structural context of a file before editing.
  - `smp_impact` — assess what breaks if you change something.
  - `smp_locate` — find code by semantic description (graph RAG).
  - `smp_search` — full-text search the codebase graph.
  - `smp_flow` — trace data/execution flow between entities.

## Rules
1. Always read a file before editing it.
2. Before making changes to a file, use `smp_context` to understand its role \
and dependencies (when SMP is available).
3. For risky changes, use `smp_impact` first.
4. After making edits, verify correctness (read the result, run tests if applicable).
5. Be concise in your explanations. Show your work through tool calls, not \
lengthy prose.
6. When you need to find code, prefer `smp_locate` (semantic) or `smp_search` \
(exact) over grep when SMP is connected — they understand the codebase structure.
7. Write clean, production-quality code. No placeholder comments.

## Workspace
Working directory: {workspace}
SMP Status: {smp_status}
"""


def _extract_file_paths(text: str) -> list[str]:
    """Best-effort extraction of file paths mentioned in user text."""
    patterns = [
        r'`([^`]+\.[a-zA-Z]{1,10})`',
        r'(?:^|\s)(\S+\.[a-zA-Z]{1,10})(?:\s|$|[,;:])',
    ]
    paths: list[str] = []
    for p in patterns:
        for m in re.finditer(p, text):
            candidate = m.group(1)
            if "/" in candidate or "\\" in candidate or "." in candidate:
                paths.append(candidate)
    return paths


class Agent:
    """The main agent that orchestrates LLM reasoning + tool execution."""

    def __init__(
        self,
        config: AgentConfig,
        smp: SMPClient,
        on_text: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end: Callable[[str, str, bool], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._smp = smp
        self._llm: LLMProvider = create_provider(config.llm)
        self._context = ContextManager(smp, config.workspace)

        async def _on_file_write(rel_path: str, content: str) -> None:
            if self._smp.is_connected:
                try:
                    await self._smp.update_file(rel_path, content)
                except Exception:
                    pass

        self._tools = ToolRegistry(config.workspace, smp, on_file_write=_on_file_write)

        # UI callbacks
        self._on_text = on_text or (lambda t: None)
        self._on_tool_start = on_tool_start or (lambda n, a: None)
        self._on_tool_end = on_tool_end or (lambda n, o, s: None)
        self._on_status = on_status or (lambda s: None)

        # Conversation history
        self._messages: list[dict[str, Any]] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def token_usage(self) -> dict[str, int]:
        return {"input": self._total_input_tokens, "output": self._total_output_tokens}

    def _build_system(self, smp_context: str = "") -> str:
        base = SYSTEM_PROMPT.format(
            workspace=self._config.workspace,
            smp_status="Connected" if self._smp.is_connected else "Offline (file tools only)",
        )
        if smp_context:
            base += "\n\n" + smp_context
        return base

    async def run(self, user_message: str) -> str:
        """Execute one user turn — may involve multiple LLM/tool iterations."""

        # Proactive SMP context enrichment
        mentioned_files = _extract_file_paths(user_message)
        smp_context = await self._context.build_system_context(mentioned_files)

        self._messages.append({"role": "user", "content": user_message})

        system = self._build_system(smp_context)
        iterations = 0
        final_text_parts: list[str] = []

        while iterations < self._config.max_iterations:
            iterations += 1
            self._on_status(f"Thinking... (step {iterations})")

            response = await self._llm.chat(
                messages=self._messages,
                tools=self._tools.get_specs(),
                system=system,
            )

            self._total_input_tokens += response.usage.get("input", 0)
            self._total_output_tokens += response.usage.get("output", 0)

            if not response.has_tool_calls:
                text = response.text
                if text:
                    final_text_parts.append(text)
                    self._on_text(text)

                self._messages.append({"role": "assistant", "content": response.text or ""})
                break

            # Build assistant message with text + tool_use blocks
            assistant_content = self._build_assistant_content(response)
            self._messages.append({"role": "assistant", "content": assistant_content})

            # Emit any text before tool calls
            for chunk in response.chunks:
                if isinstance(chunk, TextChunk) and chunk.text:
                    final_text_parts.append(chunk.text)
                    self._on_text(chunk.text)

            # Execute tool calls and collect results
            tool_results = await self._execute_tools(response.tool_calls)

            # Add tool results to history (format varies by provider)
            if self._config.llm.provider in ("anthropic", "gemini"):
                result_blocks = []
                for tc, result in zip(response.tool_calls, tool_results):
                    result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": str(result),
                        "is_error": not result.success,
                    })
                self._messages.append({"role": "user", "content": result_blocks})
            else:
                for tc, result in zip(response.tool_calls, tool_results):
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })

        return "\n".join(final_text_parts)

    def _build_assistant_content(self, response: LLMResponse) -> Any:
        """Build provider-specific assistant content with tool calls."""
        if self._config.llm.provider in ("anthropic", "gemini"):
            blocks: list[dict[str, Any]] = []
            for chunk in response.chunks:
                if isinstance(chunk, TextChunk) and chunk.text:
                    blocks.append({"type": "text", "text": chunk.text})
                elif isinstance(chunk, ToolCallChunk):
                    blocks.append({
                        "type": "tool_use",
                        "id": chunk.id,
                        "name": chunk.name,
                        "input": chunk.arguments,
                    })
            return blocks
        else:
            # OpenAI format
            msg: dict[str, Any] = {"role": "assistant"}
            texts = [c.text for c in response.chunks if isinstance(c, TextChunk)]
            msg["content"] = " ".join(texts) if texts else None
            tool_calls = response.tool_calls
            if tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
            return msg

    async def _execute_tools(self, tool_calls: list[ToolCallChunk]) -> list[Any]:
        """Execute a batch of tool calls."""
        from nemotron.tools.base import ToolResult

        results: list[ToolResult] = []
        for tc in tool_calls:
            self._on_tool_start(tc.name, tc.arguments)
            result = await self._tools.call(tc.name, tc.arguments)
            self._on_tool_end(tc.name, str(result), result.success)
            results.append(result)
        return results

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
