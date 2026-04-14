"""Context manager that enriches the agent's prompt with SMP intelligence.

Before the LLM reasons about a task, this module gathers structural context
from the SMP server so the model understands the codebase topology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemotron.memory.smp_client import SMPClient, SMPError


class ContextManager:
    """Gathers and formats SMP context for the LLM."""

    def __init__(self, smp: SMPClient, workspace: Path) -> None:
        self._smp = smp
        self._workspace = workspace
        self._cache: dict[str, Any] = {}

    def _rel(self, path: str) -> str:
        """Convert absolute path to workspace-relative."""
        try:
            return str(Path(path).relative_to(self._workspace))
        except ValueError:
            return path

    async def gather_edit_context(self, file_path: str) -> str:
        """Get rich structural context for editing a file."""
        if not self._smp.is_connected:
            return ""

        rel = self._rel(file_path)
        parts: list[str] = []

        try:
            ctx = await self._smp.get_context(rel, scope="edit", depth=2)
            self._cache[rel] = ctx

            if self_info := ctx.get("self"):
                parts.append(f"## File: {rel}")
                if role := (ctx.get("summary", {}) or {}).get("role"):
                    parts.append(f"Role: {role}")
                if blast := (ctx.get("summary", {}) or {}).get("blast_radius"):
                    parts.append(f"Blast radius: {blast} nodes")

            if funcs := ctx.get("functions_defined"):
                names = [f.get("name", "?") for f in funcs[:15]]
                parts.append(f"Functions: {', '.join(names)}")

            if imports := ctx.get("imports"):
                files = [i.get("file", i.get("name", "?")) for i in imports[:10]]
                parts.append(f"Imports from: {', '.join(files)}")

            if imported_by := ctx.get("imported_by"):
                files = [i.get("file", i.get("name", "?")) for i in imported_by[:10]]
                parts.append(f"Imported by: {', '.join(files)}")

            if warnings := ctx.get("warnings"):
                for w in warnings[:5]:
                    parts.append(f"⚠ {w}")

        except SMPError:
            pass

        return "\n".join(parts)

    async def gather_impact(self, entity_id: str) -> str:
        """Assess the impact of changing an entity."""
        if not self._smp.is_connected:
            return ""

        try:
            impact = await self._smp.assess_impact(entity_id)
            parts = [f"## Impact Analysis for {entity_id}"]

            if severity := impact.get("severity"):
                parts.append(f"Severity: {severity}")
            if affected := impact.get("affected_files"):
                parts.append(f"Affected files ({len(affected)}):")
                for f in affected[:10]:
                    parts.append(f"  - {f}")
            if recs := impact.get("recommendations"):
                parts.append("Recommendations:")
                for r in recs[:5]:
                    parts.append(f"  - {r}")

            return "\n".join(parts)
        except SMPError:
            return ""

    async def locate_code(self, description: str) -> str:
        """Find code by semantic description."""
        if not self._smp.is_connected:
            return ""

        try:
            results = await self._smp.locate(description, top_k=5)
            if isinstance(results, list):
                matches = results
            elif isinstance(results, dict):
                matches = results.get("matches", [])
            else:
                return ""

            parts = [f"## Code matching: '{description}'"]
            for m in matches[:5]:
                entity = m.get("entity", m.get("name", "?"))
                file = m.get("file", m.get("file_path", "?"))
                purpose = m.get("purpose", "")
                relevance = m.get("relevance", "")
                line = f"- **{entity}** in `{file}`"
                if relevance:
                    line += f" (relevance: {relevance:.2f})"
                if purpose:
                    line += f"\n  {purpose}"
                parts.append(line)

            return "\n".join(parts)
        except SMPError:
            return ""

    async def trace_calls(self, entity_id: str, depth: int = 3) -> str:
        """Trace the call graph from an entity."""
        if not self._smp.is_connected:
            return ""

        try:
            result = await self._smp.trace(entity_id, depth=depth)
            if isinstance(result, list):
                names = [n.get("name", "?") for n in result]
                return f"Call chain from {entity_id}: {' → '.join(names)}"
            elif isinstance(result, dict) and "tree" in result:
                return f"Call tree:\n{json.dumps(result['tree'], indent=2)}"
            return ""
        except SMPError:
            return ""

    async def build_system_context(self, mentioned_files: list[str] | None = None) -> str:
        """Build a combined SMP context block for the system prompt."""
        if not self._smp.is_connected:
            return ""

        sections: list[str] = []
        for fp in (mentioned_files or []):
            ctx = await self.gather_edit_context(fp)
            if ctx:
                sections.append(ctx)

        if not sections:
            return ""

        return (
            "<smp_context>\n"
            "The following structural intelligence was gathered from the codebase graph:\n\n"
            + "\n\n".join(sections)
            + "\n</smp_context>"
        )
