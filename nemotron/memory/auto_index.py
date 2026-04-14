"""Auto-index a workspace into SMP on agent startup."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from nemotron.memory.smp_client import SMPClient, SMPError

INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".go", ".rs", ".c", ".cpp", ".h",
    ".rb", ".php", ".cs", ".swift", ".kt",
}

IGNORE_DIRS = {
    "__pycache__", "node_modules", ".git", ".venv", "venv",
    "dist", "build", ".next", ".tox", "target", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "env",
}

MAX_FILE_SIZE = 512 * 1024  # 512 KB


def _collect_files(workspace: Path) -> list[Path]:
    """Walk workspace and collect indexable source files."""
    files: list[Path] = []
    for item in workspace.rglob("*"):
        if any(part in IGNORE_DIRS for part in item.parts):
            continue
        if item.is_file() and item.suffix in INDEXABLE_EXTENSIONS and item.stat().st_size <= MAX_FILE_SIZE:
            files.append(item)
    return sorted(files)


async def auto_index(
    smp: SMPClient,
    workspace: Path,
    on_progress: Callable[[int, int, str], None] | None = None,
    batch_size: int = 20,
) -> int:
    """Index all source files into SMP. Returns number of files indexed."""
    files = _collect_files(workspace)
    total = len(files)
    indexed = 0

    for i in range(0, total, batch_size):
        batch = files[i : i + batch_size]
        changes: list[dict[str, str]] = []

        for f in batch:
            try:
                rel = str(f.relative_to(workspace))
                content = f.read_text(encoding="utf-8", errors="replace")
                changes.append({
                    "file_path": rel,
                    "content": content,
                    "change_type": "created",
                })
            except (OSError, UnicodeDecodeError):
                continue

        if changes:
            try:
                await smp.batch_update(changes)
            except SMPError:
                # Fall back to individual updates
                for change in changes:
                    try:
                        await smp.update_file(change["file_path"], change["content"], change["change_type"])
                    except SMPError:
                        continue

        indexed += len(changes)
        if on_progress:
            last_file = str(batch[-1].relative_to(workspace)) if batch else ""
            on_progress(indexed, total, last_file)

    return indexed
