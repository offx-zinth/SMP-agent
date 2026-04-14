"""Nemotron CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nemotron",
        description="Nemotron — AI coding agent powered by Structural Memory Protocol",
    )
    parser.add_argument(
        "-w", "--workspace",
        type=Path,
        default=None,
        help="Workspace directory (defaults to current directory)",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip auto-indexing on startup",
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["anthropic", "openai", "gemini"],
        default=None,
        help="LLM provider override",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model override",
    )
    parser.add_argument(
        "--smp-url",
        default=None,
        help="SMP server URL (default: http://localhost:8420)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="nemotron 0.1.0",
    )

    args = parser.parse_args()

    # Apply CLI overrides via env vars before config loads
    import os
    if args.provider:
        os.environ["NEMOTRON_LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["NEMOTRON_MODEL"] = args.model
    if args.smp_url:
        os.environ["SMP_URL"] = args.smp_url
    if args.no_index:
        os.environ["NEMOTRON_AUTO_INDEX"] = "false"
    if args.verbose:
        os.environ["NEMOTRON_VERBOSE"] = "true"

    workspace = args.workspace or Path.cwd()
    if not workspace.is_dir():
        print(f"Error: workspace '{workspace}' is not a directory", file=sys.stderr)
        sys.exit(1)

    from nemotron.ui.terminal import TerminalUI
    ui = TerminalUI(workspace=workspace)
    asyncio.run(ui.run())


if __name__ == "__main__":
    main()
