"""Rich terminal UI for the Nemotron coding agent.

Provides a modern, interactive REPL with:
  - Colored output with markdown rendering
  - Tool execution progress indicators
  - Token usage tracking
  - SMP connection status
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from nemotron.agent import Agent
from nemotron.config import AgentConfig, load_config
from nemotron.llm.provider import create_provider
from nemotron.memory.auto_index import auto_index
from nemotron.memory.smp_client import SMPClient

THEME = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "tool.name": "bold magenta",
    "tool.arg": "dim",
    "agent": "bold cyan",
    "user": "bold green",
    "status": "dim white",
})

PROMPT_STYLE = Style.from_dict({
    "prompt": "#00d7ff bold",
    "path": "#888888",
})

WELCOME = """\
[bold cyan]
 ███╗   ██╗███████╗███╗   ███╗ ██████╗ ████████╗██████╗  ██████╗ ███╗   ██╗
 ████╗  ██║██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║
 ██╔██╗ ██║█████╗  ██╔████╔██║██║   ██║   ██║   ██████╔╝██║   ██║██╔██╗ ██║
 ██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██╔══██╗██║   ██║██║╚██╗██║
 ██║ ╚████║███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║  ██║╚██████╔╝██║ ╚████║
 ╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═══╝
[/bold cyan]
[dim]AI Coding Agent powered by Structural Memory Protocol (SMP)[/dim]
"""

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/help[/cyan]       Show this help
  [cyan]/clear[/cyan]      Clear conversation history
  [cyan]/status[/cyan]     Show SMP and token status
  [cyan]/index[/cyan]      Re-index the workspace into SMP
  [cyan]/compact[/cyan]    Summarize and compact conversation history
  [cyan]/quit[/cyan]       Exit Nemotron

[bold]Tips:[/bold]
  - Nemotron automatically queries the SMP graph before editing files.
  - Mention file paths to get proactive structural context.
  - Use multi-line input by ending a line with \\\\ .
"""


class TerminalUI:
    """Interactive REPL for the Nemotron agent."""

    def __init__(self, workspace: Path | None = None) -> None:
        self._console = Console(theme=THEME)
        self._workspace = workspace or Path.cwd()
        self._config: AgentConfig | None = None
        self._smp: SMPClient | None = None
        self._agent: Agent | None = None
        self._running = False
        self._current_spinner: str = ""
        self._tool_count = 0
        self._start_time = 0.0

    def _status(self, msg: str) -> None:
        self._current_spinner = msg

    def _on_text(self, text: str) -> None:
        self._console.print(Markdown(text))

    def _on_tool_start(self, name: str, args: dict[str, Any]) -> None:
        self._tool_count += 1
        arg_summary = _summarize_args(name, args)
        self._console.print(
            Text.assemble(
                ("  ", ""),
                (f"  {name}", "tool.name"),
                (f" {arg_summary}", "tool.arg"),
            )
        )

    def _on_tool_end(self, name: str, output: str, success: bool) -> None:
        style = "success" if success else "error"
        icon = "+" if success else "x"
        # Show truncated output for non-success or verbose
        if not success:
            lines = output.strip().splitlines()
            preview = lines[0][:120] if lines else ""
            self._console.print(f"    [{style}]{icon} {preview}[/{style}]")

    async def _initialize(self) -> None:
        """Load config, connect SMP, build agent."""
        self._config = load_config(self._workspace)

        # Connect to SMP
        self._console.print("[info]Connecting to SMP server...[/info]")
        self._smp = SMPClient(
            base_url=self._config.smp.url,
            timeout=self._config.smp.timeout,
        )
        connected = await self._smp.connect()

        if connected:
            self._console.print(f"[success]SMP connected[/success] at {self._config.smp.url}")
        else:
            self._console.print(
                f"[warning]SMP server not available at {self._config.smp.url} — "
                f"running in file-only mode. Start SMP with 'docker compose up -d' for full intelligence.[/warning]"
            )

        # Auto-index if SMP connected and enabled
        if connected and self._config.smp.auto_index:
            self._console.print("[info]Indexing workspace into SMP...[/info]")
            count = await auto_index(
                self._smp,
                self._workspace,
                on_progress=lambda done, total, f: self._console.print(
                    f"  [dim]{done}/{total} files indexed... {f}[/dim]",
                    highlight=False,
                ),
            )
            self._console.print(f"[success]Indexed {count} files into structural memory[/success]")

        # Create agent
        if not self._config.llm.api_key:
            key_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
            key_name = key_map.get(self._config.llm.provider, "API_KEY")
            self._console.print(
                f"[error]No API key found. Set {key_name} in .env or environment.[/error]"
            )
            sys.exit(1)

        self._agent = Agent(
            config=self._config,
            smp=self._smp,
            on_text=self._on_text,
            on_tool_start=self._on_tool_start,
            on_tool_end=self._on_tool_end,
            on_status=self._status,
        )

    async def _handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = command.strip().lower()

        if cmd == "/help":
            self._console.print(HELP_TEXT)
            return True

        if cmd == "/clear":
            if self._agent:
                self._agent.reset()
            self._console.print("[info]Conversation cleared.[/info]")
            return True

        if cmd == "/status":
            self._print_status()
            return True

        if cmd == "/index":
            if self._smp and self._smp.is_connected:
                self._console.print("[info]Re-indexing workspace...[/info]")
                count = await auto_index(self._smp, self._workspace)
                self._console.print(f"[success]Indexed {count} files[/success]")
            else:
                self._console.print("[warning]SMP not connected.[/warning]")
            return True

        if cmd == "/compact":
            if self._agent:
                self._agent.reset()
                self._console.print("[info]Conversation compacted (history cleared for fresh context).[/info]")
            return True

        if cmd in ("/quit", "/exit", "/q"):
            self._running = False
            return True

        return False

    def _print_status(self) -> None:
        table = Table(title="Nemotron Status", show_header=False, border_style="dim")
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Workspace", str(self._workspace))
        table.add_row("LLM Provider", self._config.llm.provider if self._config else "?")
        table.add_row("Model", self._config.llm.model if self._config else "?")

        smp_status = "Connected" if (self._smp and self._smp.is_connected) else "Offline"
        table.add_row("SMP", smp_status)

        if self._agent:
            usage = self._agent.token_usage
            table.add_row("Tokens (in/out)", f"{usage['input']:,} / {usage['output']:,}")

        self._console.print(table)

    async def run(self) -> None:
        """Main REPL loop."""
        self._console.print(WELCOME)
        await self._initialize()

        # Prompt setup
        history_file = self._workspace / ".nemotron_history"
        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_file)),
        )

        rel_ws = self._workspace.name
        self._running = True
        self._console.print(f"\n[dim]Workspace: {self._workspace}[/dim]")
        self._console.print("[dim]Type /help for commands.[/dim]\n")

        while self._running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt(
                        HTML(f"<prompt>nemotron</prompt><path>:{rel_ws}</path><prompt> > </prompt>"),
                        style=PROMPT_STYLE,
                        multiline=False,
                    ),
                )
            except (EOFError, KeyboardInterrupt):
                self._console.print("\n[dim]Goodbye![/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                handled = await self._handle_command(user_input)
                if handled:
                    continue

            # Run the agent
            self._tool_count = 0
            self._start_time = time.time()

            try:
                self._console.print()
                await self._agent.run(user_input)
            except KeyboardInterrupt:
                self._console.print("\n[warning]Interrupted.[/warning]")
            except Exception as e:
                self._console.print(f"\n[error]Error: {e}[/error]")

            # Footer with stats
            elapsed = time.time() - self._start_time
            usage = self._agent.token_usage
            self._console.print(
                f"\n[dim]--- {self._tool_count} tool calls | "
                f"{elapsed:.1f}s | "
                f"{usage['input']:,}+{usage['output']:,} tokens ---[/dim]\n"
            )

        # Cleanup
        if self._smp:
            await self._smp.close()


def _summarize_args(tool_name: str, args: dict[str, Any]) -> str:
    """Create a short summary of tool arguments for display."""
    if tool_name in ("read_file", "write_file", "edit_file"):
        return args.get("path", "")
    if tool_name == "shell":
        cmd = args.get("command", "")
        return cmd[:80] + ("..." if len(cmd) > 80 else "")
    if tool_name == "grep":
        return f"/{args.get('pattern', '')}/ in {args.get('path', '.')}"
    if tool_name == "glob":
        return args.get("pattern", "")
    if tool_name == "list_dir":
        return args.get("path", ".")
    if tool_name.startswith("smp_"):
        for key in ("query", "description", "file_path", "entity", "start"):
            if key in args:
                return str(args[key])[:80]
    parts = [f"{k}={v}" for k, v in list(args.items())[:3] if k != "content"]
    return ", ".join(parts)
