# Nemotron

**AI Coding Agent powered by Structural Memory Protocol (SMP)**

Nemotron is a terminal-based coding agent — similar to Claude Code — that uses [SMP](./smp/) as its core intelligence layer. Instead of treating code as flat text, Nemotron understands your codebase as a **graph of entities, relationships, and semantic meanings**.

---

## What Makes Nemotron Different

| Traditional Agent | Nemotron + SMP |
|---|---|
| Reads files one at a time | Understands the full dependency graph |
| Grep for code search | Semantic search via graph RAG (SeedWalkEngine) |
| No memory between edits | Graph memory updated after every change |
| Blind to impact | Assesses blast radius before editing |
| Flat text context | Structural context: imports, callers, tests, patterns |

### SMP Integration

Nemotron integrates with SMP at three levels:

1. **Proactive** — Before the LLM sees your task, Nemotron queries the SMP graph for structural context about any files you mention.
2. **Reactive** — The LLM can invoke SMP tools directly (`smp_navigate`, `smp_trace`, `smp_impact`, `smp_locate`, etc.) during its reasoning loop.
3. **Post-action** — After every file write, Nemotron pushes the change into SMP so the graph stays current.

---

## Quickstart

### 1. Prerequisites

- **Python 3.11+**
- **SMP server** running (see [SMP setup](./smp/README.md))
- **LLM API key** (Anthropic or OpenAI)

### 2. Install

```bash
cd nemotron
pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your API key and SMP URL
```

### 4. Start SMP (if not running)

```bash
cd smp
docker compose up -d
```

### 5. Run

```bash
# From your project directory
nemotron

# Or specify a workspace
nemotron -w /path/to/your/project
```

---

## CLI Options

```
nemotron [-w WORKSPACE] [-p PROVIDER] [-m MODEL] [--smp-url URL] [--no-index] [-v]

Options:
  -w, --workspace PATH    Workspace directory (default: current dir)
  -p, --provider          LLM provider: anthropic | openai
  -m, --model             Model override
  --smp-url               SMP server URL (default: http://localhost:8420)
  --no-index              Skip auto-indexing on startup
  -v, --verbose           Verbose output
```

## Interactive Commands

| Command     | Description |
|-------------|-------------|
| `/help`     | Show help |
| `/clear`    | Clear conversation history |
| `/status`   | Show SMP connection and token usage |
| `/index`    | Re-index the workspace into SMP |
| `/compact`  | Compact conversation history |
| `/quit`     | Exit |

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    TERMINAL UI                          │
│  prompt_toolkit REPL + rich rendering                  │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│                    AGENT LOOP                           │
│  ReAct cycle: Reason → Tool Call → Observe → Repeat    │
│                                                        │
│  ┌──────────────┐  ┌────────────────┐                  │
│  │  LLM Provider│  │ Context Manager│◄── SMP proactive │
│  │  (Anthropic/ │  │ (enriches      │    context        │
│  │   OpenAI)    │  │  system prompt)│                   │
│  └──────────────┘  └────────────────┘                   │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│                  TOOL REGISTRY                          │
│                                                        │
│  File Tools           Shell          SMP Tools          │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ read_file  │  │ shell    │  │ smp_navigate     │   │
│  │ write_file │  │          │  │ smp_trace        │   │
│  │ edit_file  │  └──────────┘  │ smp_context      │   │
│  │ list_dir   │                │ smp_impact       │   │
│  │ glob       │                │ smp_locate       │   │
│  │ grep       │                │ smp_search       │   │
│  └────────────┘                │ smp_flow         │   │
│                                └────────┬─────────┘   │
└─────────────────────────────────────────┼─────────────┘
                                          │
┌─────────────────────────────────────────▼─────────────┐
│                    SMP SERVER                          │
│  Neo4j Graph + ChromaDB Vectors + Merkle Index        │
│  JSON-RPC 2.0 at http://localhost:8420/rpc            │
└───────────────────────────────────────────────────────┘
```

---

## Available Tools

### File Operations
| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers |
| `write_file` | Write/create files |
| `edit_file` | Surgical string replacement in files |
| `list_dir` | List directory contents |
| `glob` | Find files by pattern |
| `grep` | Regex search across files |

### Shell
| Tool | Description |
|------|-------------|
| `shell` | Execute shell commands (git, npm, pytest, etc.) |

### SMP Structural Memory
| Tool | Description |
|------|-------------|
| `smp_navigate` | Look up any entity and its relationships |
| `smp_trace` | Follow call chains (who calls what) |
| `smp_context` | Full structural context for a file (before editing) |
| `smp_impact` | Blast radius assessment for changes |
| `smp_locate` | Semantic code search via graph RAG |
| `smp_search` | Full-text BM25 search on the graph |
| `smp_flow` | Trace data/execution flow between entities |

---

## How SMP Makes the Agent Smarter

### Example: "Add rate limiting to the login endpoint"

**Without SMP** — the agent greps for "login", reads a few files, makes edits.

**With SMP:**
1. `smp_locate("login endpoint")` — finds the exact function via semantic search
2. `smp_context("src/auth/routes.py")` — learns the file's role, dependencies, who imports it
3. `smp_impact("routes.py::login")` — sees that 12 files depend on this, tests exist at `tests/test_auth.py`
4. `smp_trace("login", direction="incoming")` — discovers the middleware chain
5. Makes the edit with full structural understanding
6. `smp/update` — graph is updated automatically after the write

---

## Project Structure

```
nemotron/
├── nemotron/
│   ├── __init__.py
│   ├── main.py           # CLI entry point
│   ├── config.py          # Configuration from env/.env
│   ├── agent.py           # Core ReAct agent loop
│   ├── llm/
│   │   ├── __init__.py
│   │   └── provider.py    # Anthropic + OpenAI providers
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── smp_client.py  # Async SMP JSON-RPC client
│   │   ├── context.py     # Proactive context enrichment
│   │   └── auto_index.py  # Workspace indexing on startup
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py        # Tool protocol + result types
│   │   ├── file_ops.py    # File read/write/edit/search
│   │   ├── shell.py       # Shell command execution
│   │   ├── smp_tools.py   # SMP graph query tools
│   │   └── registry.py    # Tool collection + dispatch
│   └── ui/
│       ├── __init__.py
│       └── terminal.py    # Rich REPL interface
├── smp/                   # SMP server (submodule/dependency)
├── pyproject.toml
├── .env.example
└── README.md
```

---

## License

MIT
