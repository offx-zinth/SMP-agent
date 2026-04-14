"""Configuration for the Nemotron agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    api_key: str = ""
    model: str = ""
    max_tokens: int = 8192
    temperature: float = 0.0

    # Defaults per provider
    _defaults: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "anthropic": {"model": "claude-sonnet-4-20250514"},
            "openai": {"model": "gpt-4o"},
            "gemini": {"model": "gemini-2.5-flash"},
        },
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.model:
            self.model = self._defaults.get(self.provider, {}).get("model", "gpt-4o")


@dataclass
class SMPConfig:
    url: str = "http://localhost:8420"
    auto_index: bool = True
    timeout: float = 30.0


@dataclass
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    smp: SMPConfig = field(default_factory=SMPConfig)
    workspace: Path = field(default_factory=Path.cwd)
    max_iterations: int = 50
    verbose: bool = False


def load_config(workspace: Path | None = None) -> AgentConfig:
    """Load configuration from environment variables and .env file."""
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    provider = os.getenv("NEMOTRON_LLM_PROVIDER", "anthropic")

    key_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
    api_key = os.getenv(key_map.get(provider, "OPENAI_API_KEY"), "")

    llm = LLMConfig(
        provider=provider,
        api_key=api_key,
        model=os.getenv("NEMOTRON_MODEL", ""),
    )

    smp = SMPConfig(
        url=os.getenv("SMP_URL", "http://localhost:8420"),
        auto_index=os.getenv("NEMOTRON_AUTO_INDEX", "true").lower() == "true",
    )

    return AgentConfig(
        llm=llm,
        smp=smp,
        workspace=workspace or Path.cwd(),
        max_iterations=int(os.getenv("NEMOTRON_MAX_ITERATIONS", "50")),
        verbose=os.getenv("NEMOTRON_VERBOSE", "false").lower() == "true",
    )
