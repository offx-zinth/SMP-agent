"""LLM provider implementations — Anthropic, OpenAI, and Gemini.

All providers are wrapped behind a common interface so the agent loop
doesn't care which backend is active.  Each provider converts tool specs
into the format its API expects and normalises the response back into a
list of message chunks the agent can process uniformly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from nemotron.config import LLMConfig
from nemotron.tools.base import ToolSpec


# -- Unified message types ---------------------------------------------------

@dataclass
class TextChunk:
    text: str


@dataclass
class ToolCallChunk:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    chunks: list[TextChunk | ToolCallChunk] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(c.text for c in self.chunks if isinstance(c, TextChunk))

    @property
    def tool_calls(self) -> list[ToolCallChunk]:
        return [c for c in self.chunks if isinstance(c, ToolCallChunk)]

    @property
    def has_tool_calls(self) -> bool:
        return any(isinstance(c, ToolCallChunk) for c in self.chunks)


# -- Provider interface ------------------------------------------------------

class LLMProvider:
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str = "",
    ) -> LLMResponse:
        raise NotImplementedError


# -- Anthropic ---------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=config.api_key)
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str = "",
    ) -> LLMResponse:
        tool_schemas = [t.to_anthropic_schema() for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        if self._temperature > 0:
            kwargs["temperature"] = self._temperature

        resp = await self._client.messages.create(**kwargs)

        chunks: list[TextChunk | ToolCallChunk] = []
        for block in resp.content:
            if block.type == "text":
                chunks.append(TextChunk(text=block.text))
            elif block.type == "tool_use":
                chunks.append(ToolCallChunk(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMResponse(
            chunks=chunks,
            stop_reason=resp.stop_reason or "end_turn",
            usage={"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
        )


# -- OpenAI -------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=config.api_key)
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str = "",
    ) -> LLMResponse:
        tool_schemas = [t.to_openai_schema() for t in tools] if tools else []

        # Prepend system message
        all_messages = list(messages)
        if system:
            all_messages.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": all_messages,
            "max_tokens": self._max_tokens,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        if self._temperature > 0:
            kwargs["temperature"] = self._temperature

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        chunks: list[TextChunk | ToolCallChunk] = []
        if msg.content:
            chunks.append(TextChunk(text=msg.content))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                chunks.append(ToolCallChunk(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage = {}
        if resp.usage:
            usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}

        return LLMResponse(
            chunks=chunks,
            stop_reason=choice.finish_reason or "stop",
            usage=usage,
        )


# -- Gemini (google-genai) ---------------------------------------------------

class GeminiProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        from google import genai

        self._client = genai.Client(api_key=config.api_key)
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._call_counter = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        system: str = "",
    ) -> LLMResponse:
        from google.genai import types

        contents = _build_gemini_contents(messages)

        tool_declarations = [t.to_gemini_declaration() for t in tools]
        gemini_tools = [types.Tool(function_declarations=tool_declarations)] if tool_declarations else []

        config = types.GenerateContentConfig(
            systemInstruction=system or None,
            tools=gemini_tools or None,
            maxOutputTokens=self._max_tokens,
            temperature=self._temperature,
            automaticFunctionCalling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        chunks: list[TextChunk | ToolCallChunk] = []
        if resp.candidates and resp.candidates[0].content:
            for part in resp.candidates[0].content.parts:
                if part.text:
                    chunks.append(TextChunk(text=part.text))
                elif part.function_call:
                    fc = part.function_call
                    self._call_counter += 1
                    chunks.append(ToolCallChunk(
                        id=fc.id or f"call_{fc.name}_{self._call_counter}",
                        name=fc.name or "",
                        arguments=dict(fc.args) if fc.args else {},
                    ))

        usage: dict[str, int] = {}
        if resp.usage_metadata:
            usage = {
                "input": resp.usage_metadata.prompt_token_count or 0,
                "output": resp.usage_metadata.candidates_token_count or 0,
            }

        stop = "end_turn"
        if resp.candidates:
            reason = resp.candidates[0].finish_reason
            if reason:
                stop = str(reason)

        return LLMResponse(chunks=chunks, stop_reason=stop, usage=usage)


def _build_gemini_contents(messages: list[dict[str, Any]]) -> list[Any]:
    """Convert our internal message format to Gemini Content objects."""
    from google.genai import types

    contents: list[types.Content] = []
    for msg in messages:
        role = msg["role"]
        raw = msg.get("content", "")

        # Anthropic-style tool_result blocks → Gemini tool role
        if role == "user" and isinstance(raw, list) and raw and isinstance(raw[0], dict) and raw[0].get("type") == "tool_result":
            parts: list[types.Part] = []
            for block in raw:
                parts.append(types.Part.from_function_response(
                    name=block.get("tool_use_id", ""),
                    response={"result": block.get("content", "")},
                ))
            contents.append(types.Content(role="tool", parts=parts))
            continue

        # OpenAI-style tool message
        if role == "tool":
            tool_id = msg.get("tool_call_id", "")
            parts = [types.Part.from_function_response(
                name=tool_id,
                response={"result": str(raw)},
            )]
            contents.append(types.Content(role="tool", parts=parts))
            continue

        # Assistant message with Anthropic-style tool_use blocks
        if role == "assistant" and isinstance(raw, list):
            parts = []
            for block in raw:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(types.Part.from_text(text=block["text"]))
                    elif block.get("type") == "tool_use":
                        parts.append(types.Part.from_function_call(
                            name=block["name"],
                            args=block.get("input", {}),
                        ))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue

        # Regular text messages
        gemini_role = "model" if role == "assistant" else "user"
        text = raw if isinstance(raw, str) else json.dumps(raw)
        if text:
            contents.append(types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=text)],
            ))

    return contents


# -- Factory -----------------------------------------------------------------

def create_provider(config: LLMConfig) -> LLMProvider:
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    elif config.provider == "openai":
        return OpenAIProvider(config)
    elif config.provider == "gemini":
        return GeminiProvider(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
