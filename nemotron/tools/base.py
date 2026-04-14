"""Base tool interface and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolResult:
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.error

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        return self.output


@dataclass
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolSpec:
    """Describes a tool for the LLM."""
    name: str
    description: str
    parameters: list[ToolParam]

    def to_openai_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            schema: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                schema["enum"] = p.enum
            props[p.name] = schema
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            schema: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                schema["enum"] = p.enum
            props[p.name] = schema
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

    def to_gemini_declaration(self) -> Any:
        """Build a google.genai FunctionDeclaration for this tool."""
        from google.genai import types

        _type_map = {
            "string": types.Type.STRING,
            "integer": types.Type.INTEGER,
            "number": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN,
            "array": types.Type.ARRAY,
            "object": types.Type.OBJECT,
        }

        props: dict[str, types.Schema] = {}
        required: list[str] = []
        for p in self.parameters:
            schema_kwargs: dict[str, Any] = {
                "type": _type_map.get(p.type, types.Type.STRING),
                "description": p.description,
            }
            if p.enum:
                schema_kwargs["enum"] = p.enum
            props[p.name] = types.Schema(**schema_kwargs)
            if p.required:
                required.append(p.name)

        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties=props,
                required=required,
            ),
        )


class Tool(Protocol):
    """Protocol for all tools."""

    @property
    def spec(self) -> ToolSpec: ...

    async def execute(self, **kwargs: Any) -> ToolResult: ...
