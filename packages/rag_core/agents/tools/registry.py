from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeVar, cast

from packages.rag_core.agents.tools.base import ToolConfig
from packages.rag_core.agents.tools.errors import (
    DuplicateToolError,
    InvalidToolError,
    ToolTypeMismatchError,
    UnknownToolError,
)

T = TypeVar("T")
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """A logical tool configuration paired with its implementation object."""

    config: ToolConfig
    implementation: object


class ToolRegistry:
    """Typed lookup registry for retrievers, generators, and future RAG tools.

    The registry deliberately does not force all tools into one invocation
    signature. Retrieval, reranking, generation, and grading keep their own
    focused protocols while pipelines resolve them by stable logical names.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, *, config: ToolConfig, implementation: object) -> None:
        name = config.name.strip()
        if not name:
            raise InvalidToolError("Tool name cannot be empty.")
        if name != config.name or not _TOOL_NAME_PATTERN.fullmatch(name):
            raise InvalidToolError(
                f"Tool name {config.name!r} must match {_TOOL_NAME_PATTERN.pattern!r}.",
            )
        if name in self._tools:
            raise DuplicateToolError(f"Tool {name!r} is already registered.")
        if not config.kind.strip():
            raise InvalidToolError(f"Tool {name!r} must declare a kind.")
        if not config.version.strip():
            raise InvalidToolError(f"Tool {name!r} must declare a version.")
        if not config.description.strip():
            raise InvalidToolError(f"Tool {name!r} must declare a description.")
        if implementation is None:
            raise InvalidToolError(f"Tool {name!r} must have an implementation.")

        self._tools[name] = RegisteredTool(config=config, implementation=implementation)

    def resolve(self, name: str, *, expected_type: type[T] | None = None) -> T:
        normalized_name = name.strip()
        registration = self._tools.get(normalized_name)
        if registration is None:
            available = ", ".join(self._tools) or "none"
            raise UnknownToolError(f"Unknown tool {normalized_name!r}. Available tools: {available}.")

        implementation = registration.implementation
        if expected_type is not None and not isinstance(implementation, expected_type):
            raise ToolTypeMismatchError(
                f"Tool {normalized_name!r} is {type(implementation).__name__}, "
                f"expected {expected_type.__name__}.",
            )
        return cast(T, implementation)

    def configs(self, *, kind: str | None = None) -> tuple[ToolConfig, ...]:
        configurations = (registration.config for registration in self._tools.values())
        if kind is None:
            return tuple(configurations)
        return tuple(config for config in configurations if config.kind == kind)
