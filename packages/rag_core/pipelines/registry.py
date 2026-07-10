from __future__ import annotations

import re
from dataclasses import dataclass

from packages.rag_core.pipelines.base import PipelineConfig, PipelineFactory, RetrievalPipeline
from packages.rag_core.pipelines.errors import DuplicatePipelineError, InvalidPipelineError, UnknownPipelineError


@dataclass(frozen=True, slots=True)
class RegisteredPipeline:
    """A pipeline configuration paired with its construction factory."""

    config: PipelineConfig
    factory: PipelineFactory


class PipelineRegistry:
    """Registry for discovering and constructing selectable query pipelines.

    Factories are stored instead of pipeline instances so a future pipeline can
    have request-scoped state without changing the API or evaluation boundary.
    """

    def __init__(self, *, default_pipeline_name: str) -> None:
        self._default_pipeline_name = _require_name(default_pipeline_name, field_name="default pipeline name")
        self._registrations: dict[str, RegisteredPipeline] = {}

    @property
    def default_pipeline_name(self) -> str:
        return self._default_pipeline_name

    def register(self, *, config: PipelineConfig, factory: PipelineFactory) -> None:
        name = _require_name(config.name, field_name="pipeline name")
        if name in self._registrations:
            raise DuplicatePipelineError(f"Pipeline {name!r} is already registered.")
        if not config.version.strip():
            raise InvalidPipelineError(f"Pipeline {name!r} must declare a version.")
        if not config.description.strip():
            raise InvalidPipelineError(f"Pipeline {name!r} must declare a description.")
        if len(set(config.tool_names)) != len(config.tool_names):
            raise InvalidPipelineError(f"Pipeline {name!r} declares the same tool more than once.")
        for tool_name in config.tool_names:
            _require_name(tool_name, field_name=f"tool name used by pipeline {name!r}")
        if not callable(factory):
            raise InvalidPipelineError(f"Pipeline {name!r} must have a callable factory.")

        self._registrations[name] = RegisteredPipeline(config=config, factory=factory)

    def build(self, pipeline_name: str | None = None) -> RetrievalPipeline:
        selected_name = self.resolve_name(pipeline_name)
        registration = self._registrations[selected_name]
        pipeline = registration.factory()
        if pipeline.name != registration.config.name or pipeline.version != registration.config.version:
            raise InvalidPipelineError(
                f"Pipeline factory for {selected_name!r} returned "
                f"{pipeline.name!r} version {pipeline.version!r}, expected "
                f"{registration.config.name!r} version {registration.config.version!r}.",
            )
        return pipeline

    def resolve_name(self, pipeline_name: str | None = None) -> str:
        selected_name = self._default_pipeline_name if pipeline_name is None else pipeline_name.strip()
        if not selected_name:
            selected_name = self._default_pipeline_name
        if selected_name not in self._registrations:
            available = ", ".join(self._registrations) or "none"
            raise UnknownPipelineError(
                f"Unknown pipeline {selected_name!r}. Available pipelines: {available}.",
            )
        return selected_name

    def configs(self) -> tuple[PipelineConfig, ...]:
        return tuple(registration.config for registration in self._registrations.values())

    def get_config(self, pipeline_name: str) -> PipelineConfig:
        selected_name = self.resolve_name(pipeline_name)
        return self._registrations[selected_name].config

    def validate(self) -> None:
        """Validate registry-wide configuration after all pipelines are registered."""

        self.resolve_name()


_PIPELINE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _require_name(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidPipelineError(f"{field_name.capitalize()} cannot be empty.")
    if normalized != value or not _PIPELINE_NAME_PATTERN.fullmatch(normalized):
        raise InvalidPipelineError(
            f"{field_name.capitalize()} {value!r} must match {_PIPELINE_NAME_PATTERN.pattern!r}.",
        )
    return normalized
