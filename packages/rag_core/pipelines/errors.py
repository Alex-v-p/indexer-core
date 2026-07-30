class PipelineRegistryError(RuntimeError):
    """Base exception for pipeline registration and selection failures."""


class DuplicatePipelineError(PipelineRegistryError):
    """Raised when a pipeline name is registered more than once."""


class InvalidPipelineError(PipelineRegistryError):
    """Raised when a pipeline registration is internally inconsistent."""


class UnknownPipelineError(PipelineRegistryError):
    """Raised when a requested pipeline is not registered."""
