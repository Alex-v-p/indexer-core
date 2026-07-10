class ToolRegistryError(RuntimeError):
    """Base exception for tool registration and lookup failures."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a tool name is registered more than once."""


class InvalidToolError(ToolRegistryError):
    """Raised when tool metadata or an implementation is invalid."""


class UnknownToolError(ToolRegistryError):
    """Raised when a requested tool is not registered."""


class ToolTypeMismatchError(ToolRegistryError):
    """Raised when a resolved tool does not match the expected concrete type."""
