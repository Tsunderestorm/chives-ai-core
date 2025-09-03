"""Base tool abstract class for all custom tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class BaseTool(StructuredTool, ABC):
    """
    Abstract base class for all custom tools.

    This class combines LangChain's StructuredTool with Python's ABC
    to provide a standardized interface for tool creation.
    """

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """
        Execute the tool with the given arguments.

        This method must be implemented by all subclasses.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tool execution result
        """
        pass

    @abstractmethod
    def _get_args_schema(self) -> Type[BaseModel]:
        """
        Get the Pydantic schema for tool arguments.

        Returns:
            Pydantic model class defining the tool's input schema
        """
        pass

    def __init__(self, **kwargs: Any):
        """
        Initialize the base tool.

        Args:
            **kwargs: Additional tool configuration
        """
        # Set the args_schema from the abstract method
        kwargs["args_schema"] = self._get_args_schema()

        # Call parent constructor
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and preprocess tool inputs.

        Override this method to add custom validation logic.

        Args:
            inputs: Raw input dictionary

        Returns:
            Validated and processed inputs

        Raises:
            ValueError: If inputs are invalid
        """
        return inputs

    def handle_error(self, error: Exception) -> str:
        """
        Handle tool execution errors.

        Override this method to customize error handling.

        Args:
            error: The exception that occurred

        Returns:
            Error message to return to the user
        """
        return f"Tool '{self.name}' failed: {str(error)}"


class BaseToolArgs(BaseModel):
    """
    Base class for tool argument schemas.

    Extend this class to define tool-specific arguments.
    """

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment
