"""Base agent abstract class for all custom agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class BaseAgent(ABC):
    """
    Abstract base class for all custom agents.

    This class provides a standardized interface for agent creation
    using LangChain's BaseLanguageModel.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the base agent.

        Args:
            llm: The language model to use for this agent
            tools: Optional list of tools available to the agent
            system_message: Optional system message to set agent behavior
            **kwargs: Additional agent configuration
        """
        self.llm = llm
        self.tools = tools or []
        self.system_message = system_message
        self.conversation_history: List[BaseMessage] = []

        # Initialize system message if provided
        if self.system_message:
            self.conversation_history.append(SystemMessage(content=self.system_message))

        # Additional initialization
        self._initialize(**kwargs)

    @abstractmethod
    def _initialize(self, **kwargs: Any) -> None:
        """
        Additional initialization logic for subclasses.

        Override this method to add custom initialization.

        Args:
            **kwargs: Additional configuration parameters
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description."""
        pass

    @abstractmethod
    def process_message(self, message: str, **kwargs: Any) -> str:
        """
        Process an incoming message and return a response.

        This is the main method that subclasses must implement.

        Args:
            message: The input message to process
            **kwargs: Additional processing parameters

        Returns:
            The agent's response
        """
        pass

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent's toolkit.

        Args:
            tool: The tool to add
        """
        if tool not in self.tools:
            self.tools.append(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent's toolkit.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        for i, tool in enumerate(self.tools):
            if str(getattr(tool, "name", tool)) == tool_name:
                self.tools.pop(i)
                return True
        return False

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        for tool in self.tools:
            if str(getattr(tool, "name", tool)) == tool_name:
                return tool
        return None

    def add_message_to_history(self, message: BaseMessage) -> None:
        """
        Add a message to the conversation history.

        Args:
            message: The message to add
        """
        self.conversation_history.append(message)

    def clear_history(self, keep_system_message: bool = True) -> None:
        """
        Clear the conversation history.

        Args:
            keep_system_message: Whether to keep the system message
        """
        if (
            keep_system_message
            and self.conversation_history
            and isinstance(self.conversation_history[0], SystemMessage)
        ):
            system_msg = self.conversation_history[0]
            self.conversation_history = [system_msg]
        else:
            self.conversation_history = []

    def get_conversation_context(self, max_messages: Optional[int] = None) -> List[BaseMessage]:
        """
        Get the conversation context for the LLM.

        Args:
            max_messages: Maximum number of recent messages to include

        Returns:
            List of messages for the LLM context
        """
        if max_messages is None:
            return self.conversation_history.copy()
        else:
            # Always include system message if it exists
            if self.conversation_history and isinstance(
                self.conversation_history[0], SystemMessage
            ):
                system_msg = [self.conversation_history[0]]
                recent_messages = self.conversation_history[1:][-max_messages:]
                return system_msg + recent_messages
            else:
                return self.conversation_history[-max_messages:]

    def format_tool_descriptions(self) -> str:
        """
        Format tool descriptions for the agent prompt.

        Returns:
            Formatted string describing available tools
        """
        if not self.tools:
            return "No tools available."

        tool_descriptions = []
        for tool in self.tools:
            # Access the property values safely
            name = str(getattr(tool, "name", tool))
            description = str(getattr(tool, "description", "No description"))
            tool_descriptions.append(f"- {name}: {description}")

        return "Available tools:\n" + "\n".join(tool_descriptions)

    def validate_message(self, message: str) -> str:
        """
        Validate and preprocess incoming messages.

        Override this method to add custom validation logic.

        Args:
            message: The incoming message

        Returns:
            The validated/processed message

        Raises:
            ValueError: If the message is invalid
        """
        if not isinstance(message, str):
            raise ValueError("Message must be a string")

        if not message.strip():
            raise ValueError("Message cannot be empty or whitespace only")

        return message.strip()

    def handle_error(self, error: Exception, context: str = "") -> str:
        """
        Handle agent execution errors.

        Override this method to customize error handling.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred

        Returns:
            Error message to return to the user
        """
        error_msg = f"Agent '{self.name}' encountered an error"
        if context:
            error_msg += f" during {context}"
        error_msg += f": {str(error)}"

        return error_msg

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', tools={len(self.tools)})"

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"description='{self.description}', "
            f"tools={len(self.tools)}, "
            f"history_length={len(self.conversation_history)}"
            f")"
        )
