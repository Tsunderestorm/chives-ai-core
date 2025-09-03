import json
import requests
import logging
import time
import os
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from pydantic import Field

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OpenAPIChatModel(BaseChatModel):
    """
    A simple chat model that calls an OpenAPI endpoint.

    Configuration can be provided via parameters or environment variables:
    - openapi_url: OPENAPI_LLM_URL
    - api_key: OPENAPI_LLM_API_KEY
    - model: OPENAPI_LLM_MODEL

    Parameters passed explicitly will override environment variables.
    """

    # Define Pydantic fields
    openapi_url: str = Field(default_factory=lambda: os.getenv("OPENAPI_LLM_URL"))
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAPI_LLM_API_KEY"))
    model: str = Field(default_factory=lambda: os.getenv("OPENAPI_LLM_MODEL"))
    tools: Optional[List[BaseTool]] = Field(default_factory=list)

    def __init__(
        self,
        openapi_url: str = None,
        api_key: str = None,
        model: str = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs,
    ):
        # Use provided values or fall back to environment variables
        url = openapi_url or os.getenv("OPENAPI_LLM_URL")
        key = api_key or os.getenv("OPENAPI_LLM_API_KEY")
        model_name = model or os.getenv("OPENAPI_LLM_MODEL")
        tool_list = tools or []

        # Validate required parameters
        if not url:
            raise ValueError(
                "openapi_url is required. Set OPENAPI_LLM_URL environment variable or pass it explicitly."
            )
        if not key:
            raise ValueError(
                "api_key is required. Set OPENAPI_LLM_API_KEY environment variable or pass it explicitly."
            )
        if not model_name:
            raise ValueError(
                "model is required. Set OPENAPI_LLM_MODEL environment variable or pass it explicitly."
            )

        # Initialize parent class with the validated values
        super().__init__(openapi_url=url, api_key=key, model=model_name, tools=tool_list, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openapi_chat"

    def add_tools(self, tools: List[BaseTool]) -> None:
        """
        Add tools to this model instance.

        Args:
            tools: List of tools to add
        """
        self.tools.extend(tools)

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a single tool to this model instance.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)

    def bind_tools(self, tools: List[BaseTool], **kwargs) -> "OpenAPIChatModel":
        """
        Bind tools to this model for LangChain compatibility.
        For simplicity, this modifies the current instance and returns self.

        Args:
            tools: List of tools to bind
            **kwargs: Additional arguments (ignored for simplicity)

        Returns:
            Self (for method chaining)
        """
        self.add_tools(tools)
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from the API."""

        # Convert LangChain messages to OpenAI standard format
        formatted_messages = self._convert_messages_to_dict(messages)

        # Format tools if provided (either from kwargs or bound tools)
        tools = kwargs.get("tools", self.tools or [])
        formatted_tools = []
        if tools:
            formatted_tools = self._format_tools(tools)

        # Create the proper payload structure expected by the API
        payload = {
            "payload": {
                "generation_settings": {
                    "temperature": kwargs.get("temperature", 0.5),
                    "parameters": {},
                },
                "model": self.model,
                "messages": formatted_messages,
            }
        }

        # Add tools to payload if provided
        if formatted_tools:
            payload["payload"]["tools"] = formatted_tools

        headers = {"api-key": self.api_key, "Content-Type": "application/json"}

        try:
            logger.info(f"📤 calling /generate with payload: {payload}")
            response = requests.post(
                self.openapi_url, json=payload, headers=headers, verify=False, timeout=30
            )

            if response.status_code != 200:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                ai_message = AIMessage(
                    content=f"HTTP Error {response.status_code}: {response.text}"
                )
                return ChatResult(generations=[ChatGeneration(message=ai_message)])

            try:
                response_json = response.json()
                generation = response_json["generations"][0]
            except (json.JSONDecodeError, KeyError) as json_error:
                logger.error(f"JSON parsing error or KeyError decoding response from LLM: {json_error}")
                ai_message = AIMessage(content=f"JSON Parse Error: {response.text}")
                return ChatResult(generations=[ChatGeneration(message=ai_message)])

            logger.info(f"📤 generation: {generation}")
            if "tool_invocations" in generation:
                tool_calls = []
                for tool_call in generation["tool_invocations"]:
                    tool_calls.append(ToolCall(name=tool_call["function"]["name"],
                                            args=json.loads(tool_call["function"]["arguments"]),
                                            id=tool_call["id"]))

                ai_message = AIMessage(content=generation["content"],
                                    additional_kwargs={"tool_calls": generation["tool_invocations"]},
                                    tool_calls=tool_calls)
            else:
                ai_message = AIMessage(content=generation["content"])
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
            

        except requests.exceptions.Timeout:
            logger.error("Request timed out after 30 seconds")
            ai_message = AIMessage(content="Request timed out")
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        except requests.exceptions.ConnectionError as conn_error:
            logger.error(f"Connection error: {conn_error}")
            ai_message = AIMessage(content=f"Connection error: {conn_error}")
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            ai_message = AIMessage(content=f"Unexpected error: {e}")
            return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _format_tools(self, tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """
        Format tools for the OpenAPI endpoint following this format:
        https://platform.openai.com/docs/guides/function-calling
        """
        formatted_tools = []
        for tool in tools:
            # Handle both dict-like objects and BaseTool objects
            if hasattr(tool, "name") and hasattr(tool, "description"):
                # BaseTool object
                name = tool.name
                description = tool.description
                # Get the args schema if available
                parameters = {}
                if hasattr(tool, "args_schema") and tool.args_schema:
                    # Convert Pydantic model to JSON schema
                    parameters = tool.args_schema.model_json_schema()
            else:
                # Dict-like object (fallback)
                name = tool.get("name", "")
                description = tool.get("description", "")
                parameters = tool.get("parameters", {})

            formatted_tool = {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            }
            formatted_tools.append(formatted_tool)
        return formatted_tools

    @staticmethod
    def _convert_messages_to_dict(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Convert LangChain messages to OpenAI API standard format following:
        https://platform.openai.com/docs/guides/structured-outputs/supported-models
        """
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            SystemMessage,
            FunctionMessage,
            ToolMessage,
            ToolCall
        )

        formatted_messages = []
        for message in messages:
            content = message.content

            # Map LangChain message types to OpenAI roles
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, FunctionMessage):
                role = "function"
            elif isinstance(message, ToolMessage):
                role = "tool"
            else:
                # Default unknown message types to "user"
                role = "user"

            formatted_message = {"role": role, "content": content}

            # Add additional fields for specific message types
            if hasattr(message, "name") and message.name:
                formatted_message["name"] = message.name
            if hasattr(message, "tool_call_id") and message.tool_call_id:
                formatted_message["tool_call_id"] = message.tool_call_id

            formatted_messages.append(formatted_message)

        return formatted_messages

    @staticmethod
    def parse_response_content(raw_content: str) -> str:
        """
        Parse the nested API response structure to extract the actual message content.

        Args:
            raw_content: The raw string content from the API response

        Returns:
            The extracted message content string
        """
        try:
            import ast

            parsed_content = ast.literal_eval(raw_content)

            # Extract the actual message content from the nested structure
            if "generations" in parsed_content and len(parsed_content["generations"]) > 0:
                return parsed_content["generations"][0].get("content", "")
            else:
                return raw_content
        except (ValueError, SyntaxError, KeyError):
            # If parsing fails, return the raw content
            return raw_content
