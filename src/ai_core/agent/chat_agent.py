"""Chat agent that can use tools for enhanced functionality."""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChatAgent(BaseAgent):
    """
    Chat agent that can use tools to enhance its responses.

    This agent demonstrates tool integration with the BaseAgent framework.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the chat agent.

        Args:
            llm: The language model to use
            tools: List of tools available to the agent
            enable_tool_use: Whether to enable tool usage
            **kwargs: Additional configuration
        """

        # Set default system message if none provided
        if "system_message" not in kwargs or kwargs["system_message"] is None:
            tool_context = ""
            if tools:
                tool_names = [str(getattr(tool, "name", tool)) for tool in tools]
                tool_context = f"\n\nYou have access to these tools: {', '.join(tool_names)}. Use them when appropriate to help answer questions."

            # TODO: Make this config-driven 
            kwargs["system_message"] = (
                "You are a helpful AI assistant. Provide clear, accurate, "
                "and helpful responses to user questions." + tool_context
            )

        super().__init__(llm=llm, tools=tools, **kwargs)
        self._initialize()

    def _initialize(self, **kwargs: Any) -> None:
        """Additional initialization for the chat agent."""
        self.tool_call_count = 0
        self.successful_tool_calls = 0
        
        # Initialize RAG engine
        self._initialize_rag_engine()

    def _initialize_rag_engine(self) -> None:
        """Initialize the RAG engine for document retrieval."""
        try:
            from ai_core.rag import RAGEngine
            from ai_core.config import settings
            
            # Log the configuration values being used
            logger.info("🔧 Initializing RAG engine with configuration:")
            logger.info(f"   CHROMA_COLLECTION_NAME: {settings.CHROMA_COLLECTION_NAME}")
            logger.info(f"   CHROMA_PERSIST_DIRECTORY: {settings.CHROMA_PERSIST_DIRECTORY}")
            logger.info(f"   CHROMA_EMBEDDING_MODEL: {settings.CHROMA_EMBEDDING_MODEL}")
            logger.info(f"   CHROMA_HOST: {settings.CHROMA_HOST}")
            logger.info(f"   CHROMA_PORT: {settings.CHROMA_PORT}")
            
            self.rag_engine = RAGEngine(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
                embedding_model=settings.CHROMA_EMBEDDING_MODEL,
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT
            )
            
            logger.info("✅ RAG engine initialized successfully")
            
        except ImportError as e:
            logger.warning(f"RAG module not available: {e}")
            self.rag_engine = None
        except Exception as e:
            logger.warning(f"Failed to initialize RAG engine: {e}")
            self.rag_engine = None

    @property
    def name(self) -> str:
        return "chat_agent"

    @property
    def description(self) -> str:
        return "A conversational agent that can use tools to provide enhanced responses"

    def process_message(self, message: str, **kwargs: Any) -> str:
        """
        Process a message and potentially use tools to enhance the response.

        Args:
            message: The input message to process
            **kwargs: Additional processing parameters

        Returns:
            The agent's response
        """
        logger.info(f"🤖 ChatAgent.process_message() - User message: '{message}'")
        
        try:
            # Validate the incoming message
            validated_message = self.validate_message(message)
            logger.debug(f"📝 Message validated: '{validated_message}'")

            # Add user message to history
            user_message = HumanMessage(content=validated_message)
            self.add_message_to_history(user_message)
            logger.debug(f"💬 Added user message to history. History length: {len(self.conversation_history)}")

            response = self._process_with_tools(validated_message)
            logger.info(f"✅ ChatAgent.process_message() - Generated response: '{response}'")
            return response

        except Exception as e:
            logger.error(f"❌ Error in ChatAgent.process_message(): {str(e)}", exc_info=True)
            return self.handle_error(e, "message processing")

    async def process_message_async(self, message: str, **kwargs: Any) -> str:
        """
        Async version of message processing with RAG enhancement.

        Args:
            message: The input message to process
            **kwargs: Additional processing parameters

        Returns:
            The agent's response
        """
        logger.info(f"🤖 ChatAgent.process_message_async() - User message: '{message}'")
        
        try:
            # Validate the incoming message
            validated_message = self.validate_message(message)
            logger.debug(f"📝 Message validated: '{validated_message}'")

            # Add user message to history
            user_message = HumanMessage(content=validated_message)
            self.add_message_to_history(user_message)
            logger.debug(f"💬 Added user message to history. History length: {len(self.conversation_history)}")

            response = await self._process_with_tools_async(validated_message)
            logger.info(f"✅ ChatAgent.process_message_async() - Generated response: '{response}'")
            return response

        except Exception as e:
            logger.error(f"❌ Error in ChatAgent.process_message_async(): {str(e)}", exc_info=True)
            return self.handle_error(e, "message processing")

    def _process_with_tools(self, message: str) -> str:
        """
        Process message with potential tool usage and RAG enhancement.

        Args:
            message: The user's message

        Returns:
            The agent's response
        """
        logger.info(f"🔧 _process_with_tools() - Processing message with {len(self.tools)} available tools")
        
        # Get conversation context
        messages_with_context = self.get_conversation_context()
        logger.debug(f"💬 Retrieved conversation context with {len(messages_with_context)} messages")

        # Perform RAG retrieval to enhance context
        rag_context = self._retrieve_relevant_documents(message)
        if rag_context:
            logger.info(f"📚 RAG retrieved relevant document chunks")
            # Add RAG context to the user message
            enhanced_message = f"{message}\n\nRelevant information from knowledge base:\n{rag_context}"
            messages_with_context[-1].content = enhanced_message
        else:
            logger.info("📚 No relevant documents found for RAG enhancement")

        # Add tool information to the context
        if self.tools:
            tool_prompt = f"\n\nAvailable tools: {self.format_tool_descriptions()}"
            messages_with_context[-1].content += tool_prompt
            logger.debug(f"🛠️ Added tool descriptions to context: {[tool.name for tool in self.tools]}")

        # Generate enhanced response with RAG context
        logger.info("🧠 Calling LLM with RAG-enhanced context...")
        llm_response = self.llm.invoke(messages_with_context)
        logger.info(f"📤 LLM raw response: {llm_response}")
        
        # Parse the response
        response_content = self._parse_llm_response(llm_response)
        logger.info(f"📄 Parsed content: '{response_content}'")
        
        # Execute tool calls if any
        tool_output = self._execute_tool_calls(llm_response)
        response = f"{response_content}\n\n{tool_output}" if tool_output else response_content

        self.add_message_to_history(AIMessage(content=response))
        return response

    async def _process_with_tools_async(self, message: str) -> str:
        """
        Async version of message processing with RAG enhancement.

        Args:
            message: The user's message

        Returns:
            The agent's response
        """
        logger.info(f"🔧 _process_with_tools_async() - Processing message with {len(self.tools)} available tools")
        
        # Get conversation context
        messages_with_context = self.get_conversation_context()
        logger.debug(f"💬 Retrieved conversation context with {len(messages_with_context)} messages")

        # Perform RAG retrieval to enhance context (async)
        rag_context = await self._retrieve_relevant_documents_async(message)
        if rag_context:
            logger.info(f"📚 RAG retrieved relevant document chunks")
            # Add RAG context to the user message
            enhanced_message = f"{message}\n\nRelevant information from knowledge base:\n{rag_context}"
            messages_with_context[-1].content = enhanced_message
        else:
            logger.info("📚 No relevant documents found for RAG enhancement")

        # Add tool information to the context
        if self.tools:
            tool_prompt = f"\n\nAvailable tools: {self.format_tool_descriptions()}"
            messages_with_context[-1].content += tool_prompt
            logger.debug(f"🛠️ Added tool descriptions to context: {[tool.name for tool in self.tools]}")

        # Generate enhanced response with RAG context
        logger.info("🧠 Calling LLM with RAG-enhanced context...")
        llm_response = self.llm.invoke(messages_with_context)
        logger.info(f"📤 LLM raw response: {llm_response}")
        
        # Parse the response
        response_content = self._parse_llm_response(llm_response)
        logger.info(f"📄 Parsed content: '{response_content}'")
        
        # Execute tool calls if any
        tool_output = self._execute_tool_calls(llm_response)
        response = f"{response_content}\n\n{tool_output}" if tool_output else response_content

        self.add_message_to_history(AIMessage(content=response))
        return response

    def _parse_llm_response(self, llm_response: Any) -> str:
        """
        Parse the raw LLM response to extract content and tool invocations.

        Args:
            llm_response: The raw response from the LLM

        Returns:
            Dictionary containing parsed content and tool invocations
        """
        logger.info(f"🔍 _parse_llm_response() - Parsing response")
        
        try: 
            content = self.llm.parse_response_content(llm_response.content)
        except Exception as e:
            logger.error(f"❌ Error parsing LLM response: {str(e)}", exc_info=True)
            return ""
        return content

    def _execute_tool_calls(self, raw_response: Any) -> str:
        """
        Execute tool calls from the raw LLM response.

        Args:
            raw_response: The raw response from the LLM

        Returns:
            Combined output from all tool calls, or empty string if no tools called
        """
        logger.info(f"🔧 _execute_tool_calls() - Checking for tool calls")
        
        try:
            # Check if the response has tool calls
            if not hasattr(raw_response, 'tool_calls') or not raw_response.tool_calls:
                logger.info("🚫 No tool calls found in response")
                return ""
            
            tool_outputs = []
            logger.info(f"🚀 Found {len(raw_response.tool_calls)} tool calls")
            
            for i, tool_call in enumerate(raw_response.tool_calls):
                logger.info(f"🔧 Executing tool call {i+1}: {tool_call}")
                
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                if not tool_name:
                    logger.warning(f"⚠️ Tool call {i+1} missing name, skipping")
                    continue
                
                # Get the tool by name
                tool = self.get_tool_by_name(tool_name)
                if not tool:
                    logger.error(f"❌ Tool '{tool_name}' not found")
                    tool_outputs.append(f"Error: Tool '{tool_name}' not found")
                    continue
                
                # Execute the tool
                try:
                    logger.info(f"🔧 Invoking tool '{tool_name}' with args: {tool_args}")
                    output = tool.invoke(tool_args)
                    logger.info(f"✅ Tool '{tool_name}' output: '{output}'")
                    tool_outputs.append(str(output))
                    
                    # Update statistics
                    self.tool_call_count += 1
                    self.successful_tool_calls += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error executing tool '{tool_name}': {str(e)}")
                    tool_outputs.append(f"Error executing tool '{tool_name}': {str(e)}")
                    self.tool_call_count += 1
            
            # Combine all tool outputs
            combined_output = "\n\n".join(tool_outputs) if tool_outputs else ""
            logger.info(f"🔧 Combined tool output: '{combined_output}'")
            return combined_output
            
        except Exception as e:
            logger.error(f"❌ Error in _execute_tool_calls: {str(e)}", exc_info=True)
            return f"Error executing tool calls: {str(e)}"

    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by its name.

        Args:
            tool_name: Name of the tool to find

        Returns:
            The tool if found, None otherwise
        """
        for tool in self.tools:
            if getattr(tool, 'name', None) == tool_name:
                return tool
        return None

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        logger.info(f"🔧 _execute_tool() - Executing tool '{tool_name}' with args: {args}")
        
        try:
            tool = self.get_tool(tool_name)
            if not tool:
                logger.error(f"❌ Tool '{tool_name}' not found in available tools: {[t.name for t in self.tools]}")
                return f"Error: Tool '{tool_name}' not found"

            logger.debug(f"✅ Found tool: {tool}")
            self.tool_call_count += 1
            logger.debug(f"📊 Tool call count: {self.tool_call_count}")
            
            result = tool._run(**args)
            self.successful_tool_calls += 1
            logger.info(f"✅ Tool '{tool_name}' executed successfully. Result: '{result}'")
            logger.debug(f"📊 Successful tool calls: {self.successful_tool_calls}")

            return str(result)

        except Exception as e:
            logger.error(f"❌ Error executing tool '{tool_name}': {str(e)}", exc_info=True)
            return f"Error executing tool '{tool_name}': {str(e)}"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        return {
            "name": self.name,
            "conversation_length": len(self.conversation_history),
            "tools_available": len(self.tools),
            "tool_names": [str(getattr(tool, "name", tool)) for tool in self.tools],
            "tool_calls_attempted": self.tool_call_count,
            "successful_tool_calls": self.successful_tool_calls,
            "tool_success_rate": (
                self.successful_tool_calls / self.tool_call_count if self.tool_call_count > 0 else 0
            ),
        }

    async def _retrieve_relevant_documents_async(self, query: str, max_results: int = 3) -> Optional[str]:
        """Async version of document retrieval using the RAG engine."""
        if self.rag_engine:
            try:
                return await self.rag_engine.retrieve_relevant_documents_async(query, max_results)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                return None
        else:
            logger.warning("RAG engine not initialized. Cannot retrieve documents.")
            return None
