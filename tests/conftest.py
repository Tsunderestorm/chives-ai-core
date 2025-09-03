"""Pytest configuration and fixtures for chives-backend tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Generator, Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set test environment variables before importing modules
os.environ["OPENAPI_LLM_URL"] = "http://test-llm-api.com"
os.environ["OPENAPI_LLM_API_KEY"] = "test-api-key"
os.environ["OPENAPI_LLM_MODEL"] = "test-model"
os.environ["CHROMA_PERSIST_DIRECTORY"] = tempfile.mkdtemp()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("ai_core.config.settings") as mock:
        mock.PROJECT_NAME = "Test Chives Backend"
        mock.VERSION = "0.1.0-test"
        mock.API_V1_STR = "/api/v1"
        mock.HOST = "0.0.0.0"
        mock.PORT = 8000
        mock.DEBUG = True
        mock.ALLOWED_HOSTS = ["*"]
        mock.OPENAPI_LLM_URL = "http://test-llm-api.com"
        mock.OPENAPI_LLM_API_KEY = "test-api-key"
        mock.OPENAPI_LLM_MODEL = "test-model"
        mock.CHROMA_COLLECTION_NAME = "test-documents"
        mock.CHROMA_PERSIST_DIRECTORY = tempfile.mkdtemp()
        mock.CHROMA_EMBEDDING_MODEL = "test-model"
        mock.CHROMA_HOST = None
        mock.CHROMA_PORT = None
        yield mock


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    message = AIMessage(content="Test response from LLM")
    generation = ChatGeneration(message=message)
    return ChatResult(generations=[generation])


@pytest.fixture
def mock_openapi_chat_model(mock_llm_response):
    """Mock OpenAPIChatModel for testing."""
    mock_llm = Mock()
    mock_llm.openapi_url = "http://test-llm-api.com"
    mock_llm.api_key = "test-api-key"
    mock_llm.model = "test-model"
    mock_llm.tools = []
    
    # Mock methods
    mock_llm.invoke.return_value = mock_llm_response
    mock_llm.add_tool = Mock()
    mock_llm.add_tools = Mock()
    mock_llm.bind_tools = Mock(return_value=mock_llm)
    mock_llm.parse_response_content = Mock(return_value="Test response content")
    
    return mock_llm


@pytest.fixture
def mock_test_tool():
    """Mock test tool for testing."""
    mock_tool = Mock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool for testing purposes"
    mock_tool.invoke.return_value = "✅ TestTool echoes: test message"
    mock_tool._run.return_value = "✅ TestTool echoes: test message"
    return mock_tool


@pytest.fixture
def mock_chat_agent(mock_openapi_chat_model, mock_test_tool):
    """Mock chat agent for testing."""
    with patch("ai_core.agent.chat_agent.ChatAgent") as mock_agent_class:
        mock_agent = Mock()
        mock_agent.llm = mock_openapi_chat_model
        mock_agent.tools = [mock_test_tool]
        mock_agent.system_message = "Test system message"
        mock_agent.conversation_history = []
        mock_agent.name = "test_chat_agent"
        mock_agent.description = "Test chat agent"
        mock_agent.tool_call_count = 0
        mock_agent.successful_tool_calls = 0
        mock_agent.rag_engine = None
        
        # Mock methods
        mock_agent.process_message.return_value = "Test agent response"
        mock_agent.process_message_async = AsyncMock(return_value="Test agent response")
        mock_agent.add_message_to_history = Mock()
        mock_agent.validate_message.return_value = "test message"
        mock_agent.get_conversation_context.return_value = [HumanMessage(content="test")]
        mock_agent.format_tool_descriptions.return_value = "Available tools:\n- test_tool: A test tool for testing purposes"
        mock_agent.get_stats.return_value = {
            "name": "test_chat_agent",
            "conversation_length": 0,
            "tools_available": 1,
            "tool_names": ["test_tool"],
            "tool_calls_attempted": 0,
            "successful_tool_calls": 0,
            "tool_success_rate": 0
        }
        
        mock_agent_class.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for testing."""
    mock_rag = Mock()
    mock_rag.retrieve_relevant_documents.return_value = "Test relevant documents"
    mock_rag.retrieve_relevant_documents_async = AsyncMock(return_value="Test relevant documents")
    return mock_rag


@pytest.fixture
def test_client(mock_settings, mock_openapi_chat_model, mock_chat_agent):
    """FastAPI test client with mocked dependencies."""
    with patch("ai_core.config.settings", mock_settings), \
         patch("ai_core.llm.OpenAPIChatModel.OpenAPIChatModel", return_value=mock_openapi_chat_model), \
         patch("ai_core.agent.ChatAgent", return_value=mock_chat_agent), \
         patch("ai_core.tools.TestTool"):
        
        from chives_backend.main import app
        client = TestClient(app)
        yield client


@pytest.fixture
def sample_messages():
    """Sample message data for testing."""
    return [
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm doing well, thank you!"),
        HumanMessage(content="Can you help me with a test?"),
    ]


@pytest.fixture
def sample_tool_call():
    """Sample tool call data for testing."""
    return {
        "name": "test_tool",
        "args": {"message": "test message"}
    }


@pytest.fixture
def mock_requests():
    """Mock requests module for API testing."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "generations": [
                {
                    "content": "Test LLM response",
                    "tool_invocations": []
                }
            ]
        }
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def temp_chroma_dir():
    """Temporary directory for ChromaDB testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


# Async test configuration
pytest_plugins = ("pytest_asyncio",)

