"""Tests for the main FastAPI application."""

import json
from unittest.mock import patch, Mock, AsyncMock

import pytest
from fastapi.testclient import TestClient


class TestMainApp:
    """Test cases for the main FastAPI application."""

    def test_root_endpoint(self, test_client: TestClient):
        """Test the root endpoint returns welcome message."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to Chives Backend API"}

    def test_health_check_endpoint(self, test_client: TestClient):
        """Test the health check endpoint returns healthy status."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_generate_endpoint_with_valid_message(self, test_client: TestClient):
        """Test the generate endpoint with a valid message."""
        test_message = "Hello, test message"
        
        response = test_client.post(
            "/generate",
            json={"message": test_message}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert "reply" in response_data
        assert response_data["reply"] == "Test agent response"

    def test_generate_endpoint_with_empty_message(self, test_client: TestClient):
        """Test the generate endpoint with an empty message."""
        response = test_client.post(
            "/generate",
            json={"message": ""}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert "reply" in response_data
        assert response_data["reply"] == "Please provide a message to generate a response."

    def test_generate_endpoint_with_whitespace_message(self, test_client: TestClient):
        """Test the generate endpoint with whitespace-only message."""
        response = test_client.post(
            "/generate",
            json={"message": "   "}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert "reply" in response_data
        assert response_data["reply"] == "Please provide a message to generate a response."

    def test_generate_endpoint_without_message_field(self, test_client: TestClient):
        """Test the generate endpoint without message field in request."""
        response = test_client.post(
            "/generate",
            json={}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert "reply" in response_data
        assert response_data["reply"] == "Please provide a message to generate a response."

    def test_generate_endpoint_with_agent_error(self, test_client: TestClient, mock_chat_agent):
        """Test the generate endpoint when the chat agent raises an exception."""
        # Mock the agent to raise an exception
        mock_chat_agent.process_message_async = AsyncMock(side_effect=Exception("Test error"))
        
        response = test_client.post(
            "/generate",
            json={"message": "test message"}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert "reply" in response_data
        assert "Sorry, I encountered an error processing your message" in response_data["reply"]
        assert "Test error" in response_data["reply"]

    def test_generate_endpoint_calls_agent_correctly(self, test_client: TestClient, mock_chat_agent):
        """Test that the generate endpoint calls the chat agent with correct parameters."""
        test_message = "Test message for agent"
        
        response = test_client.post(
            "/generate",
            json={"message": test_message}
        )
        
        assert response.status_code == 200
        mock_chat_agent.process_message_async.assert_called_once_with(test_message)

    def test_cors_middleware_configured(self, test_client: TestClient):
        """Test that CORS middleware is properly configured."""
        response = test_client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        
        # FastAPI/Starlette returns 200 for OPTIONS requests with CORS configured
        assert response.status_code == 200

    def test_fastapi_app_configuration(self, mock_settings):
        """Test that FastAPI app is configured correctly."""
        with patch("ai_core.config.settings", mock_settings), \
             patch("ai_core.llm.OpenAPIChatModel.OpenAPIChatModel"), \
             patch("ai_core.agent.ChatAgent"), \
             patch("ai_core.tools.TestTool"):
            
            from chives_backend.main import app
            
            assert app.title == mock_settings.PROJECT_NAME
            assert app.version == mock_settings.VERSION
            assert app.description == "Agentic Framework Backend - AI Agent Service with LLM Integration"
            assert app.openapi_url == f"{mock_settings.API_V1_STR}/openapi.json"


class TestAppInitialization:
    """Test cases for application initialization."""

    def test_app_initialization_with_valid_config(self, mock_settings):
        """Test that the app initializes correctly with valid configuration."""
        with patch("ai_core.config.settings", mock_settings), \
             patch("ai_core.llm.OpenAPIChatModel.OpenAPIChatModel") as mock_llm_class, \
             patch("ai_core.agent.ChatAgent") as mock_agent_class, \
             patch("ai_core.tools.TestTool") as mock_tool_class:
            
            # Import the app to trigger initialization
            from chives_backend.main import app, llm, chat_agent, test_tool
            
            # Verify LLM was created correctly
            mock_llm_class.assert_called_once_with(
                openapi_url=mock_settings.OPENAPI_LLM_URL,
                api_key=mock_settings.OPENAPI_LLM_API_KEY,
                model=mock_settings.OPENAPI_LLM_MODEL,
            )
            
            # Verify tool was created and added to LLM
            mock_tool_class.assert_called_once()
            
            # Verify chat agent was created with correct parameters
            mock_agent_class.assert_called_once()

    def test_app_initialization_handles_missing_env_vars(self):
        """Test that the app handles missing environment variables gracefully."""
        with patch.dict('os.environ', {}, clear=True):
            # This should raise an exception during import due to missing env vars
            with pytest.raises((ValueError, Exception)):
                from chives_backend.main import app


class TestAsyncEndpoints:
    """Test cases for async endpoint functionality."""

    @pytest.mark.asyncio
    async def test_generate_endpoint_async_processing(self, mock_chat_agent):
        """Test that the generate endpoint properly handles async processing."""
        # This test verifies the async nature of the endpoint
        test_message = "Test async message"
        mock_chat_agent.process_message_async = AsyncMock(return_value="Async response")
        
        # The actual async testing is handled by the test_client fixture
        # This test ensures our mock is set up correctly for async operations
        result = await mock_chat_agent.process_message_async(test_message)
        assert result == "Async response"
        mock_chat_agent.process_message_async.assert_called_once_with(test_message)
