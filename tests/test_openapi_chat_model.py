"""Tests for the OpenAPIChatModel."""

import json
import os
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ai_core.llm.OpenAPIChatModel import OpenAPIChatModel


class MockTool(BaseTool):
    """Mock tool for testing."""
    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, input_text: str) -> str:
        return f"Mock response: {input_text}"

    class ArgsSchema(BaseModel):
        input_text: str = Field(description="Input text for the tool")

    args_schema = ArgsSchema


class TestOpenAPIChatModel:
    """Test cases for OpenAPIChatModel."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.test_env = {
            "OPENAPI_LLM_URL": "https://test-api.example.com",
            "OPENAPI_LLM_API_KEY": "test-api-key-123",
            "OPENAPI_LLM_MODEL": "test-gpt-model",
        }

    def test_initialization_with_parameters(self):
        """Test model initialization with explicit parameters."""
        model = OpenAPIChatModel(
            openapi_url="https://api.example.com",
            api_key="test-key",
            model="gpt-4",
        )
        
        assert model.openapi_url == "https://api.example.com"
        assert model.api_key == "test-key"
        assert model.model == "gpt-4"
        assert model.tools == []

    def test_initialization_with_env_variables(self):
        """Test model initialization with environment variables."""
        with patch.dict(os.environ, self.test_env, clear=True):
            model = OpenAPIChatModel()
            
            assert model.openapi_url == self.test_env["OPENAPI_LLM_URL"]
            assert model.api_key == self.test_env["OPENAPI_LLM_API_KEY"]
            assert model.model == self.test_env["OPENAPI_LLM_MODEL"]

    def test_initialization_parameters_override_env(self):
        """Test that explicit parameters override environment variables."""
        with patch.dict(os.environ, self.test_env, clear=True):
            model = OpenAPIChatModel(
                openapi_url="https://override.example.com",
                api_key="override-key",
                model="override-model",
            )
            
            assert model.openapi_url == "https://override.example.com"
            assert model.api_key == "override-key"
            assert model.model == "override-model"

    def test_initialization_missing_url_raises_error(self):
        """Test that missing URL raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="openapi_url is required"):
                OpenAPIChatModel(api_key="test", model="test")

    def test_initialization_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="api_key is required"):
                OpenAPIChatModel(openapi_url="https://test.com", model="test")

    def test_initialization_missing_model_raises_error(self):
        """Test that missing model raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="model is required"):
                OpenAPIChatModel(openapi_url="https://test.com", api_key="test")

    def test_llm_type_property(self):
        """Test the _llm_type property."""
        with patch.dict(os.environ, self.test_env, clear=True):
            model = OpenAPIChatModel()
            assert model._llm_type == "openapi_chat"

    def test_add_tool(self):
        """Test adding a single tool."""
        with patch.dict(os.environ, self.test_env, clear=True):
            model = OpenAPIChatModel()
            tool = MockTool()
            
            model.add_tool(tool)
            
            assert len(model.tools) == 1
            assert model.tools[0] is tool

    def test_add_tools(self):
        """Test adding multiple tools."""
        with patch.dict(os.environ, self.test_env, clear=True):
            model = OpenAPIChatModel()
            tools = [MockTool(), MockTool()]
            
            model.add_tools(tools)
            
            assert len(model.tools) == 2

    def test_bind_tools(self):
        """Test binding tools to the model."""
        with patch.dict(os.environ, self.test_env, clear=True):
            model = OpenAPIChatModel()
            tools = [MockTool()]
            
            result = model.bind_tools(tools)
            
            assert result is model  # Should return self
            assert len(model.tools) == 1

    @patch('requests.post')
    def test_generate_successful_request(self, mock_post):
        """Test successful API request."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generations": [
                    {
                        "content": "Test response content",
                        "tool_invocations": []
                    }
                ]
            }
            mock_post.return_value = mock_response
            
            model = OpenAPIChatModel()
            messages = [HumanMessage(content="Test message")]
            
            result = model._generate(messages)
            
            assert isinstance(result, ChatResult)
            assert len(result.generations) == 1
            assert isinstance(result.generations[0].message, AIMessage)
            assert result.generations[0].message.content == "Test response content"

    @patch('requests.post')
    def test_generate_with_tools(self, mock_post):
        """Test API request with tools."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Mock successful response with tool invocations
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generations": [
                    {
                        "content": "Test response with tools",
                        "tool_invocations": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "mock_tool",
                                    "arguments": '{"input_text": "test"}'
                                }
                            }
                        ]
                    }
                ]
            }
            mock_post.return_value = mock_response
            
            model = OpenAPIChatModel()
            tool = MockTool()
            model.add_tool(tool)
            
            messages = [HumanMessage(content="Use the tool")]
            result = model._generate(messages, tools=[tool])
            
            # Verify the request payload includes tools
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            
            assert 'tools' in payload['payload']
            assert len(payload['payload']['tools']) == 1
            assert payload['payload']['tools'][0]['function']['name'] == 'mock_tool'

    @patch('requests.post')
    def test_generate_http_error(self, mock_post):
        """Test handling of HTTP errors."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Mock HTTP error response
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response
            
            model = OpenAPIChatModel()
            messages = [HumanMessage(content="Test message")]
            
            result = model._generate(messages)
            
            assert isinstance(result, ChatResult)
            assert "HTTP Error 500" in result.generations[0].message.content

    @patch('requests.post')
    def test_generate_json_parse_error(self, mock_post):
        """Test handling of JSON parsing errors."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Mock response with invalid JSON
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "Invalid JSON response"
            mock_post.return_value = mock_response
            
            model = OpenAPIChatModel()
            messages = [HumanMessage(content="Test message")]
            
            result = model._generate(messages)
            
            assert isinstance(result, ChatResult)
            assert "JSON Parse Error" in result.generations[0].message.content

    @patch('requests.post')
    def test_generate_timeout_error(self, mock_post):
        """Test handling of timeout errors."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Mock timeout error
            mock_post.side_effect = requests.exceptions.Timeout()
            
            model = OpenAPIChatModel()
            messages = [HumanMessage(content="Test message")]
            
            result = model._generate(messages)
            
            assert isinstance(result, ChatResult)
            assert "Request timed out" in result.generations[0].message.content

    @patch('requests.post')
    def test_generate_connection_error(self, mock_post):
        """Test handling of connection errors."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Mock connection error
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            model = OpenAPIChatModel()
            messages = [HumanMessage(content="Test message")]
            
            result = model._generate(messages)
            
            assert isinstance(result, ChatResult)
            assert "Connection error" in result.generations[0].message.content

    def test_format_tools_with_basetool(self):
        """Test tool formatting with BaseTool objects."""
        with patch.dict(os.environ, self.test_env, clear=True):
            model = OpenAPIChatModel()
            tool = MockTool()
            
            formatted_tools = model._format_tools([tool])
            
            assert len(formatted_tools) == 1
            assert formatted_tools[0]['type'] == 'function'
            assert formatted_tools[0]['function']['name'] == 'mock_tool'
            assert formatted_tools[0]['function']['description'] == 'A mock tool for testing'
            assert 'parameters' in formatted_tools[0]['function']

    def test_format_tools_with_dict(self):
        """Test tool formatting with dictionary objects."""
        with patch.dict(os.environ, self.test_env, clear=True):
            model = OpenAPIChatModel()
            tool_dict = {
                'name': 'test_tool',
                'description': 'A test tool',
                'parameters': {'type': 'object', 'properties': {}}
            }
            
            formatted_tools = model._format_tools([tool_dict])
            
            assert len(formatted_tools) == 1
            assert formatted_tools[0]['function']['name'] == 'test_tool'
            assert formatted_tools[0]['function']['description'] == 'A test tool'

    def test_convert_messages_to_dict_human_message(self):
        """Test message conversion for HumanMessage."""
        messages = [HumanMessage(content="Hello")]
        result = OpenAPIChatModel._convert_messages_to_dict(messages)
        
        assert len(result) == 1
        assert result[0]['role'] == 'user'
        assert result[0]['content'] == 'Hello'

    def test_convert_messages_to_dict_ai_message(self):
        """Test message conversion for AIMessage."""
        messages = [AIMessage(content="Hi there")]
        result = OpenAPIChatModel._convert_messages_to_dict(messages)
        
        assert len(result) == 1
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == 'Hi there'

    def test_convert_messages_to_dict_system_message(self):
        """Test message conversion for SystemMessage."""
        messages = [SystemMessage(content="You are helpful")]
        result = OpenAPIChatModel._convert_messages_to_dict(messages)
        
        assert len(result) == 1
        assert result[0]['role'] == 'system'
        assert result[0]['content'] == 'You are helpful'

    def test_convert_messages_to_dict_multiple_messages(self):
        """Test message conversion for multiple messages."""
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User message"),
            AIMessage(content="Assistant response"),
        ]
        result = OpenAPIChatModel._convert_messages_to_dict(messages)
        
        assert len(result) == 3
        assert result[0]['role'] == 'system'
        assert result[1]['role'] == 'user'
        assert result[2]['role'] == 'assistant'

    def test_parse_response_content_valid_json(self):
        """Test parsing valid response content."""
        raw_content = "{'generations': [{'content': 'Test response'}]}"
        result = OpenAPIChatModel.parse_response_content(raw_content)
        
        assert result == "Test response"

    def test_parse_response_content_invalid_json(self):
        """Test parsing invalid response content."""
        raw_content = "Invalid JSON content"
        result = OpenAPIChatModel.parse_response_content(raw_content)
        
        assert result == "Invalid JSON content"

    def test_parse_response_content_missing_generations(self):
        """Test parsing response content without generations."""
        raw_content = "{'some_other_field': 'value'}"
        result = OpenAPIChatModel.parse_response_content(raw_content)
        
        assert result == "{'some_other_field': 'value'}"

    @patch('requests.post')
    def test_generate_request_payload_structure(self, mock_post):
        """Test that the request payload has the correct structure."""
        with patch.dict(os.environ, self.test_env, clear=True):
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generations": [{"content": "Test response", "tool_invocations": []}]
            }
            mock_post.return_value = mock_response
            
            model = OpenAPIChatModel()
            messages = [HumanMessage(content="Test message")]
            
            model._generate(messages, temperature=0.7)
            
            # Verify the request structure
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # Check URL and headers
            assert call_args[0][0] == self.test_env["OPENAPI_LLM_URL"]
            assert call_args[1]['headers']['api-key'] == self.test_env["OPENAPI_LLM_API_KEY"]
            assert call_args[1]['headers']['Content-Type'] == 'application/json'
            
            # Check payload structure
            payload = call_args[1]['json']
            assert 'payload' in payload
            assert 'generation_settings' in payload['payload']
            assert 'model' in payload['payload']
            assert 'messages' in payload['payload']
            
            # Check specific values
            assert payload['payload']['model'] == self.test_env["OPENAPI_LLM_MODEL"]
            assert payload['payload']['generation_settings']['temperature'] == 0.7
            assert len(payload['payload']['messages']) == 1
            assert payload['payload']['messages'][0]['role'] == 'user'
            assert payload['payload']['messages'][0]['content'] == 'Test message'
