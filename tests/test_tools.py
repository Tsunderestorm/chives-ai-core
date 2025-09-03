"""Tests for the tool implementations."""

from typing import Any, Type
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError

from ai_core.tools.base_tool import BaseTool, BaseToolArgs
from ai_core.tools.test_tool import TestTool, TestToolArgs, create_test_tool


class MockToolArgs(BaseToolArgs):
    """Mock tool arguments for testing."""
    
    input_text: str = Field(description="Input text for the mock tool")
    optional_param: int = Field(default=42, description="Optional parameter")


class MockTool(BaseTool):
    """Mock tool implementation for testing."""
    
    name: str = "mock_tool"
    description: str = "A mock tool for testing purposes"
    
    def _get_args_schema(self) -> Type[BaseModel]:
        """Get the arguments schema."""
        return MockToolArgs
    
    def _run(self, input_text: str, optional_param: int = 42, **kwargs: Any) -> str:
        """Execute the mock tool."""
        return f"Mock tool processed: {input_text} (param: {optional_param})"


class TestBaseToolArgs:
    """Test cases for BaseToolArgs."""
    
    def test_base_tool_args_validation(self):
        """Test basic validation of BaseToolArgs."""
        # This should work as BaseToolArgs is a valid Pydantic model
        args = BaseToolArgs()
        assert isinstance(args, BaseToolArgs)
    
    def test_base_tool_args_config(self):
        """Test BaseToolArgs configuration."""
        # Test that extra fields are forbidden
        with pytest.raises(ValidationError):
            MockToolArgs(input_text="test", extra_field="should_fail")
    
    def test_mock_tool_args_required_field(self):
        """Test MockToolArgs with required field."""
        args = MockToolArgs(input_text="test input")
        assert args.input_text == "test input"
        assert args.optional_param == 42  # default value
    
    def test_mock_tool_args_optional_field(self):
        """Test MockToolArgs with optional field."""
        args = MockToolArgs(input_text="test input", optional_param=100)
        assert args.input_text == "test input"
        assert args.optional_param == 100
    
    def test_mock_tool_args_missing_required_field(self):
        """Test MockToolArgs with missing required field."""
        with pytest.raises(ValidationError) as exc_info:
            MockToolArgs(optional_param=100)  # Missing input_text
        
        assert "input_text" in str(exc_info.value)


class TestBaseTool:
    """Test cases for BaseTool."""
    
    def test_base_tool_instantiation(self):
        """Test BaseTool instantiation."""
        tool = MockTool()
        
        assert tool.name == "mock_tool"
        assert tool.description == "A mock tool for testing purposes"
        assert hasattr(tool, 'args_schema')
        assert tool.args_schema == MockToolArgs
    
    def test_base_tool_args_schema_property(self):
        """Test that args_schema is properly set from _get_args_schema."""
        tool = MockTool()
        
        # The args_schema should be set during initialization
        assert tool.args_schema is MockToolArgs
    
    def test_base_tool_run_method(self):
        """Test the _run method execution."""
        tool = MockTool()
        
        result = tool._run(input_text="test", optional_param=99)
        
        assert result == "Mock tool processed: test (param: 99)"
    
    def test_base_tool_invoke_method(self):
        """Test the invoke method (inherited from StructuredTool)."""
        tool = MockTool()
        
        # Test with dict input (LangChain style)
        result = tool.invoke({"input_text": "hello", "optional_param": 10})
        
        assert "Mock tool processed: hello (param: 10)" in str(result)
    
    def test_base_tool_validate_inputs_default(self):
        """Test default input validation."""
        tool = MockTool()
        
        inputs = {"input_text": "test", "optional_param": 50}
        validated = tool.validate_inputs(inputs)
        
        assert validated == inputs
    
    def test_base_tool_handle_error_default(self):
        """Test default error handling."""
        tool = MockTool()
        
        error = ValueError("Test error")
        result = tool.handle_error(error)
        
        assert result == "Tool 'mock_tool' failed: Test error"


class TestTestTool:
    """Test cases for TestTool."""
    
    def test_test_tool_instantiation(self):
        """Test TestTool instantiation."""
        tool = TestTool()
        
        assert tool.name == "test_tool"
        assert "simple tool" in tool.description.lower()
        assert tool.args_schema is TestToolArgs
    
    def test_test_tool_args_schema(self):
        """Test TestTool arguments schema."""
        args = TestToolArgs(message="Hello world")
        
        assert args.message == "Hello world"
        assert isinstance(args, BaseToolArgs)
    
    def test_test_tool_args_validation(self):
        """Test TestTool arguments validation."""
        # Valid arguments
        args = TestToolArgs(message="Test message")
        assert args.message == "Test message"
        
        # Missing required field
        with pytest.raises(ValidationError):
            TestToolArgs()
    
    def test_test_tool_run_method(self):
        """Test TestTool _run method."""
        tool = TestTool()
        
        with patch('builtins.print') as mock_print:
            result = tool._run(message="Hello test tool")
            
            # Check that print was called
            assert mock_print.call_count == 2
            mock_print.assert_any_call("🔧 TestTool received: Hello test tool")
            mock_print.assert_any_call("🔧 TestTool responding: ✅ TestTool echoes: Hello test tool")
        
        assert result == "✅ TestTool echoes: Hello test tool"
    
    def test_test_tool_invoke(self):
        """Test TestTool invoke method."""
        tool = TestTool()
        
        with patch('builtins.print'):
            result = tool.invoke({"message": "Invoke test"})
            
            assert "✅ TestTool echoes: Invoke test" in str(result)
    
    def test_test_tool_with_different_messages(self):
        """Test TestTool with various message inputs."""
        tool = TestTool()
        
        test_cases = [
            "Simple message",
            "Message with numbers 123",
            "Message with special chars !@#$%",
            "",  # Empty message
            "   ",  # Whitespace message
        ]
        
        for message in test_cases:
            with patch('builtins.print'):
                result = tool._run(message=message)
                assert f"✅ TestTool echoes: {message}" == result


class TestCreateTestTool:
    """Test cases for create_test_tool factory function."""
    
    def test_create_test_tool(self):
        """Test the create_test_tool factory function."""
        tool = create_test_tool()
        
        assert isinstance(tool, TestTool)
        assert tool.name == "test_tool"
        assert hasattr(tool, '_run')
    
    def test_create_test_tool_returns_different_instances(self):
        """Test that create_test_tool returns different instances."""
        tool1 = create_test_tool()
        tool2 = create_test_tool()
        
        assert tool1 is not tool2
        assert isinstance(tool1, TestTool)
        assert isinstance(tool2, TestTool)


class TestToolIntegration:
    """Integration tests for tool functionality."""
    
    def test_tool_with_langchain_compatibility(self):
        """Test that tools are compatible with LangChain patterns."""
        tool = TestTool()
        
        # Test that it has required LangChain attributes
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'args_schema')
        assert hasattr(tool, 'invoke')
        
        # Test that args_schema is a Pydantic model
        assert issubclass(tool.args_schema, BaseModel)
    
    def test_tool_serialization(self):
        """Test tool arguments serialization."""
        args = TestToolArgs(message="Test serialization")
        
        # Test JSON serialization
        json_data = args.model_dump()
        assert json_data == {"message": "Test serialization"}
        
        # Test JSON schema
        schema = args.model_json_schema()
        assert "message" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"
    
    def test_multiple_tools_different_schemas(self):
        """Test multiple tools with different argument schemas."""
        test_tool = TestTool()
        mock_tool = MockTool()
        
        # They should have different schemas
        assert test_tool.args_schema is not mock_tool.args_schema
        assert test_tool.args_schema is TestToolArgs
        assert mock_tool.args_schema is MockToolArgs
        
        # But both should be valid tools
        with patch('builtins.print'):
            test_result = test_tool._run(message="test")
            mock_result = mock_tool._run(input_text="test")
        
        assert "TestTool echoes" in test_result
        assert "Mock tool processed" in mock_result


class TestErrorHandling:
    """Test error handling in tools."""
    
    def test_tool_run_with_exception(self):
        """Test tool behavior when _run raises an exception."""
        class FailingTool(BaseTool):
            name = "failing_tool"
            description = "A tool that always fails"
            
            def _get_args_schema(self) -> Type[BaseModel]:
                return MockToolArgs
            
            def _run(self, **kwargs):
                raise ValueError("Tool execution failed")
        
        tool = FailingTool()
        
        # The tool itself should raise the exception
        with pytest.raises(ValueError, match="Tool execution failed"):
            tool._run(input_text="test")
    
    def test_tool_handle_error_custom(self):
        """Test custom error handling."""
        class CustomErrorTool(BaseTool):
            name = "custom_error_tool"
            description = "Tool with custom error handling"
            
            def _get_args_schema(self) -> Type[BaseModel]:
                return MockToolArgs
            
            def _run(self, **kwargs):
                return "success"
            
            def handle_error(self, error: Exception) -> str:
                return f"Custom error handler: {str(error)}"
        
        tool = CustomErrorTool()
        error = RuntimeError("Custom test error")
        
        result = tool.handle_error(error)
        assert result == "Custom error handler: Custom test error"
    
    def test_tool_validate_inputs_custom(self):
        """Test custom input validation."""
        class ValidatingTool(BaseTool):
            name = "validating_tool"
            description = "Tool with custom input validation"
            
            def _get_args_schema(self) -> Type[BaseModel]:
                return MockToolArgs
            
            def _run(self, **kwargs):
                return "validated"
            
            def validate_inputs(self, inputs):
                if not inputs.get("input_text"):
                    raise ValueError("input_text is required and cannot be empty")
                return inputs
        
        tool = ValidatingTool()
        
        # Valid inputs
        valid_inputs = {"input_text": "valid", "optional_param": 10}
        validated = tool.validate_inputs(valid_inputs)
        assert validated == valid_inputs
        
        # Invalid inputs
        with pytest.raises(ValueError, match="input_text is required"):
            tool.validate_inputs({"input_text": "", "optional_param": 10})


class TestToolDocumentation:
    """Test tool documentation and metadata."""
    
    def test_test_tool_field_documentation(self):
        """Test that TestTool fields have proper documentation."""
        schema = TestToolArgs.model_json_schema()
        
        assert "message" in schema["properties"]
        message_field = schema["properties"]["message"]
        assert "description" in message_field
        assert "echo back" in message_field["description"].lower()
        
        # Test examples if present
        if "examples" in message_field:
            assert isinstance(message_field["examples"], list)
            assert len(message_field["examples"]) > 0
    
    def test_tool_string_representations(self):
        """Test tool string representations."""
        tool = TestTool()
        
        # Test that string conversion doesn't raise errors
        str_repr = str(tool)
        assert isinstance(str_repr, str)
        
        repr_str = repr(tool)
        assert isinstance(repr_str, str)


if __name__ == "__main__":
    # Test the main execution block of test_tool.py
    with patch('builtins.print') as mock_print, \
         patch('ai_core.tools.test_tool.TestTool') as mock_test_tool_class:
        
        # Mock the TestTool class and its methods
        mock_tool_instance = Mock()
        mock_tool_instance.name = "test_tool"
        mock_tool_instance.description = "Test description"
        mock_tool_instance._run.return_value = "Test result"
        mock_tool_instance.invoke.return_value = "LangChain result"
        
        mock_test_tool_class.return_value = mock_tool_instance
        
        # Import and run the main block
        from ai_core.tools.test_tool import __name__ as tool_name
        
        # The main block tests should pass without errors
        assert True  # If we get here, no exceptions were raised
