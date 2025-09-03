"""Test tool implementation for demonstration purposes."""

from typing import Any, Type

from pydantic import BaseModel, Field

try:
    from .base_tool import BaseTool, BaseToolArgs
except ImportError:
    # For direct execution
    from base_tool import BaseTool, BaseToolArgs


class TestToolArgs(BaseToolArgs):
    """Arguments for the test tool."""

    message: str = Field(
        description="The message to echo back",
        examples=["Hello world!", "Testing the tool framework"],
    )


class TestTool(BaseTool):
    """A simple test tool that echoes back a message."""

    name: str = "test_tool"
    description: str = "A simple tool that takes a string message and prints it back. Useful for testing the tool framework."

    def _get_args_schema(self) -> Type[BaseModel]:
        """Get the arguments schema for this tool."""
        return TestToolArgs

    def _run(self, message: str, **kwargs: Any) -> str:
        """Execute the tool logic.

        Args:
            message: The message to echo back
            **kwargs: Additional keyword arguments

        Returns:
            The echoed message with a prefix
        """
        # Print to console for debugging
        print(f"🔧 TestTool received: {message}")

        # Echo back the message with some formatting
        response = f"✅ TestTool echoes: {message}"

        print(f"🔧 TestTool responding: {response}")

        return response


# Example usage function
def create_test_tool() -> TestTool:
    """Factory function to create a test tool instance."""
    return TestTool()


# Example of how to use the tool directly
if __name__ == "__main__":
    # Create the tool
    tool = create_test_tool()

    # Display tool info
    print(f"🔧 Tool name: {tool.name}")
    print(f"📝 Description: {tool.description}")
    print()

    # Test direct execution
    print("🚀 Testing direct execution:")
    result = tool._run("Hello from test tool!")
    print(f"Result: {result}")
    print()

    # Test LangChain-style invocation
    print("🚀 Testing LangChain invocation:")
    try:
        langchain_result = tool.invoke({"message": "Hello from LangChain!"})
        print(f"LangChain result: {langchain_result}")
    except Exception as e:
        print(f"❌ LangChain invocation error: {e}")

    print("\n✅ TestTool demonstration complete!")
