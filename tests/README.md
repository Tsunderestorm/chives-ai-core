# Chives Backend Test Suite

This directory contains comprehensive tests for the chives-backend application components.

## 🧪 Test Structure

```
tests/
├── __init__.py                     # Package initialization
├── conftest.py                     # Pytest configuration and fixtures
├── run_tests.py                    # Interactive test runner
├── test_main.py                    # FastAPI application tests
├── test_config.py                  # Configuration and settings tests
├── test_openapi_chat_model.py      # LLM integration tests
├── test_chat_agent.py              # Chat agent functionality tests
├── test_tools.py                   # Tool implementation tests
├── test_chunking.py                # Chunking strategy tests
└── README.md                       # This file
```

## 🚀 Running Tests

### Option 1: Direct Poetry Commands
```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=chives_backend --cov=ai_core

# Run specific test file
poetry run pytest tests/test_main.py -v

# Run with HTML coverage report
poetry run pytest tests/ --cov=chives_backend --cov=ai_core --cov-report=html
```

### Option 2: Direct Pytest (if you have pytest installed)
```bash
# Set required environment variables first
export OPENAPI_LLM_URL="http://test-api.com"
export OPENAPI_LLM_API_KEY="test-key"
export OPENAPI_LLM_MODEL="test-model"

pytest tests/ -v
```

### Option 3: Using Test Categories
```bash
# Run only integration tests
poetry run pytest tests/ -m integration

# Run only unit tests (exclude integration)
poetry run pytest tests/ -m "not integration"

# Run fast tests only (exclude slow tests)
poetry run pytest tests/ -m "not slow"
```

## 📋 Test Coverage

### **FastAPI Application** (`test_main.py`)
- ✅ Root endpoint (`/`)
- ✅ Health check endpoint (`/health`) 
- ✅ Generate endpoint (`/generate`)
- ✅ Error handling and validation
- ✅ CORS middleware configuration
- ✅ Async endpoint functionality
- ✅ Application initialization

### **Configuration** (`test_config.py`)
- ✅ Settings validation and defaults
- ✅ Environment variable loading
- ✅ Required field validation
- ✅ Type conversion (int, bool, list)
- ✅ Error handling for missing config
- ✅ Settings singleton behavior

### **LLM Integration** (`test_openapi_chat_model.py`)
- ✅ Model initialization and validation
- ✅ API request/response handling
- ✅ Tool binding and formatting
- ✅ Message conversion (LangChain ↔ OpenAI)
- ✅ Error handling (HTTP, JSON, timeout, connection)
- ✅ Tool invocation with structured schemas

### **Chat Agent** (`test_chat_agent.py`)
- ✅ Agent initialization and configuration
- ✅ Message processing (sync and async)
- ✅ Tool execution and error handling
- ✅ RAG integration and document retrieval
- ✅ Conversation history management
- ✅ Statistics and performance tracking

### **Tool System** (`test_tools.py`)
- ✅ BaseTool abstract class functionality
- ✅ TestTool implementation
- ✅ Argument validation with Pydantic
- ✅ LangChain compatibility
- ✅ Error handling and custom validation
- ✅ Tool serialization and schemas

### **Chunking Strategies** (`test_chunking.py`)
- ✅ Chunk data class and metadata handling
- ✅ ChunkingStrategy abstract base class
- ✅ FixedSizeChunker with overlap and boundary respect
- ✅ TokenBasedFixedSizeChunker with tiktoken integration
- ✅ SemanticChunker with embedding-based splitting
- ✅ SemanticChunkingConfig configuration management
- ✅ Content preservation and metadata consistency
- ✅ Integration testing across strategies

## 🔧 Test Configuration

### Fixtures (`conftest.py`)
The test suite includes comprehensive fixtures for:

- **Mock Settings**: Test configuration with safe defaults
- **Mock LLM**: Mocked OpenAPIChatModel for isolated testing
- **Mock Tools**: Test tools that don't make external calls
- **Mock Chat Agent**: Configured agent with predictable responses
- **FastAPI Test Client**: Fully configured test client
- **Sample Data**: Message history, tool calls, and test payloads

### Environment Setup
Tests automatically set up safe test environment variables and clean up after each test.

### Async Testing
Full support for async testing with `pytest-asyncio` plugin.

## 📊 Coverage Reports

Generate detailed coverage reports:

```bash
# Terminal coverage report
poetry run pytest tests/ --cov=chives_backend --cov=ai_core --cov-report=term

# HTML coverage report
poetry run pytest tests/ --cov=chives_backend --cov=ai_core --cov-report=html

# Both terminal and HTML
poetry run pytest tests/ --cov=chives_backend --cov=ai_core --cov-report=term --cov-report=html
```

HTML reports are generated in the `htmlcov/` directory.

## 🏷️ Test Markers

The test suite uses pytest markers for categorization:

- `integration`: Tests that involve multiple components or external systems
- `slow`: Tests that take more time to execute
- Custom markers can be added in `pyproject.toml`

## 🛠️ Dependencies

Test dependencies are managed in `pyproject.toml`:

- `pytest`: Core testing framework
- `pytest-asyncio`: Async test support
- `pytest-mock`: Enhanced mocking capabilities
- `httpx`: Async HTTP client for FastAPI testing

## 🚨 Common Issues & Solutions

### Environment Variables
If you see validation errors about missing environment variables:
```bash
export OPENAPI_LLM_URL="http://test-api.com"
export OPENAPI_LLM_API_KEY="test-key"
export OPENAPI_LLM_MODEL="test-model"
```

### Poetry Environment
Ensure you're in a poetry environment:
```bash
poetry install
poetry shell
```

### Import Errors
Make sure the source code is importable:
```bash
# Run from project root
cd /path/to/chives-backend
python -c "import chives_backend; import ai_core"
```

## 🧹 Code Quality Integration

The test runner can also run code quality checks:

```bash
# Check code formatting
poetry run black --check src/ tests/

# Check import sorting  
poetry run isort --check-only src/ tests/

# Run linting
poetry run flake8 src/ tests/

# Type checking
poetry run mypy src/
```

## 🎯 Best Practices

1. **Isolation**: Each test is independent and doesn't rely on others
2. **Mocking**: External dependencies are mocked to prevent side effects
3. **Clear Names**: Test names describe what functionality is being tested
4. **Arrange-Act-Assert**: Tests follow a clear structure
5. **Edge Cases**: Tests cover both happy path and error scenarios
6. **Async Support**: Full support for testing async functionality

## 📈 Extending the Test Suite

When adding new features to chives-backend:

1. **Add Tests**: Create tests in the appropriate `test_*.py` file
2. **Update Fixtures**: Add new fixtures to `conftest.py` if needed
3. **Add Markers**: Use appropriate markers for categorization
4. **Document**: Update this README with new test coverage

## 🔄 Continuous Integration

These tests are designed to be CI-friendly:

- No external dependencies (everything is mocked)
- Fast execution (most tests run in milliseconds)
- Comprehensive coverage of core functionality
- Clear pass/fail indicators

For CI systems, run:
```bash
poetry install
poetry run pytest tests/ --cov=chives_backend --cov=ai_core --cov-report=xml
```
