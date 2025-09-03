# Chives Agentic Framework

A bare-bones AI-powered agentic framework backend with LLM integration, RAG capabilities, and tool calls built for scalable agent-based applications.


## Features

- **AI Agent Framework** - Modular agent architecture with LangChain integration
- **LLM Integration** - Inherits from Langchain's BaseChatModel
- **RAG (Retrieval-Augmented Generation)** - Automatic document retrieval to enhance LLM responses
- **Vector Search** - ChromaDB integration for semantic document search
- **FastAPI** - Fast web framework for building APIs
- **Docker Support** - Containerized development and deployment
- **Poetry** - Dependency management and packaging
- **Extensible Tools** - Abstract base classes for creating custom agent tools
- **Document Ingestion** - Google Drive integration with semantic chunking

## Project Structure

```
chives-backend/
├── chives-ingestion/            # DOCUMENT INGESTION PIPELINE
│   ├── src/
│   │   └── chives_ingestion/
│   │       ├── implementations/     # Concrete implementations
│   │       │   ├── chunking/        # Text chunking strategies
│   │       │   │   ├── fixed_size_chunker.py
│   │       │   │   └── semantic_chunker.py  # Embedding-based semantic chunking
│   │       │   ├── document_retriever/   # Google Drive integration
│   │       │   ├── document_store/       # MongoDB storage
│   │       │   └── vector_store/         # ChromaDB vector storage
│   │       ├── interfaces/           # Abstract interfaces
│   │       ├── ingestion_pipeline.py # Main pipeline orchestrator
│   │       └── main.py              # CLI interface
│   ├── pyproject.toml
│   └── README.md
├── src/
│   ├── ai_core/                 # MODULAR AI FRAMEWORK
│   │   ├── agent/               # Agent implementations
│   │   │   ├── base_agent.py        # Abstract base agent
│   │   │   ├── chat_agent.py        # Conversational agent with RAG
│   │   │   └── __init__.py
│   │   ├── tools/               # Agent tools
│   │   │   ├── base_tool.py         # Abstract base tool
│   │   │   └── __init__.py
│   │   ├── llm/                 # LLM integrations
│   │   │   ├── OpenAPIChatModel.py  # Custom LLM client
│   │   │   └── __init__.py
│   │   ├── rag/                 # RAG engine
│   │   │   ├── rag_engine.py        # Vector search and document retrieval
│   │   │   └── __init__.py
│   │   ├── config.py            # AI framework configuration
│   │   └── __init__.py
│   └── chives_backend/          # APPLICATION LAYER
│       ├── __init__.py
│       ├── main.py              # FastAPI application with agent endpoints
│       ├── api/                 # API routes
│       │   ├── __init__.py
│       │   └── routes.py        # API router
│       ├── models/              # Data models
│       │   └── __init__.py
│       ├── schemas/             # Pydantic schemas
│       │   └── __init__.py
│       ├── services/            # Business logic
│       │   └── __init__.py
│       └── utils/               # Utility functions
│           └── __init__.py
├── tests/                       # Test suite
├── pyproject.toml               # Poetry configuration
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Development stack
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository with submodules**
   ```bash
   git clone --recursive <repository-url>
   cd chives-backend
   ```

2. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

#### Option 1: Docker Compose (Recommended)

```bash
# Start the application
docker-compose up

# Or run in the background
docker-compose up -d

# To rebuild after code changes
docker-compose up --build

# Stop the services
docker-compose down
```

#### Option 2: Local Development with Poetry

For development with hot-reloading:

```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env with your actual values

# Run the app locally
poetry run python -m uvicorn chives_backend.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Poetry shell
poetry shell
python -m uvicorn chives_backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**The API will be available at:** `http://localhost:8000`

### Testing the Server

Once your server is running, you can test it using these methods:

#### Quick Health Check

```bash
# Test root endpoint
curl http://localhost:8000/
# Should return: {"message":"Welcome to Chives Backend API"}

# Test health check endpoint
curl http://localhost:8000/health
# Should return: {"status":"healthy"}

# Test generate endpoint with RAG
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you tell me about the documents in your knowledge base?"}'
```

#### Web Browser

- **Main API**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health
- **Generate endpoint**: http://localhost:8000/generate
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Core Features

### 🤖 **AI Agent with RAG**

The system automatically enhances LLM responses with relevant document context:

- **Automatic Retrieval**: Searches ChromaDB for relevant documents
- **Context Injection**: Injects retrieved context into LLM prompts
- **Semantic Search**: Uses embeddings for meaning-based document retrieval

### 📚 **Document Ingestion Pipeline**

The `chives-ingestion` submodule provides:

- **Google Drive Integration**: Automatic document retrieval and sync
- **Semantic Chunking**: Embedding-based text chunking for better context
- **Vector Storage**: ChromaDB integration for semantic search
- **Multiple Formats**: Support for PDF, DOCX, text, and Google Workspace documents

### 🔍 **Vector Search & RAG**

- **ChromaDB Backend**: High-performance vector database
- **Semantic Similarity**: Cosine similarity search using sentence transformers
- **Automatic Context**: RAG engine automatically retrieves relevant documents
- **Configurable Thresholds**: Adjustable similarity thresholds for search precision

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Server
DEBUG=true
ALLOWED_HOSTS=["http://localhost:3000"]

# ChromaDB Configuration
CHROMA_COLLECTION_NAME=document_chunks
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_HOST=localhost
CHROMA_PORT=8000

# LLM Configuration
OPENAPI_LLM_API_KEY=your_openai_api_key
OPENAPI_LLM_URL=https://api.openai.com/v1
OPENAPI_LLM_MODEL=gpt-4
```

## API Documentation

Once the server is running, visit:

- **Interactive API docs (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative API docs (ReDoc)**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/api/v1/openapi.json`

## Available Endpoints

### Core System
- `GET /` - Root endpoint (welcome message)
- `GET /health` - Health check endpoint

### AI Agent with RAG
- `POST /generate` - Main agent endpoint with automatic RAG
  - **Request**: `{"message": "your message here"}`
  - **Response**: `{"reply": "agent response with retrieved context"}`

## Development

### Code Quality

The project includes several code quality tools:

```bash
# Format code with black
poetry run black .

# Sort imports with isort
poetry run isort .

# Lint with flake8
poetry run flake8 .

# Type checking with mypy
poetry run mypy src/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=chives_backend

# Run specific test file
poetry run pytest tests/test_main.py
```

### Adding Dependencies

```bash
# Add production dependency
poetry add package_name

# Add development dependency
poetry add --group dev package_name
```

## Document Ingestion

### Setting Up the Ingestion Pipeline

1. **Configure Google Drive credentials** in `chives-ingestion/.env`
2. **Set ChromaDB parameters** for vector storage
3. **Run the ingestion pipeline**:

```bash
cd chives-ingestion
poetry install
poetry run python src/chives_ingestion/main.py
```

### Chunking Strategies

The system supports multiple chunking approaches:

- **Fixed-Size**: Character-based chunking with overlap
- **Token-Based**: Token-aware chunking using tiktoken
- **Semantic**: Embedding-based chunking that respects meaning and topic boundaries

## Deployment

### Docker Production

1. **Build production image**
   ```bash
   docker build -t chives-backend .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 --env-file .env chives-backend
   ```

### Environment-specific Configuration

- Development: Use `docker-compose.yml`
- Production: Use environment variables and external ChromaDB instance

## Contributing

1. Fork the repository
2. Do as you please 
