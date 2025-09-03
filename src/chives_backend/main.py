"""Main FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# API routes removed - focusing on agentic framework
from ai_core.config import settings
from ai_core.llm.OpenAPIChatModel import OpenAPIChatModel
from ai_core.agent import ChatAgent
from ai_core.tools import TestTool

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Agentic Framework Backend - AI Agent Service with LLM Integration",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Initialize LLM
llm = OpenAPIChatModel(
    openapi_url=settings.OPENAPI_LLM_URL,
    api_key=settings.OPENAPI_LLM_API_KEY,
    model=settings.OPENAPI_LLM_MODEL,
)

# Add tools to the LLM
test_tool = TestTool()
llm.add_tool(test_tool)

chat_agent = ChatAgent(
    llm=llm,
    tools=[test_tool],
    system_message="You are a helpful AI assistant with access to tools. You have access to a test_tool that can echo messages back. Use the test_tool when users ask you to echo, repeat, or test something. Provide clear, accurate, and helpful responses to user questions.",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes removed - agent endpoints defined directly below


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to Chives Backend API"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/generate")
async def generate(request: Request) -> dict[str, str]:
    """Generate response using the chat agent."""
    body = await request.json()

    # Get the message from the request
    message_content = body.get("message", "")

    if not message_content.strip():
        return {"reply": "Please provide a message to generate a response."}

    try:
        # Use the async version of the chat agent to process the message
        response = await chat_agent.process_message_async(message_content)

        print(f"User: {message_content}")
        print(f"Agent: {response}")

        return {"reply": response}

    except Exception as e:
        print(f"Error in generate endpoint: {e}")
        return {"reply": f"Sorry, I encountered an error processing your message: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "chives_backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
