import time
import uuid
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import run_orchestrator
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Multi-Agent Orchestration API starting...")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Multi-Agent Orchestration API",
    description="LangGraph-powered multi-agent system with Researcher, Analyst, and Writer agents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    task: str
    session_id: str | None = None


class TaskResponse(BaseModel):
    session_id: str
    task: str
    result: str
    iterations: int
    latency_ms: float


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "multi-agent-orchestration"}


@app.post("/run", response_model=TaskResponse)
def run_task(request: TaskRequest):
    """
    Submit a task to the multi-agent pipeline.
    Flows through: Researcher → Analyst → Writer.
    """
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    start = time.perf_counter()
    try:
        result = run_orchestrator(task=request.task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")
    latency_ms = (time.perf_counter() - start) * 1000

    return TaskResponse(
        session_id=session_id,
        task=request.task,
        result=result["final_output"],
        iterations=result["iterations"],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/agents")
def list_agents():
    return {
        "agents": [
            {
                "name": "Researcher",
                "role": "Gathers information using search and summarization tools.",
                "tools": ["search_tool", "summarizer_tool"],
            },
            {
                "name": "Analyst",
                "role": "Synthesizes research and extracts key insights.",
                "tools": ["calculator_tool", "summarizer_tool"],
            },
            {
                "name": "Writer",
                "role": "Produces the final response for the user.",
                "tools": ["summarizer_tool"],
            },
        ]
    }
