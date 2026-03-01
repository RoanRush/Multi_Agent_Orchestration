# 🤖 Multi AI Agent Orchestration

A **LangGraph-powered multi-agent system** that orchestrates specialized AI agents - Researcher, Analyst, and Writer - to collaboratively solve complex tasks with lower average response time.

## Architecture

```
User Request
     │
     ▼
┌──────────────┐
│  Researcher  │  ── search_tool, summarizer_tool
└──────┬───────┘
       │ research_output
       ▼
┌──────────────┐
│   Analyst    │  ── calculator_tool, summarizer_tool
└──────┬───────┘
       │ analysis_output
       ▼
┌──────────────┐
│    Writer    │  ── summarizer_tool
└──────┬───────┘
       │
       ▼
  Final Response
```

## Features

- **LangGraph orchestration** with typed shared state and conditional routing
- **3 specialized agents** with distinct roles and tool sets
- **Custom tools**: Web search (DuckDuckGo), safe math calculator, LLM summarizer
- **FastAPI backend** with `/run`, `/agents`, and `/health` endpoints
- **Docker-ready** for one-command deployment

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/RoanRush/multi-agent-orchestration.git
cd multi-agent-orchestration

# 2. Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn main:app --reload
```

API will be live at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

## Docker

```bash
docker build -t multi-agent-orchestration .
docker run -p 8000:8000 --env-file .env multi-agent-orchestration
```

## Example Request

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Explain the pros and cons of using microservices vs monolithic architecture"}'
```

```json
{
  "session_id": "abc-123",
  "task": "Explain the pros and cons...",
  "result": "## Microservices vs Monolithic Architecture\n\n...",
  "iterations": 3,
  "latency_ms": 1850
}
```

## API Endpoints

| Method | Endpoint  | Description                            |
|--------|-----------|----------------------------------------|
| GET    | `/health` | Health check                           |
| GET    | `/agents` | List all agents and their tools        |
| POST   | `/run`    | Submit a task to the orchestration pipeline |

## Tech Stack

- **LangGraph** - agent orchestration and state management
- **LangChain** - agent and tool abstractions
- **OpenAI GPT-4o-mini** - LLM backbone
- **FastAPI** - REST API backend
- **DuckDuckGo Search** - real-time web search
- **Docker** - containerized deployment
