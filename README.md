# training-chatbot-backend

FastAPI backend for a RAG-assisted multi-agent research chatbot built with Google's [Agent Development Kit (ADK)](https://github.com/google/adk-python). Designed for the Hawaii State House Majority Staff Office to help staff members query internal training documents, policies, and procedures.

## Architecture

All user queries flow through an **Intent Orchestrator** first, which validates relevance and enriches the query before routing to the appropriate agent.

```
User Query
    |
    v
Intent Orchestrator (Flash)
    |
    ├── Conversational (Flash)         — auto-routed for greetings/small talk
    ├── Quick Search (Flash)           — single-agent RAG search
    ├── Quick Search Pro (Pro)         — single-agent RAG search, higher quality
    ├── Search Escalate (Pro)          — re-runs deeper after an unsatisfactory Flash answer
    ├── Quiz Generator (Flash)         — generates multiple-choice quizzes from the corpus
    └── Deep Research (Flash + Pro)    — multi-agent pipeline
            ├── Query Analyzer (Flash)
            ├── Question Expander (Flash)
            ├── Dynamic Research Squad (Flash) — N parallel researchers
            └── Research Compiler (Pro)
```

All routes accept an optional `conversationHistory` parameter for multi-turn context.

## Project structure

```
.
├── Dockerfile                  # Builds & runs the FastAPI app
├── .env / .env.example         # Environment configuration
├── fastapi/
│   ├── main.py                 # FastAPI app entry, middleware, root docs page, /api/warmup
│   ├── config.py               # Centralised env loading (.env + ADC)
│   ├── models.py               # Pydantic request/response models
│   ├── requirements.txt
│   ├── middleware/
│   │   ├── auth.py             # x-api-key gate
│   │   └── rate_limiter.py     # Sliding-window rate limiter
│   ├── routes/
│   │   ├── intent.py           # POST /api/intent
│   │   ├── conversational.py   # POST /api/conversational
│   │   ├── quick_search.py     # POST /api/quick-search
│   │   ├── quick_search_pro.py # POST /api/quick-search-pro
│   │   ├── search_escalate.py  # POST /api/search-escalate
│   │   ├── research.py         # POST /api/research (SSE)
│   │   └── quiz.py             # POST /api/quiz
│   ├── agents/
│   │   ├── intent_orchestrator.py
│   │   ├── conversational.py
│   │   ├── quick_search.py
│   │   ├── escalation_search.py
│   │   ├── quiz_generator.py
│   │   ├── rag_tool.py         # Vertex AI RAG retrieval (raw chunks)
│   │   ├── runner_helper.py    # InMemoryRunner ephemeral session helper
│   │   └── deep_research/
│   │       ├── pipeline.py
│   │       ├── dynamic_research_squad.py
│   │       └── runner.py
│   └── adk_agents/             # Standalone wrappers for `adk web` testing
└── migration/                  # Historical migration notes
```

## Authentication & CORS

Two layers protect every API route:

1. **CORS origin check** — only requests from origins listed in `ALLOWED_ORIGINS` (or with no `Origin` header) are allowed.
2. **API key** — all requests must include a valid `x-api-key` header matching the `API_KEY` env var. The `/`, `/docs`, and `/api/warmup` paths are exempt.

**Example:**
```bash
curl -X POST https://your-host/api/intent \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key-here" \
  -d '{"query": "test"}'
```

The frontend (`training-chatbot-frontend`) uses a server-side proxy that injects `x-api-key` from its own env so the key never reaches the browser.

## API routes

Every POST route accepts an optional `conversationHistory` array:

```json
{
  "query": "what about ethics rules?",
  "conversationHistory": [
    { "role": "user", "content": "tell me about onboarding" },
    { "role": "assistant", "content": "House Majority onboarding covers..." }
  ]
}
```

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | API documentation page (auth-exempt) |
| `/api/warmup` | GET | Lightweight ping; warms ADC + Vertex on first hit (auth-exempt) |
| `/api/intent` | POST | Validates and enriches the query; returns `confirm` / `chat` / `clarify` / `reject` |
| `/api/conversational` | POST | Greetings, small talk, system-capability questions (no RAG) |
| `/api/quick-search` | POST | Single-pass RAG search using the Flash model |
| `/api/quick-search-pro` | POST | Single-pass RAG search using the Pro model |
| `/api/search-escalate` | POST | Re-runs the search deeper (5 sub-queries, Pro model) when the user marks the Flash answer unsatisfactory. Body adds `previousAnswer: string`. |
| `/api/research` | POST (SSE) | Full deep-research pipeline; streams events |
| `/api/quiz` | POST | Generates a structured multiple-choice quiz on a topic |

### Deep research SSE events

| Event | Payload | Description |
|-------|---------|-------------|
| `log` | `{ agent, message, promptTokens, responseTokens, totalTokens, timestamp, researcherIndex? }` | Per-agent / per-tool token usage |
| `step` | `{ field, value }` | Pipeline step completed (`enrichedQuery`, `researchQuestions`, `answer`) |
| `researchers_init` | `{ count, labels }` | Names of parallel researchers |
| `researcher_done` | `{ index, label, value }` | A researcher finished |
| `error` | `{ error, detail }` | Pipeline error |
| `done` | `{}` | Stream complete |

## Agents

| Agent | Model env var | Role |
|-------|---------------|------|
| `intent_orchestrator` | `GEN_FAST_MODEL` | Validates relevance, enriches queries, routes conversational queries |
| `conversational_agent` | `GEN_FAST_MODEL` | Greetings, small talk (no tools) |
| `quick_search_agent` | `GEN_FAST_MODEL` (Flash) or `GEN_PRO_MODEL` (Pro) | Single-pass RAG search |
| `escalation_search_agent` | `GEN_PRO_MODEL` | Deeper re-search after unsatisfactory Flash answer |
| `quiz_generator` | `GEN_FAST_MODEL` | Structured quiz JSON output |
| Deep research pipeline | `GEN_FAST_MODEL` + `GEN_REPORT_MODEL` | Multi-agent: analyzer → expander → parallel researchers → compiler |

Each RAG-using agent follows a three-phase pattern: **plan internally → exactly N retrievals → plan output → answer**, and parses real source references (page numbers, policy IDs, URLs) directly from the retrieved chunk text rather than the corpus filename.

## Environment variables

Copy `.env.example` to `.env` and fill in your values.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GCP_PROJECT` | Yes | — | Google Cloud project ID |
| `GCP_LOCATION` | | `us-west1` | Vertex AI region for **models**. Use `global` for Gemini 3 previews. |
| `GOOGLE_GENAI_USE_VERTEXAI` | | `TRUE` | Required for RAG corpus access |
| `GEN_FAST_MODEL` | | `gemini-2.5-flash` | Fast model (intent, conversational, quick search, query analyzer) |
| `GEN_REPORT_MODEL` | | `gemini-2.5-pro` | Report model (research compiler) |
| `GEN_PRO_MODEL` | | `GEN_REPORT_MODEL` | Pro model (quick-search-pro, escalation) |
| `RAG_CORPUS` | Yes | — | Full Vertex AI RAG corpus resource name |
| `API_KEY` | Yes | — | Secret key for `x-api-key` header validation |
| `ALLOWED_ORIGINS` | | `*` | Comma-separated CORS origins |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | | — | Single-line service-account JSON key, used when ADC isn't available (e.g. in containers) |

`RAG_CORPUS` format: `projects/<PROJECT>/locations/<REGION>/ragCorpora/<CORPUS_ID>`. The RAG tool auto-detects the corpus's region from this URI, so models can run on `GCP_LOCATION=global` while the corpus lives in (e.g.) `us-west1`.

## Local development

```bash
cd fastapi
python3.12 -m venv .venv          # 3.10+ required (3.9 will fail)
source .venv/bin/activate
pip install -r requirements.txt

# Auth: either of the two
gcloud auth application-default login          # uses your user creds
# — or paste the SA JSON into GOOGLE_APPLICATION_CREDENTIALS_JSON in .env

uvicorn main:app --reload --port 3001
```

Open `http://localhost:3001/` for the API docs page.

### Testing individual agents with `adk web`

Each agent has a thin wrapper under `fastapi/adk_agents/` that exposes a `root_agent`. From inside `fastapi/`:

```bash
adk web adk_agents
```

This launches Google's ADK web UI; pick any agent from the dropdown to chat with it in isolation.

## Deployment (Docker)

The root `Dockerfile` builds the FastAPI app:

```bash
docker build -t hmso-training-backend .
docker run --rm -p 3001:3001 --env-file .env hmso-training-backend
```

For Coolify or any container host, point at the repo root and use the existing `Dockerfile`. The container runs `uvicorn main:app` on port 3001.

### Cold starts

Heavy imports (`google-adk`, `vertexai`) make the first request slow on a cold container. Two mitigations are built in:

- **Startup warmup** ([`fastapi/main.py`](fastapi/main.py)) — `@app.on_event("startup")` resolves Google ADC and calls `vertexai.init()` so the first real request doesn't pay that cost.
- **`/api/warmup`** — auth-exempt GET endpoint the frontend can ping on page load to wake the container before the user sends a chat.

If you're seeing cold-start 500s, configure your container host to keep at least one replica warm.

## Token tracking

All agent events and RAG tool calls report token usage. The helper `extract_usage_tokens(event)` in [`fastapi/agents/runner_helper.py`](fastapi/agents/runner_helper.py) safely coerces the various `usage_metadata` fields (some of which can be `None` on tool-only turns) to `int`. Every route returns a `logs` array with one entry per agent/tool call.

## Conversation history

Every route accepts an optional `conversationHistory` array. It's formatted into a text prefix injected into the agent's user message, so each agent sees prior turns as context. The frontend tracks history per-session and forwards it on every request.
