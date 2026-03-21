# FastAPI Migration Plan

Migrate the training-chatbot-backend from Next.js (TypeScript ADK) to FastAPI (Python ADK). All SSE events, agent prompts, response formats, and API contracts must be preserved exactly so the frontend requires zero changes.

---

## Current System Summary

- 6 API endpoints (5 JSON, 1 SSE stream)
- 6 agents (intent orchestrator, conversational, quick search, escalation search, deep research pipeline with 4 sub-agents)
- RAG tool via custom FunctionTool wrapping Vertex AI
- Token usage tracking on every agent event and RAG call
- Conversation history support on all routes
- CORS + API key auth + rate limiting middleware

---

## API Contract — MUST NOT CHANGE

The frontend expects these exact request/response shapes. Any deviation breaks the frontend.

### POST /api/intent

**Request:**
```json
{
  "query": "string",
  "conversationHistory": [{ "role": "user"|"assistant", "content": "string" }]  // optional
}
```

**Response:**
```json
{
  "action": "confirm" | "clarify" | "reject" | "chat",
  "enrichedQuery": "string",
  "message": "string",
  "logs": [LogEntry]
}
```

### POST /api/conversational

**Request:**
```json
{
  "query": "string",
  "conversationHistory": []  // optional
}
```

**Response:**
```json
{
  "answer": "string",
  "logs": [LogEntry]
}
```

### POST /api/quick-search

**Request:**
```json
{
  "query": "string",
  "context": "string",           // optional, enriched query from intent
  "conversationHistory": []      // optional
}
```

**Response:**
```json
{
  "answer": "string",
  "logs": [LogEntry]
}
```

### POST /api/quick-search-pro

Same request/response as `/api/quick-search`. Uses Pro model instead of Flash.

### POST /api/search-escalate

**Request:**
```json
{
  "query": "string",
  "previousAnswer": "string",   // required — the Flash answer user rejected
  "context": "string",          // optional
  "conversationHistory": []     // optional
}
```

**Response:**
```json
{
  "answer": "string",
  "logs": [LogEntry]
}
```

### POST /api/research (SSE Stream)

**Request:**
```json
{
  "query": "string",
  "context": "string",          // optional
  "conversationHistory": []     // optional
}
```

**Response:** Server-Sent Events stream with these exact event types:

```
event: log
data: {"agent":"string","message":"string","promptTokens":0,"responseTokens":0,"totalTokens":0,"timestamp":1234,"researcherIndex":0}

event: step
data: {"field":"enrichedQuery"|"researchQuestions"|"answer","value":"string"}

event: researchers_init
data: {"count":5,"labels":["question 1","question 2",...]}

event: researcher_done
data: {"index":0,"label":"question text","value":"findings text"}

event: error
data: {"error":"string","detail":"string"}

event: done
data: {}
```

### LogEntry Shape (used in all JSON responses)

```json
{
  "agent": "string",
  "message": "string",
  "promptTokens": 0,
  "responseTokens": 0,
  "totalTokens": 0,
  "timestamp": 1234567890,
  "researcherIndex": 0          // optional, only for researcher logs
}
```

---

## SSE Event Sequence — MUST BE PRESERVED EXACTLY

The frontend renders a live pipeline UI based on the exact order and content of these events:

```
1. event: log          → intent/query_analyzer logs (with token counts)
2. event: step         → field: "enrichedQuery"  (query analyzer output)
3. event: log          → question_expander logs
4. event: step         → field: "researchQuestions" (5 numbered questions)
5. event: researchers_init → { count: N, labels: [question strings] }
6. [For each researcher, in completion order:]
   a. event: log       → researcher RAG call logs (researcherIndex: N)
   b. event: log       → researcher agent logs (researcherIndex: N)
   c. event: researcher_done → { index: N, label: "question", value: "findings" }
7. event: log          → research_compiler logs
8. event: step         → field: "answer" (final compiled report)
9. event: done         → {}
```

**Critical details:**
- Researcher logs MUST include `researcherIndex` field for the frontend to bucket them per researcher
- Researchers run in parallel — events arrive in completion order, not index order
- RAG tool logs (`retrieve_from_rag`) are merged with agent logs sorted by timestamp
- Noise events (no tokens, no state writes) are filtered out
- `researchers_init` MUST fire before any `researcher_done` events
- `done` event MUST be the last event in the stream
- On error, emit `event: error` with `{ error, detail }` then close

---

## Middleware — ALL routes share these behaviors

### CORS
- Read `ALLOWED_ORIGINS` env var (comma-separated, `*` = allow all)
- Set `Access-Control-Allow-Origin` to request origin if in allowed list
- Set `Access-Control-Allow-Methods: POST, OPTIONS`
- Set `Access-Control-Allow-Headers: Content-Type, Authorization, x-api-key`
- Set `Access-Control-Max-Age: 86400`
- Return 403 if origin not allowed

### API Key Auth
- Read `API_KEY` env var
- Check `x-api-key` header matches
- Return 401 if invalid/missing (when API_KEY is set)

### Rate Limiting
- Sliding window: 20 requests per 10 seconds per process
- Return 429 with `Retry-After: 10` header when exceeded

### Request Validation
- All routes require `query` string in body
- `/api/search-escalate` also requires `previousAnswer` string
- Return 400 on invalid/missing body fields
- Return 500 if `GCP_PROJECT` or `RAG_CORPUS` env vars missing

---

## Phase 1: Project Setup

```
training-chatbot-backend/
├── main.py                    # FastAPI app, CORS, startup
├── requirements.txt
├── Dockerfile
├── .env.example
├── middleware/
│   ├── auth.py                # API key validation
│   └── rate_limiter.py        # Sliding window rate limiter
├── models.py                  # Pydantic request/response models
├── agents/
│   ├── intent_orchestrator.py
│   ├── conversational.py
│   ├── quick_search.py
│   ├── escalation_search.py
│   └── deep_research/
│       ├── pipeline.py        # SequentialAgent
│       ├── query_analyzer.py
│       ├── question_expander.py
│       ├── dynamic_research_squad.py
│       └── runner.py          # streamDeepResearch async generator
└── routes/
    ├── intent.py
    ├── conversational.py
    ├── quick_search.py
    ├── quick_search_pro.py
    ├── search_escalate.py
    └── research.py            # SSE streaming
```

### Dependencies
```
fastapi>=0.115
uvicorn[standard]>=0.30
google-adk>=0.4
google-cloud-aiplatform>=1.70
vertexai>=1.70
pydantic>=2.0
slowapi>=0.2        # optional, or implement custom rate limiter
python-dotenv>=1.0
```

### Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=3001
EXPOSE 3001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"]
```

---

## Phase 2: Pydantic Models (models.py)

```python
from pydantic import BaseModel
from typing import Literal, Optional

class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class LogEntry(BaseModel):
    agent: str
    message: str
    promptTokens: int
    responseTokens: int
    totalTokens: int
    timestamp: int
    researcherIndex: Optional[int] = None

class IntentRequest(BaseModel):
    query: str
    conversationHistory: list[ConversationMessage] = []

class IntentResponse(BaseModel):
    action: Literal["confirm", "clarify", "reject", "chat"]
    enrichedQuery: str
    message: str
    logs: list[LogEntry]

class SearchRequest(BaseModel):
    query: str
    context: Optional[str] = None
    conversationHistory: list[ConversationMessage] = []

class EscalationRequest(BaseModel):
    query: str
    previousAnswer: str
    context: Optional[str] = None
    conversationHistory: list[ConversationMessage] = []

class SearchResponse(BaseModel):
    answer: str
    logs: list[LogEntry]
```

---

## Phase 3: Agent Migration

### Conversation History Helper
```python
def format_conversation_history(history: list[ConversationMessage]) -> str:
    if not history:
        return ""
    lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in history]
    return f"CONVERSATION HISTORY (use this for context — the user may refer to earlier messages):\n" + "\n".join(lines) + "\n\n"
```

### 3a. Intent Orchestrator
- **Source:** `app/lib/agents/intent-orchestrator/agent.ts`
- LlmAgent with `model=GEN_FAST_MODEL`, no tools
- Output key: `intent_result`
- **PRESERVE EXACT INSTRUCTION PROMPT** (the full prompt from lines 41-98 of agent.ts)
- Parse JSON from response, extract `{ action, enrichedQuery, message }`
- Fallback to `{ action: "clarify", enrichedQuery: "", message: "Could you please rephrase..." }` on parse failure

### 3b. Conversational Agent
- **Source:** `app/lib/agents/conversational/agent.ts`
- LlmAgent with `model=GEN_FAST_MODEL`, no tools
- Output key: `chat_answer`
- **PRESERVE EXACT INSTRUCTION PROMPT** (lines 38-54)

### 3c. Quick Search Agent
- **Source:** `app/lib/agents/quick-search/agent.ts`
- LlmAgent with `model=GEN_FAST_MODEL`, RAG tool
- Output key: `quick_answer`
- **PRESERVE EXACT INSTRUCTION PROMPT** (lines 49-68)
- Python ADK has native `VertexAiRagRetrieval` — use it instead of custom FunctionTool
- Token tracking: capture RAG call tokens via callback, merge with agent logs sorted by timestamp
- Context injection: if `context` provided, prepend `INTENT ANALYSIS...` block to user message

### 3d. Escalation Search Agent
- **Source:** `app/lib/agents/escalation-search/agent.ts`
- LlmAgent with `model=GEN_PRO_MODEL`, RAG tool
- Output key: `escalation_answer`
- **PRESERVE EXACT INSTRUCTION PROMPT** (lines 54-75)
- Message format: `{historyPrefix}INTENT ANALYSIS:...\n\nPREVIOUS ANSWER (the user was NOT satisfied with this):\n---\n{previousAnswer}\n---\n\nUSER QUESTION:\n{query}`

### 3e. Deep Research Pipeline

#### SequentialAgent (pipeline.py)
- **Source:** `app/lib/agents/deep-research/agent.ts`
- 4 sub-agents in sequence: query_analyzer → question_expander → dynamic_research_squad → research_compiler
- Uses session state to pass data between agents

#### Query Analyzer
- LlmAgent, `model=fastModel`, output key: `enriched_query`
- **PRESERVE EXACT INSTRUCTION PROMPT** (lines 35-54 of agent.ts)

#### Question Expander
- LlmAgent, `model=advancedModel`, output key: `research_questions`
- **PRESERVE EXACT INSTRUCTION PROMPT** (lines 65-80 of agent.ts)
- Reads `{enriched_query}` from session state via template substitution

#### Dynamic Research Squad (MOST COMPLEX — custom BaseAgent)
- **Source:** `app/lib/agents/deep-research/dynamic-research-squad.ts`
- Custom `BaseAgent` subclass, NOT an LlmAgent
- **MUST PRESERVE:**
  1. Question parsing: try JSON `{ "questions": [...] }` first, fallback to text lines ending with `?`
  2. Hard cap at 10 questions
  3. Fallback: if 0 questions parsed, run single "General Research" researcher
  4. Emit `researcher_count` + `researcher_labels` to state (triggers `researchers_init` SSE event)
  5. Launch ALL researchers in parallel (async tasks)
  6. As each completes (in completion order, NOT index order):
     - Emit researcher logs with `researcherIndex` field
     - Emit `researcher_N` to state (triggers `researcher_done` SSE event)
  7. After all done, combine findings with `---` separator and write to `section_findings_all`
  8. Each researcher: LlmAgent + RAG tool, formulates 3-5 queries, calls RAG for each
  9. **PRESERVE EXACT RESEARCHER INSTRUCTION PROMPT** (lines 132-165 of dynamic-research-squad.ts)

#### Report Compiler
- LlmAgent, `model=reportModel`, output key: `final_report`
- **PRESERVE EXACT INSTRUCTION PROMPT** (lines 100-112 of agent.ts)
- Reads `{enriched_query}`, `{research_questions}`, `{section_findings_all}` from state

#### Runner (runner.py) — SSE Event Generator
- **Source:** `app/lib/agents/deep-research/runner.ts`
- Async generator that yields `PipelineEvent` objects
- **State key → SSE event mapping (MUST PRESERVE):**
  - `enriched_query` → `step` event with `field: "enrichedQuery"`
  - `research_questions` → `step` event with `field: "researchQuestions"`
  - `final_report` → `step` event with `field: "answer"`
  - `researcher_labels` → `researchers_init` event with `{ count, labels }`
  - `researcher_N` → `researcher_done` event with `{ index, label, value }`
  - `researcher_log` → `log` event with `researcherIndex` field
  - `researcher_count` → consumed with `researcher_labels`, not emitted separately
- **Log filtering rules:**
  - Skip events where `author == "user"`
  - Skip noise events (no tokens AND no meaningful state writes)
  - State keys matching `researcher_log|researcher_count|researcher_labels|researcher_\d+` are handled specially, not included in generic log detail

---

## Phase 4: FastAPI Routes

### SSE Streaming Route (research.py)
```python
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json

router = APIRouter()

@router.post("/api/research")
async def research(request: SearchRequest):
    async def event_generator():
        async for event in stream_deep_research(request.query, config):
            if event["type"] == "log":
                yield f"event: log\ndata: {json.dumps(event['data'])}\n\n"
            elif event["type"] == "step":
                yield f"event: step\ndata: {json.dumps(event['data'])}\n\n"
            elif event["type"] == "researchers_init":
                yield f"event: researchers_init\ndata: {json.dumps(event['data'])}\n\n"
            elif event["type"] == "researcher_done":
                yield f"event: researcher_done\ndata: {json.dumps(event['data'])}\n\n"
            elif event["type"] == "error":
                yield f"event: error\ndata: {json.dumps(event['data'])}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            **cors_headers(request),
        }
    )
```

### JSON Routes
All JSON routes follow the same pattern:
1. Validate request body (Pydantic handles this)
2. Run the agent
3. Return `{ answer, logs }` or `{ action, enrichedQuery, message, logs }`

---

## Phase 5: Environment Variables — NO CHANGES

```bash
GCP_PROJECT=capstone-2026-489309
GCP_LOCATION=us-west1
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GEN_FAST_MODEL=gemini-2.5-flash
GEN_ADVANCED_MODEL=gemini-2.5-flash
GEN_REPORT_MODEL=gemini-2.5-pro
RAG_CORPUS=projects/capstone-2026-489309/locations/us-west1/ragCorpora/6917529027641081856
API_KEY=<same key>
ALLOWED_ORIGINS=http://localhost:3000,https://hmso-training.vercel.app,https://hmso-training.ics.hawaii.edu
GOOGLE_APPLICATION_CREDENTIALS_JSON=<same service account JSON>
```

---

## Phase 6: Deployment

- [ ] Update Dockerfile in repo
- [ ] Coolify: same domain path (`/backend`), same port (3001)
- [ ] Same env vars — no changes needed
- [ ] Keep the Traefik `flushinterval=100ms` label for SSE streaming
- [ ] Frontend: ZERO changes — same `BACKEND_URL`, same API contract

---

## Phase 7: Verification Checklist

### Per-route testing:
- [ ] `POST /api/intent` — returns correct `action` for confirm/chat/clarify/reject cases
- [ ] `POST /api/conversational` — returns greeting responses with logs
- [ ] `POST /api/quick-search` — returns RAG-grounded answer with sources and logs
- [ ] `POST /api/quick-search-pro` — same as above but with Pro model
- [ ] `POST /api/search-escalate` — returns deeper answer, receives `previousAnswer` in body
- [ ] `POST /api/research` — SSE stream with all event types in correct order

### SSE verification:
- [ ] `researchers_init` fires with correct count and labels
- [ ] Each `researcher_done` includes correct index, label, and findings
- [ ] Researcher logs include `researcherIndex` field
- [ ] `step` events fire for `enrichedQuery`, `researchQuestions`, and `answer`
- [ ] `done` event is always the last event
- [ ] `error` event fires on pipeline failure with `{ error, detail }`
- [ ] RAG token logs are merged into timeline sorted by timestamp

### Middleware verification:
- [ ] CORS: blocked origin returns 403
- [ ] API key: missing/wrong key returns 401
- [ ] Rate limit: 21st request in 10s returns 429 with `Retry-After: 10`
- [ ] Missing `GCP_PROJECT` returns 500
- [ ] Missing `RAG_CORPUS` returns 500
- [ ] Empty/missing `query` returns 400
- [ ] Missing `previousAnswer` on escalate returns 400

### Frontend integration:
- [ ] Quick search works end-to-end (user types → intent → auto-confirm → Flash answer → satisfaction check)
- [ ] Escalation works (user clicks "Search deeper" → Pro answer appears)
- [ ] Deep research works (live pipeline UI: steps, researchers, final report)
- [ ] Agent log panel shows correct token counts and agent names
- [ ] Conversation history is maintained across turns

---

## Agent Instruction Prompts — PRESERVE VERBATIM

All agent instruction prompts must be copied exactly from the TypeScript source files. Do not paraphrase, summarize, or "improve" them. The prompts are tuned for the specific RAG corpus and user base.

| Agent | Source File | Lines |
|---|---|---|
| Intent Orchestrator | `intent-orchestrator/agent.ts` | 41–98 |
| Conversational | `conversational/agent.ts` | 38–54 |
| Quick Search | `quick-search/agent.ts` | 49–68 |
| Escalation Search | `escalation-search/agent.ts` | 54–75 |
| Query Analyzer | `deep-research/agent.ts` | 35–54 |
| Question Expander | `deep-research/agent.ts` | 65–80 |
| Researcher | `deep-research/dynamic-research-squad.ts` | 132–165 |
| Report Compiler | `deep-research/agent.ts` | 100–112 |
