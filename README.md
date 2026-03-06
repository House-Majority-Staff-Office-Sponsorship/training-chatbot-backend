# training-chatbot-backend

Next.js API backend for a RAG-assisted multi-agent research chatbot built with Google's [Agent Development Kit (ADK)](https://github.com/google/adk-node). Designed for the House Majority Staff Office to help staff members query internal training documents, policies, and procedures.

## Architecture

All user queries flow through an **Intent Orchestrator** first, which validates relevance and enriches the query before routing to either a quick search, deep research pipeline, or conversational response.

```
User Query
    |
    v
Intent Orchestrator (Flash)
    |
    ├── Conversational (Flash)     — auto-routed for greetings/small talk
    ├── Quick Search (Flash)       — single-agent RAG search
    ├── Quick Search (Pro)         — single-agent RAG search with pro model
    └── Deep Research (Flash+Pro)  — multi-agent pipeline
            ├── Query Analyzer (Flash)
            ├── Question Expander (Flash)
            ├── Dynamic Research Squad (Flash) — 5 parallel researchers
            └── Research Compiler (Pro)
```

All routes accept an optional `conversationHistory` parameter for multi-turn context.

## Authentication & CORS

All API routes are protected with two layers:

1. **CORS origin check** — Only requests from origins listed in `ALLOWED_ORIGINS` (or with no `Origin` header, i.e. same-origin/server-side) are allowed. Unauthorized origins receive a `403`.
2. **API key** — All cross-origin requests must include a valid `x-api-key` header matching the `API_KEY` environment variable. Missing or invalid keys receive a `401`. Same-origin requests (from the built-in UI) skip this check since the key is passed from the server component.

**Headers required for external consumers:**
```
Content-Type: application/json
x-api-key: <your-api-key>
```

**Example:**
```bash
curl -X POST https://training-chatbot-backend.vercel.app/api/intent \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key-here" \
  -d '{"query": "test"}'
```

The frontend app (`training-chatbot-frontend`) uses a server-side proxy that reads `BACKEND_API_KEY` from its own environment and injects the `x-api-key` header before forwarding requests — the key is never exposed to the browser.

## API Routes

### Conversation History

Every route accepts an optional `conversationHistory` array in the request body. This allows the agents to reference prior turns for context-aware responses.

```json
{
  "query": "what about the ethics rules?",
  "conversationHistory": [
    { "role": "user", "content": "tell me about onboarding" },
    { "role": "assistant", "content": "House Majority onboarding covers..." }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `conversationHistory` | `Array<{ role: "user" \| "assistant", content: string }>` | Optional. Prior conversation turns for context. |

---

### `POST /api/intent`

Gatekeeper that validates and enriches user queries before research begins.

**Request:**
```json
{ "query": "tell me about drafting procedures", "conversationHistory": [] }
```

**Response:**
```json
{
  "action": "confirm",
  "enrichedQuery": "Detailed analysis with House Majority context and sub-questions...",
  "message": "You're asking about legislative drafting procedures. Shall I proceed?",
  "logs": [{ "agent": "intent_orchestrator", "promptTokens": 771, "responseTokens": 218, ... }]
}
```

| Field | Description |
|-------|-------------|
| `action` | `"confirm"` (proceed), `"chat"` (conversational), `"clarify"` (need more info), or `"reject"` (off-topic) |
| `enrichedQuery` | Contextualized query for downstream agents (empty for chat/clarify/reject) |
| `message` | User-facing message (empty for chat — use `/api/conversational` instead) |
| `logs` | Token usage logs from the agent |

---

### `POST /api/conversational`

Simple conversational agent — no RAG search. Handles greetings, small talk, and system capability questions. Auto-routed when the intent orchestrator returns `action: "chat"`.

**Request:**
```json
{ "query": "hello", "conversationHistory": [] }
```

**Response:**
```json
{
  "answer": "Hello! I'm the House Majority Training Assistant. I can help you...",
  "logs": [{ "agent": "conversational_agent", "promptTokens": 200, "responseTokens": 50, ... }]
}
```

---

### `POST /api/quick-search`

Single-pass search using one LlmAgent (Flash) with a RAG tool. Fast and cost-effective.

**Request:**
```json
{ "query": "what are the travel reimbursement rules?", "context": "optional enriched context", "conversationHistory": [] }
```

**Response:**
```json
{
  "answer": "Staff travel reimbursement is governed by...\n\n## Sources\n...",
  "logs": [
    { "agent": "quick_search_agent", "promptTokens": 500, "responseTokens": 30, ... },
    { "agent": "retrieve_from_rag", "promptTokens": 12, "responseTokens": 150, ... }
  ]
}
```

---

### `POST /api/quick-search-pro`

Identical to `/api/quick-search` but uses the Pro model (`gemini-2.5-pro`). More reliable at following instructions and producing higher-quality answers, but slower and more expensive.

Same request/response format as `/api/quick-search`.

---

### `POST /api/research`

Multi-agent deep research pipeline. Returns a **Server-Sent Events (SSE)** stream for real-time progress updates.

**Request:**
```json
{ "query": "tell me about drafting procedures", "context": "optional enriched context", "conversationHistory": [] }
```

**SSE Events:**

| Event | Payload | Description |
|-------|---------|-------------|
| `log` | `{ agent, message, promptTokens, responseTokens, totalTokens, timestamp, researcherIndex? }` | Token usage for each agent/tool call |
| `step` | `{ field, value }` | Pipeline step completed (enrichedQuery, researchQuestions, answer) |
| `researchers_init` | `{ count, labels }` | Number and names of parallel researchers spawned |
| `researcher_done` | `{ index, label, value }` | Individual researcher completed with findings |
| `error` | `{ error, detail }` | Pipeline error |
| `done` | `{}` | Stream complete |

**Pipeline stages:**

1. **Query Analyzer** (Flash) — enriches the raw user query with organizational context
2. **Question Expander** (Flash) — generates exactly 5 targeted research sub-questions
3. **Dynamic Research Squad** (Flash) — spawns one researcher per question, all running in parallel. Each researcher makes 3-5 RAG corpus queries and synthesizes findings.
4. **Research Compiler** (Pro) — compiles all findings into a structured report with executive summary and sources

## Agents

| Agent | Model | Role |
|-------|-------|------|
| `intent_orchestrator` | Flash | Validates relevance, enriches queries, gates access, routes conversational queries |
| `conversational_agent` | Flash | Handles greetings, small talk, and system capability questions (no tools) |
| `quick_search_agent` | Flash or Pro | Single-pass RAG search and answer synthesis |
| `query_analyzer` | Flash | Enriches raw queries with organizational context |
| `question_expander` | Flash | Breaks enriched query into 5 sub-questions |
| `dynamic_research_squad` | Flash | Spawns parallel researchers with RAG tools |
| `research_compiler` | Pro | Compiles all research into final report with sources |

## Project Structure

```
app/
  api/
    intent/              — POST /api/intent (query validation)
    conversational/      — POST /api/conversational (chat responses)
    quick-search/        — POST /api/quick-search (Flash single-pass)
    quick-search-pro/    — POST /api/quick-search-pro (Pro single-pass)
    research/            — POST /api/research (deep research SSE stream)
  lib/
    types.ts              — Shared types (ConversationMessage, formatConversationHistory)
    agents/
      intent-orchestrator/  — Intent validation and query enrichment
      conversational/       — Simple LLM chat agent (no tools)
      quick-search/         — Single-agent RAG search
      deep-research/
        agent.ts              — Pipeline definition (SequentialAgent)
        dynamic-research-squad.ts — Parallel researcher spawner (BaseAgent)
        runner.ts             — Streaming and batch runners
      shared-tools/
        rag-tool.ts           — Vertex AI RAG FunctionTool with token tracking
    rate-limiter.ts         — Sliding-window rate limiter
  page.tsx                — Server component that passes API_KEY to HomeClient
  HomeClient.tsx          — Dev UI with mode tabs and terminal log panel
```

## Environment Variables

Copy `.env.example` to `.env.local` and fill in your values:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GCP_PROJECT` | Yes | | Google Cloud project ID |
| `GCP_LOCATION` | | `us-central1` | Vertex AI region |
| `GOOGLE_GENAI_USE_VERTEXAI` | | `TRUE` | Use Vertex AI backend (required for RAG) |
| `GEN_FAST_MODEL` | | `gemini-2.0-flash` | Flash model for intent, query analysis, researchers |
| `GEN_REPORT_MODEL` | | `gemini-2.5-pro` | Pro model for research compiler and quick-search-pro |
| `RAG_CORPUS` | Yes | | Full Vertex AI RAG corpus resource name |
| `API_KEY` | Yes | | Secret key for `x-api-key` header validation |
| `ALLOWED_ORIGINS` | | `*` | Comma-separated CORS origins |

`RAG_CORPUS` format: `projects/<PROJECT>/locations/<LOCATION>/ragCorpora/<CORPUS_ID>`

## Conversation History

All routes support multi-turn conversations via the `conversationHistory` parameter. This is formatted as a text prefix injected into the agent's user message, so each agent sees prior turns as context.

- The **dev UI** (`page.tsx`) tracks history in-memory and sends it with every API call automatically
- **External consumers** pass the array directly in the request body
- History is per-session — it resets on page reload in the dev UI
- Each successful response appends a user/assistant pair to the history

## Token Tracking

All agents and RAG tool calls report token usage:

- **ADK agent events** — tracked via `event.usageMetadata` (promptTokenCount, candidatesTokenCount, totalTokenCount)
- **RAG tool calls** — tracked via `onTokenUsage` callback from the Vertex AI `generateContent` response inside the FunctionTool
- **Frontend log panel** — terminal-style display showing per-agent and per-researcher consolidated token counts

Each deep research query typically involves:
- 1 intent call
- 1 query analyzer call
- 1 question expander call
- 5 researchers x (3-5 RAG queries + 4 LLM calls each) = ~35 API calls
- 1 research compiler call

## Local Development

```bash
npm install
cp .env.example .env.local  # fill in your values
gcloud auth application-default login  # authenticate for Vertex AI
npm run dev
```

Available at `http://localhost:3000` with a dev UI providing three modes:
- **Quick Search (Flash)** — fast single-pass search
- **Quick Search (Pro)** — higher quality single-pass search
- **Deep Research (Flash + Pro)** — full multi-agent pipeline with streaming progress

Conversational queries (greetings, small talk) are auto-routed by the intent orchestrator regardless of which mode is selected.

## Deployment

Configured for Vercel (`vercel.json`). Push to the linked GitHub repo and Vercel builds automatically. Set environment variables in the Vercel project settings.
