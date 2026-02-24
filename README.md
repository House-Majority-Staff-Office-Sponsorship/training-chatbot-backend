# training-chatbot-backend

Next.js API backend that powers a RAG-assisted Gemini chatbot. The frontend lives in a separate repo; this project exposes HTTP endpoints that any external site can call.

## API endpoints

### `POST /api/research` _(deep research agent)_

Invokes a multi-agent deep research pipeline inspired by Google's Agent Development Kit (ADK) architecture. A single API call drives three cooperating agents:

1. **IntentExtractor** – uses the fast Gemini model to decompose the user query into 2–4 focused sub-queries and key topics.
2. **RagRetriever** – queries the Vertex AI RAG corpus for **each** sub-query in **parallel**, so every angle of the question is addressed with grounded context.
3. **Synthesizer** – uses the advanced Gemini model to merge all individual findings into one coherent, comprehensive answer.

**Request body** (JSON):

```json
{ "query": "What are the rules for staff travel reimbursement?" }
```

**Successful response** (`200 OK`):

```json
{
  "answer": "Staff travel reimbursement is governed by ...",
  "subQueries": [
    "What are the travel reimbursement rules for House staff?",
    "What expenses are eligible for staff travel reimbursement?",
    "How do staff members submit travel reimbursement requests?"
  ],
  "groundingMetadata": [{ ... }, { ... }, { ... }]
}
```

| Field | Description |
|-------|-------------|
| `answer` | Synthesized, comprehensive answer |
| `subQueries` | Sub-queries issued against the RAG corpus |
| `groundingMetadata` | Per-sub-query grounding/source metadata from Vertex AI RAG (entry may be `null`) |

**Error responses**:

| Status | Meaning |
|--------|---------|
| `400` | Missing or empty `query` field |
| `500` | Required environment variable not set |
| `502` | Upstream Vertex AI / Gemini error |

CORS pre-flight requests (`OPTIONS`) are handled automatically.

## Environment variables

Copy `.env.example` to `.env.local` (local dev) or add them in the Vercel project dashboard:

| Variable | Required | Description |
|----------|----------|-------------|
| `GCP_PROJECT` | ✅ | Google Cloud project ID |
| `GCP_LOCATION` | | Vertex AI region (default: `us-central1`) |
| `GEN_FAST_MODEL` | | Fast model for IntentExtractor agent (default: `gemini-2.0-flash`) |
| `GEN_ADVANCED_MODEL` | | Advanced model for RagRetriever & Synthesizer agents (default: `gemini-2.5-pro`) |
| `RAG_CORPUS` | ✅ | Full Vertex AI RAG corpus resource name |
| `ALLOWED_ORIGINS` | | Comma-separated CORS origins (default: `*`) |

`RAG_CORPUS` format: `projects/<PROJECT>/locations/<LOCATION>/ragCorpora/<CORPUS_ID>`

## Local development

```bash
npm install
cp .env.example .env.local  # then fill in your values
npm run dev
```

Both endpoints will be available at:
- `http://localhost:3000/api/research` (deep research agent)

## Deployment

The project is configured for Vercel (`vercel.json`). Push to the linked GitHub repo and Vercel will build and deploy automatically. Set the environment variables in the Vercel project settings.
