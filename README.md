# training-chatbot-backend

Next.js API backend that powers a RAG-assisted Gemini chatbot. The frontend lives in a separate repo; this project exposes a single HTTP endpoint that any external site can call.

## API endpoint

### `POST /api/chat`

**Request body** (JSON):

```json
{ "query": "What is the onboarding process for new staff?" }
```

**Successful response** (`200 OK`):

```json
{
  "answer": "The onboarding process ...",
  "groundingMetadata": { ... }
}
```

`groundingMetadata` contains source citations returned by the Vertex AI RAG engine (may be `null` if no metadata was returned).

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
| `GEMINI_MODEL` | | Gemini model name (default: `gemini-1.5-pro`) |
| `RAG_CORPUS` | ✅ | Full Vertex AI RAG corpus resource name |
| `ALLOWED_ORIGINS` | | Comma-separated CORS origins (default: `*`) |

`RAG_CORPUS` format: `projects/<PROJECT>/locations/<LOCATION>/ragCorpora/<CORPUS_ID>`

## Local development

```bash
npm install
cp .env.example .env.local  # then fill in your values
npm run dev
```

The API will be available at `http://localhost:3000/api/chat`.

## Deployment

The project is configured for Vercel (`vercel.json`). Push to the linked GitHub repo and Vercel will build and deploy automatically. Set the environment variables in the Vercel project settings.
