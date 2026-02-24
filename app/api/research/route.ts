import { NextRequest, NextResponse } from "next/server";
import { runDeepResearch } from "@/app/lib/agents/orchestrator";

// ---------------------------------------------------------------------------
// Environment variables
// ---------------------------------------------------------------------------
const GCP_PROJECT = process.env.GCP_PROJECT ?? "";
const GCP_LOCATION = process.env.GCP_LOCATION ?? "us-central1";
// Fast model used for intent extraction (low-latency)
const GEN_FAST_MODEL =
  process.env.GEN_FAST_MODEL ?? process.env.GEMINI_MODEL ?? "gemini-2.0-flash";
// Advanced model used for RAG retrieval and synthesis (higher quality)
const GEN_ADVANCED_MODEL =
  process.env.GEN_ADVANCED_MODEL ??
  process.env.GEMINI_MODEL ??
  "gemini-2.5-pro";
// Full resource name: projects/<project>/locations/<loc>/ragCorpora/<id>
const RAG_CORPUS = process.env.RAG_CORPUS ?? "";
// Optional: comma-separated list of allowed origins for CORS
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS ?? "*")
  .split(",")
  .map((o) => o.trim());

// ---------------------------------------------------------------------------
// CORS helpers
// ---------------------------------------------------------------------------
function corsHeaders(origin: string | null): Record<string, string> {
  const allowed =
    ALLOWED_ORIGINS.includes("*") ||
    (origin !== null && ALLOWED_ORIGINS.includes(origin));

  return {
    "Access-Control-Allow-Origin": allowed ? (origin ?? "*") : "",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
  };
}

// ---------------------------------------------------------------------------
// OPTIONS – pre-flight
// ---------------------------------------------------------------------------
export async function OPTIONS(req: NextRequest) {
  const origin = req.headers.get("origin");
  return new NextResponse(null, { status: 204, headers: corsHeaders(origin) });
}

// ---------------------------------------------------------------------------
// POST /api/research
//
// Triggers the full deep research pipeline:
//   1. IntentExtractor  – decomposes query into sub-queries (fast model)
//   2. RagRetriever     – queries RAG corpus per sub-query in parallel
//   3. Synthesizer      – merges all findings into one coherent answer
//
// Body:    { query: string }
// Returns: { answer: string, subQueries: string[], groundingMetadata: any[] }
// ---------------------------------------------------------------------------
export async function POST(req: NextRequest) {
  const origin = req.headers.get("origin");
  const headers = corsHeaders(origin);

  // ---- Validate required env vars ----------------------------------------
  if (!GCP_PROJECT) {
    return NextResponse.json(
      { error: "Server misconfiguration: GCP_PROJECT is not set." },
      { status: 500, headers }
    );
  }
  if (!RAG_CORPUS) {
    return NextResponse.json(
      { error: "Server misconfiguration: RAG_CORPUS is not set." },
      { status: 500, headers }
    );
  }

  // ---- Parse request body -------------------------------------------------
  let query: string;
  try {
    const body = await req.json();
    if (typeof body?.query !== "string" || body.query.trim() === "") {
      return NextResponse.json(
        { error: "Request body must contain a non-empty `query` string." },
        { status: 400, headers }
      );
    }
    query = body.query.trim();
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON in request body." },
      { status: 400, headers }
    );
  }

  // ---- Run the deep research pipeline -------------------------------------
  try {
    const result = await runDeepResearch(query, {
      project: GCP_PROJECT,
      location: GCP_LOCATION,
      fastModel: GEN_FAST_MODEL,
      advancedModel: GEN_ADVANCED_MODEL,
      ragCorpus: RAG_CORPUS,
    });

    return NextResponse.json(result, { status: 200, headers });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[/api/research] Deep research pipeline error:", message);
    return NextResponse.json(
      { error: "Failed to complete the deep research pipeline." },
      { status: 502, headers }
    );
  }
}
