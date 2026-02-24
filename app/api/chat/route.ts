import { NextRequest, NextResponse } from "next/server";
import { VertexAI } from "@google-cloud/vertexai";

// ---------------------------------------------------------------------------
// Environment variables (set these in Vercel project settings or a .env file)
// ---------------------------------------------------------------------------
const GCP_PROJECT = process.env.GCP_PROJECT ?? "";
const GCP_LOCATION = process.env.GCP_LOCATION ?? "us-central1";
const GEMINI_MODEL = process.env.GEMINI_MODEL ?? "gemini-1.5-pro";
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
// POST /api/chat
// Body: { query: string }
// Returns: { answer: string, sources?: object[] }
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

  // ---- Call Vertex AI with RAG retrieval + Gemini generation --------------
  try {
    const vertexAI = new VertexAI({
      project: GCP_PROJECT,
      location: GCP_LOCATION,
    });

    const model = vertexAI.getGenerativeModel({
      model: GEMINI_MODEL,
      // Attach the RAG corpus so Gemini performs grounded retrieval
      tools: [
        {
          retrieval: {
            vertexRagStore: {
              ragResources: [{ ragCorpus: RAG_CORPUS }],
            },
          },
        },
      ],
    });

    const result = await model.generateContent({
      contents: [
        {
          role: "user",
          parts: [{ text: query }],
        },
      ],
    });

    const response = result.response;
    const answer =
      response.candidates?.[0]?.content?.parts?.[0]?.text ?? "(no response)";

    // Extract grounding metadata / sources when available
    const groundingMetadata =
      response.candidates?.[0]?.groundingMetadata ?? null;

    return NextResponse.json(
      { answer, groundingMetadata },
      { status: 200, headers }
    );
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[/api/chat] Vertex AI error:", message);
    return NextResponse.json(
      { error: "Failed to generate a response from the AI model." },
      { status: 502, headers }
    );
  }
}
