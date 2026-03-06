import "@/app/lib/gcp-auth";
import { NextRequest, NextResponse } from "next/server";
import { runIntentOrchestrator } from "@/app/lib/agents/intent-orchestrator/agent";
import { createRateLimiter } from "@/app/lib/rate-limiter";
import { type ConversationMessage } from "@/app/lib/types";

// ---------------------------------------------------------------------------
// Environment variables
// ---------------------------------------------------------------------------
const GCP_PROJECT = process.env.GCP_PROJECT ?? "";
const GCP_LOCATION = process.env.GCP_LOCATION ?? "us-central1";
if (GCP_PROJECT) process.env.GOOGLE_CLOUD_PROJECT ??= GCP_PROJECT;
if (GCP_LOCATION) process.env.GOOGLE_CLOUD_LOCATION ??= GCP_LOCATION;
const GEN_FAST_MODEL =
  process.env.GEN_FAST_MODEL ?? process.env.GEMINI_MODEL ?? "gemini-2.0-flash";
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS ?? "*")
  .split(",")
  .map((o) => o.trim());

// ---------------------------------------------------------------------------
// Rate limiter — 20 requests per 10 seconds
// ---------------------------------------------------------------------------
const isRateLimited = createRateLimiter(20, 10_000);

// ---------------------------------------------------------------------------
// CORS helpers
// ---------------------------------------------------------------------------
function isOriginAllowed(origin: string | null): boolean {
  if (origin === null) return true; // server-to-server (no Origin header)
  return ALLOWED_ORIGINS.includes("*") || ALLOWED_ORIGINS.includes(origin);
}

function corsHeaders(origin: string | null): Record<string, string> {
  const allowed = isOriginAllowed(origin);

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
// POST /api/intent
//
// Gatekeeper route. Validates relevance and enriches queries before they
// reach the quick-search or deep-research pipelines.
//
// Body:    { query: string }
// Returns: { action: "proceed"|"clarify"|"reject", enrichedQuery: string, message: string }
// ---------------------------------------------------------------------------
export async function POST(req: NextRequest) {
  const origin = req.headers.get("origin");
  const headers = corsHeaders(origin);

  if (!isOriginAllowed(origin)) {
    return NextResponse.json(
      { error: "Forbidden: origin not allowed." },
      { status: 403, headers }
    );
  }

  if (isRateLimited()) {
    return NextResponse.json(
      { error: "Too many requests. Please try again shortly." },
      { status: 429, headers: { ...headers, "Retry-After": "10" } }
    );
  }

  if (!GCP_PROJECT) {
    return NextResponse.json(
      { error: "Server misconfiguration: GCP_PROJECT is not set." },
      { status: 500, headers }
    );
  }

  let query: string;
  let conversationHistory: ConversationMessage[] = [];
  try {
    const body = await req.json();
    if (typeof body?.query !== "string" || body.query.trim() === "") {
      return NextResponse.json(
        { error: "Request body must contain a non-empty `query` string." },
        { status: 400, headers }
      );
    }
    query = body.query.trim();
    if (Array.isArray(body.conversationHistory)) {
      conversationHistory = body.conversationHistory;
    }
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON in request body." },
      { status: 400, headers }
    );
  }

  try {
    const result = await runIntentOrchestrator(query, {
      project: GCP_PROJECT,
      location: GCP_LOCATION,
      model: GEN_FAST_MODEL,
      conversationHistory,
    });

    return NextResponse.json(result, { status: 200, headers });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[/api/intent] error:", message);
    return NextResponse.json(
      { error: "Failed to process intent.", detail: message },
      { status: 502, headers }
    );
  }
}
