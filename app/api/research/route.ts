import "@/app/lib/gcp-auth";
import { NextRequest, NextResponse } from "next/server";
import { streamDeepResearch } from "@/app/lib/agents/deep-research/runner";
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
const GEN_REPORT_MODEL =
  process.env.GEN_REPORT_MODEL ?? "gemini-2.5-pro";
const RAG_CORPUS = process.env.RAG_CORPUS ?? "";
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
// POST /api/research
//
// Returns a Server-Sent Events stream. Each event has:
//   event: step
//   data: { "field": "<resultKey>", "value": "<content>" }
//
// Final event:
//   event: done
//   data: {}
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
  if (!RAG_CORPUS) {
    return NextResponse.json(
      { error: "Server misconfiguration: RAG_CORPUS is not set." },
      { status: 500, headers }
    );
  }

  let query: string;
  let context: string | undefined;
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
    if (typeof body.context === "string" && body.context.trim()) {
      context = body.context.trim();
    }
    if (Array.isArray(body.conversationHistory)) {
      conversationHistory = body.conversationHistory;
    }
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON in request body." },
      { status: 400, headers }
    );
  }

  // If intent orchestrator provided enriched context, prepend it to the query
  const enrichedQuery = context
    ? `INTENT ANALYSIS (use this to guide your research — all queries relate to House Majority Staff Office):\n${context}\n\nUSER QUESTION:\n${query}`
    : query;

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        const gen = streamDeepResearch(enrichedQuery, {
          project: GCP_PROJECT,
          location: GCP_LOCATION,
          fastModel: GEN_FAST_MODEL,
          advancedModel: GEN_REPORT_MODEL,
          reportModel: GEN_REPORT_MODEL,
          ragCorpus: RAG_CORPUS,
          conversationHistory,
        });

        for await (const event of gen) {
          if (event.type === "log") {
            const payload = JSON.stringify({
              agent: event.agent,
              message: event.message,
              promptTokens: event.promptTokens,
              responseTokens: event.responseTokens,
              totalTokens: event.totalTokens,
              timestamp: event.timestamp,
              ...(event.researcherIndex != null && { researcherIndex: event.researcherIndex }),
            });
            controller.enqueue(encoder.encode(`event: log\ndata: ${payload}\n\n`));
          } else if (event.type === "step") {
            const payload = JSON.stringify({ field: event.field, value: event.value });
            controller.enqueue(encoder.encode(`event: step\ndata: ${payload}\n\n`));
          } else if (event.type === "researchers_init") {
            const payload = JSON.stringify({ count: event.count, labels: event.labels });
            controller.enqueue(encoder.encode(`event: researchers_init\ndata: ${payload}\n\n`));
          } else if (event.type === "researcher_done") {
            const payload = JSON.stringify({ index: event.index, label: event.label, value: event.value });
            controller.enqueue(encoder.encode(`event: researcher_done\ndata: ${payload}\n\n`));
          }
        }

        controller.enqueue(encoder.encode(`event: done\ndata: {}\n\n`));
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        console.error("[/api/research] SSE pipeline error:", message);
        const payload = JSON.stringify({ error: "Pipeline failed.", detail: message });
        controller.enqueue(encoder.encode(`event: error\ndata: ${payload}\n\n`));
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      ...headers,
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
