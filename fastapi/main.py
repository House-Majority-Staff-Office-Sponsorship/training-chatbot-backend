"""
FastAPI backend for the HMSO Training Chatbot.
Drop-in replacement for the Next.js API routes.
"""

import os
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Ensure the fastapi directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ALLOWED_ORIGINS
from middleware.auth import ApiKeyMiddleware
from middleware.rate_limiter import RateLimiterMiddleware
from routes import intent, conversational, quick_search, quick_search_pro, search_escalate, research, quiz


app = FastAPI(
    title="House Majority Training Assistant — API",
    description="Backend service for the HMSO Training Chatbot",
    docs_url="/docs",
    redoc_url=None,
)

# ── Middleware (order matters: outermost first) ──────────────────────────
app.add_middleware(RateLimiterMiddleware, max_requests=20, window_ms=10_000)
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-api-key"],
    max_age=86400,
)

# ── Routes ───────────────────────────────────────────────────────────────
app.include_router(intent.router)
app.include_router(conversational.router)
app.include_router(quick_search.router)
app.include_router(quick_search_pro.router)
app.include_router(search_escalate.router)
app.include_router(research.router)
app.include_router(quiz.router)


# ── Root — API Documentation page ───────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HMSO Training Assistant — API</title>
  <style>
    body { max-width: 700px; margin: 0 auto; padding: 3rem 1.5rem; font-family: system-ui, sans-serif; color: #1e293b; line-height: 1.7; }
    h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
    h2 { font-size: 1.1rem; margin-bottom: 0.5rem; }
    p.sub { color: #64748b; margin-bottom: 2rem; font-size: 0.875rem; }
    code { background: #f1f5f9; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.8125rem; }
    pre { background: #1e1e1e; color: #d4d4d4; padding: 1rem; border-radius: 6px; font-size: 0.8rem; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.8125rem; }
    th { border-bottom: 2px solid #e2e8f0; text-align: left; padding: 0.5rem 0.75rem; }
    td { border-bottom: 1px solid #f1f5f9; padding: 0.5rem 0.75rem; }
    .method { background: #dbeafe; color: #1e40af; padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.75rem; }
    .muted { color: #64748b; }
    footer { border-top: 1px solid #e2e8f0; padding-top: 1rem; font-size: 0.75rem; color: #94a3b8; }
    ul { font-size: 0.875rem; padding-left: 1.25rem; margin: 0; }
  </style>
</head>
<body>
  <h1>House Majority Training Assistant &mdash; API</h1>
  <p class="sub">Backend service for the HMSO Training Chatbot (FastAPI)</p>

  <section>
    <h2>Connecting</h2>
    <p style="font-size:0.875rem">All endpoints require an API key passed via the <code>x-api-key</code> header.</p>
    <pre>curl -X POST &lt;backend-url&gt;/api/intent \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: &lt;your-api-key&gt;" \\
  -d '{"query": "What is the onboarding process?"}'</pre>
  </section>

  <section>
    <h2>Available Endpoints</h2>
    <table>
      <thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td><code class="method">POST</code></td><td><code>/api/intent</code></td><td class="muted">Intent orchestrator — classifies and enriches the query</td></tr>
        <tr><td><code class="method">POST</code></td><td><code>/api/quick-search</code></td><td class="muted">Quick search using Flash model + RAG</td></tr>
        <tr><td><code class="method">POST</code></td><td><code>/api/quick-search-pro</code></td><td class="muted">Quick search using Pro model + RAG</td></tr>
        <tr><td><code class="method">POST</code></td><td><code>/api/search-escalate</code></td><td class="muted">Escalation search (Pro) after an unsatisfactory Flash answer</td></tr>
        <tr><td><code class="method">POST</code></td><td><code>/api/conversational</code></td><td class="muted">Conversational follow-up (no RAG)</td></tr>
        <tr><td><code class="method">POST</code></td><td><code>/api/research</code></td><td class="muted">Deep research pipeline (SSE stream)</td></tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2>Allowed Origins (CORS)</h2>
    <ul>
      <li><code>https://hmso-training.vercel.app</code></li>
      <li><code>https://hmso-training.ics.hawaii.edu</code></li>
      <li><code>http://localhost:3000</code> (development)</li>
    </ul>
    <p style="font-size:0.8125rem" class="muted">Requests from other origins are rejected unless a valid <code>x-api-key</code> is provided.</p>
  </section>

  <section>
    <h2>Interactive Docs</h2>
    <p style="font-size:0.875rem">Visit <a href="/docs">/docs</a> for the Swagger UI.</p>
  </section>

  <footer>Sponsored by the House of Majority Staff Office (HMSO)</footer>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
