"""
FastAPI backend for the HMSO Training Chatbot.
Drop-in replacement for the Next.js API routes.
"""

import os
import sys
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# Ensure the fastapi directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ALLOWED_ORIGINS, GCP_PROJECT, GCP_LOCATION
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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-api-key"],
    max_age=86400,
)

# ── Global exception handler — surface the real error in the response ──────
@app.exception_handler(Exception)
async def _surface_exception(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[error] {request.method} {request.url.path}\n{tb}", flush=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "detail": str(exc),
        },
    )


# ── Startup: eager-initialize Vertex auth so first real request isn't slow ──
@app.on_event("startup")
async def _warm_vertex_ai():
    """Resolve ADC credentials and init Vertex at boot, not on first request.

    Cold-boot first request can take 10-30s because Google ADC token fetch
    + vertexai.init() are deferred until a tool calls them. Doing it here
    shifts that cost to container startup."""
    try:
        import google.auth
        google.auth.default()  # forces ADC credential resolution
        if GCP_PROJECT:
            import vertexai as vtx
            vtx.init(project=GCP_PROJECT, location=GCP_LOCATION)
        print("[startup] Vertex AI pre-initialized", flush=True)
    except Exception as e:
        print(f"[startup] Vertex warmup skipped: {e}", flush=True)


# ── Warmup endpoint (cheap, auth-exempt, pingable from the frontend) ────────
@app.get("/api/warmup")
async def warmup():
    return {"status": "ok"}


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
  <title>HMSO Training Assistant — API Documentation</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:system-ui,-apple-system,sans-serif;color:#1e293b;line-height:1.7;background:#f8fafc}
    a{color:#2563eb;text-decoration:none}a:hover{text-decoration:underline}

    /* Layout */
    .layout{display:flex;min-height:100vh}
    .sidebar{position:fixed;top:0;left:0;width:240px;height:100vh;background:#1a2332;color:#94a3b8;overflow-y:auto;padding:1.5rem 0;z-index:10}
    .sidebar .logo{padding:0 1.25rem 1.25rem;border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:1rem}
    .sidebar .logo h2{color:#fff;font-size:.875rem;font-weight:600;margin-bottom:.125rem}
    .sidebar .logo p{font-size:.7rem;color:#64748b}
    .sidebar nav{padding:0 .75rem}
    .sidebar .nav-group{margin-bottom:1rem}
    .sidebar .nav-label{font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#475569;padding:.5rem .5rem .25rem}
    .sidebar .nav-link{display:block;font-size:.8rem;padding:.4rem .75rem;border-radius:6px;color:#94a3b8;transition:all .15s}
    .sidebar .nav-link:hover{background:rgba(255,255,255,.06);color:#e2e8f0;text-decoration:none}
    .sidebar .nav-link.active{background:rgba(59,130,246,.15);color:#93c5fd}
    .main{margin-left:240px;flex:1;padding:2.5rem 3rem 4rem}
    .content{max-width:820px}

    /* Typography */
    h1{font-size:1.75rem;font-weight:700;margin-bottom:.25rem}
    .subtitle{color:#64748b;font-size:.9rem;margin-bottom:2.5rem}
    h2{font-size:1.25rem;font-weight:700;margin:2.5rem 0 .75rem;padding-top:1.5rem;border-top:1px solid #e2e8f0;scroll-margin-top:1.5rem}
    h2:first-of-type{border-top:none;margin-top:0;padding-top:0}
    h3{font-size:1rem;font-weight:600;margin:1.5rem 0 .5rem;color:#334155}
    p{font-size:.875rem;margin-bottom:.75rem}

    /* Code */
    code{background:#f1f5f9;padding:.15rem .4rem;border-radius:4px;font-size:.8rem;font-family:'SF Mono',Monaco,Consolas,monospace}
    pre{background:#1e1e1e;color:#d4d4d4;padding:1.25rem;border-radius:8px;font-size:.8rem;overflow-x:auto;margin:.75rem 0 1.25rem;line-height:1.6}
    pre code{background:none;padding:0}

    /* Tables */
    table{width:100%;border-collapse:collapse;font-size:.8125rem;margin:.5rem 0 1.25rem}
    th{text-align:left;padding:.625rem .75rem;border-bottom:2px solid #e2e8f0;font-weight:600;color:#475569;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em}
    td{padding:.625rem .75rem;border-bottom:1px solid #f1f5f9}
    .method{display:inline-block;background:#dbeafe;color:#1e40af;padding:.1rem .45rem;border-radius:4px;font-size:.7rem;font-weight:600;font-family:monospace}
    .method-get{background:#d1fae5;color:#065f46}

    /* Cards */
    .card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1.25rem;margin:.75rem 0}
    .card h4{font-size:.875rem;font-weight:600;margin-bottom:.5rem}

    /* Endpoint detail */
    .endpoint{background:#fff;border:1px solid #e2e8f0;border-radius:10px;margin:1rem 0;overflow:hidden}
    .endpoint-header{display:flex;align-items:center;gap:.75rem;padding:.875rem 1.25rem;background:#f8fafc;border-bottom:1px solid #f1f5f9}
    .endpoint-header code{font-size:.85rem;font-weight:600;background:none;padding:0}
    .endpoint-body{padding:1.25rem}
    .endpoint-body p{font-size:.8125rem;color:#475569}

    /* Badge */
    .badge{display:inline-block;font-size:.65rem;font-weight:600;padding:.15rem .5rem;border-radius:999px;text-transform:uppercase;letter-spacing:.04em}
    .badge-blue{background:#dbeafe;color:#1e40af}
    .badge-green{background:#d1fae5;color:#065f46}
    .badge-purple{background:#ede9fe;color:#5b21b6}
    .badge-amber{background:#fef3c7;color:#92400e}

    /* Muted */
    .muted{color:#64748b}
    .small{font-size:.8125rem}

    /* Responsive */
    @media(max-width:768px){
      .sidebar{position:relative;width:100%;height:auto;max-height:none}
      .layout{flex-direction:column}
      .main{margin-left:0;padding:1.5rem}
    }
  </style>
</head>
<body>
<div class="layout">

  <!-- Sidebar Navigation -->
  <div class="sidebar">
    <div class="logo">
      <h2>HMSO Training API</h2>
      <p>v1.0 &mdash; FastAPI</p>
    </div>
    <nav>
      <div class="nav-group">
        <div class="nav-label">Getting Started</div>
        <a href="#overview" class="nav-link">Overview</a>
        <a href="#authentication" class="nav-link">Authentication</a>
        <a href="#rate-limiting" class="nav-link">Rate Limiting</a>
        <a href="#cors" class="nav-link">CORS</a>
      </div>
      <div class="nav-group">
        <div class="nav-label">Chat Endpoints</div>
        <a href="#intent" class="nav-link">Intent Orchestrator</a>
        <a href="#conversational" class="nav-link">Conversational</a>
        <a href="#quick-search" class="nav-link">Quick Search</a>
        <a href="#quick-search-pro" class="nav-link">Quick Search Pro</a>
        <a href="#search-escalate" class="nav-link">Escalation Search</a>
        <a href="#research" class="nav-link">Deep Research (SSE)</a>
      </div>
      <div class="nav-group">
        <div class="nav-label">Quiz Endpoints</div>
        <a href="#quiz-generate" class="nav-link">Generate Quiz</a>
      </div>
      <div class="nav-group">
        <div class="nav-label">Resources</div>
        <a href="#models" class="nav-link">Models</a>
        <a href="#errors" class="nav-link">Error Handling</a>
        <a href="/docs" class="nav-link">Swagger UI &rarr;</a>
      </div>
    </nav>
  </div>

  <!-- Main Content -->
  <div class="main">
    <div class="content">

      <h1>House Majority Training Assistant</h1>
      <p class="subtitle">API documentation for the HMSO Training Chatbot backend service.</p>

      <!-- Overview -->
      <h2 id="overview">Overview</h2>
      <p>This API powers the House Majority Staff Office Training Chatbot. It provides RAG-grounded search over official training documents, multi-agent deep research, conversational chat, and AI-generated quizzes.</p>
      <div class="card">
        <h4>Base URL</h4>
        <code>https://hmso-training.ics.hawaii.edu/backend</code>
      </div>
      <p>All endpoints accept JSON request bodies and return JSON responses, except <code>/api/research</code> which returns a Server-Sent Events (SSE) stream.</p>

      <!-- Authentication -->
      <h2 id="authentication">Authentication</h2>
      <p>All requests require an API key passed via the <code>x-api-key</code> header.</p>
      <pre><code>curl -X POST https://hmso-training.ics.hawaii.edu/backend/api/intent \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: YOUR_API_KEY" \\
  -d '{"query": "What is the onboarding process?"}'</code></pre>
      <table>
        <thead><tr><th>Header</th><th>Required</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td><code>x-api-key</code></td><td>Yes</td><td>Your API key for authentication</td></tr>
          <tr><td><code>Content-Type</code></td><td>Yes</td><td>Must be <code>application/json</code></td></tr>
        </tbody>
      </table>
      <p class="muted small">Requests with an invalid or missing API key receive a <code>401 Unauthorized</code> response.</p>

      <!-- Rate Limiting -->
      <h2 id="rate-limiting">Rate Limiting</h2>
      <p>The API enforces a sliding-window rate limit of <strong>20 requests per 10 seconds</strong> per process.</p>
      <table>
        <thead><tr><th>Status</th><th>Header</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td><code>429</code></td><td><code>Retry-After: 10</code></td><td>Too many requests &mdash; wait and retry</td></tr>
        </tbody>
      </table>

      <!-- CORS -->
      <h2 id="cors">CORS</h2>
      <p>Cross-origin requests are restricted to the following origins:</p>
      <table>
        <thead><tr><th>Origin</th><th>Environment</th></tr></thead>
        <tbody>
          <tr><td><code>https://hmso-training.ics.hawaii.edu</code></td><td><span class="badge badge-green">Production</span></td></tr>
          <tr><td><code>http://localhost:3000</code></td><td><span class="badge badge-amber">Development</span></td></tr>
        </tbody>
      </table>
      <p class="muted small">Requests from other origins are rejected unless a valid <code>x-api-key</code> is provided.</p>

      <!-- Intent -->
      <h2 id="intent">Intent Orchestrator</h2>
      <p>Validates and classifies user queries before routing to the appropriate search agent. Returns one of four actions: <code>confirm</code>, <code>chat</code>, <code>clarify</code>, or <code>reject</code>.</p>
      <div class="endpoint">
        <div class="endpoint-header"><span class="method">POST</span><code>/api/intent</code></div>
        <div class="endpoint-body">
          <h4>Request Body</h4>
          <pre><code>{
  "query": "What are the ethics rules?",
  "conversationHistory": []   // optional
}</code></pre>
          <h4>Response</h4>
          <pre><code>{
  "action": "confirm",
  "enrichedQuery": "Detailed query with context...",
  "message": "Summary shown to user...",
  "logs": [{ "agent": "...", "promptTokens": 0, "responseTokens": 0, "totalTokens": 0, "timestamp": 0 }]
}</code></pre>
          <p><strong>Actions:</strong> <code>confirm</code> (proceed with search), <code>chat</code> (greeting/small talk), <code>clarify</code> (ask follow-up), <code>reject</code> (off-topic)</p>
        </div>
      </div>

      <!-- Conversational -->
      <h2 id="conversational">Conversational</h2>
      <p>Handles greetings, small talk, and questions about what the system can do. No RAG search &mdash; pure LLM chat.</p>
      <div class="endpoint">
        <div class="endpoint-header"><span class="method">POST</span><code>/api/conversational</code></div>
        <div class="endpoint-body">
          <h4>Request Body</h4>
          <pre><code>{
  "query": "What can you help me with?",
  "conversationHistory": []   // optional
}</code></pre>
          <h4>Response</h4>
          <pre><code>{
  "answer": "I can help you with...",
  "logs": [...]
}</code></pre>
        </div>
      </div>

      <!-- Quick Search -->
      <h2 id="quick-search">Quick Search</h2>
      <p>Single-pass RAG search using the <span class="badge badge-blue">Gemini 2.5 Flash</span> model. Fast answers for everyday questions.</p>
      <div class="endpoint">
        <div class="endpoint-header"><span class="method">POST</span><code>/api/quick-search</code></div>
        <div class="endpoint-body">
          <h4>Request Body</h4>
          <pre><code>{
  "query": "What is the onboarding process?",
  "context": "enriched query from intent",  // optional
  "conversationHistory": []                  // optional
}</code></pre>
          <h4>Response</h4>
          <pre><code>{
  "answer": "The onboarding process includes...\n\n## Sources\n...",
  "logs": [...]
}</code></pre>
        </div>
      </div>

      <!-- Quick Search Pro -->
      <h2 id="quick-search-pro">Quick Search Pro</h2>
      <p>Same as Quick Search but uses the <span class="badge badge-purple">Gemini 2.5 Pro</span> model for deeper, more thorough answers.</p>
      <div class="endpoint">
        <div class="endpoint-header"><span class="method">POST</span><code>/api/quick-search-pro</code></div>
        <div class="endpoint-body">
          <p>Same request/response format as <a href="#quick-search">Quick Search</a>.</p>
        </div>
      </div>

      <!-- Escalation Search -->
      <h2 id="search-escalate">Escalation Search</h2>
      <p>Re-searches with the Pro model after the initial Flash answer was insufficient. Takes the previous answer into account to go deeper.</p>
      <div class="endpoint">
        <div class="endpoint-header"><span class="method">POST</span><code>/api/search-escalate</code></div>
        <div class="endpoint-body">
          <h4>Request Body</h4>
          <pre><code>{
  "query": "What are the ethics rules?",
  "previousAnswer": "The initial Flash answer...",  // required
  "context": "enriched query",                       // optional
  "conversationHistory": []                          // optional
}</code></pre>
          <h4>Response</h4>
          <pre><code>{
  "answer": "More comprehensive answer...\n\n## Sources\n...",
  "logs": [...]
}</code></pre>
        </div>
      </div>

      <!-- Deep Research -->
      <h2 id="research">Deep Research (SSE)</h2>
      <p>Multi-agent research pipeline that breaks the query into sub-questions, researches each in parallel, and compiles a comprehensive report. Returns a <strong>Server-Sent Events</strong> stream.</p>
      <div class="endpoint">
        <div class="endpoint-header"><span class="method">POST</span><code>/api/research</code></div>
        <div class="endpoint-body">
          <h4>Request Body</h4>
          <pre><code>{
  "query": "What is the legislative process?",
  "context": "enriched query",       // optional
  "conversationHistory": []          // optional
}</code></pre>
          <h4>SSE Event Types</h4>
          <table>
            <thead><tr><th>Event</th><th>Data</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td><code>log</code></td><td><code>{ agent, message, promptTokens, responseTokens, totalTokens, timestamp, researcherIndex? }</code></td><td>Agent activity log with token usage</td></tr>
              <tr><td><code>step</code></td><td><code>{ field, value }</code></td><td>Pipeline step output. Fields: <code>enrichedQuery</code>, <code>researchQuestions</code>, <code>answer</code></td></tr>
              <tr><td><code>researchers_init</code></td><td><code>{ count, labels }</code></td><td>Parallel researchers spawned</td></tr>
              <tr><td><code>researcher_done</code></td><td><code>{ index, label, value }</code></td><td>Individual researcher completed</td></tr>
              <tr><td><code>error</code></td><td><code>{ error, detail }</code></td><td>Pipeline failure</td></tr>
              <tr><td><code>done</code></td><td><code>{}</code></td><td>Stream complete</td></tr>
            </tbody>
          </table>
          <h4>Pipeline Stages</h4>
          <p>1. <strong>Query Analyzer</strong> (Flash) &rarr; 2. <strong>Question Expander</strong> (Pro) &rarr; 3. <strong>Parallel Research Squad</strong> (5 researchers, Pro) &rarr; 4. <strong>Report Compiler</strong> (Pro)</p>
        </div>
      </div>

      <!-- Quiz Generate -->
      <h2 id="quiz-generate">Generate Quiz</h2>
      <p>Searches the RAG corpus and generates structured multiple-choice quiz questions with correct answers and source citations.</p>
      <div class="endpoint">
        <div class="endpoint-header"><span class="method">POST</span><code>/api/quiz/generate</code></div>
        <div class="endpoint-body">
          <h4>Request Body</h4>
          <pre><code>{
  "topic": "ethics rules and code of conduct",
  "numQuestions": 5    // optional, default 5
}</code></pre>
          <h4>Response</h4>
          <pre><code>{
  "quiz": {
    "title": "House Staff Ethics Rules",
    "questions": [
      {
        "id": 1,
        "question": "What should staff do if unsure about a gift?",
        "options": ["Accept it", "Decline it", "Consult Ethics Committee", "Ask supervisor"],
        "correct": 2,
        "source": "House Ethics Manual, &sect;3.4"
      }
    ]
  },
  "logs": [...]
}</code></pre>
          <p class="muted small">Each question has exactly 4 options. The <code>correct</code> field is the 0-based index of the correct answer.</p>
        </div>
      </div>

      <!-- Models -->
      <h2 id="models">Models</h2>
      <p>The API uses Google Vertex AI Gemini models, configurable via environment variables.</p>
      <table>
        <thead><tr><th>Variable</th><th>Default</th><th>Used By</th></tr></thead>
        <tbody>
          <tr><td><code>GEN_FAST_MODEL</code></td><td><code>gemini-2.5-flash</code></td><td>Intent, Conversational, Quick Search, Quiz, Query Analyzer</td></tr>
          <tr><td><code>GEN_REPORT_MODEL</code></td><td><code>gemini-2.5-pro</code></td><td>Quick Search Pro, Escalation, Research Squad, Report Compiler</td></tr>
        </tbody>
      </table>

      <!-- Errors -->
      <h2 id="errors">Error Handling</h2>
      <p>All endpoints return standard HTTP status codes with JSON error bodies.</p>
      <table>
        <thead><tr><th>Status</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><code>400</code></td><td>Invalid request body or missing required fields</td></tr>
          <tr><td><code>401</code></td><td>Invalid or missing API key</td></tr>
          <tr><td><code>403</code></td><td>Origin not in allowed CORS list</td></tr>
          <tr><td><code>429</code></td><td>Rate limit exceeded</td></tr>
          <tr><td><code>500</code></td><td>Server misconfiguration (missing GCP_PROJECT or RAG_CORPUS)</td></tr>
          <tr><td><code>502</code></td><td>Upstream Vertex AI error</td></tr>
        </tbody>
      </table>
      <h4>Error Response Format</h4>
      <pre><code>{
  "error": "Short error message",
  "detail": "Additional context"    // optional
}</code></pre>

      <br><br>
      <footer style="border-top:1px solid #e2e8f0;padding-top:1rem;font-size:.75rem;color:#94a3b8">
        Sponsored by the House of Majority Staff Office (HMSO) &mdash; <a href="/docs" style="color:#94a3b8">Swagger UI</a>
      </footer>
    </div>
  </div>

</div>

<script>
// Highlight active sidebar link on scroll
const links = document.querySelectorAll('.sidebar .nav-link[href^="#"]');
const sections = [...links].map(l => document.getElementById(l.getAttribute('href').slice(1))).filter(Boolean);
function updateActive() {
  let current = '';
  for (const s of sections) {
    if (s.getBoundingClientRect().top <= 80) current = s.id;
  }
  links.forEach(l => {
    l.classList.toggle('active', l.getAttribute('href') === '#' + current);
  });
}
window.addEventListener('scroll', updateActive, { passive: true });
document.querySelector('.main').addEventListener('scroll', updateActive, { passive: true });
updateActive();
</script>
</body>
</html>"""


# ── Catch-all — redirect unknown paths to docs ──────────────────────────
@app.api_route("/{path:path}", methods=["GET"], include_in_schema=False)
async def catch_all(path: str):
    return RedirectResponse(url="/")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
