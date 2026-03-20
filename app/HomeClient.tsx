"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type Mode = "quick" | "quick-pro" | "deep";

interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

interface LogEntry {
  agent: string;
  message: string;
  promptTokens: number;
  responseTokens: number;
  totalTokens: number;
  timestamp: number;
  researcherIndex?: number;
}

interface DeepResult {
  enrichedQuery: string;
  researchQuestions: string;
  sectionFindings: string;
  answer: string;
}

type StepKey = keyof DeepResult;

interface ResearcherState {
  label: string;
  findings: string;
  done: boolean;
}

const EMPTY_DEEP: DeepResult = {
  enrichedQuery: "",
  researchQuestions: "",
  sectionFindings: "",
  answer: "",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";

export default function Home({ apiKey }: { apiKey: string }) {
  const apiHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    ...(apiKey && { "x-api-key": apiKey }),
  };
  const [mode, setMode] = useState<Mode>("quick");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<ConversationMessage[]>([]);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Intent orchestrator state
  const [intentMessage, setIntentMessage] = useState<string | null>(null);
  const [intentChecking, setIntentChecking] = useState(false);
  const [pendingConfirm, setPendingConfirm] = useState<{
    query: string;
    enrichedQuery: string;
    message: string;
  } | null>(null);

  // Quick search state (Flash first-pass)
  const [quickAnswer, setQuickAnswer] = useState<string | null>(null);

  // Escalation flow state: Flash answer → satisfaction check → Pro
  const [satisfaction, setSatisfaction] = useState<"pending" | "satisfied" | "escalated" | null>(null);
  const [proAnswer, setProAnswer] = useState<string | null>(null);
  const [escalationContext, setEscalationContext] = useState<{ query: string; context: string } | null>(null);

  // Deep research state
  const [deepResult, setDeepResult] = useState<DeepResult>(EMPTY_DEEP);
  const [completedSteps, setCompletedSteps] = useState<Set<StepKey>>(new Set());
  const [expandedStep, setExpandedStep] = useState<StepKey | null>(null);

  // Dynamic researchers state
  const [researchers, setResearchers] = useState<ResearcherState[]>([]);
  const [expandedResearchers, setExpandedResearchers] = useState<Set<number>>(new Set());

  // Log panel state
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logOpen, setLogOpen] = useState(true);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  const addLogs = useCallback((entries: LogEntry[]) => {
    setLogs((prev) => [...prev, ...entries]);
  }, []);

  const addLog = useCallback((entry: LogEntry) => {
    setLogs((prev) => [...prev, entry]);
  }, []);

  // Auto-scroll log panel
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const abortRef = useRef<AbortController | null>(null);

  // ---- Quick Search -------------------------------------------------------
  const runQuickSearch = useCallback(async (q: string, context: string, pro = false) => {
    const start = Date.now();
    try {
      const res = await fetch(pro ? `${BASE}/api/quick-search-pro` : `${BASE}/api/quick-search`, {
        method: "POST",
        headers: apiHeaders,
        body: JSON.stringify({ query: q, context, conversationHistory }),
      });
      const data = await res.json();
      if (data.logs) addLogs(data.logs);
      if (data.error) {
        setError(data.error + (data.detail ? `: ${data.detail}` : ""));
      } else {
        setQuickAnswer(data.answer);
        // If this was a Flash first-pass (not pro), enable satisfaction check
        if (!pro) {
          setSatisfaction("pending");
          setEscalationContext({ query: q, context });
        }
        setConversationHistory((prev) => [
          ...prev,
          { role: "user", content: q },
          { role: "assistant", content: data.answer },
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setElapsed(Date.now() - start);
      setLoading(false);
    }
  }, [conversationHistory]);

  // ---- Conversational -----------------------------------------------------
  const runConversational = useCallback(async (q: string) => {
    const start = Date.now();
    try {
      const res = await fetch(`${BASE}/api/conversational`, {
        method: "POST",
        headers: apiHeaders,
        body: JSON.stringify({ query: q, conversationHistory }),
      });
      const data = await res.json();
      if (data.logs) addLogs(data.logs);
      if (data.error) {
        setError(data.error + (data.detail ? `: ${data.detail}` : ""));
      } else {
        setQuickAnswer(data.answer);
        setConversationHistory((prev) => [
          ...prev,
          { role: "user", content: q },
          { role: "assistant", content: data.answer },
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setElapsed(Date.now() - start);
      setLoading(false);
    }
  }, [conversationHistory]);

  // ---- Escalation Search (Pro) --------------------------------------------
  const runEscalationSearch = useCallback(async (q: string, context: string, previousAnswer: string) => {
    const start = Date.now();
    setLoading(true);
    setError(null);
    setElapsed(null);
    setSatisfaction("escalated");
    try {
      const res = await fetch(`${BASE}/api/search-escalate`, {
        method: "POST",
        headers: apiHeaders,
        body: JSON.stringify({ query: q, context, previousAnswer, conversationHistory }),
      });
      const data = await res.json();
      if (data.logs) addLogs(data.logs);
      if (data.error) {
        setError(data.error + (data.detail ? `: ${data.detail}` : ""));
      } else {
        setProAnswer(data.answer);
        setConversationHistory((prev) => [
          ...prev,
          { role: "assistant", content: data.answer },
        ]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setElapsed(Date.now() - start);
      setLoading(false);
    }
  }, [conversationHistory]);

  // ---- Deep Research (SSE) ------------------------------------------------
  const runDeepResearch = useCallback(async (q: string, context: string) => {
    const start = Date.now();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${BASE}/api/research`, {
        method: "POST",
        headers: apiHeaders,
        body: JSON.stringify({ query: q, context, conversationHistory }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        setError(data.error || `HTTP ${res.status}`);
        setElapsed(Date.now() - start);
        setLoading(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            const raw = line.slice(6);
            try {
              const payload = JSON.parse(raw);

              if (eventType === "log") {
                addLog(payload as LogEntry);
              } else if (eventType === "step" && payload.field && payload.value) {
                const field = payload.field as StepKey;
                setDeepResult((prev) => ({ ...prev, [field]: payload.value }));
                setCompletedSteps((prev) => new Set(prev).add(field));
                setExpandedStep(field);
                if (field === "answer") {
                  setConversationHistory((prev) => [
                    ...prev,
                    { role: "user", content: q },
                    { role: "assistant", content: payload.value },
                  ]);
                }
              } else if (eventType === "researchers_init" && payload.labels) {
                const labels = payload.labels as string[];
                console.log("[researchers_init]", { count: payload.count, labels });
                setResearchers(
                  labels.map((label: string) => ({ label, findings: "", done: false }))
                );
              } else if (eventType === "researcher_done") {
                const idx = payload.index as number;
                setResearchers((prev) => {
                  const next = [...prev];
                  if (next[idx]) {
                    next[idx] = { ...next[idx], findings: payload.value, done: true };
                  }
                  return next;
                });
                setExpandedResearchers((prev) => new Set(prev).add(idx));
              } else if (eventType === "error") {
                setError(payload.error + (payload.detail ? `: ${payload.detail}` : ""));
              }
            } catch {
              // ignore malformed JSON
            }
            eventType = "";
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setElapsed(Date.now() - start);
      setLoading(false);
      abortRef.current = null;
    }
  }, [conversationHistory]);

  // ---- Proceed after confirmation -----------------------------------------
  const handleConfirm = useCallback(async () => {
    if (!pendingConfirm || loading) return;
    const { query: q, enrichedQuery: ctx } = pendingConfirm;
    setPendingConfirm(null);
    setIntentMessage(null);
    setLoading(true);
    setError(null);
    setElapsed(null);
    setQuickAnswer(null);
    setDeepResult(EMPTY_DEEP);
    setCompletedSteps(new Set());
    setExpandedStep(null);
    setResearchers([]);
    setExpandedResearchers(new Set());

    if (mode === "quick" || mode === "quick-pro") {
      await runQuickSearch(q, ctx, mode === "quick-pro");
    } else {
      await runDeepResearch(q, ctx);
    }
  }, [pendingConfirm, loading, mode, runConversational, runQuickSearch, runDeepResearch]);

  // ---- Intent Orchestrator → Confirmation ---------------------------------
  const handleSubmit = useCallback(async () => {
    const q = query.trim();
    if (!q || loading) return;

    setLoading(true);
    setIntentChecking(false);
    setIntentMessage(null);
    setPendingConfirm(null);
    setError(null);
    setElapsed(null);
    setQuickAnswer(null);
    setSatisfaction(null);
    setProAnswer(null);
    setEscalationContext(null);
    setDeepResult(EMPTY_DEEP);
    setCompletedSteps(new Set());
    setExpandedStep(null);
    setResearchers([]);
    setExpandedResearchers(new Set());
    setLogs([]);

    setIntentChecking(true);

    try {
      const intentRes = await fetch(`${BASE}/api/intent`, {
        method: "POST",
        headers: apiHeaders,
        body: JSON.stringify({ query: q, conversationHistory }),
      });
      const intent = await intentRes.json();
      setIntentChecking(false);

      // Collect logs from intent orchestrator
      if (intent.logs) addLogs(intent.logs);

      if (intent.error) {
        setError(intent.error + (intent.detail ? `: ${intent.detail}` : ""));
        setLoading(false);
        return;
      }

      if (intent.action === "reject") {
        setIntentMessage(intent.message);
        setLoading(false);
        return;
      }

      if (intent.action === "clarify") {
        setIntentMessage(intent.message);
        setLoading(false);
        return;
      }

      // Conversational — route directly, no confirmation needed
      if (intent.action === "chat") {
        await runConversational(q);
        return;
      }

      setPendingConfirm({
        query: q,
        enrichedQuery: intent.enrichedQuery || q,
        message: intent.message,
      });
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
      setIntentChecking(false);
    }
  }, [query, loading, mode, conversationHistory, runConversational]);

  const resetAll = () => {
    setPendingConfirm(null);
    setIntentMessage(null);
    setError(null);
    setElapsed(null);
    setQuickAnswer(null);
    setSatisfaction(null);
    setProAnswer(null);
    setEscalationContext(null);
    setDeepResult(EMPTY_DEEP);
    setCompletedSteps(new Set());
    setExpandedStep(null);
    setResearchers([]);
    setExpandedResearchers(new Set());
    setLogs([]);
  };

  // Derived state
  const researchersDone = researchers.length > 0 && researchers.every((r) => r.done);
  const doneCount = researchers.filter((r) => r.done).length;
  const LETTERS = "abcdefghijklmnopqrstuvwxyz";

  // ---- Render -------------------------------------------------------------
  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: "2rem 1rem", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: "1.5rem", marginBottom: "0.25rem" }}>House Majority Training Assistant</h1>
      <p style={{ color: "#666", marginBottom: "1.5rem", fontSize: "0.875rem" }}>
        Ask questions about House Majority training documents, policies, and procedures
      </p>

      {/* Terminal Log Panel */}
      {logs.length > 0 && (
        <div style={{
          marginBottom: "1rem",
          border: "1px solid #333",
          borderRadius: 6,
          overflow: "hidden",
          fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
          fontSize: "0.75rem",
          backgroundColor: "#1e1e1e",
          color: "#d4d4d4",
        }}>
          <div
            onClick={() => setLogOpen((v) => !v)}
            style={{
              padding: "0.5rem 0.75rem",
              backgroundColor: "#2d2d2d",
              borderBottom: logOpen ? "1px solid #444" : "none",
              cursor: "pointer",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              userSelect: "none",
            }}
          >
            <span style={{ color: "#569cd6" }}>Agent Logs</span>
            <span style={{ color: "#808080", fontSize: "0.7rem" }}>
              {logs.length} events | {logs.reduce((s, l) => s + l.totalTokens, 0).toLocaleString()} total tokens
              {" "}{logOpen ? "\u25B2" : "\u25BC"}
            </span>
          </div>
          {logOpen && (
            <div style={{
              maxHeight: 200,
              overflowY: "auto",
              padding: "0.5rem 0.75rem",
            }}>
              {(() => {
                // Consolidate researcher logs into summary lines
                type DisplayLine = { key: string; time: string; agent: string; detail: string; tokens: string };
                const lines: DisplayLine[] = [];
                const researcherBuckets: Map<number, { logs: LogEntry[]; firstTime: number; lastTime: number }> = new Map();

                for (const log of logs) {
                  if (log.researcherIndex != null) {
                    const bucket = researcherBuckets.get(log.researcherIndex);
                    if (bucket) {
                      bucket.logs.push(log);
                      bucket.lastTime = Math.max(bucket.lastTime, log.timestamp);
                    } else {
                      researcherBuckets.set(log.researcherIndex, { logs: [log], firstTime: log.timestamp, lastTime: log.timestamp });
                    }
                  } else {
                    // Flush any completed researcher buckets that ended before this log
                    for (const [idx, bucket] of researcherBuckets) {
                      if (bucket.lastTime <= log.timestamp) {
                        const calls = bucket.logs.filter((l) => l.totalTokens > 0).length;
                        const totalIn = bucket.logs.reduce((s, l) => s + l.promptTokens, 0);
                        const totalOut = bucket.logs.reduce((s, l) => s + l.responseTokens, 0);
                        const totalTok = bucket.logs.reduce((s, l) => s + l.totalTokens, 0);
                        const label = researchers[idx]?.label ?? `Researcher ${idx}`;
                        const shortLabel = label.length > 60 ? label.slice(0, 57) + "..." : label;
                        const time = new Date(bucket.lastTime).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
                        lines.push({
                          key: `r-${lines.length}`,
                          time,
                          agent: `researcher_${idx}`,
                          detail: shortLabel,
                          tokens: totalTok > 0 ? `[${calls} calls, ${totalIn}in/${totalOut}out = ${totalTok} tokens]` : "",
                        });
                        researcherBuckets.delete(idx);
                      }
                    }
                    const time = new Date(log.timestamp).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
                    const detail = log.message !== log.agent ? log.message.replace(log.agent, "").trim() : "";
                    const tokens = log.totalTokens > 0 ? `[${log.promptTokens}in/${log.responseTokens}out = ${log.totalTokens} tokens]` : "";
                    lines.push({ key: `l-${lines.length}`, time, agent: log.agent, detail, tokens });
                  }
                }
                // Flush remaining researcher buckets
                for (const [idx, bucket] of researcherBuckets) {
                  const calls = bucket.logs.filter((l) => l.totalTokens > 0).length;
                  const totalIn = bucket.logs.reduce((s, l) => s + l.promptTokens, 0);
                  const totalOut = bucket.logs.reduce((s, l) => s + l.responseTokens, 0);
                  const totalTok = bucket.logs.reduce((s, l) => s + l.totalTokens, 0);
                  const label = researchers[idx]?.label ?? `Researcher ${idx}`;
                  const shortLabel = label.length > 60 ? label.slice(0, 57) + "..." : label;
                  const time = new Date(bucket.lastTime).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
                  lines.push({
                    key: `r-${lines.length}`,
                    time,
                    agent: `researcher_${idx}`,
                    detail: shortLabel,
                    tokens: totalTok > 0 ? `[${calls} calls, ${totalIn}in/${totalOut}out = ${totalTok} tokens]` : "",
                  });
                }

                return lines.map((line) => (
                  <div key={line.key} style={{ lineHeight: 1.8, whiteSpace: "nowrap" }}>
                    <span style={{ color: "#808080" }}>[{line.time}]</span>{" "}
                    <span style={{ color: "#4ec9b0" }}>{line.agent}</span>{" "}
                    {line.detail && <span style={{ color: "#d4d4d4" }}>{line.detail} </span>}
                    {line.tokens && <span style={{ color: "#ce9178" }}>{line.tokens}</span>}
                  </div>
                ));
              })()}
              <div ref={logEndRef} />
            </div>
          )}
        </div>
      )}

      {/* Mode toggle */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
        {(["quick", "quick-pro", "deep"] as const).map((m) => {
          const label = m === "quick" ? "Quick Search (Flash)" : m === "quick-pro" ? "Quick Search (Pro)" : "Deep Research (Flash + Pro)";
          const activeColor = m === "quick" ? "#2563eb" : m === "quick-pro" ? "#0891b2" : "#7c3aed";
          return (
            <button
              key={m}
              onClick={() => { setMode(m); resetAll(); }}
              disabled={loading}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "0.8125rem",
                fontWeight: mode === m ? 600 : 400,
                color: mode === m ? "#fff" : "#374151",
                backgroundColor: mode === m ? activeColor : "#f3f4f6",
                border: "1px solid",
                borderColor: mode === m ? "transparent" : "#d1d5db",
                borderRadius: 6,
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Query input */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !pendingConfirm && handleSubmit()}
          placeholder="Ask about training docs, policies, procedures..."
          disabled={loading}
          style={{
            flex: 1,
            padding: "0.625rem 0.75rem",
            fontSize: "0.875rem",
            border: "1px solid #ccc",
            borderRadius: 6,
            outline: "none",
          }}
        />
        <button
          onClick={pendingConfirm ? undefined : handleSubmit}
          disabled={loading || !query.trim() || !!pendingConfirm}
          style={{
            padding: "0.625rem 1.25rem",
            fontSize: "0.875rem",
            fontWeight: 600,
            color: "#fff",
            backgroundColor: loading ? "#999" : (mode === "deep" ? "#7c3aed" : mode === "quick-pro" ? "#0891b2" : "#2563eb"),
            border: "none",
            borderRadius: 6,
            cursor: loading ? "wait" : (pendingConfirm ? "not-allowed" : "pointer"),
          }}
        >
          {loading ? "Running..." : (mode === "deep" ? "Research" : "Search")}
        </button>
      </div>

      {/* Intent checking indicator */}
      {intentChecking && (
        <div style={{ padding: "1rem", textAlign: "center", color: "#666", fontSize: "0.875rem" }}>
          Validating your question...
        </div>
      )}

      {/* Confirmation prompt */}
      {pendingConfirm && !loading && (
        <div style={{
          padding: "1rem",
          marginBottom: "1rem",
          backgroundColor: "#eff6ff",
          border: "1px solid #bfdbfe",
          borderRadius: 6,
          fontSize: "0.875rem",
          lineHeight: 1.6,
        }}>
          <p style={{ color: "#1e40af", marginBottom: "0.75rem" }}>
            {pendingConfirm.message}
          </p>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              onClick={handleConfirm}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "0.8125rem",
                fontWeight: 600,
                color: "#fff",
                backgroundColor: mode === "deep" ? "#7c3aed" : mode === "quick-pro" ? "#0891b2" : "#2563eb",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              Yes, proceed
            </button>
            <button
              onClick={() => { setPendingConfirm(null); }}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "0.8125rem",
                fontWeight: 500,
                color: "#374151",
                backgroundColor: "#f3f4f6",
                border: "1px solid #d1d5db",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              No, let me rephrase
            </button>
          </div>
        </div>
      )}

      {/* Intent clarify/reject message */}
      {intentMessage && !loading && !pendingConfirm && (
        <div style={{
          padding: "0.75rem 1rem",
          marginBottom: "1rem",
          backgroundColor: "#fefce8",
          border: "1px solid #fde68a",
          borderRadius: 6,
          color: "#92400e",
          fontSize: "0.875rem",
          lineHeight: 1.6,
        }}>
          {intentMessage}
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: "0.75rem 1rem",
          marginBottom: "1rem",
          backgroundColor: "#fef2f2",
          border: "1px solid #fecaca",
          borderRadius: 6,
          color: "#991b1b",
          fontSize: "0.875rem",
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Elapsed */}
      {elapsed !== null && !loading && (
        <p style={{ fontSize: "0.75rem", color: "#999", marginBottom: "1rem" }}>
          Completed in {(elapsed / 1000).toFixed(1)}s
        </p>
      )}

      {/* ── Quick Search Result ── */}
      {quickAnswer && (
        <div style={{
          padding: "1rem",
          backgroundColor: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: 6,
          fontSize: "0.8125rem",
          lineHeight: 1.6,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}>
          {quickAnswer}
        </div>
      )}

      {/* ── Satisfaction Check ── */}
      {satisfaction === "pending" && !loading && quickAnswer && (
        <div style={{
          padding: "1rem",
          marginTop: "0.75rem",
          backgroundColor: "#f0fdf4",
          border: "1px solid #bbf7d0",
          borderRadius: 6,
          fontSize: "0.875rem",
          lineHeight: 1.6,
        }}>
          <p style={{ color: "#166534", marginBottom: "0.75rem" }}>
            Does this answer your question?
          </p>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              onClick={() => setSatisfaction("satisfied")}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "0.8125rem",
                fontWeight: 600,
                color: "#fff",
                backgroundColor: "#16a34a",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              Yes
            </button>
            <button
              onClick={() => {
                if (escalationContext && quickAnswer) {
                  runEscalationSearch(escalationContext.query, escalationContext.context, quickAnswer);
                }
              }}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "0.8125rem",
                fontWeight: 600,
                color: "#fff",
                backgroundColor: "#0891b2",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              No, search deeper
            </button>
          </div>
        </div>
      )}

      {/* ── Escalation: Pro Search in progress ── */}
      {satisfaction === "escalated" && loading && (
        <div style={{ padding: "1rem", marginTop: "0.75rem", textAlign: "center", color: "#0891b2", fontSize: "0.875rem" }}>
          Searching deeper with Pro model...
        </div>
      )}

      {/* ── Escalation: Pro Answer ── */}
      {proAnswer && (
        <div style={{
          padding: "1rem",
          marginTop: "0.75rem",
          backgroundColor: "#fff",
          border: "1px solid #06b6d4",
          borderRadius: 6,
          fontSize: "0.8125rem",
          lineHeight: 1.6,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}>
          <div style={{ fontSize: "0.75rem", color: "#0891b2", fontWeight: 600, marginBottom: "0.5rem" }}>
            Pro Search Result
          </div>
          {proAnswer}
        </div>
      )}

      {/* ── Deep Research Live Pipeline ── */}
      {mode === "deep" && (loading && !intentChecking || completedSteps.size > 0 || researchers.length > 0) && (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>

          {/* Step 1: Query Analyzer */}
          {(() => {
            const isDone = completedSteps.has("enrichedQuery");
            const isActive = loading && !isDone;
            const isExpanded = expandedStep === "enrichedQuery";
            return (
              <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, overflow: "hidden", opacity: !isDone && !isActive ? 0.5 : 1 }}>
                <button
                  onClick={() => isDone && setExpandedStep(isExpanded ? null : "enrichedQuery")}
                  style={{ width: "100%", padding: "0.625rem 1rem", display: "flex", justifyContent: "space-between", alignItems: "center", backgroundColor: isExpanded ? "#f0f4ff" : "#fafafa", border: "none", cursor: isDone ? "pointer" : "default", fontSize: "0.875rem", fontWeight: 500 }}
                >
                  <span>
                    <span style={{ marginRight: "0.5rem", color: isDone ? "#16a34a" : isActive ? "#2563eb" : "#9ca3af" }}>
                      {isDone ? "\u2713" : isActive ? "\u25CF" : "\u2013"}
                    </span>
                    1. Query Analyzer
                    {isActive && <span style={{ marginLeft: "0.5rem", color: "#2563eb", fontSize: "0.75rem" }}>running...</span>}
                  </span>
                  {isDone && <span style={{ color: "#999", fontSize: "0.75rem" }}>{isExpanded ? "\u25B2" : "\u25BC"}</span>}
                </button>
                {isExpanded && isDone && (
                  <div style={{ padding: "1rem", backgroundColor: "#fff", borderTop: "1px solid #e5e7eb", fontSize: "0.8125rem", lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: 500, overflowY: "auto" }}>
                    {deepResult.enrichedQuery || "(empty)"}
                  </div>
                )}
              </div>
            );
          })()}

          {/* Step 2: Question Expander */}
          {(() => {
            const isDone = completedSteps.has("researchQuestions");
            const isActive = loading && !isDone && completedSteps.has("enrichedQuery");
            const isExpanded = expandedStep === "researchQuestions";
            return (
              <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, overflow: "hidden", opacity: !isDone && !isActive ? 0.5 : 1 }}>
                <button
                  onClick={() => isDone && setExpandedStep(isExpanded ? null : "researchQuestions")}
                  style={{ width: "100%", padding: "0.625rem 1rem", display: "flex", justifyContent: "space-between", alignItems: "center", backgroundColor: isExpanded ? "#f0f4ff" : "#fafafa", border: "none", cursor: isDone ? "pointer" : "default", fontSize: "0.875rem", fontWeight: 500 }}
                >
                  <span>
                    <span style={{ marginRight: "0.5rem", color: isDone ? "#16a34a" : isActive ? "#2563eb" : "#9ca3af" }}>
                      {isDone ? "\u2713" : isActive ? "\u25CF" : "\u2013"}
                    </span>
                    2. Question Expander
                    {isActive && <span style={{ marginLeft: "0.5rem", color: "#2563eb", fontSize: "0.75rem" }}>running...</span>}
                  </span>
                  {isDone && <span style={{ color: "#999", fontSize: "0.75rem" }}>{isExpanded ? "\u25B2" : "\u25BC"}</span>}
                </button>
                {isExpanded && isDone && (
                  <div style={{ padding: "1rem", backgroundColor: "#fff", borderTop: "1px solid #e5e7eb", fontSize: "0.8125rem", lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: 500, overflowY: "auto" }}>
                    {deepResult.researchQuestions || "(empty)"}
                  </div>
                )}
              </div>
            );
          })()}

          {/* Step 3: Parallel Research — always visible */}
          {(() => {
            const isActive = loading && completedSteps.has("researchQuestions") && !completedSteps.has("answer");
            return (
              <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, overflow: "hidden", opacity: researchersDone || isActive || researchers.length > 0 ? 1 : 0.5 }}>
                {/* Header */}
                <div style={{ padding: "0.625rem 1rem", display: "flex", justifyContent: "space-between", alignItems: "center", backgroundColor: "#fafafa", fontSize: "0.875rem", fontWeight: 500 }}>
                  <span>
                    <span style={{ marginRight: "0.5rem", color: researchersDone ? "#16a34a" : isActive ? "#2563eb" : "#9ca3af" }}>
                      {researchersDone ? "\u2713" : isActive ? "\u25CF" : "\u2013"}
                    </span>
                    3. Parallel Research
                    {researchers.length > 0 ? (
                      <span style={{ marginLeft: "0.5rem", color: researchersDone ? "#16a34a" : "#2563eb", fontSize: "0.75rem" }}>
                        {doneCount}/{researchers.length} complete
                      </span>
                    ) : isActive && (
                      <span style={{ marginLeft: "0.5rem", color: "#2563eb", fontSize: "0.75rem" }}>
                        spawning agents...
                      </span>
                    )}
                  </span>
                </div>

                {/* Individual researchers */}
                {researchers.map((r, idx) => {
                  const isResExpanded = expandedResearchers.has(idx);
                  const letter = LETTERS[idx] || String(idx);
                  return (
                    <div key={idx} style={{ borderTop: "1px solid #f0f0f0" }}>
                      <button
                        onClick={() => r.done && setExpandedResearchers((prev) => {
                          const next = new Set(prev);
                          if (next.has(idx)) next.delete(idx); else next.add(idx);
                          return next;
                        })}
                        style={{
                          width: "100%",
                          padding: "0.5rem 1rem 0.5rem 2rem",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          backgroundColor: isResExpanded ? "#f0f4ff" : "#fff",
                          border: "none",
                          cursor: r.done ? "pointer" : "default",
                          fontSize: "0.8125rem",
                          fontWeight: 400,
                        }}
                      >
                        <span style={{ textAlign: "left" }}>
                          <span style={{ marginRight: "0.5rem", color: r.done ? "#16a34a" : loading ? "#2563eb" : "#d1d5db", fontSize: "0.75rem" }}>
                            {r.done ? "\u2713" : loading ? "\u25CF" : "\u25CB"}
                          </span>
                          <span style={{ color: "#6b7280", marginRight: "0.25rem" }}>3{letter}.</span>
                          {r.label}
                          {!r.done && loading && (
                            <span style={{ marginLeft: "0.5rem", color: "#2563eb", fontSize: "0.7rem" }}>
                              researching...
                            </span>
                          )}
                        </span>
                        {r.done && (
                          <span style={{ color: "#999", fontSize: "0.7rem" }}>
                            {isResExpanded ? "\u25B2" : "\u25BC"}
                          </span>
                        )}
                      </button>
                      {isResExpanded && r.done && (() => {
                        const resLogs = logs.filter((l) => l.researcherIndex === idx);
                        return (
                          <div style={{
                            borderTop: "1px solid #f0f0f0",
                            maxHeight: 400,
                            overflowY: "auto",
                          }}>
                            {/* Event log for this researcher */}
                            {resLogs.length > 0 && (
                              <div style={{
                                padding: "0.5rem 1rem 0.5rem 2rem",
                                backgroundColor: "#1e1e1e",
                                fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
                                fontSize: "0.7rem",
                                lineHeight: 1.7,
                                color: "#d4d4d4",
                              }}>
                                {resLogs.map((log, li) => {
                                  const hasTokens = log.totalTokens > 0;
                                  const isLast = li === resLogs.length - 1;
                                  const label = log.message !== log.agent ? log.message.replace(log.agent, "").trim() : "";
                                  return (
                                    <div key={li} style={{ whiteSpace: "nowrap" }}>
                                      <span style={{ color: "#16a34a", marginRight: "0.4rem" }}>{"\u2713"}</span>
                                      <span style={{ color: isLast ? "#4ec9b0" : "#808080" }}>
                                        {hasTokens ? (label ? `${label} ` : "") : (label || "tool response")}
                                      </span>
                                      {hasTokens && (
                                        <span style={{ color: "#ce9178" }}>
                                          [{log.promptTokens}in/{log.responseTokens}out = {log.totalTokens}]
                                        </span>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                            {/* Findings */}
                            <div style={{
                              padding: "0.75rem 1rem 0.75rem 2rem",
                              backgroundColor: "#fafbff",
                              fontSize: "0.8125rem",
                              lineHeight: 1.6,
                              whiteSpace: "pre-wrap",
                              wordBreak: "break-word",
                            }}>
                              {r.findings || "(empty)"}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  );
                })}
              </div>
            );
          })()}

          {/* Step 4: Final Report — always visible */}
          {(() => {
            const isDone = completedSteps.has("answer");
            const isActive = loading && !isDone && researchersDone;
            const isExpanded = expandedStep === "answer";
            return (
              <div style={{ border: "1px solid #e5e7eb", borderRadius: 6, overflow: "hidden", opacity: isDone || isActive ? 1 : 0.5 }}>
                <button
                  onClick={() => isDone && setExpandedStep(isExpanded ? null : "answer")}
                  style={{ width: "100%", padding: "0.625rem 1rem", display: "flex", justifyContent: "space-between", alignItems: "center", backgroundColor: isExpanded ? "#f0f4ff" : "#fafafa", border: "none", cursor: isDone ? "pointer" : "default", fontSize: "0.875rem", fontWeight: 500 }}
                >
                  <span>
                    <span style={{ marginRight: "0.5rem", color: isDone ? "#16a34a" : isActive ? "#2563eb" : "#9ca3af" }}>
                      {isDone ? "\u2713" : isActive ? "\u25CF" : "\u2013"}
                    </span>
                    4. Final Report
                    {isActive && <span style={{ marginLeft: "0.5rem", color: "#2563eb", fontSize: "0.75rem" }}>composing...</span>}
                  </span>
                  {isDone && <span style={{ color: "#999", fontSize: "0.75rem" }}>{isExpanded ? "\u25B2" : "\u25BC"}</span>}
                </button>
                {isExpanded && isDone && (
                  <div style={{ padding: "1rem", backgroundColor: "#fff", borderTop: "1px solid #e5e7eb", fontSize: "0.8125rem", lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: 500, overflowY: "auto" }}>
                    {deepResult.answer || "(empty)"}
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}

      {/* Loading indicator for quick search */}
      {mode !== "deep" && loading && !intentChecking && (
        <div style={{ padding: "2rem", textAlign: "center", color: "#666" }}>
          Searching...
        </div>
      )}
    </main>
  );
}
