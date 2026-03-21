export default function Page() {
  return (
    <main
      style={{
        maxWidth: 700,
        margin: "0 auto",
        padding: "3rem 1.5rem",
        fontFamily: "system-ui, sans-serif",
        color: "#1e293b",
        lineHeight: 1.7,
      }}
    >
      <h1 style={{ fontSize: "1.5rem", marginBottom: "0.25rem" }}>
        House Majority Training Assistant &mdash; API
      </h1>
      <p style={{ color: "#64748b", marginBottom: "2rem", fontSize: "0.875rem" }}>
        Backend service for the HMSO Training Chatbot
      </p>

      <section style={{ marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>Connecting</h2>
        <p style={{ fontSize: "0.875rem", marginBottom: "0.75rem" }}>
          All endpoints require an API key passed via the{" "}
          <code style={{ backgroundColor: "#f1f5f9", padding: "0.15rem 0.4rem", borderRadius: 4, fontSize: "0.8125rem" }}>
            x-api-key
          </code>{" "}
          header.
        </p>
        <pre
          style={{
            backgroundColor: "#1e1e1e",
            color: "#d4d4d4",
            padding: "1rem",
            borderRadius: 6,
            fontSize: "0.8rem",
            overflowX: "auto",
          }}
        >{`curl -X POST <backend-url>/api/intent \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: <your-api-key>" \\
  -d '{"query": "What is the onboarding process?"}'`}</pre>
      </section>

      <section style={{ marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>Available Endpoints</h2>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "0.8125rem",
          }}
        >
          <thead>
            <tr style={{ borderBottom: "2px solid #e2e8f0", textAlign: "left" }}>
              <th style={{ padding: "0.5rem 0.75rem" }}>Method</th>
              <th style={{ padding: "0.5rem 0.75rem" }}>Endpoint</th>
              <th style={{ padding: "0.5rem 0.75rem" }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["POST", "/api/intent", "Intent orchestrator — classifies and enriches the query"],
              ["POST", "/api/quick-search", "Quick search using Flash model + RAG"],
              ["POST", "/api/quick-search-pro", "Quick search using Pro model + RAG"],
              ["POST", "/api/search-escalate", "Escalation search (Pro) after an unsatisfactory Flash answer"],
              ["POST", "/api/conversational", "Conversational follow-up (no RAG)"],
              ["POST", "/api/research", "Deep research pipeline (SSE stream)"],
            ].map(([method, path, desc]) => (
              <tr key={path} style={{ borderBottom: "1px solid #f1f5f9" }}>
                <td style={{ padding: "0.5rem 0.75rem" }}>
                  <code style={{ backgroundColor: "#dbeafe", color: "#1e40af", padding: "0.1rem 0.35rem", borderRadius: 3, fontSize: "0.75rem" }}>
                    {method}
                  </code>
                </td>
                <td style={{ padding: "0.5rem 0.75rem" }}>
                  <code style={{ fontSize: "0.8rem" }}>{path}</code>
                </td>
                <td style={{ padding: "0.5rem 0.75rem", color: "#64748b" }}>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section style={{ marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>Allowed Origins (CORS)</h2>
        <ul style={{ fontSize: "0.875rem", paddingLeft: "1.25rem", margin: 0 }}>
          <li><code style={{ fontSize: "0.8rem" }}>https://hmso-training.ics.hawaii.edu</code></li>
          <li><code style={{ fontSize: "0.8rem" }}>http://localhost:3000</code> (development)</li>
        </ul>
        <p style={{ fontSize: "0.8125rem", color: "#64748b", marginTop: "0.5rem" }}>
          Requests from other origins are rejected unless a valid{" "}
          <code style={{ fontSize: "0.8rem" }}>x-api-key</code> is provided.
        </p>
      </section>

      <footer style={{ borderTop: "1px solid #e2e8f0", paddingTop: "1rem", fontSize: "0.75rem", color: "#94a3b8" }}>
        Sponsored by the House of Majority Staff Office (HMSO)
      </footer>
    </main>
  );
}
