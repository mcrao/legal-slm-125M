"use client";

import { useRef, useState } from "react";
import { GEMMA_URL, RAFT_EXAMPLES, RAFT_URL, type Engine } from "@/app/lib/model";

type Status = "idle" | "waking" | "streaming" | "done" | "error";

export default function Raft() {
  const [context, setContext] = useState<string>(RAFT_EXAMPLES[0].context);
  const [question, setQuestion] = useState<string>(RAFT_EXAMPLES[0].question);
  const [answer, setAnswer] = useState<string>("");
  const [engine, setEngine] = useState<Engine>("slm");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const busy = status === "waking" || status === "streaming";

  function loadExample(i: number) {
    if (busy) return;
    setContext(RAFT_EXAMPLES[i].context);
    setQuestion(RAFT_EXAMPLES[i].question);
    setAnswer("");
    setStatus("idle");
    setError("");
  }

  async function run() {
    if (!question.trim() || busy) return;
    setAnswer("");
    setError("");
    setStatus("waking");
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const base = engine === "gemma" ? GEMMA_URL : RAFT_URL;
      const res = await fetch(`${base}/raft`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context, question, max_new_tokens: 180, temperature: 0.5 }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`server ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let first = true;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          try {
            const obj = JSON.parse(line.slice(5).trim());
            if (obj.token) {
              if (first) { first = false; setStatus("streaming"); }
              setAnswer((a) => a + obj.token);
            }
            if (obj.done) setStatus("done");
          } catch { /* ignore */ }
        }
      }
      setStatus((s) => (s === "streaming" || s === "waking" ? "done" : s));
    } catch (e) {
      if ((e as Error).name === "AbortError") setStatus("done");
      else { setError("Could not reach the model. It may be waking up. Try again in a moment."); setStatus("error"); }
    }
  }

  return (
    <div className="paper-card" style={{ padding: "clamp(1.25rem, 3vw, 2rem)" }}>
      {/* model toggle */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        <span className="eyebrow">Answer from context</span>
        <div style={{ display: "flex", alignItems: "center", gap: "0.7rem", flexWrap: "wrap" }}>
          <div className="seg">
            <button data-active={engine === "slm"} onClick={() => !busy && setEngine("slm")} disabled={busy}>Our SLM · 125M</button>
            <button data-active={engine === "gemma"} onClick={() => !busy && setEngine("gemma")} disabled={busy}>Gemma 2 · 2B</button>
          </div>
          <span className="mono" style={{ fontSize: "0.68rem", color: "var(--faint)" }}>
            {engine === "slm" ? "full FT · hosted CPU" : "QLoRA · hosted GPU"}
          </span>
        </div>
      </div>

      <div className="eyebrow" style={{ marginBottom: "0.7rem" }}>Try an example</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1.3rem" }}>
        {RAFT_EXAMPLES.map((ex, i) => (
          <button key={ex.label} className="chip" onClick={() => loadExample(i)} disabled={busy}>
            {ex.label}
          </button>
        ))}
      </div>

      <label className="eyebrow" style={{ color: "var(--muted)", display: "block", marginBottom: "0.5rem" }}>Context (paste documents)</label>
      <textarea
        value={context}
        onChange={(e) => setContext(e.target.value)}
        rows={6}
        spellCheck={false}
        placeholder="Paste the context the model should answer from. Add unrelated text too — it should ignore it."
        style={taStyle}
      />

      <label className="eyebrow" style={{ color: "var(--muted)", display: "block", margin: "1.1rem 0 0.5rem" }}>Question</label>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(); } }}
        rows={2}
        spellCheck={false}
        placeholder="Ask a question answerable from the context above…"
        style={taStyle}
      />

      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginTop: "1.2rem" }}>
        {!busy ? (
          <button className="btn-primary" onClick={run} disabled={!question.trim()}>Answer&nbsp;from&nbsp;context&nbsp;→</button>
        ) : (
          <button className="btn-primary" onClick={() => abortRef.current?.abort()} style={{ background: "var(--brass)", borderColor: "var(--brass)", boxShadow: "none" }}>Stop</button>
        )}
        <span className="mono" style={{ fontSize: "0.74rem", color: status === "error" ? "var(--brass)" : "var(--faint)" }}>
          {status === "waking" ? "waking the model…" : status === "streaming" ? "grounding…" : status === "done" ? "done" : ""}
        </span>
      </div>

      <div
        aria-live="polite"
        style={{
          marginTop: "1.3rem", minHeight: "6rem", background: "var(--paper-3)",
          border: "1px solid var(--line)", borderLeft: "2px solid var(--green)", borderRadius: 4,
          padding: "1.1rem 1.2rem", fontFamily: "var(--font-mono)", fontSize: "0.9rem",
          lineHeight: 1.7, whiteSpace: "pre-wrap", color: "var(--ink)",
        }}
      >
        {answer ? renderQuotes(answer) : (
          <span style={{ color: "var(--faint)" }}>The grounded answer appears here. It quotes the context it relies on, then gives the final answer.</span>
        )}
        {busy && <span className="caret" />}
        {error && <span style={{ display: "block", marginTop: "0.8rem", color: "var(--brass)", fontFamily: "var(--font-sans)", fontSize: "0.85rem" }}>{error}</span>}
      </div>

      <p style={{ marginTop: "1rem", fontSize: "0.8rem", color: "var(--faint)", lineHeight: 1.6 }}>
        RAFT (Retrieval-Augmented Fine-Tuning): the model answers from the context you provide and ignores distractors.{" "}
        {engine === "slm"
          ? "Our 125M model answers from context, but is too small to reliably know when the answer isn't there — ask about something absent and it will confidently make one up. That limit is the point."
          : "Gemma 2B was QLoRA-tuned on the same data (0.79% of weights) to decline when the answer isn't in the context, instead of guessing. Wakes on your first request."}
        {" "}Not legal or financial advice.
      </p>
    </div>
  );
}

// Render ##begin_quote##…##end_quote## as a highlighted inline quote.
function renderQuotes(text: string) {
  const parts = text.split(/(##begin_quote##|##end_quote##)/g);
  const out: React.ReactNode[] = [];
  let inQuote = false;
  let key = 0;
  for (const p of parts) {
    if (p === "##begin_quote##") { inQuote = true; continue; }
    if (p === "##end_quote##") { inQuote = false; continue; }
    if (!p) continue;
    out.push(
      inQuote
        ? <mark key={key++} style={{ background: "rgba(46,90,67,0.14)", color: "var(--ink)", padding: "0 2px", borderRadius: 2 }}>{p}</mark>
        : <span key={key++}>{p}</span>,
    );
  }
  return out;
}

const taStyle: React.CSSProperties = {
  width: "100%", resize: "vertical", background: "var(--paper-3)",
  border: "1px solid var(--line-2)", borderRadius: 4, padding: "0.9rem 1rem",
  fontFamily: "var(--font-mono)", fontSize: "0.9rem", color: "var(--ink)",
  lineHeight: 1.6, outline: "none",
};
