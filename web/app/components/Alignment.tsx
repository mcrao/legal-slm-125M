"use client";

import { useRef, useState } from "react";
import { ALIGN_SIZES, alignModelId, UNIVERSAL_URL } from "@/app/lib/model";

type Status = "idle" | "waking" | "streaming" | "done" | "error";
type Method = "dpo" | "rlaif";
type Stage = "sft" | "raft";

const RAFT_CTX =
  "[1] The Company entered into a five-year lease for its headquarters at an annual rent of $2.4 million.\n[2] The board declared a quarterly dividend of $0.15 per share.";

export default function Alignment() {
  const [method, setMethod] = useState<Method>("dpo");
  const [stage, setStage] = useState<Stage>("sft");
  const [sizeKey, setSizeKey] = useState<string>(ALIGN_SIZES[0].key);
  const [message, setMessage] = useState("What must a plaintiff prove in a breach of contract claim?");
  const [context, setContext] = useState(RAFT_CTX);
  const [question, setQuestion] = useState("What is the annual rent for the headquarters lease?");
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const busy = status === "waking" || status === "streaming";

  const size = ALIGN_SIZES.find((s) => s.key === sizeKey) ?? ALIGN_SIZES[0];
  const modelId = alignModelId(size.prefix, stage, method);

  async function run() {
    if (busy) return;
    setAnswer("");
    setError("");
    setStatus("waking");
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const endpoint = stage === "sft" ? "chat" : "raft";
    const body = stage === "sft"
      ? { model_id: modelId, message, max_new_tokens: 180, temperature: 0.7 }
      : { model_id: modelId, context, question, max_new_tokens: 180, temperature: 0.5 };
    try {
      const res = await fetch(`${UNIVERSAL_URL}/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`server ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "", first = true;
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
            if (obj.token) { if (first) { first = false; setStatus("streaming"); } setAnswer((a) => a + obj.token); }
            if (obj.done) setStatus("done");
          } catch { /* ignore */ }
        }
      }
      setStatus((s) => (s === "streaming" || s === "waking" ? "done" : s));
    } catch (e) {
      if ((e as Error).name === "AbortError") setStatus("done");
      else { setError("Could not reach the model. The GPU may be waking (~1 min) — try again."); setStatus("error"); }
    }
  }

  return (
    <div className="paper-card" style={{ padding: "clamp(1.25rem, 3vw, 2rem)" }}>
      {/* pickers */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "1.4rem", marginBottom: "1.3rem" }}>
        <Picker label="Method" value={method} onChange={(v) => setMethod(v as Method)}
                options={[["dpo", "DPO"], ["rlaif", "RLAIF"]]} disabled={busy} />
        <Picker label="Built on" value={stage} onChange={(v) => setStage(v as Stage)}
                options={[["sft", "SFT"], ["raft", "RAFT"]]} disabled={busy} />
        <Picker label="Model" value={sizeKey} onChange={setSizeKey}
                options={ALIGN_SIZES.map((s) => [s.key, s.label] as [string, string])} disabled={busy} />
      </div>

      <div className="mono" style={{ fontSize: "0.68rem", color: "var(--faint)", marginBottom: "1.1rem" }}>
        {modelId.replace("jonam-ai/", "")}
      </div>

      {stage === "sft" ? (
        <textarea value={message} onChange={(e) => setMessage(e.target.value)} rows={2} spellCheck={false}
          placeholder="Ask a legal or financial question…" style={ta} />
      ) : (
        <>
          <label className="eyebrow" style={{ color: "var(--muted)", display: "block", marginBottom: "0.5rem" }}>Context</label>
          <textarea value={context} onChange={(e) => setContext(e.target.value)} rows={4} spellCheck={false} style={ta} />
          <label className="eyebrow" style={{ color: "var(--muted)", display: "block", margin: "1rem 0 0.5rem" }}>Question</label>
          <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={2} spellCheck={false} style={ta} />
        </>
      )}

      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginTop: "1.2rem" }}>
        {!busy ? (
          <button className="btn-primary" onClick={run}>Run {method.toUpperCase()}&nbsp;→</button>
        ) : (
          <button className="btn-primary" onClick={() => abortRef.current?.abort()} style={{ background: "var(--brass)", borderColor: "var(--brass)", boxShadow: "none" }}>Stop</button>
        )}
        <span className="mono" style={{ fontSize: "0.74rem", color: status === "error" ? "var(--brass)" : "var(--faint)" }}>
          {status === "waking" ? "waking the GPU…" : status === "streaming" ? "generating…" : status === "done" ? "done" : ""}
        </span>
      </div>

      <div aria-live="polite" style={{ marginTop: "1.3rem", minHeight: "6rem", background: "var(--paper-3)", border: "1px solid var(--line)", borderLeft: "2px solid var(--brass)", borderRadius: 4, padding: "1.1rem 1.2rem", fontFamily: "var(--font-mono)", fontSize: "0.9rem", lineHeight: 1.7, whiteSpace: "pre-wrap", color: "var(--ink)" }}>
        {answer ? renderQuotes(answer) : <span style={{ color: "var(--faint)" }}>The preference-optimized model&apos;s answer appears here.</span>}
        {busy && <span className="caret" />}
        {error && <span style={{ display: "block", marginTop: "0.8rem", color: "var(--brass)", fontFamily: "var(--font-sans)", fontSize: "0.85rem" }}>{error}</span>}
      </div>

      <p style={{ marginTop: "1rem", fontSize: "0.8rem", color: "var(--faint)", lineHeight: 1.6 }}>
        Every combination is a real model on Hugging Face: DPO or RLAIF (reward model + GRPO), on top of the
        SFT or RAFT version, at 125M, 500M, or 2B. Preference optimization mostly tunes tone and structure,
        so read for <em>quality</em>, not just correctness. Not legal or financial advice.
      </p>
    </div>
  );
}

function Picker({ label, value, onChange, options, disabled }: {
  label: string; value: string; onChange: (v: string) => void; options: [string, string][]; disabled: boolean;
}) {
  return (
    <div>
      <div className="eyebrow" style={{ color: "var(--muted)", marginBottom: "0.5rem" }}>{label}</div>
      <div className="seg">
        {options.map(([v, l]) => (
          <button key={v} data-active={value === v} onClick={() => !disabled && onChange(v)} disabled={disabled}>{l}</button>
        ))}
      </div>
    </div>
  );
}

function renderQuotes(text: string) {
  const parts = text.split(/(##begin_quote##|##end_quote##)/g);
  const out: React.ReactNode[] = [];
  let inQuote = false, key = 0;
  for (const p of parts) {
    if (p === "##begin_quote##") { inQuote = true; continue; }
    if (p === "##end_quote##") { inQuote = false; continue; }
    if (!p) continue;
    out.push(inQuote
      ? <mark key={key++} style={{ background: "rgba(46,90,67,0.14)", color: "var(--ink)", padding: "0 2px", borderRadius: 2 }}>{p}</mark>
      : <span key={key++}>{p}</span>);
  }
  return out;
}

const ta: React.CSSProperties = {
  width: "100%", resize: "vertical", background: "var(--paper-3)", border: "1px solid var(--line-2)",
  borderRadius: 4, padding: "0.8rem 1rem", fontFamily: "var(--font-mono)", fontSize: "0.9rem",
  color: "var(--ink)", lineHeight: 1.6, outline: "none",
};
