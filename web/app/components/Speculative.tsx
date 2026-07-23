"use client";

import { useState } from "react";
import { SPEC_PRESETS, SPEC_URL } from "@/app/lib/model";

type Tok = { text: string; source: "draft" | "target" };
type Result = {
  prompt: string;
  target_only: { output: string; tokens_per_sec: number; seconds: number; tokens: number };
  speculative: {
    output: string; tokens_per_sec: number; seconds: number; tokens: number;
    acceptance_rate: number; token_provenance: Tok[];
  };
  speedup: number;
};

export default function Speculative() {
  const [prompt, setPrompt] = useState<string>(SPEC_PRESETS[0].prompt);
  const [status, setStatus] = useState<"idle" | "waking" | "running" | "done" | "error">("idle");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const busy = status === "waking" || status === "running";

  async function run() {
    if (!prompt.trim() || busy) return;
    setError("");
    setStatus("waking");
    try {
      const res = await fetch(`${SPEC_URL}/spec`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: 96 }),
      });
      if (!res.ok) throw new Error(`server ${res.status}`);
      setStatus("running");
      const data = (await res.json()) as Result;
      setResult(data);
      setStatus("done");
    } catch {
      setError("Couldn't reach the GPU. The 7B model takes ~2 min to wake on the first call — try again shortly.");
      setStatus("error");
    }
  }

  const s = result?.speculative;
  const maxTps = result ? Math.max(result.target_only.tokens_per_sec, result.speculative.tokens_per_sec) : 1;
  const accept = s ? Math.round(s.acceptance_rate * 100) : 0;

  return (
    <div className="paper-card" style={{ padding: "clamp(1.25rem, 3vw, 2rem)" }}>
      <div className="eyebrow" style={{ marginBottom: "0.5rem" }}>Speculative decoding · Qwen 2.5 7B + 0.5B</div>
      <p style={{ margin: "0 0 1.3rem", color: "var(--muted)", fontSize: "0.95rem", lineHeight: 1.6 }}>
        The small 0.5B <b>draft</b> proposes several tokens at once; the big 7B <b>target</b> verifies
        them in a single pass and keeps the longest prefix it agrees with. Same output as the 7B alone,
        but the speedup rides entirely on how often the draft is right.
      </p>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.9rem" }}>
        {SPEC_PRESETS.map((p) => (
          <button key={p.label} className="chip" disabled={busy} onClick={() => setPrompt(p.prompt)}>
            {p.label}
          </button>
        ))}
      </div>
      <textarea
        value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={2} spellCheck={false}
        style={{ width: "100%", resize: "vertical", background: "var(--paper-3)", border: "1px solid var(--line-2)", borderRadius: 4, padding: "0.8rem 1rem", fontFamily: "var(--font-mono)", fontSize: "0.88rem", color: "var(--ink)", lineHeight: 1.6, outline: "none" }}
      />

      <button className="btn-primary" onClick={run} disabled={busy || !prompt.trim()} style={{ marginTop: "1.1rem" }}>
        {status === "waking" ? "waking 7B GPU (~2 min first time)…" : status === "running" ? "decoding…" : "Run both →"}
      </button>

      {error && <p style={{ margin: "1rem 0 0", color: "var(--brass)", fontSize: "0.85rem" }}>{error}</p>}

      {result && s && (
        <div style={{ marginTop: "1.75rem" }}>
          <Bar label="Target only (7B, one token at a time)" tps={result.target_only.tokens_per_sec}
               secs={result.target_only.seconds} pct={(result.target_only.tokens_per_sec / maxTps) * 100} tone="var(--brass)" />
          <Bar label="Speculative (7B verifies 0.5B)" tps={s.tokens_per_sec}
               secs={s.seconds} pct={(s.tokens_per_sec / maxTps) * 100} tone="var(--green)" />

          <div style={{ display: "flex", flexWrap: "wrap", gap: "1.5rem", alignItems: "baseline", margin: "1.2rem 0 0.2rem" }}>
            <div>
              <span className="stat-num" style={{ fontSize: "2.2rem", color: result.speedup >= 1 ? "var(--green)" : "var(--brass)" }}>{result.speedup}×</span>
              <span style={{ color: "var(--muted)", fontSize: "0.9rem", marginLeft: "0.5rem" }}>throughput</span>
            </div>
            <div>
              <span className="stat-num" style={{ fontSize: "2.2rem", color: "var(--ink)" }}>{accept}%</span>
              <span style={{ color: "var(--muted)", fontSize: "0.9rem", marginLeft: "0.5rem" }}>draft tokens accepted</span>
            </div>
          </div>
          <p className="mono" style={{ fontSize: "0.72rem", color: "var(--faint)", margin: "0.6rem 0 0", lineHeight: 1.5 }}>
            {accept >= 55
              ? "High acceptance → the draft nails predictable text, so speculation pays off."
              : "Low acceptance → the draft and target disagree often, so verification overhead outweighs the savings. Speedup is prompt-dependent."}
          </p>

          {/* provenance */}
          <div style={{ margin: "1.3rem 0 0.5rem", display: "flex", gap: "1.2rem", fontSize: "0.72rem", color: "var(--muted)" }}>
            <span><Swatch tone="var(--green)" /> proposed by draft, accepted</span>
            <span><Swatch tone="var(--brass)" /> produced by target</span>
          </div>
          <div style={{ background: "var(--paper-3)", border: "1px solid var(--line)", borderRadius: 4, padding: "1rem 1.1rem", fontFamily: "var(--font-mono)", fontSize: "0.85rem", lineHeight: 1.9, whiteSpace: "pre-wrap", maxHeight: "16rem", overflowY: "auto" }}>
            {s.token_provenance.map((t, i) => (
              <span key={i} style={{
                background: t.source === "draft" ? "rgba(46,90,67,0.18)" : "rgba(168,129,74,0.16)",
                color: "var(--ink)", borderRadius: 2, padding: "0.05em 0",
                boxShadow: t.source === "draft" ? "inset 0 -2px 0 var(--green)" : "inset 0 -2px 0 var(--brass)",
              }}>{t.text}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Bar({ label, tps, secs, pct, tone }: { label: string; tps: number; secs: number; pct: number; tone: string }) {
  return (
    <div style={{ marginBottom: "0.85rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.3rem", gap: "1rem" }}>
        <span style={{ fontSize: "0.9rem", color: "var(--ink)" }}>{label}</span>
        <span className="mono" style={{ fontSize: "0.82rem", color: tone, whiteSpace: "nowrap" }}>
          {tps} tok/s <span style={{ color: "var(--faint)" }}>· {secs}s</span>
        </span>
      </div>
      <div style={{ height: 10, background: "var(--paper-3)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden" }}>
        <div style={{ width: `${Math.max(2, pct)}%`, height: "100%", background: tone, opacity: 0.55, transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
}

function Swatch({ tone }: { tone: string }) {
  return <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: tone, opacity: 0.6, marginRight: 4, verticalAlign: "middle" }} />;
}
