"use client";

import { useState } from "react";
import { KV_URL } from "@/app/lib/model";

type Result = {
  batch_size: number;
  max_new_tokens: number;
  output: string;
  with_cache: { seconds: number; tokens_per_sec: number };
  without_cache: { seconds: number; tokens_per_sec: number };
  speedup: number;
  total_tokens: number;
};

export default function KvCache() {
  const [batch, setBatch] = useState(16);
  const [status, setStatus] = useState<"idle" | "waking" | "running" | "done" | "error">("idle");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const busy = status === "waking" || status === "running";

  async function run() {
    if (busy) return;
    setError("");
    setStatus("waking");
    try {
      const res = await fetch(`${KV_URL}/bench`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_size: batch, max_new_tokens: 128 }),
      });
      if (!res.ok) throw new Error(`server ${res.status}`);
      setStatus("running");
      const data = (await res.json()) as Result;
      setResult(data);
      setStatus("done");
    } catch {
      setError("Couldn't reach the benchmark GPU. It may be waking up (~30s) — try again.");
      setStatus("error");
    }
  }

  const maxTps = result ? Math.max(result.with_cache.tokens_per_sec, result.without_cache.tokens_per_sec) : 1;

  return (
    <div className="paper-card" style={{ padding: "clamp(1.25rem, 3vw, 2rem)" }}>
      <div className="eyebrow" style={{ marginBottom: "0.5rem" }}>KV cache · 125M SLM</div>
      <p style={{ margin: "0 0 1.4rem", color: "var(--muted)", fontSize: "0.95rem", lineHeight: 1.6 }}>
        Without a KV cache the model recomputes attention over the whole growing sequence every
        step. With it, each step reuses the stored keys/values. Raise the batch size (more
        concurrent &quot;users&quot;) and watch the gap explode.
      </p>

      {/* batch slider */}
      <label className="eyebrow" style={{ color: "var(--muted)", display: "block", marginBottom: "0.6rem" }}>
        Batch size (concurrent sequences): <span className="mono" style={{ color: "var(--green)" }}>{batch}</span>
      </label>
      <input
        type="range" min={1} max={64} step={1} value={batch} disabled={busy}
        onChange={(e) => setBatch(Number(e.target.value))}
        style={{ width: "100%", accentColor: "var(--green)", cursor: busy ? "default" : "pointer" }}
      />
      <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: "0.66rem", color: "var(--faint)", marginTop: "0.25rem" }}>
        <span>1</span><span>128 tokens each · greedy</span><span>64</span>
      </div>

      <button className="btn-primary" onClick={run} disabled={busy} style={{ marginTop: "1.3rem" }}>
        {status === "waking" ? "waking GPU…" : status === "running" ? "benchmarking…" : "Run benchmark →"}
      </button>

      {error && <p style={{ margin: "1rem 0 0", color: "var(--brass)", fontSize: "0.85rem" }}>{error}</p>}

      {result && (
        <div style={{ marginTop: "1.75rem" }}>
          <Bar label="With KV cache" tps={result.with_cache.tokens_per_sec} secs={result.with_cache.seconds}
               pct={(result.with_cache.tokens_per_sec / maxTps) * 100} tone="var(--green)" />
          <Bar label="Without KV cache" tps={result.without_cache.tokens_per_sec} secs={result.without_cache.seconds}
               pct={(result.without_cache.tokens_per_sec / maxTps) * 100} tone="var(--brass)" />

          <div style={{ display: "flex", alignItems: "baseline", gap: "0.6rem", margin: "1.3rem 0 0.4rem" }}>
            <span className="stat-num" style={{ fontSize: "2.4rem", color: "var(--green)" }}>{result.speedup}×</span>
            <span style={{ color: "var(--muted)", fontSize: "0.95rem" }}>
              faster with the cache at batch {result.batch_size}
            </span>
          </div>

          <div className="mono" style={{ marginTop: "1.1rem", background: "var(--paper-3)", border: "1px solid var(--line)", borderRadius: 4, padding: "0.9rem 1rem", fontSize: "0.82rem", color: "var(--ink-soft)", lineHeight: 1.55, maxHeight: "8rem", overflowY: "auto" }}>
            <span className="mono" style={{ display: "block", fontSize: "0.6rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--faint)", marginBottom: "0.4rem" }}>
              identical output, both paths
            </span>
            {result.output}
          </div>
        </div>
      )}
    </div>
  );
}

function Bar({ label, tps, secs, pct, tone }: { label: string; tps: number; secs: number; pct: number; tone: string }) {
  return (
    <div style={{ marginBottom: "0.85rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.3rem" }}>
        <span style={{ fontSize: "0.9rem", color: "var(--ink)" }}>{label}</span>
        <span className="mono" style={{ fontSize: "0.82rem", color: tone }}>
          {tps.toLocaleString()} tok/s <span style={{ color: "var(--faint)" }}>· {secs}s</span>
        </span>
      </div>
      <div style={{ height: 10, background: "var(--paper-3)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden" }}>
        <div style={{ width: `${Math.max(2, pct)}%`, height: "100%", background: tone, opacity: 0.55, transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
}
