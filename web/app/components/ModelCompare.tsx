import { COMPARE, HF_GEMMA_RAFT_URL, HF_GEMMA_SFT_URL, HF_RAFT_URL, HF_SFT_URL } from "@/app/lib/model";

const LINKS: Record<string, { sft: string; raft: string }> = {
  slm: { sft: HF_SFT_URL, raft: HF_RAFT_URL },
  gemma: { sft: HF_GEMMA_SFT_URL, raft: HF_GEMMA_RAFT_URL },
};

export default function ModelCompare() {
  return (
    <div style={{ display: "grid", gap: "1.5rem", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
      {(["slm", "gemma"] as const).map((id) => {
        const m = COMPARE[id];
        const accent = id === "slm" ? "var(--green)" : "var(--brass)";
        return (
          <div key={id} className="paper-card" style={{ padding: "clamp(1.25rem, 2.5vw, 1.75rem)", display: "flex", flexDirection: "column" }}>
            {/* header */}
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
              <h3 className="display" style={{ fontSize: "1.5rem", margin: 0 }}>{m.name}</h3>
              <span className="mono" style={{ fontSize: "0.66rem", letterSpacing: "0.08em", textTransform: "uppercase", color: accent, border: `1px solid ${accent}`, borderRadius: 999, padding: "0.15rem 0.6rem", opacity: 0.85 }}>
                {m.tag}
              </span>
            </div>
            <div className="mono" style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: "0.55rem", lineHeight: 1.6 }}>
              <span style={{ color: "var(--ink)" }}>{m.params}</span> parameters · {m.arch}
            </div>

            {/* phase rows */}
            <div style={{ marginTop: "1.25rem", display: "grid", gap: "0.6rem" }}>
              {m.phases.map((p) => {
                const link = p.phase === "SFT" ? LINKS[id].sft : p.phase === "RAFT" ? LINKS[id].raft : null;
                return (
                  <div key={p.phase} style={{ background: "var(--paper-3)", border: "1px solid var(--line)", borderLeft: `2px solid ${accent}`, borderRadius: 4, padding: "0.8rem 0.95rem" }}>
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "0.5rem" }}>
                      <span style={{ fontWeight: 500, color: "var(--ink)" }}>
                        {p.phase}
                        {link && (
                          <a href={link} target="_blank" rel="noopener" className="link-underline" style={{ color: accent, fontSize: "0.7rem", marginLeft: "0.45rem" }}>↗</a>
                        )}
                      </span>
                      <span className="mono" style={{ fontSize: "0.7rem", color: "var(--muted)" }}>{p.method}</span>
                    </div>
                    <dl style={{ margin: "0.6rem 0 0", display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: "0.5rem 1rem" }}>
                      <Cell k="Trainable" v={p.trainable} />
                      <Cell k="Train tokens" v={p.tokens} sub={p.note} />
                      <Cell k="Compute" v={p.gpu} />
                      <Cell k="Cost" v={p.cost} accent={accent} />
                    </dl>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Cell({ k, v, sub, accent }: { k: string; v: string; sub?: string; accent?: string }) {
  return (
    <div>
      <dt className="mono" style={{ fontSize: "0.6rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--faint)" }}>{k}</dt>
      <dd className="mono" style={{ margin: "0.15rem 0 0", fontSize: "0.86rem", color: accent ?? "var(--ink)" }}>
        {v}
        {sub && <span style={{ color: "var(--faint)", fontSize: "0.68rem" }}> · {sub}</span>}
      </dd>
    </div>
  );
}
