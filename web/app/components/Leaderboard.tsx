"use client";

import { useMemo, useState } from "react";
import { LEADER_METRICS, LEADERBOARD, type LeaderRow } from "@/app/lib/model";

type Key = (typeof LEADER_METRICS)[number]["key"];

const famTone = (f: string) => (f === "slm" ? "var(--green)" : "var(--brass)");
const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function Leaderboard() {
  const rows = LEADERBOARD.models;
  const [sortKey, setSortKey] = useState<Key | "none">("none");

  const metaByKey = Object.fromEntries(LEADER_METRICS.map((m) => [m.key, m])) as Record<Key, (typeof LEADER_METRICS)[number]>;

  // per-column min/max for bar scaling + best value
  const stats = useMemo(() => {
    const s: Record<string, { min: number; max: number; best: number }> = {};
    for (const m of LEADER_METRICS) {
      const vals = rows.map((r) => r[m.key] as number);
      const min = Math.min(...vals), max = Math.max(...vals);
      s[m.key] = { min, max, best: m.better === "low" ? min : max };
    }
    return s;
  }, [rows]);

  const sorted = useMemo(() => {
    if (sortKey === "none") return rows;
    const better = metaByKey[sortKey].better;
    return [...rows].sort((a, b) => {
      const av = a[sortKey] as number, bv = b[sortKey] as number;
      return better === "low" ? av - bv : bv - av;
    });
  }, [rows, sortKey, metaByKey]);

  if (rows.length === 0) {
    return (
      <div className="paper-card" style={{ padding: "2rem", textAlign: "center", color: "var(--faint)" }}>
        Leaderboard is being computed. Run <span className="mono">modal run leaderboard_eval.py::run</span> and paste the results.
      </div>
    );
  }

  // Accuracy / refusal are true 0–100% metrics, so the bar is the ABSOLUTE percentage
  // (37% fills 37% of the track). Bits/byte is an unbounded score with no natural 100%,
  // so its bar is relative to the worst value in the column. Green always marks the best.
  function barWidth(key: Key, v: number) {
    if (key === "bits_per_byte") {
      const { max } = stats[key];
      return max <= 0 ? 3 : 3 + (v / max) * 97;
    }
    return Math.max(2, Math.min(100, v * 100));
  }

  return (
    <div className="paper-card" style={{ padding: "clamp(1rem, 2.5vw, 1.5rem)" }}>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.86rem", minWidth: 720 }}>
          <thead>
            <tr>
              <Th style={{ textAlign: "left", minWidth: 180 }}>Model</Th>
              {LEADER_METRICS.map((m) => (
                <Th key={m.key} title={m.hint} onClick={() => setSortKey(m.key)}
                    active={sortKey === m.key}>
                  {m.label} {m.better === "low" ? "↓" : "↑"}
                </Th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.id} style={{ borderTop: "1px solid var(--line)" }}>
                <td style={{ padding: "0.7rem 0.6rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: famTone(r.family), flexShrink: 0 }} />
                    <span style={{ color: "var(--ink)", fontWeight: 500 }}>{r.name}</span>
                    <span className="mono" style={{ fontSize: "0.64rem", color: "var(--faint)", border: "1px solid var(--line-2)", borderRadius: 999, padding: "0.05rem 0.4rem" }}>{r.params}</span>
                  </div>
                  <div className="mono" style={{ fontSize: "0.64rem", color: "var(--faint)", marginTop: "0.2rem", marginLeft: "1rem" }}>{r.arch} · {r.note}</div>
                </td>
                {LEADER_METRICS.map((m) => {
                  const v = r[m.key] as number;
                  const isBest = v === stats[m.key].best;
                  const disp = m.key === "bits_per_byte" ? v.toFixed(2) : pct(v);
                  return (
                    <td key={m.key} style={{ padding: "0.7rem 0.6rem", verticalAlign: "middle" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <div style={{ flex: 1, height: 6, background: "var(--paper-3)", borderRadius: 3, overflow: "hidden", minWidth: 40 }}>
                          <div style={{ width: `${barWidth(m.key, v)}%`, height: "100%", background: isBest ? "var(--green)" : "var(--line-2)", opacity: isBest ? 0.7 : 1, transition: "width 0.4s" }} />
                        </div>
                        {isBest && <span className="mono" style={{ fontSize: "0.55rem", letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--green)" }}>best</span>}
                        <span className="mono tnum" style={{ fontSize: "0.8rem", minWidth: "3.5ch", textAlign: "right", color: isBest ? "var(--green)" : "var(--ink-soft)", fontWeight: isBest ? 600 : 400 }}>{disp}</span>
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* metric legend */}
      <div style={{ marginTop: "1.25rem", display: "grid", gap: "0.4rem" }}>
        {LEADER_METRICS.map((m) => (
          <p key={m.key} style={{ margin: 0, fontSize: "0.76rem", color: "var(--faint)", lineHeight: 1.5 }}>
            <span className="mono" style={{ color: "var(--muted)" }}>{m.label} {m.better === "low" ? "↓" : "↑"}</span> — {m.hint}
          </p>
        ))}
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.72rem", color: "var(--faint)", lineHeight: 1.5 }}>
          <span style={{ color: "var(--green)", fontWeight: 600 }}>Green</span> marks the best score in each
          column — that means the <b>lowest</b> bits/byte (the arrow is ↓) and the <b>highest</b> accuracy and
          refusal (↑). Accuracy and refusal bars are absolute (0–100%); the bits/byte bar is relative to the
          worst value, since it has no natural ceiling. Click a column header to sort.
        </p>
      </div>
    </div>
  );
}

function Th({ children, style, onClick, active, title }: { children: React.ReactNode; style?: React.CSSProperties; onClick?: () => void; active?: boolean; title?: string }) {
  return (
    <th
      onClick={onClick}
      title={title}
      className="section-num"
      style={{ padding: "0 0.6rem 0.6rem", textAlign: "right", whiteSpace: "nowrap", cursor: onClick ? "pointer" : "default", color: active ? "var(--ink)" : "var(--faint)", userSelect: "none", ...style }}
    >
      {children}
    </th>
  );
}
