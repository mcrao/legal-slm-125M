"use client";

import { useMemo, useState } from "react";
import { JUDGE_LEADERBOARD, JUDGE_METRICS, type JudgeRow } from "@/app/lib/model";

type Key = (typeof JUDGE_METRICS)[number]["key"];
const famTone = (f: string) => (f === "slm" ? "var(--green)" : "var(--brass)");

export default function JudgeLeaderboard() {
  const rows = JUDGE_LEADERBOARD.models;
  const [sortKey, setSortKey] = useState<Key>("judge_overall");
  const metaByKey = Object.fromEntries(JUDGE_METRICS.map((m) => [m.key, m])) as Record<Key, (typeof JUDGE_METRICS)[number]>;

  const stats = useMemo(() => {
    const s: Record<string, { max: number; best: number }> = {};
    for (const m of JUDGE_METRICS) {
      const vals = rows.map((r) => r[m.key] as number | null).filter((v): v is number => v != null);
      if (!vals.length) { s[m.key] = { max: 1, best: 0 }; continue; }
      s[m.key] = { max: Math.max(...vals), best: m.better === "low" ? Math.min(...vals) : Math.max(...vals) };
    }
    return s;
  }, [rows]);

  const sorted = useMemo(() => {
    const better = metaByKey[sortKey].better;
    return [...rows].sort((a, b) => {
      const av = (a[sortKey] as number) ?? -1, bv = (b[sortKey] as number) ?? -1;
      return better === "low" ? av - bv : bv - av;
    });
  }, [rows, sortKey, metaByKey]);

  if (rows.length === 0) {
    return <div className="paper-card" style={{ padding: "2rem", textAlign: "center", color: "var(--faint)" }}>
      Judge scores are being computed. Run <span className="mono">modal run judge_eval.py::run</span>.
    </div>;
  }

  function barWidth(m: (typeof JUDGE_METRICS)[number], v: number | null) {
    if (v == null) return 0;
    return m.scale === "out10" ? Math.max(2, (v / 10) * 100) : Math.max(2, Math.min(100, v * 100));
  }
  function disp(m: (typeof JUDGE_METRICS)[number], v: number | null) {
    if (v == null) return "—";
    return m.scale === "out10" ? v.toFixed(2) : `${Math.round(v * 100)}%`;
  }

  return (
    <div className="paper-card" style={{ padding: "clamp(1rem, 2.5vw, 1.5rem)" }}>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.86rem", minWidth: 760 }}>
          <thead>
            <tr>
              <th className="section-num" style={{ textAlign: "left", padding: "0 0.6rem 0.6rem", minWidth: 190 }}>Model</th>
              {JUDGE_METRICS.map((m) => (
                <th key={m.key} title={m.hint} onClick={() => setSortKey(m.key)} className="section-num"
                    style={{ textAlign: "right", padding: "0 0.5rem 0.6rem", whiteSpace: "nowrap", cursor: "pointer",
                             color: sortKey === m.key ? "var(--ink)" : "var(--faint)", userSelect: "none" }}>
                  {m.label} {m.better === "low" ? "↓" : "↑"}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.id} style={{ borderTop: "1px solid var(--line)" }}>
                <td style={{ padding: "0.65rem 0.6rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: famTone(r.family), flexShrink: 0 }} />
                    <span style={{ color: "var(--ink)", fontWeight: 500 }}>{r.name}</span>
                    <span className="mono" style={{ fontSize: "0.62rem", color: "var(--faint)", border: "1px solid var(--line-2)", borderRadius: 999, padding: "0.05rem 0.4rem" }}>{r.params}</span>
                  </div>
                </td>
                {JUDGE_METRICS.map((m) => {
                  const v = r[m.key] as number | null;
                  const isBest = v != null && v === stats[m.key].best;
                  return (
                    <td key={m.key} style={{ padding: "0.65rem 0.5rem", verticalAlign: "middle" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.45rem" }}>
                        <div style={{ flex: 1, height: 6, background: "var(--paper-3)", borderRadius: 3, overflow: "hidden", minWidth: 34 }}>
                          <div style={{ width: `${barWidth(m, v)}%`, height: "100%", background: isBest ? "var(--green)" : "var(--line-2)", opacity: isBest ? 0.7 : 1, transition: "width 0.4s" }} />
                        </div>
                        {isBest && <span className="mono" style={{ fontSize: "0.52rem", letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--green)" }}>best</span>}
                        <span className="mono tnum" style={{ fontSize: "0.78rem", minWidth: "3.2ch", textAlign: "right", color: isBest ? "var(--green)" : "var(--ink-soft)", fontWeight: isBest ? 600 : 400 }}>{disp(m, v)}</span>
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: "1.25rem", display: "grid", gap: "0.4rem" }}>
        {JUDGE_METRICS.map((m) => (
          <p key={m.key} style={{ margin: 0, fontSize: "0.76rem", color: "var(--faint)", lineHeight: 1.5 }}>
            <span className="mono" style={{ color: "var(--muted)" }}>{m.label} {m.better === "low" ? "↓" : "↑"}</span> — {m.hint}
          </p>
        ))}
        <p style={{ margin: "0.5rem 0 0", fontSize: "0.72rem", color: "var(--faint)", lineHeight: 1.5 }}>
          Judge: <span className="mono">{JUDGE_LEADERBOARD.judge_model}</span> at temperature 0, blind and pointwise,
          graded against gold answers + evidence from a held-out corpus set (it made none of our training data).
          <span style={{ color: "var(--green)", fontWeight: 600 }}> Green</span> marks the best in each column.
          Re-judge self-agreement: <b>{Math.round(JUDGE_LEADERBOARD.self_agreement_exact * 100)}%</b> exact,
          {" "}<b>{Math.round(JUDGE_LEADERBOARD.self_agreement_within1 * 100)}%</b> within one point.
        </p>
      </div>
    </div>
  );
}
