"use client";

import { useState } from "react";
import JudgeLeaderboard from "@/app/components/JudgeLeaderboard";
import Leaderboard from "@/app/components/Leaderboard";

export default function LeaderboardTabs() {
  const [view, setView] = useState<"quality" | "capability">("quality");
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", flexWrap: "wrap", marginBottom: "1.1rem" }}>
        <span className="mono" style={{ fontSize: "0.72rem", color: "var(--faint)" }}>
          {view === "quality"
            ? "Quality — an independent LLM judge scores each answer /10 against a gold reference"
            : "Capability — tokenizer-fair language modeling + exact-match accuracy"}
        </span>
        <div className="seg">
          <button data-active={view === "quality"} onClick={() => setView("quality")}>Quality · LLM-judge</button>
          <button data-active={view === "capability"} onClick={() => setView("capability")}>Capability</button>
        </div>
      </div>
      {view === "quality" ? <JudgeLeaderboard /> : <Leaderboard />}
    </div>
  );
}
