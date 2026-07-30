import { EXPENSES } from "@/app/lib/model";

const sum = (rows: readonly { cost: number }[]) => rows.reduce((a, r) => a + r.cost, 0);

export default function Expenses() {
  const trainedTotal = sum(EXPENSES.trained);
  const sharedTotal = sum(EXPENSES.shared);
  const grand = trainedTotal + sharedTotal;

  return (
    <div className="paper-card" style={{ padding: "clamp(1.25rem, 3vw, 2rem)" }}>
      <div style={{ display: "grid", gap: "1.75rem", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }}>
        <Table title="Models we trained" rows={EXPENSES.trained} total={trainedTotal} tone="var(--green)" />
        <Table title="Shared data & infrastructure" rows={EXPENSES.shared} total={sharedTotal} tone="var(--brass)" />
      </div>

      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "1rem", marginTop: "1.75rem", paddingTop: "1.25rem", borderTop: "2px solid var(--line-2)", flexWrap: "wrap" }}>
        <div>
          <div className="section-num">Grand total, everything</div>
          <div className="mono" style={{ fontSize: "0.72rem", color: "var(--faint)", marginTop: "0.25rem" }}>
            21 models · data → pretrain → SFT → RAFT → DPO → RLAIF
          </div>
        </div>
        <span className="stat-num" style={{ fontSize: "2.6rem", color: "var(--ink)" }}>~${grand.toFixed(0)}</span>
      </div>

      <p style={{ margin: "1.1rem 0 0", fontSize: "0.8rem", color: "var(--faint)", lineHeight: 1.6 }}>
        Estimated from GPU type × wall-clock on Modal (H100 ~$3.95/hr, A100-40 ~$2.10/hr, L4 ~$0.80/hr)
        plus OpenRouter and Gemini usage. The pretraining alone is ~45% of it; every fine-tune after
        that — SFT, RAFT, six DPO models, two reward models, six RLAIF models — together cost less than
        the base model did. The borrowed bases (the mentor&apos;s 125M and 500M, and Gemma 2B) cost us
        nothing to make — that is the whole economic case for building on pretrained models.
      </p>
    </div>
  );
}

function Table({ title, rows, total, tone }: {
  title: string; rows: readonly { name: string; detail: string; cost: number }[]; total: number; tone: string;
}) {
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: "0.9rem" }}>{title}</div>
      <div style={{ display: "grid", gap: "0.55rem" }}>
        {rows.map((r) => (
          <div key={r.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "1rem" }}>
            <div style={{ minWidth: 0 }}>
              <div style={{ color: "var(--ink)", fontSize: "0.9rem" }}>{r.name}</div>
              <div className="mono" style={{ fontSize: "0.66rem", color: "var(--faint)" }}>{r.detail}</div>
            </div>
            <span className="mono tnum" style={{ fontSize: "0.85rem", color: "var(--ink-soft)", whiteSpace: "nowrap" }}>
              ${r.cost.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.9rem", paddingTop: "0.6rem", borderTop: "1px solid var(--line)" }}>
        <span className="mono" style={{ fontSize: "0.72rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)" }}>subtotal</span>
        <span className="mono tnum" style={{ fontSize: "0.95rem", color: tone, fontWeight: 600 }}>${total.toFixed(2)}</span>
      </div>
    </div>
  );
}
