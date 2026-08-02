import Alignment from "@/app/components/Alignment";
import Chat from "@/app/components/Chat";
import Expenses from "@/app/components/Expenses";
import KvCache from "@/app/components/KvCache";
import LeaderboardTabs from "@/app/components/LeaderboardTabs";
import ModelCompare from "@/app/components/ModelCompare";
import Nav from "@/app/components/Nav";
import Playground from "@/app/components/Playground";
import Raft from "@/app/components/Raft";
import Speculative from "@/app/components/Speculative";
import { DonutMix, TrainingCurve } from "@/app/components/Visuals";
import { ARCH, HERO_STATS, HF_RAFT_URL, HF_SFT_URL, HF_URL, NUMBERS, RAFT_STATS, SFT_STATS } from "@/app/lib/model";

export default function Home() {
  return (
    <main style={{ position: "relative", zIndex: 2 }}>
      <Nav />
      <Hero />

      <section id="map" style={{ borderTop: "1px solid var(--line)" }}>
        <div className="wrap" style={{ paddingTop: "clamp(3rem, 7vw, 5.5rem)", paddingBottom: "clamp(2rem, 5vw, 4rem)" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "1.1rem", marginBottom: "1.5rem" }}>
            <span className="section-num">00</span>
            <div className="rule-brass" style={{ transform: "translateY(-4px)" }} />
            <span className="eyebrow">The map</span>
          </div>
          <h2 className="display" style={{ fontSize: "clamp(1.9rem, 4.5vw, 3rem)", marginBottom: "0.5rem", maxWidth: "24ch" }}>
            Everything we built, at a glance
          </h2>
          <p style={lead}>
            Three origins — a 125M we pretrained from noise, the mentor&apos;s 500M we adopted, and Google&apos;s
            Gemma&nbsp;2&nbsp;2B off the shelf — each run through the same five-phase pipeline: pretrain, SFT,
            RAFT, DPO, and RLAIF. Twenty-two models in all. Everything below is live.
          </p>
          <figure style={{ margin: "1.9rem 0 0", borderRadius: 8, overflow: "hidden", border: "1px solid var(--line-2)", background: "#f4f1ea" }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/flow.png" alt="Model family tree: three origins (125M from scratch, mentor 500M adopted, Gemma 2B off-the-shelf), each taken through pretrain, SFT, RAFT, DPO and RLAIF"
                 style={{ width: "100%", height: "auto", display: "block" }} />
          </figure>
        </div>
      </section>

      <Section n="01" eyebrow="Playground" title="Complete the passage">
        <p style={lead}>
          Give it the opening of a brief, a filing, or an opinion, then watch the model
          continue it, one token at a time. These are the real 125M model weights,
          generating live.
        </p>
        <div style={{ marginTop: "2rem" }}>
          <Playground />
        </div>
      </Section>

      <Section n="02" eyebrow="Chat" title="Now ask it a question">
        <p style={lead}>
          Fine-tuned on 5,846 grounded legal &amp; financial Q&amp;A pairs, the same 125M
          model stops rambling and starts <em>answering</em>. It is a separate model,{" "}
          <a href={HF_SFT_URL} target="_blank" rel="noopener" className="link-underline" style={{ color: "var(--green)" }}>
            legal-slm-125m-sft ↗
          </a>
          , that streams its reply as you watch.
        </p>
        <div style={{ marginTop: "1.5rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1px", background: "var(--line)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden", marginBottom: "1.75rem" }}>
          {SFT_STATS.map((s) => (
            <div key={s.k} style={{ background: "var(--paper-2)", padding: "1rem 1.1rem" }}>
              <div className="section-num" style={{ marginBottom: "0.35rem" }}>{s.k}</div>
              <div className="stat-num" style={{ fontSize: "1.15rem", color: "var(--ink)" }}>{s.v}</div>
              <div className="mono" style={{ fontSize: "0.68rem", color: "var(--faint)", marginTop: "0.2rem" }}>{s.note}</div>
            </div>
          ))}
        </div>
        <Chat />
      </Section>

      <Section n="03" eyebrow="RAFT" title="Now ground it in your context">
        <p style={lead}>
          One more layer.{" "}
          <a href={HF_RAFT_URL} target="_blank" rel="noopener" className="link-underline" style={{ color: "var(--green)" }}>
            legal-slm-125m-raft ↗
          </a>{" "}
          was <em>RAFT-tuned</em> (Retrieval-Augmented Fine-Tuning) to answer from context
          <em> you</em> provide, quote the exact source, and ignore unrelated distractor text.
          Paste a passage, add some noise, and ask.
        </p>
        <div style={{ marginTop: "1.5rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1px", background: "var(--line)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden", marginBottom: "1.75rem" }}>
          {RAFT_STATS.map((s) => (
            <div key={s.k} style={{ background: "var(--paper-2)", padding: "1rem 1.1rem" }}>
              <div className="section-num" style={{ marginBottom: "0.35rem" }}>{s.k}</div>
              <div className="stat-num" style={{ fontSize: "1.15rem", color: "var(--ink)" }}>{s.v}</div>
              <div className="mono" style={{ fontSize: "0.68rem", color: "var(--faint)", marginTop: "0.2rem" }}>{s.note}</div>
            </div>
          ))}
        </div>
        <Raft />
      </Section>

      <Section n="04" eyebrow="Alignment" title="Preference-tuned: DPO and RLAIF">
        <p style={lead}>
          After SFT and RAFT, one more layer: align the model to preference. Pick a method — direct
          preference optimization (DPO), or a reward model plus GRPO (RLAIF) — a base stage to build on,
          and a size. Twelve real models, live on the GPU.
        </p>
        <div style={{ marginTop: "1.9rem" }}>
          <Alignment />
        </div>
      </Section>

      <Section n="05" eyebrow="Compare" title="Same data, two very different models">
        <p style={lead}>
          We ran the <em>identical</em> SFT and RAFT datasets through a real pretrained
          model — <a href="https://huggingface.co/google/gemma-2-2b-it" target="_blank" rel="noopener" className="link-underline" style={{ color: "var(--brass)" }}>Gemma&nbsp;2&nbsp;2B ↗</a> —
          using QLoRA (4-bit base, only 0.79% of weights trained). Pick it in the model
          selectors of the Chat, RAFT, and Alignment panels above, next to our 125M and 500M.
          Here is exactly what each phase costs and trains.
        </p>
        <div style={{ marginTop: "1.9rem" }}>
          <ModelCompare />
        </div>
        <p style={{ ...lead, marginTop: "1.6rem" }}>
          The trade is the whole lesson. Our 125M was built from a random init for ~$36 and
          is small enough to run in a browser tab — but it is a toy. Gemma borrows a $millions
          Google pretraining for free, trains <b>20.8M</b> adapter weights instead of all
          2.6B, and answers far more fluently — but it needs a GPU to serve, and every token
          it &quot;knows&quot; came from someone else&apos;s pretraining, not ours.
        </p>
      </Section>

      <Section n="06" eyebrow="Leaderboard" title="Every model, one scoreboard">
        <p style={lead}>
          Every model on the same held-out sets, in two views. <strong style={{ fontWeight: 500, color: "var(--green)" }}>Quality</strong> is
          an independent LLM judge (DeepSeek-V3, which made none of our training data) scoring each answer out
          of 10 against a gold reference and its evidence — reference-grounded, so it is checkable rather than
          vibes, and it finally makes the DPO/RLAIF work visible. <strong style={{ fontWeight: 500, color: "var(--green)" }}>Capability</strong> is
          the tokenizer-fair side: bits-per-byte language modeling plus exact-match accuracy. Both include
          Gemma 2 2B off the shelf, so you can see what a real pretrained model scores with zero legal training.
        </p>
        <div style={{ marginTop: "1.9rem" }}>
          <LeaderboardTabs />
        </div>
        <p style={{ ...lead, marginTop: "1.6rem" }}>
          Two things jump out. <strong style={{ fontWeight: 500, color: "var(--ink)" }}>Grounded accuracy
          climbs at every stage</strong> — our tiny model goes 1.4% → 7.1% → 24.3% across base → SFT → RAFT,
          and Gemma lands highest at 37%. But watch the <strong style={{ fontWeight: 500, color: "var(--ink)" }}>faithful-refusal
          column</strong>: Gemma off-the-shelf already declines 90% of unanswerable questions, our SFT step
          <em> destroys</em> that instinct (it learns to always answer, 0%), and only RAFT with abstention
          examples brings it back — to 100% for Gemma. And you can watch abstention <em>emerge with scale</em>:
          the 125M RAFT never learns to refuse (0%), the 500M RAFT starts to (20%), Gemma nails it (100%).
          Faithfulness isn&apos;t a prompt trick — past a point it&apos;s a capability that costs parameters.
        </p>
        <p style={{ ...lead, marginTop: "1.1rem" }}>
          The <strong style={{ fontWeight: 500, color: "var(--ink)" }}>+ DPO</strong> and{" "}
          <strong style={{ fontWeight: 500, color: "var(--ink)" }}>+ RLAIF</strong> rows are a full
          preference-optimization sweep on top of every SFT and RAFT model (DPO on ~4.7k AI-labeled
          pairs; RLAIF = a Bradley-Terry reward model then GRPO). Read them honestly: these methods
          tune <em>response quality and style</em> — helpfulness, structure, and, where the pairs teach
          it, faithfulness — which strict exact-match accuracy barely registers. So the capability
          columns hold roughly steady (they mostly don&apos;t regress), faithful refusal is preserved
          (Gemma RAFT stays at 100% through both), and where a model over-optimizes you can see it: Gemma
          RAFT + DPO&apos;s bits/byte balloons to 1.59 as it leans hard into the preferred abstain-heavy
          style. Quantifying the quality gain itself would need an LLM-judge win-rate — the natural next
          column.
        </p>
      </Section>

      <Section n="07" eyebrow="Cost" title="What the whole thing cost">
        <p style={lead}>
          Every model on the board, every dataset, every GPU-hour — tracked. Building a base
          model from scratch is the expensive part; adapting one is cheap. Here is the honest
          ledger.
        </p>
        <div style={{ marginTop: "1.9rem" }}>
          <Expenses />
        </div>
      </Section>

      <Section n="08" eyebrow="Inference" title="Making generation fast">
        <p style={lead}>
          Two optimizations every real serving stack uses, running live on GPUs. The first is
          exact and free; the second is exact but only pays off when a small model can guess
          what a big one will say.
        </p>

        <div style={{ marginTop: "2rem", display: "grid", gap: "1.5rem", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))" }}>
          <KvCache />
          <Speculative />
        </div>

        <p style={{ ...lead, marginTop: "1.6rem" }}>
          The lesson in both: throughput is not one number. The KV cache turns a quadratic
          cost linear, and its payoff grows with how many users you batch together. Speculative
          decoding trades extra draft compute for fewer expensive target steps, and only wins
          when the draft agrees often enough — try a numbered list versus a poem and watch the
          acceptance rate, and the speedup, move together.
        </p>
      </Section>

      <Section n="09" eyebrow="The numbers" title="Small model, honest accounting">
        <div style={grid3}>
          {NUMBERS.map((x) => (
            <div key={x.k} style={numCell}>
              <div className="section-num" style={{ marginBottom: "0.5rem" }}>{x.k}</div>
              <div className="stat-num" style={{ fontSize: "1.9rem", color: "var(--ink)" }}>{x.v}</div>
              <div className="mono" style={{ fontSize: "0.74rem", color: "var(--faint)", marginTop: "0.25rem" }}>{x.note}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section n="10" eyebrow="Training" title="Perplexity, falling">
        <p style={lead}>
          Held-out perplexity measured on a 20.6-million-token validation set the model
          never trained on. Two epochs, 7,778 optimizer steps, from a random start to{" "}
          <strong style={{ fontWeight: 500, color: "var(--green)" }}>9.13</strong>.
        </p>
        <div className="paper-card" style={{ marginTop: "1.75rem", padding: "1.5rem 1.25rem" }}>
          <TrainingCurve />
        </div>
      </Section>

      <Section n="11" eyebrow="Architecture" title="A Llama, in miniature">
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: "2.5rem", alignItems: "start" }}>
          <dl style={{ margin: 0, display: "grid", gap: 0 }}>
            {ARCH.map((a, i) => (
              <div key={a.k} style={{ display: "flex", justifyContent: "space-between", gap: "1rem", padding: "0.7rem 0", borderTop: i === 0 ? "none" : "1px solid var(--line)" }}>
                <dt style={{ color: "var(--muted)" }}>{a.k}</dt>
                <dd className="mono" style={{ margin: 0, color: "var(--ink)", fontSize: "0.85rem", textAlign: "right" }}>{a.v}</dd>
              </div>
            ))}
          </dl>
          <LayerStack />
        </div>
      </Section>

      <Section n="12" eyebrow="The corpus" title="Two billion tokens, hand-cleaned">
        <p style={lead}>
          Streamed from public datasets, then run through a deterministic pipeline:
          rule-based cleaning, an OCR-garble gate, MinHash-LSH near-duplicate removal,
          and 13-gram decontamination against the CaseHOLD and LexGLUE benchmarks.
        </p>
        <div className="paper-card" style={{ marginTop: "1.75rem", padding: "2rem" }}>
          <DonutMix />
        </div>
      </Section>

      <Section n="13" eyebrow="Caveats" title="What this is, and is not">
        <div style={{ display: "grid", gap: "1.1rem" }}>
          <Caveat>
            It is a <b>base (pretrained) model</b>, a next-token predictor. It has never
            been instruction-tuned, aligned, or shown a single question-answer pair.
          </Caveat>
          <Caveat>
            It will <b>fabricate</b> case names, docket numbers, statutes and financial
            figures with total confidence. Everything it writes is fiction shaped like law.
          </Caveat>
          <Caveat>
            English only, 1,024-token context, 125M parameters. It is a study in doing a
            lot with very little. It is not a product, and never legal or financial advice.
          </Caveat>
        </div>
      </Section>

      <Footer />
    </main>
  );
}

/* ---------------- sections ---------------- */

function Hero() {
  return (
    <header id="top" style={{ position: "relative", overflow: "hidden" }}>
      <div className="wrap" style={{ paddingTop: "clamp(3.5rem, 9vw, 7rem)", paddingBottom: "clamp(3rem, 7vw, 5.5rem)" }}>
        <div className="rise">
          <div className="eyebrow" style={{ marginBottom: "1.4rem" }}>A 125-million-parameter base language model</div>
          <h1 className="display" style={{ fontSize: "clamp(2.6rem, 7vw, 5rem)", maxWidth: "26ch" }}>
            Legal &amp; financial language, learned from&nbsp;nothing.
          </h1>
          <p style={{ marginTop: "1.75rem", maxWidth: "62ch", fontSize: "1.1rem", color: "var(--muted)", lineHeight: 1.65 }}>
            Trained from a random initialization on <b style={{ color: "var(--ink-soft)", fontWeight: 500 }}>2.04&nbsp;billion tokens</b> of US
            case law, SEC filings and educational web text, then asked to keep writing.
          </p>
          <div style={{ marginTop: "2.25rem", display: "flex", gap: "0.9rem", flexWrap: "wrap", alignItems: "center" }}>
            <a href="#play" className="btn-primary" style={{ display: "inline-block" }}>Try it live ↓</a>
            <a href="#chat" className="btn-secondary" style={{ display: "inline-block" }}>Chat with the fine-tuned model →</a>
          </div>
        </div>

        <div style={{ marginTop: "clamp(3rem, 7vw, 5rem)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1px", background: "var(--line)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden" }}>
          {HERO_STATS.map((s) => (
            <div key={s.label} style={{ background: "var(--paper-2)", padding: "1.4rem 1.25rem" }}>
              <div className="stat-num" style={{ fontSize: "2rem", color: "var(--ink)" }}>{s.value}</div>
              <div className="mono" style={{ fontSize: "0.72rem", color: "var(--faint)", marginTop: "0.3rem", letterSpacing: "0.03em" }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
}

function Section({ n, eyebrow, title, children }: { n: string; eyebrow: string; title: string; children: React.ReactNode }) {
  const anchor = eyebrow === "Playground" ? "play" : eyebrow === "Chat" ? "chat" : eyebrow === "RAFT" ? "raft" : eyebrow === "Alignment" ? "align" : eyebrow === "Compare" ? "compare" : eyebrow === "Leaderboard" ? "leaderboard" : eyebrow === "Cost" ? "cost" : eyebrow === "Inference" ? "inference" : eyebrow === "Architecture" ? "arch" : undefined;
  return (
    <section id={anchor} style={{ borderTop: "1px solid var(--line)" }}>
      <div className="wrap" style={{ paddingTop: "clamp(3rem, 7vw, 5.5rem)", paddingBottom: "clamp(3rem, 7vw, 5.5rem)" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "1.1rem", marginBottom: "1.5rem" }}>
          <span className="section-num">{n}</span>
          <div className="rule-brass" style={{ transform: "translateY(-4px)" }} />
          <span className="eyebrow">{eyebrow}</span>
        </div>
        <h2 className="display" style={{ fontSize: "clamp(1.9rem, 4.5vw, 3rem)", marginBottom: "0.5rem", maxWidth: "24ch" }}>{title}</h2>
        {children}
      </div>
    </section>
  );
}

function LayerStack() {
  return (
    <div className="paper-card" style={{ padding: "1.5rem", display: "flex", flexDirection: "column", gap: "0.9rem" }}>
      <Tag>tokens → 16,384 BPE embedding</Tag>
      <div style={{ display: "grid", gap: "5px" }}>
        {Array.from({ length: 12 }).map((_, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <span className="mono" style={{ fontSize: "0.62rem", color: "var(--faint)", width: 18 }}>{String(i + 1).padStart(2, "0")}</span>
            <div style={{ flex: 1, height: 16, borderRadius: 2, background: "var(--paper-3)", border: "1px solid var(--line-2)", position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", inset: 0, background: `linear-gradient(90deg, var(--green) ${18 + i}%, transparent ${18 + i}%)`, opacity: 0.16 }} />
            </div>
          </div>
        ))}
      </div>
      <Tag>RMSNorm → tied LM head → logits</Tag>
      <div className="mono" style={{ fontSize: "0.68rem", color: "var(--faint)", textAlign: "center", marginTop: "0.2rem" }}>
        12 decoder blocks · RoPE · SwiGLU
      </div>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <div className="mono" style={{ fontSize: "0.68rem", color: "var(--muted)", textAlign: "center", padding: "0.5rem", background: "var(--paper-3)", border: "1px solid var(--line)", borderRadius: 3 }}>
      {children}
    </div>
  );
}

function Caveat({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: "0.9rem", alignItems: "flex-start" }}>
      <span style={{ color: "var(--brass)", fontFamily: "var(--font-serif)", fontSize: "1.4rem", lineHeight: 1, transform: "translateY(2px)" }}>§</span>
      <p style={{ margin: 0, color: "var(--ink-soft)", lineHeight: 1.6 }}>{children}</p>
    </div>
  );
}

function Footer() {
  return (
    <footer style={{ borderTop: "1px solid var(--line)", background: "var(--paper-3)" }}>
      <div className="wrap" style={{ padding: "2.5rem 1.75rem", display: "flex", flexWrap: "wrap", gap: "1.5rem", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div className="mono" style={{ fontSize: "0.8rem", letterSpacing: "0.14em", color: "var(--ink)" }}>
            LEGAL·SLM·<span style={{ color: "var(--green)" }}>125</span>
          </div>
          <p style={{ margin: "0.5rem 0 0", fontSize: "0.82rem", color: "var(--faint)", maxWidth: "40ch" }}>
            Weights on Hugging Face · inference on Modal · built from scratch, data to deploy.
          </p>
        </div>
        <div style={{ display: "flex", gap: "1.5rem", fontSize: "0.85rem" }}>
          <a href={HF_URL} target="_blank" rel="noopener" className="link-underline">Model ↗</a>
          <a href="#top" className="link-underline">Back to top ↑</a>
        </div>
      </div>
    </footer>
  );
}

/* ---------------- shared styles ---------------- */
const lead: React.CSSProperties = { fontSize: "1.08rem", color: "var(--muted)", lineHeight: 1.7 };
const grid3: React.CSSProperties = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "1px", background: "var(--line)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden" };
const numCell: React.CSSProperties = { background: "var(--paper-2)", padding: "1.6rem 1.5rem" };
