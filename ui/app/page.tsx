"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type SummarizeResp = {
  summary: string;
  budget_chosen: string;
  latency_ms_estimated: number;
  model_size_mb_estimated: number;
  on_pareto_frontier: boolean;
  disclaimer: string;
};

const DEMO_DOC = `Dubai's Roads Authority announced a revised policy. The change takes effect next quarter and applies to all licensed entities. Officials cited the need for greater transparency and consumer protection. Implementation guidance will be published on the official portal. The story remains developing.`;

const PARETO_TIERS = [
  { budget: "tight", latency: 35, rouge: 0.22, size: 35, on: true },
  { budget: "small", latency: 60, rouge: 0.28, size: 60, on: true },
  { budget: "medium", latency: 95, rouge: 0.34, size: 110, on: true },
  { budget: "large", latency: 180, rouge: 0.36, size: 220, on: false },
];

export default function Home() {
  const [doc, setDoc] = useState(DEMO_DOC);
  const [budgetMs, setBudgetMs] = useState(100);
  const [maxSizeMb, setMaxSizeMb] = useState(100);
  const [resp, setResp] = useState<SummarizeResp | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    const r = await fetch(`${API}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc, budget_ms: budgetMs, max_size_mb: maxSizeMb }),
    });
    setResp(await r.json());
    setLoading(false);
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">On-Device SLM Summarization</h1>
      <p className="opacity-70 mb-6">
        Pareto-aware summarizer. Production runs a quantized SLM (GGUF int4/int8) on-device;
        this UI uses an extractive fallback so it works in a notebook or edge sandbox.
      </p>

      <textarea
        value={doc}
        onChange={(e) => setDoc(e.target.value)}
        className="w-full h-32 rounded-xl border p-3 text-sm"
      />
      <div className="mt-3 grid grid-cols-3 gap-3 max-w-xl">
        <label>
          <div className="text-xs uppercase opacity-60">Latency budget (ms)</div>
          <input
            type="number"
            value={budgetMs}
            min={10}
            max={2000}
            onChange={(e) => setBudgetMs(parseInt(e.target.value || "100"))}
            className="rounded-xl border px-3 py-2 w-full"
          />
        </label>
        <label>
          <div className="text-xs uppercase opacity-60">Max size (MB)</div>
          <input
            type="number"
            value={maxSizeMb}
            min={10}
            max={1000}
            onChange={(e) => setMaxSizeMb(parseInt(e.target.value || "100"))}
            className="rounded-xl border px-3 py-2 w-full"
          />
        </label>
        <button
          onClick={run}
          disabled={loading}
          className="rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50 self-end"
        >
          {loading ? "Summarizing..." : "Summarize"}
        </button>
      </div>

      {resp && (
        <>
          <div className="mt-6 grid grid-cols-3 gap-4">
            <Stat label="Tier" value={resp.budget_chosen} />
            <Stat label="Latency (est.)" value={resp.latency_ms_estimated.toFixed(0) + " ms"} />
            <Stat label="Size (est.)" value={resp.model_size_mb_estimated + " MB"} />
          </div>
          <div className="mt-4 rounded-2xl border p-4">
            <div className="text-xs uppercase opacity-60 mb-1">
              Summary {resp.on_pareto_frontier ? "(on Pareto frontier)" : "(off Pareto)"}
            </div>
            <p>{resp.summary}</p>
          </div>
          <p className="mt-3 text-xs opacity-60 italic">{resp.disclaimer}</p>
        </>
      )}

      <div className="mt-8 rounded-2xl border p-4">
        <div className="text-xs uppercase opacity-60 mb-2">Pareto frontier — latency vs ROUGE-L</div>
        <ParetoChart />
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border p-4">
      <div className="text-xs uppercase tracking-wide opacity-60">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
    </div>
  );
}

function ParetoChart() {
  const W = 520, H = 220, pad = 36;
  const xs = PARETO_TIERS.map((p) => p.latency);
  const ys = PARETO_TIERS.map((p) => p.rouge);
  const xMax = Math.max(...xs) * 1.1;
  const yMax = Math.max(...ys) * 1.1;
  const x = (v: number) => pad + (v / xMax) * (W - 2 * pad);
  const y = (v: number) => H - pad - (v / yMax) * (H - 2 * pad);
  const frontier = PARETO_TIERS.filter((p) => p.on);
  return (
    <svg width={W} height={H} className="border rounded">
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="black" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="black" />
      <text x={W / 2} y={H - 6} fontSize="10" textAnchor="middle">latency p95 (ms)</text>
      <text x={10} y={H / 2} fontSize="10" transform={`rotate(-90 10 ${H/2})`}>ROUGE-L</text>
      {/* frontier polyline */}
      <polyline
        points={frontier.map((p) => `${x(p.latency)},${y(p.rouge)}`).join(" ")}
        fill="none"
        stroke="#10b981"
        strokeWidth={2}
      />
      {PARETO_TIERS.map((p, i) => (
        <g key={i}>
          <circle cx={x(p.latency)} cy={y(p.rouge)} r={6} fill={p.on ? "#10b981" : "#ef4444"} />
          <text x={x(p.latency) + 8} y={y(p.rouge) - 6} fontSize="10">
            {p.budget} · {p.size} MB
          </text>
        </g>
      ))}
    </svg>
  );
}
