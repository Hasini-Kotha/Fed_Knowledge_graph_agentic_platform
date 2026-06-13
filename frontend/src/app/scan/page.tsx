"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Search, Shield, Loader2, Network } from "lucide-react"
import { api } from "@/lib/api"
import type { TransactionResult } from "@/lib/types"
import { dispatchOverride } from "@/lib/events"
import { getVerifiedClaims } from "@/components/sidebar"

const decisionColors: Record<string, { bg: string; text: string }> = {
  ALLOW: { bg: "bg-emerald-500/10", text: "text-emerald-400" },
  FLAG: { bg: "bg-amber-500/10", text: "text-amber-400" },
  BLOCK: { bg: "bg-red-500/10", text: "text-red-400" },
}

function RiskGauge({ score }: { score: number }) {
  const circumference = 251
  const offset = circumference - score * circumference
  const color =
    score > 0.7 ? "#ef4444" : score > 0.35 ? "#f59e0b" : "#10b981"

  return (
    <div className="flex flex-col items-center">
      <svg width="160" height="90" viewBox="0 0 200 110">
        <path
          d="M 10 100 A 80 80 0 0 1 190 100"
          fill="none"
          stroke="#1e293b"
          strokeWidth="12"
          strokeLinecap="round"
        />
        <path
          d="M 10 100 A 80 80 0 0 1 190 100"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s ease-out" }}
        />
      </svg>
      <span
        className="text-3xl font-bold font-mono mt-1"
        style={{ color }}
      >
        {(score * 100).toFixed(0)}
      </span>
      <span className="text-[9px] text-[#64748b] uppercase tracking-wider mt-0.5">
        Risk Score
      </span>
    </div>
  )
}

function MiniGraph({ graph }: { graph: TransactionResult["graph"] }) {
  if (!graph.nodes.length) return null
  return (
    <div className="space-y-2">
      {graph.nodes.map((n) => (
        <div
          key={n.id}
          className="flex items-center justify-between text-xs py-1 border-b border-[#1e293b] last:border-0"
        >
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                n.risk > 0.6 ? "bg-red-500" : n.risk > 0.35 ? "bg-amber-500" : "bg-emerald-500"
              }`}
            />
            <span className="text-[#94a3b8] font-mono text-[10px]">{n.label}</span>
          </div>
          <span className="text-[10px] text-[#64748b] font-mono">
            {(n.risk * 100).toFixed(2)}%
          </span>
        </div>
      ))}
      <div className="pt-1">
        <span className="text-[9px] text-[#475569] uppercase tracking-wider">
          Edges: {graph.edges.length}
        </span>
      </div>
    </div>
  )
}

export default function ScanPage() {
  const router = useRouter()
  const [form, setForm] = useState({
    transactionId: "",
    amount: "",
    location: "",
    ip: "",
    merchant: "",
  })
  const [scanning, setScanning] = useState(false)
  const [result, setResult] = useState<TransactionResult | null>(null)
  const [error, setError] = useState("")

  // Redirect guard
  useEffect(() => {
    const claims = getVerifiedClaims()
    if (!claims) {
      router.push("/login")
    } else if (claims.role !== "fl_client") {
      router.push("/admin")
    }
  }, [router])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }))
  }

  const handleScan = async () => {
    setScanning(true)
    setError("")
    setResult(null)
    try {
      const res = await api.scanTransaction({
        transactionId: form.transactionId || undefined,
        amount: parseFloat(form.amount) || 0,
        location: form.location,
        ip: form.ip,
        merchant: form.merchant,
      })
      setResult(res)
    } catch (err: unknown) {
      if (err instanceof Error && err.message.includes("404")) {
        setError(`Transaction "${form.transactionId}" not found in the system`)
      } else {
        setError("Analysis failed. Please try again.")
      }
    } finally {
      setScanning(false)
    }
  }

  const handleOverride = async (decision: string) => {
    if (!result) return
    try {
      const updated = await api.overrideDecision(result.id, decision)
      setResult(updated)
      dispatchOverride({
        transactionId: result.id,
        decision,
        timestamp: updated.timestamp,
      })
    } catch {
      setError("Override failed. Please try again.")
    }
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-[#f1f5f9]">
          Transaction Scan
        </h1>
        <p className="text-sm text-[#64748b] mt-0.5">
          Analyze a single transaction through the full 5-stage pipeline
        </p>
      </div>

      {/* Input Form */}
      <div className="glass-panel p-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-[9px] text-[#64748b] uppercase tracking-wider font-medium">
              Transaction ID
            </label>
            <input
              name="transactionId"
              value={form.transactionId}
              onChange={handleChange}
              placeholder="Optional"
              className="bg-[#020617] border border-[#1e293b] text-sm text-[#f1f5f9] px-3 py-2 rounded-md focus:outline-none focus:border-cyan-500/50 transition-colors placeholder:text-[#475569]"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-[9px] text-[#64748b] uppercase tracking-wider font-medium">
              Amount ($)
            </label>
            <input
              name="amount"
              value={form.amount}
              onChange={handleChange}
              placeholder="0.00"
              className="bg-[#020617] border border-[#1e293b] text-sm text-[#f1f5f9] px-3 py-2 rounded-md focus:outline-none focus:border-cyan-500/50 transition-colors placeholder:text-[#475569]"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-[9px] text-[#64748b] uppercase tracking-wider font-medium">
              Location
            </label>
            <input
              name="location"
              value={form.location}
              onChange={handleChange}
              placeholder="e.g. US, HK"
              className="bg-[#020617] border border-[#1e293b] text-sm text-[#f1f5f9] px-3 py-2 rounded-md focus:outline-none focus:border-cyan-500/50 transition-colors placeholder:text-[#475569]"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-[9px] text-[#64748b] uppercase tracking-wider font-medium">
              IP Address
            </label>
            <input
              name="ip"
              value={form.ip}
              onChange={handleChange}
              placeholder="e.g. 192.168.1.1"
              className="bg-[#020617] border border-[#1e293b] text-sm text-[#f1f5f9] px-3 py-2 rounded-md focus:outline-none focus:border-cyan-500/50 transition-colors placeholder:text-[#475569]"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-[9px] text-[#64748b] uppercase tracking-wider font-medium">
              Merchant
            </label>
            <input
              name="merchant"
              value={form.merchant}
              onChange={handleChange}
              placeholder="e.g. AMAZON.COM"
              className="bg-[#020617] border border-[#1e293b] text-sm text-[#f1f5f9] px-3 py-2 rounded-md focus:outline-none focus:border-cyan-500/50 transition-colors placeholder:text-[#475569]"
            />
          </div>
        </div>
        <button
          onClick={handleScan}
          disabled={scanning}
          className="mt-4 w-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 py-2.5 rounded-md text-sm font-semibold uppercase tracking-wider hover:bg-cyan-500/20 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {scanning ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Search size={16} />
              Analyze Transaction
            </>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="glass-panel p-4 border-red-500/30 bg-red-500/5">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4 animate-fadeIn">
          {/* Decision Banner */}
          <div
            className={`glass-panel p-4 flex items-center justify-between ${
              result.decision === "BLOCK"
                ? "border-red-500/30"
                : result.decision === "FLAG"
                ? "border-amber-500/30"
                : "border-emerald-500/30"
            }`}
          >
            <div className="flex items-center gap-3">
              <Shield size={24} />
              <div>
                <span
                  className={`text-lg font-bold ${
                    decisionColors[result.decision]?.text ?? ""
                  }`}
                >
                  {result.decision}
                </span>
                <p className="text-xs text-[#64748b]">
                  Confidence: {(result.confidence * 100).toFixed(0)}%
                </p>
              </div>
            </div>
            <span className="text-[10px] text-[#64748b] font-mono">
              {new Date(result.timestamp).toLocaleString()}
            </span>
          </div>

          {/* Grid: Gauge + Factors + Graph + Rationale */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Risk Gauge */}
            <div className="glass-panel p-4 flex flex-col items-center justify-center">
              <RiskGauge score={result.riskScore} />
            </div>

            {/* Factors */}
            <div className="glass-panel p-4">
              <h3 className="text-xs font-medium text-[#f1f5f9] uppercase tracking-wider mb-3">
                Key Factors
              </h3>
              <div className="space-y-2">
                {result.factors.map((f, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between py-1.5 border-b border-[#1e293b] last:border-0"
                  >
                    <span className="text-xs text-[#94a3b8]">{f.name}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-[#1e293b] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-cyan-500 rounded-full"
                          style={{ width: `${f.contribution * 100}%` }}
                        />
                      </div>
                      <span className="text-[10px] font-mono text-[#64748b] w-8 text-right">
                        {(f.contribution * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* KG Mini-Graph */}
            <div className="glass-panel p-4">
              <div className="flex items-center gap-2 mb-3">
                <Network size={14} className="text-cyan-400" />
                <h3 className="text-xs font-medium text-[#f1f5f9] uppercase tracking-wider">
                  Knowledge Graph
                </h3>
              </div>
              <MiniGraph graph={result.graph} />
            </div>

            {/* Agent Rationale */}
            <div className="glass-panel p-4">
              <h3 className="text-xs font-medium text-[#f1f5f9] uppercase tracking-wider mb-3">
                Agent Rationale
              </h3>
              <div className="space-y-1.5">
                {result.rationale.map((line, i) => {
                  const isDecision = line.startsWith("[DECISION]")
                  const isAction = line.startsWith("[ACTION]")
                  const isObs = line.startsWith("[OBSERVATION]")
                  return (
                    <p
                      key={i}
                      className={`text-[11px] font-mono leading-relaxed ${
                        isDecision
                          ? "text-cyan-400"
                          : isAction
                          ? "text-[#94a3b8]"
                          : isObs
                          ? "text-amber-400"
                          : "text-[#64748b]"
                      }`}
                    >
                      {line}
                    </p>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="glass-panel p-4 flex items-center justify-end gap-3">
            <span className="text-[10px] text-[#64748b] uppercase tracking-wider mr-auto">
              Override Decision
            </span>
            <button onClick={() => handleOverride("ALLOW")} className="px-4 py-1.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-md text-xs font-semibold uppercase tracking-wider hover:bg-emerald-500/20 transition-all">
              Allow
            </button>
            <button onClick={() => handleOverride("FLAG")} className="px-4 py-1.5 bg-amber-500/10 border border-amber-500/30 text-amber-400 rounded-md text-xs font-semibold uppercase tracking-wider hover:bg-amber-500/20 transition-all">
              Flag
            </button>
            <button onClick={() => handleOverride("BLOCK")} className="px-4 py-1.5 bg-red-500/10 border border-red-500/30 text-red-400 rounded-md text-xs font-semibold uppercase tracking-wider hover:bg-red-500/20 transition-all">
              Block
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
