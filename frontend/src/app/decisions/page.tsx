"use client"

import { useEffect, useState } from "react"
import { Filter, Loader2 } from "lucide-react"
import { api } from "@/lib/api"
import type { Decision } from "@/lib/types"

const decisionColors: Record<string, string> = {
  ALLOW: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  FLAG: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  BLOCK: "bg-red-500/10 text-red-400 border-red-500/20",
}

export default function DecisionsPage() {
  const [decisions, setDecisions] = useState<Decision[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState("all")
  const [minRisk, setMinRisk] = useState(0)
  const [detailId, setDetailId] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    api
      .getDecisions({ decision: filter, minRisk })
      .then(setDecisions)
      .finally(() => setLoading(false))
  }, [filter, minRisk])

  const detail = detailId ? decisions.find((d) => d.id === detailId) : null

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-[#f1f5f9]">
          Decision Log
        </h1>
        <p className="text-sm text-[#64748b] mt-0.5">
          Audit trail of all agent decisions
        </p>
      </div>

      {/* Filters */}
      <div className="glass-panel p-4">
        <div className="flex items-center gap-4 flex-wrap">
          <Filter size={14} className="text-[#64748b]" />
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[#64748b] uppercase tracking-wider font-medium">
              Decision
            </span>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="bg-[#020617] border border-[#1e293b] text-xs text-[#f1f5f9] px-3 py-1.5 rounded-md focus:outline-none focus:border-cyan-500/50"
            >
              <option value="all">All</option>
              <option value="ALLOW">Allow</option>
              <option value="FLAG">Flag</option>
              <option value="BLOCK">Block</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[#64748b] uppercase tracking-wider font-medium">
              Min Risk
            </span>
            <input
              type="number"
              min={0}
              max={100}
              value={minRisk * 100}
              onChange={(e) => setMinRisk((parseFloat(e.target.value) || 0) / 100)}
              className="bg-[#020617] border border-[#1e293b] text-xs text-[#f1f5f9] px-3 py-1.5 rounded-md w-16 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <span className="text-[10px] text-[#64748b] font-mono ml-auto">
            {decisions.length} results
          </span>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={20} className="animate-spin text-[#64748b]" />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Table */}
          <div className="lg:col-span-2 glass-panel">
            <div className="overflow-x-auto max-h-[70vh] overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[10px] text-[#64748b] uppercase tracking-wider border-b border-[#1e293b] sticky top-0 bg-[#0d1526]">
                    <th className="text-left px-4 py-2.5 font-medium">Tx ID</th>
                    <th className="text-left px-4 py-2.5 font-medium">Merchant</th>
                    <th className="text-right px-4 py-2.5 font-medium">Amount</th>
                    <th className="text-right px-4 py-2.5 font-medium">Risk</th>
                    <th className="text-center px-4 py-2.5 font-medium">Decision</th>
                    <th className="text-right px-4 py-2.5 font-medium">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {decisions.map((d) => (
                    <tr
                      key={d.id}
                      onClick={() => setDetailId(d.id)}
                      className={`border-b border-[#1e293b] hover:bg-[#1e293b]/40 transition-colors cursor-pointer ${
                        detailId === d.id ? "bg-cyan-500/5" : ""
                      }`}
                    >
                      <td className="px-4 py-3 font-mono text-xs text-[#94a3b8]">
                        {d.transactionId}
                      </td>
                      <td className="px-4 py-3 text-xs text-[#cbd5e1]">{d.merchant}</td>
                      <td className="px-4 py-3 text-right font-mono text-xs">
                        ${d.amount.toLocaleString()}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs">
                        {(d.riskScore * 100).toFixed(0)}%
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider border ${
                            decisionColors[d.decision]
                          }`}
                        >
                          {d.decision}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[10px] text-[#64748b]">
                        {new Date(d.timestamp).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Detail Panel */}
          <div className="glass-panel p-4">
            {detail ? (
              <div className="space-y-4">
                <h3 className="text-xs font-medium text-[#f1f5f9] uppercase tracking-wider">
                  Decision Details
                </h3>
                <div className="space-y-3">
                  <div>
                    <span className="text-[9px] text-[#64748b] uppercase tracking-wider block mb-0.5">
                      Transaction
                    </span>
                    <span className="text-xs font-mono text-[#94a3b8]">{detail.transactionId}</span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[#64748b] uppercase tracking-wider block mb-0.5">
                      Merchant
                    </span>
                    <span className="text-xs text-[#cbd5e1]">{detail.merchant}</span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[#64748b] uppercase tracking-wider block mb-0.5">
                      Amount
                    </span>
                    <span className="text-xs font-mono text-[#f1f5f9]">
                      ${detail.amount.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[#64748b] uppercase tracking-wider block mb-0.5">
                      Risk Score
                    </span>
                    <span
                      className={`text-sm font-bold font-mono ${
                        detail.riskScore > 0.7
                          ? "text-red-400"
                          : detail.riskScore > 0.35
                          ? "text-amber-400"
                          : "text-emerald-400"
                      }`}
                    >
                      {(detail.riskScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[#64748b] uppercase tracking-wider block mb-0.5">
                      Confidence
                    </span>
                    <span className="text-xs font-mono text-[#94a3b8]">
                      {(detail.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[#64748b] uppercase tracking-wider block mb-0.5">
                      Decision
                    </span>
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider border ${
                        decisionColors[detail.decision]
                      }`}
                    >
                      {detail.decision}
                    </span>
                  </div>
                  <div>
                    <span className="text-[9px] text-[#64748b] uppercase tracking-wider block mb-0.5">
                      Key Factors
                    </span>
                    <div className="space-y-1">
                      {detail.factors.slice(0, 3).map((f, i) => (
                        <div key={i} className="text-[10px] text-[#94a3b8] flex justify-between">
                          <span>{f.name}</span>
                          <span className="font-mono">{(f.contribution * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-48 text-center">
                <p className="text-xs text-[#64748b]">
                  Select a transaction to view details
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
