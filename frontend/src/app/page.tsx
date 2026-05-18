"use client"

import { useEffect, useState } from "react"
import {
  TrendingUp,
  ShieldBan,
  Clock,
  BrainCircuit,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react"
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import { api } from "@/lib/api"
import type { SystemStats, Alert } from "@/lib/types"

function statCard(
  label: string,
  value: string,
  icon: React.ReactNode,
  trend: { dir: "up" | "down"; label: string },
  accent: string
) {
  return (
    <div className="glass-panel p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[#64748b] uppercase tracking-wider font-medium">
          {label}
        </span>
        <span className={`text-${accent}`}>{icon}</span>
      </div>
      <span className="text-2xl font-bold font-mono">{value}</span>
      <div className="flex items-center gap-1 text-[10px]">
        {trend.dir === "up" ? (
          <ArrowUpRight size={12} className="text-emerald-500" />
        ) : (
          <ArrowDownRight size={12} className="text-red-500" />
        )}
        <span
          className={
            trend.dir === "up" ? "text-emerald-500" : "text-red-500"
          }
        >
          {trend.label}
        </span>
      </div>
    </div>
  )
}

const COLORS = ["#10b981", "#f59e0b", "#ef4444"]

export default function DashboardPage() {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function load() {
      const [s, a] = await Promise.all([
        api.getSystemStats(),
        api.getRecentAlerts(),
      ])
      if (cancelled) return
      setStats(s)
      setAlerts(a)
      setLoading(false)
    }

    load()
    const interval = setInterval(load, 10_000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-6 h-6 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const pieData = stats
    ? [
        { name: "Approved", value: stats.approvalBreakdown.approvedPercent },
        { name: "Flagged", value: stats.approvalBreakdown.flaggedPercent },
        { name: "Blocked", value: stats.approvalBreakdown.blockedPercent },
      ]
    : []

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Page Header */}
      <div>
        <h1 className="text-xl font-semibold text-[#f1f5f9]">Dashboard</h1>
        <p className="text-sm text-[#64748b] mt-0.5">
          Real-time fraud intelligence overview
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCard(
          "Fraud Detection Rate",
          `${stats?.fraudRate.toFixed(1)}%`,
          <TrendingUp size={16} />,
          { dir: "up", label: "+0.3% this week" },
          "cyan-400"
        )}
        {statCard(
          "Auto-Blocked Today",
          (stats?.blockedToday ?? 0).toLocaleString(),
          <ShieldBan size={16} />,
          { dir: "up", label: "+2.4% vs yesterday" },
          "red-400"
        )}
        {statCard(
          "Pending Review",
          (stats?.pendingReview ?? 0).toString(),
          <Clock size={16} />,
          { dir: "down", label: "-1.2% clearance rate" },
          "amber-400"
        )}
        {statCard(
          "Model Accuracy",
          `${stats?.modelAccuracy.toFixed(1)}%`,
          <BrainCircuit size={16} />,
          { dir: "up", label: "+0.1% this round" },
          "emerald-400"
        )}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Transaction Approval Breakdown */}
        <div className="glass-panel p-4">
          <h2 className="text-sm font-medium text-[#f1f5f9] mb-1">
            Approval Breakdown
          </h2>
          <p className="text-[10px] text-[#64748b] mb-3">
            Out of {(stats?.totalTransactions ?? 0).toLocaleString()}{" "}
            total transactions
          </p>
          <div className="h-48 min-h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={48}
                  outerRadius={68}
                  dataKey="value"
                  stroke="none"
                >
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#0d1526",
                    border: "1px solid #1e293b",
                    borderRadius: 6,
                    fontSize: 11,
                  }}
                  formatter={(v) => [`${Number(v).toFixed(1)}%`]}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-4 mt-1">
            {pieData.map((d, i) => (
              <div key={d.name} className="flex items-center gap-1.5">
                <span
                  className="w-2 h-2 rounded-sm"
                  style={{ background: COLORS[i] }}
                />
                <span className="text-[10px] text-[#94a3b8] font-mono">
                  {d.value.toFixed(1)}% {d.name}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="lg:col-span-2 glass-panel p-4">
          <h2 className="text-sm font-medium text-[#f1f5f9] mb-4">
            Recent Activity
          </h2>
          <div className="space-y-3">
            {(() => {
              const activities = [
                { color: "bg-cyan-500", text: "Transaction scanned — Elevated risk pattern detected" },
                { color: "bg-red-500", text: "Transaction auto-blocked — Account takeover pattern identified" },
                { color: "bg-emerald-500", text: "FL training round completed — Model accuracy improved" },
                { color: "bg-amber-500", text: "Knowledge graph updated — New entity relationships added" },
                { color: "bg-cyan-500", text: "Batch analysis complete — Results ready for review" },
                { color: "bg-emerald-500", text: "Model sync successful — Updated weights distributed to clients" },
              ]
              const now = Date.now()
              return activities.map((s, i) => {
                const minAgo = Math.floor(Math.random() * 40) + 1 + i * 3
                const time = minAgo < 60 ? `${minAgo}m ago` : `${Math.floor(minAgo / 60)}h ago`
                return (
                  <div
                    key={i}
                    className="flex items-start gap-3 py-2 border-b border-[#1e293b] last:border-0"
                  >
                    <div className={`w-2 h-2 rounded-full ${s.color} mt-1.5 flex-shrink-0`} />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-[#cbd5e1] truncate">{s.text}</p>
                      <span className="text-[9px] text-[#64748b] font-mono">{time}</span>
                    </div>
                  </div>
                )
              })
            })()}
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="glass-panel">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e293b]">
          <h2 className="text-sm font-medium text-[#f1f5f9]">
            Recent Alerts
          </h2>
          <span className="text-[10px] text-[#64748b] font-mono">
            Top 100
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[10px] text-[#64748b] uppercase tracking-wider border-b border-[#1e293b]">
                <th className="text-left px-4 py-2.5 font-medium">Tx ID</th>
                <th className="text-left px-4 py-2.5 font-medium">Merchant</th>
                <th className="text-right px-4 py-2.5 font-medium">Amount</th>
                <th className="text-right px-4 py-2.5 font-medium">Risk</th>
                <th className="text-center px-4 py-2.5 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((a) => (
                <tr
                  key={a.id}
                  className="border-b border-[#1e293b] hover:bg-[#1e293b]/40 transition-colors"
                >
                  <td className="px-4 py-3 font-mono text-xs text-[#94a3b8]">
                    {a.transactionId}
                  </td>
                  <td className="px-4 py-3 text-xs text-[#cbd5e1]">
                    {a.merchant}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-xs">
                    ${a.amount.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="font-mono text-xs">
                      {(a.riskScore * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider ${
                        a.decision === "BLOCK"
                          ? "bg-red-500/10 text-red-400"
                          : a.decision === "FLAG"
                          ? "bg-amber-500/10 text-amber-400"
                          : "bg-emerald-500/10 text-emerald-400"
                      }`}
                    >
                      {a.decision}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
