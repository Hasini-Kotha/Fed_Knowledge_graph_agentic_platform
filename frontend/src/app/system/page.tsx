"use client"

import { useEffect, useState } from "react"
import {
  BarChart3,
  Network,
  Cable,
  Database,
  Cpu,
  Loader2,
} from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import { api } from "@/lib/api"
import type { FLStatus } from "@/lib/types"

interface MCPStatus {
  name: string
  status: "ONLINE" | "SYNCING" | "OFFLINE"
  latency: string
}

interface KGStats {
  nodes: number
  edges: number
  communities: number
  lastUpdated: string
}

export default function SystemPage() {
  const [fl, setFl] = useState<FLStatus | null>(null)
  const [mcp, setMcp] = useState<MCPStatus[]>([])
  const [kg, setKg] = useState<KGStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      const [f, m, k] = await Promise.all([
        api.getFLStatus(),
        api.getMCPStatus(),
        api.getKGStats(),
      ])
      setFl(f)
      setMcp(m)
      setKg(k)
      setLoading(false)
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={20} className="animate-spin text-[#64748b]" />
      </div>
    )
  }

  const chartData = (fl?.history ?? []).map((h) => ({
    round: `R${h.round}`,
    accuracy: +(h.accuracy * 100).toFixed(1),
  }))

  const statusColor = (s: string) => {
    switch (s) {
      case "ONLINE":
        return "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"
      case "SYNCING":
        return "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]"
      default:
        return "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"
    }
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-[#f1f5f9]">
          System Health
        </h1>
        <p className="text-sm text-[#64748b] mt-0.5">
          Federated learning, knowledge graph, and MCP bridge status
        </p>
      </div>

      {/* FL Section */}
      <div className="glass-panel p-4">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 size={16} className="text-cyan-400" />
          <h2 className="text-sm font-medium text-[#f1f5f9]">
            Federated Learning
          </h2>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
          <div className="bg-[#020617] rounded-lg p-3 border border-[#1e293b]">
            <span className="text-[9px] text-[#64748b] uppercase tracking-wider">
              Current Round
            </span>
            <p className="text-lg font-bold font-mono text-[#f1f5f9] mt-1">
              {fl?.round ?? "—"}
            </p>
          </div>
          <div className="bg-[#020617] rounded-lg p-3 border border-[#1e293b]">
            <span className="text-[9px] text-[#64748b] uppercase tracking-wider">
              Clients Connected
            </span>
            <p className="text-lg font-bold font-mono text-[#f1f5f9] mt-1">
              {fl?.clients ?? "—"}
            </p>
          </div>
          <div className="bg-[#020617] rounded-lg p-3 border border-[#1e293b]">
            <span className="text-[9px] text-[#64748b] uppercase tracking-wider">
              Model Loss
            </span>
            <p className="text-lg font-bold font-mono text-[#f1f5f9] mt-1">
              {fl?.loss.toFixed(4) ?? "—"}
            </p>
          </div>
        </div>
        <div className="h-52 min-h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <XAxis
                dataKey="round"
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={[80, 100]}
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  background: "#0d1526",
                  border: "1px solid #1e293b",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
              <Line
                type="monotone"
                dataKey="accuracy"
                stroke="#06b6d4"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* KG Section */}
      <div className="glass-panel p-4">
        <div className="flex items-center gap-2 mb-4">
          <Database size={16} className="text-cyan-400" />
          <h2 className="text-sm font-medium text-[#f1f5f9]">
            Knowledge Graph
          </h2>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-[#020617] rounded-lg p-3 border border-[#1e293b]">
            <span className="text-[9px] text-[#64748b] uppercase tracking-wider">
              Nodes
            </span>
            <p className="text-lg font-bold font-mono text-[#f1f5f9] mt-1">
              {kg?.nodes.toLocaleString() ?? "—"}
            </p>
          </div>
          <div className="bg-[#020617] rounded-lg p-3 border border-[#1e293b]">
            <span className="text-[9px] text-[#64748b] uppercase tracking-wider">
              Edges
            </span>
            <p className="text-lg font-bold font-mono text-[#f1f5f9] mt-1">
              {kg?.edges.toLocaleString() ?? "—"}
            </p>
          </div>
          <div className="bg-[#020617] rounded-lg p-3 border border-[#1e293b]">
            <span className="text-[9px] text-[#64748b] uppercase tracking-wider">
              Communities
            </span>
            <p className="text-lg font-bold font-mono text-[#f1f5f9] mt-1">
              {kg?.communities ?? "—"}
            </p>
          </div>
          <div className="bg-[#020617] rounded-lg p-3 border border-[#1e293b]">
            <span className="text-[9px] text-[#64748b] uppercase tracking-wider">
              Last Updated
            </span>
            <p className="text-sm font-mono text-[#94a3b8] mt-1">
              {kg?.lastUpdated
                ? new Date(kg.lastUpdated).toLocaleDateString()
                : "—"}
            </p>
          </div>
        </div>
      </div>

      {/* MCP Bridge — POC Section */}
      <div className="glass-panel p-4 border-cyan-500/20">
        <div className="flex items-center gap-2 mb-4">
          <Cable size={16} className="text-cyan-400" />
          <h2 className="text-sm font-medium text-[#f1f5f9]">
            MCP Bridge
          </h2>
          <span className="text-[9px] text-[#475569] uppercase tracking-wider font-mono ml-auto">
            Proof of Concept
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {mcp.map((svc) => (
            <div
              key={svc.name}
              className="bg-[#020617] rounded-lg p-3 border border-[#1e293b] flex items-center justify-between"
            >
              <div>
                <span className="text-xs text-[#94a3b8]">{svc.name}</span>
                <div className="flex items-center gap-1.5 mt-1">
                  <div className={`w-2 h-2 rounded-full ${statusColor(svc.status)}`} />
                  <span
                    className={`text-[10px] font-mono font-semibold ${
                      svc.status === "ONLINE"
                        ? "text-emerald-400"
                        : svc.status === "SYNCING"
                        ? "text-amber-400"
                        : "text-red-400"
                    }`}
                  >
                    {svc.status}
                  </span>
                </div>
              </div>
              <span className="text-[9px] font-mono text-[#475569]">
                {svc.latency}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Pipeline Health */}
      <div className="glass-panel p-4">
        <div className="flex items-center gap-2 mb-4">
          <Cpu size={16} className="text-cyan-400" />
          <h2 className="text-sm font-medium text-[#f1f5f9]">
            Pipeline Stages
          </h2>
        </div>
        <div className="space-y-3">
          {[
            { name: "Data Pipeline", desc: "Vectorization & client splits", time: "2.3s" },
            { name: "Federated Learning", desc: "FedProx aggregation", time: "47.1s" },
            { name: "Knowledge Graph", desc: "Node/edge construction", time: "12.8s" },
            { name: "Explainability", desc: "SHAP + LLM reasoning", time: "3.4s" },
            { name: "Agent Runtime", desc: "ReAct decision loop", time: "1.9s" },
          ].map((stage, i) => (
            <div
              key={stage.name}
              className="flex items-center justify-between py-2 border-b border-[#1e293b] last:border-0"
            >
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 rounded-full bg-cyan-500/10 text-cyan-400 flex items-center justify-center text-[10px] font-mono font-bold">
                  {i + 1}
                </div>
                <div>
                  <span className="text-xs text-[#f1f5f9] font-medium">
                    {stage.name}
                  </span>
                  <p className="text-[10px] text-[#64748b]">{stage.desc}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span className="text-[10px] font-mono text-[#64748b]">{stage.time}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
