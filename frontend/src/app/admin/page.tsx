"use client"

import { useState, useEffect, Fragment } from "react"
import { useRouter } from "next/navigation"
import { 
  Shield, 
  Users, 
  Cpu, 
  Terminal, 
  Activity, 
  Plus, 
  RefreshCw, 
  CheckCircle, 
  AlertTriangle,
  Database,
  Copy,
  Trash2,
  Ban,
  X,
  ChevronDown,
  ChevronUp,
  Info
} from "lucide-react"

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  Cell
} from "recharts"

// JWT decoding utilities
function decodeJwt(token: string): any {
  try {
    const base64Url = token.split(".")[1]
    const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/")
    const jsonPayload = decodeURIComponent(
      window.atob(base64)
        .split("")
        .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
        .join("")
    )
    return JSON.parse(jsonPayload)
  } catch (e) {
    return null
  }
}

function getVerifiedClaims(): any {
  if (typeof window === "undefined") return null
  const token = localStorage.getItem("token")
  if (!token) return null
  
  const claims = decodeJwt(token)
  if (!claims) return null
  
  const now = Math.floor(Date.now() / 1000)
  if (claims.exp && claims.exp < now) {
    localStorage.removeItem("token")
    localStorage.removeItem("user_role")
    localStorage.removeItem("bank_name")
    localStorage.removeItem("client_id")
    return null
  }
  return claims
}

export default function AdminDashboard() {
  const router = useRouter()
  
  // Auth state
  const [token, setToken] = useState<string | null>(null)
  
  // Data states
  const [overview, setOverview] = useState<any>(null)
  const [analytics, setAnalytics] = useState<any>(null)
  const [clients, setClients] = useState<any[]>([])
  const [registry, setRegistry] = useState<any[]>([])
  const [logs, setLogs] = useState<any[]>([])
  const [fraudStats, setFraudStats] = useState<any>(null)
  
  // UI states
  const [activeTab, setActiveTab] = useState("overview")
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState(false)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  
  // Client creation modal & credentials state
  const [newBankName, setNewBankName] = useState("")
  const [createdCredentials, setCreatedCredentials] = useState<any | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showCredentialsModal, setShowCredentialsModal] = useState(false)
  const [copiedId, setCopiedId] = useState(false)
  const [copiedSecret, setCopiedSecret] = useState(false)
  
  // Confirmation state
  const [confirmAction, setConfirmAction] = useState<{
    type: "suspend" | "remove";
    clientId: string;
    bankName: string;
  } | null>(null)

  // Audit Ledger filtering states
  const [selectedEventType, setSelectedEventType] = useState("all")
  const [selectedOutcome, setSelectedOutcome] = useState("all")
  const [searchKeyword, setSearchKeyword] = useState("")
  const [expandedLogId, setExpandedLogId] = useState<number | null>(null)

  // Pagination
  const [logPage, setLogPage] = useState(1)
  const [totalLogs, setTotalLogs] = useState(0)

  // 1. Role / JWT Validation Guard
  useEffect(() => {
    const claims = getVerifiedClaims()
    if (!claims || claims.role !== "admin") {
      router.push("/login")
    } else {
      setToken(localStorage.getItem("token"))
    }
  }, [router])

  // Sync activeTab with URL parameter tab
  useEffect(() => {
    const handleUrlChange = () => {
      if (typeof window !== "undefined") {
        const params = new URLSearchParams(window.location.search)
        const tab = params.get("tab")
        if (tab && ["overview", "clients", "rounds", "registry", "logs"].includes(tab)) {
          setActiveTab(tab)
        } else if (!tab) {
          setActiveTab("overview")
        }
      }
    }
    handleUrlChange()
    const interval = setInterval(handleUrlChange, 150)
    return () => clearInterval(interval)
  }, [])

  // 2. Fetch all data
  useEffect(() => {
    if (!token) return
    fetchAllData()
  }, [token, logPage, selectedEventType, selectedOutcome, searchKeyword])

  // Auto-refresh data every 10 seconds
  useEffect(() => {
    if (!token) return
    const interval = setInterval(() => {
      fetchAllData(true)
    }, 10000)
    return () => clearInterval(interval)
  }, [token, logPage, selectedEventType, selectedOutcome, searchKeyword])

  const fetchAllData = async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const headers = { Authorization: `Bearer ${token}` }
      
      // Build logs URL with filters
      let logsUrl = `http://localhost:8002/api/admin/logs?page=${logPage}&limit=12`
      if (selectedEventType !== "all") logsUrl += `&event_type=${selectedEventType}`
      if (selectedOutcome !== "all") logsUrl += `&outcome=${selectedOutcome}`
      if (searchKeyword.trim() !== "") logsUrl += `&search=${encodeURIComponent(searchKeyword)}`

      const [ovRes, anRes, clRes, regRes, logRes, frRes] = await Promise.all([
        fetch("http://localhost:8002/api/admin/overview", { headers }),
        fetch("http://localhost:8002/api/admin/fl-analytics", { headers }),
        fetch("http://localhost:8002/api/admin/clients", { headers }),
        fetch("http://localhost:8002/api/admin/registry", { headers }),
        fetch(logsUrl, { headers }),
        fetch("http://localhost:8002/api/admin/fraud-analytics", { headers }),
      ])

      if (ovRes.ok) setOverview(await ovRes.json())
      if (anRes.ok) setAnalytics(await anRes.json())
      if (clRes.ok) {
        const data = await clRes.json()
        setClients(data.clients || [])
      }
      if (regRes.ok) {
        const data = await regRes.json()
        setRegistry(data.registry || [])
      }
      if (logRes.ok) {
        const data = await logRes.json()
        setLogs(data.logs || [])
        setTotalLogs(data.total || 0)
      }
      if (frRes.ok) setFraudStats(await frRes.json())
    } catch (err: any) {
      if (!silent) setErrorMsg("Failed to connect to administrative server: " + err.message)
    } finally {
      if (!silent) setLoading(false)
    }
  }

  // Client Actions
  const handleRegisterClient = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newBankName) return
    setActionLoading(true)
    setErrorMsg(null)
    setCreatedCredentials(null)
    
    try {
      const response = await fetch("http://localhost:8002/api/admin/clients", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ bank_name: newBankName })
      })
      
      if (!response.ok) throw new Error("Failed to register client bank.")
      const data = await response.json()
      setCreatedCredentials(data)
      setNewBankName("")
      setShowCreateModal(false)
      setShowCredentialsModal(true)
      fetchClientsOnly()
    } catch (err: any) {
      setErrorMsg(err.message)
    } finally {
      setActionLoading(false)
    }
  }

  const triggerSuspend = (clientId: string, bankName: string) => {
    setConfirmAction({
      type: "suspend",
      clientId,
      bankName
    })
  }

  const triggerRemove = (clientId: string, bankName: string) => {
    setConfirmAction({
      type: "remove",
      clientId,
      bankName
    })
  }

  const executeConfirmAction = async () => {
    if (!confirmAction) return
    setActionLoading(true)
    setErrorMsg(null)
    setSuccessMsg(null)

    const { type, clientId } = confirmAction
    try {
      if (type === "suspend") {
        const response = await fetch(`http://localhost:8002/api/admin/clients/${clientId}/deactivate`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` }
        })
        if (!response.ok) throw new Error("Deactivation failed.")
        setSuccessMsg(`Suspended client: ${clientId}`)
      } else if (type === "remove") {
        const response = await fetch(`http://localhost:8002/api/admin/clients/${clientId}`, {
          method: "DELETE",
          headers: { Authorization: `Bearer ${token}` }
        })
        if (!response.ok) throw new Error("Removal failed.")
        setSuccessMsg(`Permanently removed client: ${clientId}`)
      }
      fetchClientsOnly()
    } catch (err: any) {
      setErrorMsg(err.message)
    } finally {
      setActionLoading(false)
      setConfirmAction(null)
    }
  }

  const handleResetCredentials = async (clientId: string) => {
    if (!confirm(`Are you sure you want to reset credentials for ${clientId}?`)) return
    setActionLoading(true)
    setErrorMsg(null)
    setCreatedCredentials(null)
    try {
      const response = await fetch(`http://localhost:8002/api/admin/clients/${clientId}/reset-credentials`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` }
      })
      if (!response.ok) throw new Error("Credentials reset failed.")
      const data = await response.json()
      setCreatedCredentials(data)
      setShowCredentialsModal(true)
      fetchClientsOnly()
    } catch (err: any) {
      setErrorMsg(err.message)
    } finally {
      setActionLoading(false)
    }
  }

  const fetchClientsOnly = async () => {
    try {
      const response = await fetch("http://localhost:8002/api/admin/clients", {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (response.ok) {
        const data = await response.json()
        setClients(data.clients || [])
      }
    } catch (e) {}
  }

  const copyToClipboard = (text: string, type: "id" | "secret") => {
    navigator.clipboard.writeText(text)
    if (type === "id") {
      setCopiedId(true)
      setTimeout(() => setCopiedId(false), 2000)
    } else {
      setCopiedSecret(true)
      setTimeout(() => setCopiedSecret(false), 2000)
    }
  }

  if (!token) return null

  // Chart data mapping
  const riskChartData = fraudStats ? [
    { name: "HIGH", count: fraudStats.risk_distribution?.HIGH ?? fraudStats.risk_distribution?.high ?? 0 },
    { name: "MEDIUM", count: fraudStats.risk_distribution?.MEDIUM ?? fraudStats.risk_distribution?.medium ?? 0 },
    { name: "LOW", count: fraudStats.risk_distribution?.LOW ?? fraudStats.risk_distribution?.low ?? 0 },
  ] : []

  const participationChartData = analytics?.rounds_history ? 
    analytics.rounds_history.map((r: any) => ({
      round: `R${r.round_number}`,
      participants: r.participating_clients?.length ?? 0
    })).reverse() : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-800 pb-5">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-rose-500 font-mono text-xs uppercase tracking-wider font-semibold animate-pulse">
            <Shield className="w-3.5 h-3.5" />
            Administrative Security Gateway
          </div>
          <h1 className="text-3xl font-extrabold text-slate-100 tracking-tight">Admin Control Room</h1>
        </div>
        <button
          onClick={() => fetchAllData()}
          disabled={loading}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 text-sm text-slate-300 font-semibold transition-colors disabled:opacity-50 self-start"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh Console
        </button>
      </div>

      {/* Overview Cards */}
      {overview && fraudStats && analytics && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="p-4 bg-slate-900/40 border border-slate-800/80 rounded-xl flex items-center gap-3">
            <div className="p-2.5 bg-cyan-500/10 rounded-lg text-cyan-400 border border-cyan-500/20">
              <Users className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] text-slate-500 font-mono uppercase font-semibold">Active Partners</div>
              <div className="text-lg font-bold text-slate-100">{overview.active_clients} / {overview.total_clients}</div>
            </div>
          </div>

          <div className="p-4 bg-slate-900/40 border border-slate-800/80 rounded-xl flex items-center gap-3">
            <div className="p-2.5 bg-rose-500/10 rounded-lg text-rose-400 border border-rose-500/20">
              <Ban className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] text-slate-500 font-mono uppercase font-semibold">Frauds Detected</div>
              <div className="text-lg font-bold text-rose-500">{fraudStats.total_fraud_blocked}</div>
            </div>
          </div>

          <div className="p-4 bg-slate-900/40 border border-slate-800/80 rounded-xl flex items-center gap-3">
            <div className="p-2.5 bg-amber-500/10 rounded-lg text-amber-400 border border-amber-500/20">
              <AlertTriangle className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] text-slate-500 font-mono uppercase font-semibold">Flagged Review</div>
              <div className="text-lg font-bold text-amber-500">{fraudStats.total_flagged}</div>
            </div>
          </div>

          <div className="p-4 bg-slate-900/40 border border-slate-800/80 rounded-xl flex items-center gap-3">
            <div className="p-2.5 bg-indigo-500/10 rounded-lg text-indigo-400 border border-indigo-500/20">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] text-slate-500 font-mono uppercase font-semibold">Round Submissions</div>
              <div className="text-lg font-bold text-slate-100">{analytics.received_updates} / {analytics.expected_clients}</div>
            </div>
          </div>

          <div className="p-4 bg-slate-900/40 border border-slate-800/80 rounded-xl flex items-center gap-3">
            <div className="p-2.5 bg-emerald-500/10 rounded-lg text-emerald-400 border border-emerald-500/20">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] text-slate-500 font-mono uppercase font-semibold">Communities</div>
              <div className="text-lg font-bold text-emerald-400">{fraudStats.community_count || fraudStats.kg_communities}</div>
            </div>
          </div>
        </div>
      )}

      {/* Alert Banners */}
      {errorMsg && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-sm flex items-start gap-2.5 font-medium">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
      {successMsg && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-emerald-400 text-sm flex items-start gap-2.5 font-medium font-mono">
          <CheckCircle className="w-5 h-5 flex-shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}

      {/* TAB 1: OVERVIEW & STATS */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* Charts Row */}
          {fraudStats && analytics && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Chart 1: Fraud Trend */}
              <div className="p-5 bg-slate-900/40 border border-slate-800 rounded-xl space-y-4">
                <h3 className="text-xs font-bold text-slate-300 uppercase font-mono tracking-wider">
                  Fraud Detection Trend by Round
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={fraudStats.trend_by_round || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="round" stroke="#64748b" fontSize={11} className="font-mono" />
                      <YAxis stroke="#64748b" fontSize={11} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: "#0b1329", borderColor: "#1e293b", borderRadius: "8px" }} 
                        labelClassName="text-slate-400 font-mono text-xs"
                        itemStyle={{ color: "#ef4444" }}
                      />
                      <Line type="monotone" dataKey="fraud_count" name="Blocked Alerts" stroke="#ef4444" strokeWidth={2.5} dot={{ fill: "#ef4444" }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Chart 2: Risk Distribution */}
              <div className="p-5 bg-slate-900/40 border border-slate-800 rounded-xl space-y-4">
                <h3 className="text-xs font-bold text-slate-300 uppercase font-mono tracking-wider">
                  Transaction Risk Distribution
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={riskChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={11} className="font-mono" />
                      <YAxis stroke="#64748b" fontSize={11} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: "#0b1329", borderColor: "#1e293b", borderRadius: "8px" }} 
                        labelClassName="text-slate-400 font-mono text-xs"
                      />
                      <Bar dataKey="count" name="Decisions">
                        {riskChartData.map((entry, index) => {
                          const colors = ["#ef4444", "#f97316", "#10b981"]
                          return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Chart 3: Client Participation */}
              <div className="p-5 bg-slate-900/40 border border-slate-800 rounded-xl space-y-4">
                <h3 className="text-xs font-bold text-slate-300 uppercase font-mono tracking-wider">
                  Client Participation per Round
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={participationChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="round" stroke="#64748b" fontSize={11} className="font-mono" />
                      <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: "#0b1329", borderColor: "#1e293b", borderRadius: "8px" }} 
                        labelClassName="text-slate-400 font-mono text-xs"
                        itemStyle={{ color: "#06b6d4" }}
                      />
                      <Bar dataKey="participants" name="Active Nodes" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* Lower Panel: Knowledge Graph Details */}
          {fraudStats && (
            <div className="p-6 bg-slate-900/40 border border-slate-800 rounded-xl space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 font-mono uppercase">
                  <Database className="w-3.5 h-3.5 text-indigo-400" />
                  Knowledge Graph Schema Size
                </span>
                <span className="text-[10px] text-emerald-400 font-mono bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  Live Sync
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4 font-mono text-center">
                <div className="p-4 bg-slate-950/60 rounded-lg border border-slate-850">
                  <div className="text-slate-500 text-[10px] uppercase font-semibold">Nodes</div>
                  <div className="text-2xl font-bold text-indigo-300">{fraudStats.kg_nodes}</div>
                </div>
                <div className="p-4 bg-slate-950/60 rounded-lg border border-slate-850">
                  <div className="text-slate-500 text-[10px] uppercase font-semibold">Edges</div>
                  <div className="text-2xl font-bold text-indigo-300">{fraudStats.kg_edges}</div>
                </div>
                <div className="p-4 bg-slate-950/60 rounded-lg border border-slate-850">
                  <div className="text-slate-500 text-[10px] uppercase font-semibold">Clusters</div>
                  <div className="text-2xl font-bold text-indigo-300">{fraudStats.kg_communities}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: CLIENTS CONSOLE */}
      {activeTab === "clients" && (
        <div className="space-y-6">
          {/* Controls */}
          <div className="flex justify-between items-center bg-slate-900/20 p-4 border border-slate-800 rounded-xl">
            <div className="text-sm font-semibold text-slate-200">Registered Banking Partners</div>
            <button
              onClick={() => { setShowCreateModal(true); setCreatedCredentials(null); }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold transition-all"
            >
              <Plus className="w-3.5 h-3.5" />
              Register Client Bank
            </button>
          </div>

          {/* Client Table */}
          <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-900/20">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/40 text-[10px] font-mono uppercase text-slate-400 tracking-wider">
                  <th className="p-4">Bank Name / Display</th>
                  <th className="p-4">Client ID</th>
                  <th className="p-4">Access Role</th>
                  <th className="p-4">Last Connection</th>
                  <th className="p-4">Status</th>
                  <th className="p-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-xs">
                {clients.map((c) => (
                  <tr key={c.client_id} className="hover:bg-slate-900/10">
                    <td className="p-4 font-semibold text-slate-200">{c.bank_name}</td>
                    <td className="p-4 font-mono text-[#94a3b8]">{c.client_id}</td>
                    <td className="p-4 font-mono text-rose-400 uppercase tracking-wider">{c.role}</td>
                    <td className="p-4 text-slate-400 font-mono">
                      {c.last_seen ? new Date(c.last_seen).toLocaleString() : "Never"}
                    </td>
                    <td className="p-4">
                      {c.allowed ? (
                        <span className="inline-flex items-center gap-1 text-emerald-400 font-medium font-mono uppercase text-[10px] bg-emerald-500/5 px-2 py-0.5 rounded border border-emerald-500/15">
                          <CheckCircle className="w-3 h-3" /> Active
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-rose-400 font-medium font-mono uppercase text-[10px] bg-rose-500/5 px-2 py-0.5 rounded border border-rose-500/15">
                          <AlertTriangle className="w-3 h-3" /> Suspended
                        </span>
                      )}
                    </td>
                    <td className="p-4 text-right space-x-2">
                      {c.role !== "admin" && (
                        <>
                          <button
                            onClick={() => handleResetCredentials(c.client_id)}
                            disabled={actionLoading}
                            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 text-[10px] font-semibold transition-colors disabled:opacity-50"
                          >
                            Reset Key
                          </button>
                          {c.allowed && (
                            <button
                              onClick={() => triggerSuspend(c.client_id, c.bank_name)}
                              disabled={actionLoading}
                              className="px-2.5 py-1 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 rounded border border-amber-500/20 text-[10px] font-semibold transition-colors disabled:opacity-50"
                            >
                              Suspend
                            </button>
                          )}
                          <button
                            onClick={() => triggerRemove(c.client_id, c.bank_name)}
                            disabled={actionLoading}
                            className="px-2.5 py-1 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 rounded border border-rose-500/20 text-[10px] font-semibold transition-colors disabled:opacity-50"
                          >
                            Remove
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Creation modal (card overlay) */}
          {showCreateModal && (
            <div className="fixed inset-0 flex items-center justify-center bg-black/60 z-50">
              <div className="p-6 bg-[#0a101f] border border-slate-800 rounded-2xl space-y-4 max-w-md w-full mx-4">
                <div className="flex justify-between items-center border-b border-slate-850 pb-2">
                  <h3 className="text-sm font-bold text-slate-100">Register Partner Client</h3>
                  <button onClick={() => setShowCreateModal(false)} className="text-slate-500 hover:text-slate-300">
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <form onSubmit={handleRegisterClient} className="space-y-4">
                  <div>
                    <label className="text-[10px] font-mono text-slate-400 uppercase">Bank Name</label>
                    <input
                      type="text"
                      required
                      value={newBankName}
                      onChange={(e) => setNewBankName(e.target.value)}
                      placeholder="e.g. Gamma Savings Bank"
                      className="w-full mt-1.5 p-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-rose-500/50"
                    />
                  </div>
                  <div className="flex gap-2 pt-2">
                    <button
                      type="submit"
                      disabled={actionLoading}
                      className="flex-1 py-2 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded-lg text-xs transition-colors"
                    >
                      Generate Credentials
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowCreateModal(false)}
                      className="px-4 py-2 bg-slate-850 hover:bg-slate-850 text-slate-300 border border-slate-750 rounded-lg text-xs"
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {/* Credentials Display Modal */}
          {showCredentialsModal && createdCredentials && (
            <div className="fixed inset-0 flex items-center justify-center bg-black/70 z-50">
              <div className="p-6 bg-[#090e1a] border border-rose-500/30 rounded-2xl space-y-4 max-w-md w-full mx-4 shadow-[0_0_25px_rgba(239,68,68,0.15)]">
                <div className="flex justify-between items-center border-b border-slate-850 pb-2">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                    <Shield className="w-4 h-4 text-rose-500" />
                    Secure Client Credentials Generated
                  </h3>
                  <button onClick={() => setShowCredentialsModal(false)} className="text-slate-500 hover:text-slate-300">
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* DISTINCT WARNING BANNER */}
                <div className="p-3.5 bg-rose-500/10 border border-rose-500/25 rounded-lg flex items-start gap-2.5">
                  <AlertTriangle className="w-4 h-4 text-rose-500 flex-shrink-0 mt-0.5" />
                  <p className="text-[11px] text-rose-300 leading-relaxed font-semibold">
                    WARNING: The client secret key will only be displayed ONCE. Store it securely. If lost, you must reset the key.
                  </p>
                </div>

                {/* Credentials details */}
                <div className="p-4 bg-slate-950/80 border border-slate-850 rounded-xl space-y-3 font-mono text-xs">
                  <div className="space-y-1">
                    <span className="text-slate-500 text-[10px] uppercase">CLIENT_ID:</span>
                    <div className="flex items-center justify-between p-2 bg-slate-900 border border-slate-800 rounded-lg">
                      <span className="text-slate-200 font-bold select-all text-xs truncate max-w-[260px]">
                        {createdCredentials.client_id}
                      </span>
                      <button
                        onClick={() => copyToClipboard(createdCredentials.client_id, "id")}
                        className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200 transition-colors"
                        title="Copy Client ID"
                      >
                        {copiedId ? (
                          <span className="text-[10px] text-emerald-400 font-bold">Copied!</span>
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <span className="text-slate-500 text-[10px] uppercase">CLIENT_SECRET (AES KEY):</span>
                    <div className="flex items-center justify-between p-2 bg-slate-900 border border-slate-800 rounded-lg">
                      <span className="text-rose-400 font-bold select-all text-xs truncate max-w-[260px]">
                        {createdCredentials.client_secret}
                      </span>
                      <button
                        onClick={() => copyToClipboard(createdCredentials.client_secret, "secret")}
                        className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200 transition-colors"
                        title="Copy Client Secret"
                      >
                        {copiedSecret ? (
                          <span className="text-[10px] text-emerald-400 font-bold">Copied!</span>
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  </div>
                </div>

                <div className="pt-2">
                  <button
                    onClick={() => setShowCredentialsModal(false)}
                    className="w-full py-2 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded-lg text-xs transition-colors"
                  >
                    I Have Backed Up The Key
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Action Confirmation Modal */}
          {confirmAction && (
            <div className="fixed inset-0 flex items-center justify-center bg-black/60 z-50">
              <div className="p-6 bg-[#0a101f] border border-slate-800 rounded-2xl space-y-4 max-w-sm w-full mx-4">
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-500" />
                  Confirm Administrative Action
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Are you sure you want to {confirmAction.type === "suspend" ? "suspend access for" : "permanently remove"} the partner bank: 
                  <strong className="text-slate-100 block mt-1 font-mono">{confirmAction.bankName} ({confirmAction.clientId})</strong>?
                </p>
                <div className="flex gap-2 pt-2">
                  <button
                    onClick={executeConfirmAction}
                    disabled={actionLoading}
                    className={`flex-1 py-2 text-white font-bold rounded-lg text-xs transition-colors ${
                      confirmAction.type === "remove" ? "bg-rose-600 hover:bg-rose-500" : "bg-amber-600 hover:bg-amber-500"
                    }`}
                  >
                    Confirm {confirmAction.type === "suspend" ? "Suspension" : "Removal"}
                  </button>
                  <button
                    onClick={() => setConfirmAction(null)}
                    disabled={actionLoading}
                    className="px-4 py-2 bg-slate-850 hover:bg-slate-800 text-slate-300 border border-slate-750 rounded-lg text-xs"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: FL ROUNDS HISTORY */}
      {activeTab === "rounds" && (
        <div className="space-y-4">
          <div className="text-sm font-semibold text-slate-200">Aggregated Consensus Rounds Ledger</div>
          {analytics?.rounds_history && (
            <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-900/20">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-900/40 text-[10px] font-mono uppercase text-slate-400 tracking-wider">
                    <th className="p-4">Round</th>
                    <th className="p-4">Started At</th>
                    <th className="p-4">Completed At</th>
                    <th className="p-4">Participating Clients</th>
                    <th className="p-4">Submissions (Acc / Rej)</th>
                    <th className="p-4">Final Accuracy</th>
                    <th className="p-4">Final Loss</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-xs">
                  {analytics.rounds_history.map((r: any) => (
                    <tr key={r.round_number} className="hover:bg-slate-900/10">
                      <td className="p-4 font-bold text-rose-500">Round #{r.round_number}</td>
                      <td className="p-4 text-slate-400 font-mono">{new Date(r.started_at).toLocaleString()}</td>
                      <td className="p-4 text-slate-400 font-mono">
                        {r.completed_at ? new Date(r.completed_at).toLocaleString() : "Running"}
                      </td>
                      <td className="p-4">
                        <div className="flex gap-1 flex-wrap">
                          {r.participating_clients.map((c: string) => (
                            <span key={c} className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[9px]">
                              {c}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="p-4 font-mono">
                        <span className="text-emerald-400 font-bold">{r.submission_stats.accepted}</span>
                        {" / "}
                        <span className="text-rose-400 font-bold">{r.submission_stats.rejected}</span>
                      </td>
                      <td className="p-4 font-mono text-emerald-400 font-bold">
                        {r.accuracy ? `${(r.accuracy * 100).toFixed(2)}%` : "N/A"}
                      </td>
                      <td className="p-4 font-mono text-rose-400">
                        {r.loss ? r.loss.toFixed(4) : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: MODEL REGISTRY */}
      {activeTab === "registry" && (
        <div className="space-y-4">
          <div className="text-sm font-semibold text-slate-200">Global Model Snapshots</div>
          <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-900/20">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/40 text-[10px] font-mono uppercase text-slate-400 tracking-wider">
                  <th className="p-4">Model ID</th>
                  <th className="p-4">Round</th>
                  <th className="p-4">Created At</th>
                  <th className="p-4">Disk Snapshot Path</th>
                  <th className="p-4">Eval Accuracy</th>
                  <th className="p-4">Loss</th>
                  <th className="p-4 text-center">Hosts</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-xs">
                {registry.map((m) => (
                  <tr key={m.model_id} className="hover:bg-slate-900/10">
                    <td className="p-4 font-mono font-bold text-slate-200">{m.model_id}</td>
                    <td className="p-4 font-mono text-rose-500 font-bold">Round {m.round_number}</td>
                    <td className="p-4 text-slate-400 font-mono">{new Date(m.created_at).toLocaleString()}</td>
                    <td className="p-4 font-mono text-[#94a3b8] truncate max-w-[240px] select-all" title={m.snapshot_path}>
                      {m.snapshot_path}
                    </td>
                    <td className="p-4 font-mono text-emerald-400 font-bold">{(m.accuracy * 100).toFixed(2)}%</td>
                    <td className="p-4 font-mono text-rose-400">{m.loss.toFixed(4)}</td>
                    <td className="p-4 font-mono text-center text-slate-300 font-semibold">{m.participating_client_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 5: AUDIT LOGS */}
      {activeTab === "logs" && (
        <div className="space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/20 p-4 border border-slate-800 rounded-xl">
            <div className="text-sm font-semibold text-slate-200">Security Audit Logs</div>
            
            {/* Ledger Filters */}
            <div className="flex flex-wrap items-center gap-3">
              <div>
                <select
                  value={selectedEventType}
                  onChange={(e) => { setSelectedEventType(e.target.value); setLogPage(1); }}
                  className="bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-300 p-2 focus:outline-none focus:border-rose-500"
                >
                  <option value="all">All Events</option>
                  <option value="LOGIN">LOGIN</option>
                  <option value="UPLOAD">UPLOAD</option>
                  <option value="VALIDATION_FAILURE">VALIDATION_FAILURE</option>
                  <option value="WEIGHT_ALTERED">WEIGHT_ALTERED</option>
                  <option value="ADMIN_ACTION">ADMIN_ACTION</option>
                  <option value="AGGREGATION_COMPLETE">AGGREGATION_COMPLETE</option>
                </select>
              </div>

              <div>
                <select
                  value={selectedOutcome}
                  onChange={(e) => { setSelectedOutcome(e.target.value); setLogPage(1); }}
                  className="bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-300 p-2 focus:outline-none focus:border-rose-500"
                >
                  <option value="all">All Outcomes</option>
                  <option value="SUCCESS">SUCCESS</option>
                  <option value="FAILURE">FAILURE</option>
                </select>
              </div>

              <div>
                <input
                  type="text"
                  placeholder="Filter by Client ID..."
                  value={searchKeyword}
                  onChange={(e) => { setSearchKeyword(e.target.value); setLogPage(1); }}
                  className="bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-300 p-2 placeholder-slate-600 focus:outline-none focus:border-rose-500 w-44"
                />
              </div>
            </div>
          </div>

          <div className="overflow-hidden border border-slate-800 rounded-xl bg-slate-900/20">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/40 text-[10px] font-mono uppercase text-slate-400 tracking-wider">
                  <th className="p-4 w-8"></th>
                  <th className="p-4">Time</th>
                  <th className="p-4">Event</th>
                  <th className="p-4">Outcome</th>
                  <th className="p-4">Client</th>
                  <th className="p-4">Admin</th>
                  <th className="p-4">IP Address</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-xs font-mono">
                {logs.length > 0 ? (
                  logs.map((l) => {
                    const isWeightAltered = l.event_type === "WEIGHT_ALTERED"
                    const isExpanded = expandedLogId === l.log_id
                    
                    return (
                      <Fragment key={l.log_id}>
                        {/* Audit Row */}
                        <tr 
                          onClick={() => setExpandedLogId(isExpanded ? null : l.log_id)}
                          className={`hover:bg-slate-850/40 cursor-pointer transition-colors ${
                            isWeightAltered ? "bg-rose-500/10 border-l-4 border-rose-500" : ""
                          }`}
                        >
                          <td className="p-4 text-center text-slate-500">
                            {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          </td>
                          <td className="p-4 text-slate-400">{new Date(l.timestamp).toLocaleString()}</td>
                          <td className="p-4">
                            <span className={`px-2 py-0.5 rounded font-bold text-[10px] inline-flex items-center gap-1 ${
                              isWeightAltered || l.event_type === "VALIDATION_FAILURE"
                                ? "bg-rose-500/15 text-rose-400 border border-rose-500/20"
                                : l.event_type === "UPLOAD" || l.event_type === "LOGIN"
                                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/20"
                                : "bg-slate-850 text-slate-300 border border-slate-700/60"
                            }`}>
                              {isWeightAltered && <span>⚠️</span>}
                              {l.event_type}
                            </span>
                          </td>
                          <td className="p-4">
                            <span className={`font-bold ${l.outcome === "SUCCESS" ? "text-emerald-400" : "text-rose-500"}`}>
                              {l.outcome}
                            </span>
                          </td>
                          <td className="p-4 text-slate-300">{l.client_id || "—"}</td>
                          <td className="p-4 text-slate-300">{l.admin_id || "—"}</td>
                          <td className="p-4 text-slate-500">{l.ip_address}</td>
                        </tr>

                        {/* Expandable details card panel */}
                        {isExpanded && (
                          <tr>
                            <td colSpan={7} className="p-4 bg-slate-950/60 border-b border-slate-800">
                              <div className="space-y-3">
                                <div className="flex items-center gap-2 text-[10px] text-slate-400 uppercase font-mono tracking-wider font-semibold border-b border-slate-850 pb-1.5">
                                  <Terminal className="w-3.5 h-3.5 text-rose-500" />
                                  Detailed Log Audit Information
                                </div>
                                
                                {isWeightAltered && (
                                  <div className="p-2.5 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-lg text-xs leading-relaxed flex items-start gap-2 max-w-2xl">
                                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                                    <span>
                                      <strong>Security Notice:</strong> A <code>WEIGHT_ALTERED</code> audit type indicates that weights decrypted from this upload did not match the provided HMAC signature. This suggests the update binary was either corrupted in transit, altered maliciously, or signed with an invalid client secret key.
                                    </span>
                                  </div>
                                )}

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono bg-slate-900/30 p-3 rounded-lg border border-slate-850 max-w-3xl">
                                  {l.detail && typeof l.detail === "object" ? (
                                    Object.entries(l.detail).map(([key, value]) => (
                                      <div key={key} className="flex flex-col gap-0.5">
                                        <span className="text-slate-500 text-[10px] uppercase font-bold">{key.replace(/_/g, " ")}:</span>
                                        <span className="text-slate-300 font-semibold select-all">
                                          {typeof value === "object" ? JSON.stringify(value) : String(value)}
                                        </span>
                                      </div>
                                    ))
                                  ) : (
                                    <div className="col-span-2 text-slate-400">No additional details recorded.</div>
                                  )}
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    )
                  })
                ) : (
                  <tr>
                    <td colSpan={7} className="p-6 text-center text-slate-500 font-mono">
                      No matching audit ledger entries found in gateway database.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          {totalLogs > 12 && (
            <div className="flex justify-between items-center pt-2 font-mono text-xs">
              <button
                disabled={logPage === 1}
                onClick={() => setLogPage(logPage - 1)}
                className="px-3 py-1.5 bg-slate-900 border border-slate-850 hover:border-slate-750 text-slate-400 disabled:opacity-30 rounded-lg transition-colors"
              >
                Previous
              </button>
              <span className="text-slate-500 font-bold">Page {logPage} of {Math.ceil(totalLogs / 12)}</span>
              <button
                disabled={logPage * 12 >= totalLogs}
                onClick={() => setLogPage(logPage + 1)}
                className="px-3 py-1.5 bg-slate-900 border border-slate-850 hover:border-slate-750 text-slate-400 disabled:opacity-30 rounded-lg transition-colors"
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
