"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Shield, Lock, User, AlertCircle, ArrowRight } from "lucide-react"

export default function LoginPage() {
  const router = useRouter()
  const [clientId, setClientId] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Redirect if already logged in
  useEffect(() => {
    const token = localStorage.getItem("token")
    const role = localStorage.getItem("user_role")
    if (token && role) {
      if (role === "admin") router.push("/admin")
      else if (role === "fl_client") router.push("/federated-learning")
    }
  }, [router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const response = await fetch("http://localhost:8002/api/admin/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: clientId, password }),
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || "Invalid credentials. Please try again.")
      }

      const data = await response.json()
      
      // Store auth state
      localStorage.setItem("token", data.access_token)
      localStorage.setItem("user_role", data.role)
      localStorage.setItem("client_id", data.client_id)
      localStorage.setItem("bank_name", data.bank_name)

      // Redirect
      if (data.role === "admin") {
        router.push("/admin")
      } else {
        router.push("/federated-learning")
      }
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w-full max-w-md p-8 bg-[#090d16]/60 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl relative overflow-hidden">
      {/* Decorative gradient blur */}
      <div className="absolute -top-20 -left-20 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-20 -right-20 w-48 h-48 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="text-center space-y-2 mb-8">
        <div className="inline-flex p-3 bg-cyan-500/10 rounded-xl border border-cyan-500/20 text-cyan-400 mb-2">
          <Shield className="w-6 h-6 animate-pulse" />
        </div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight">TraceAI Console</h1>
        <p className="text-sm text-slate-400">
          Enter partner credentials to access secure FL gateway
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {error && (
          <div className="p-3.5 bg-rose-500/10 border border-rose-500/20 rounded-lg flex items-start gap-2.5 text-rose-400 text-xs font-medium">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
            Client ID / Admin ID
          </label>
          <div className="relative">
            <User className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-500" />
            <input
              type="text"
              required
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              placeholder="e.g. admin or bank_alpha"
              className="w-full pl-10 pr-4 py-3 bg-slate-900/40 border border-slate-800 rounded-xl text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-cyan-500/50 transition-colors"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 font-mono">
            Security Key / Password
          </label>
          <div className="relative">
            <Lock className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-500" />
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              className="w-full pl-10 pr-4 py-3 bg-slate-900/40 border border-slate-800 rounded-xl text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-cyan-500/50 transition-colors"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold rounded-xl text-sm transition-all duration-150 flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {loading ? (
            <div className="w-5 h-5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
          ) : (
            <>
              Access Gateway
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </form>
      
      {/* Footer Info */}
      <div className="text-center mt-6 pt-4 border-t border-slate-800/40">
        <p className="text-[10px] text-slate-500 font-mono">
          SECURE PROTOCOL // TLS 1.3 // ENCRYPTED SHA-256
        </p>
      </div>
    </div>
  )
}
