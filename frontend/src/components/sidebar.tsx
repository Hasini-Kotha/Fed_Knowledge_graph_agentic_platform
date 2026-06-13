"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import {
  LayoutDashboard,
  FileText,
  Settings2,
  BookOpen,
  Shield,
  Cpu,
  LogIn,
  LogOut,
  Users,
  Database,
  Terminal,
  Activity,
  Search,
  Upload
} from "lucide-react"

// Pure JS JWT decoder
export function decodeJwt(token: string): any {
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

// Get verified claims, verify expiration, clear if expired
export function getVerifiedClaims(): any {
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

export function Sidebar() {
  const pathname = usePathname()
  const isLoginPage = pathname === "/login"
  if (isLoginPage) return null

  const claims = getVerifiedClaims()
  const role = claims?.role

  if (role === "admin") {
    return <AdminSidebar />
  }
  return <ClientSidebar />
}

// ---------------------------------------------------------------------------
// ADMIN SIDEBAR
// ---------------------------------------------------------------------------
function AdminSidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const [adminName, setAdminName] = useState<string>("Administrator")
  const [activeTab, setActiveTab] = useState<string>("overview")

  useEffect(() => {
    const claims = getVerifiedClaims()
    if (!claims || claims.role !== "admin") {
      router.push("/login")
      return
    }
    setAdminName(claims.bank_name || claims.sub || "Administrator")
  }, [pathname, router])

  // Sync active tab with search parameter in the URL
  useEffect(() => {
    const handleUrlChange = () => {
      if (typeof window !== "undefined") {
        const params = new URLSearchParams(window.location.search)
        const tab = params.get("tab") || "overview"
        setActiveTab(tab)
      }
    }
    handleUrlChange()
    const interval = setInterval(handleUrlChange, 150)
    return () => clearInterval(interval)
  }, [])

  const handleLogout = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("user_role")
    localStorage.removeItem("bank_name")
    localStorage.removeItem("client_id")
    router.push("/login")
  }

  const adminNavItems = [
    { tab: "overview", label: "Overview & Stats", icon: LayoutDashboard },
    { tab: "clients", label: "Clients Console", icon: Users },
    { tab: "rounds", label: "FL Rounds History", icon: Cpu },
    { tab: "registry", label: "Model Registry", icon: Database },
    { tab: "logs", label: "Audit Ledger", icon: Terminal },
  ]

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-[#090f1a] border-r border-slate-800 flex flex-col z-50 font-sans">
      {/* Logo */}
      <div className="h-14 flex items-center gap-3 px-5 border-b border-slate-850">
        <div className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-[0_0_10px_rgba(239,68,68,0.5)] animate-pulse flex-shrink-0" />
        <span className="text-md font-extrabold text-slate-100 tracking-tight font-mono">
          SECURE<span className="text-rose-500">GATEWAY</span>
        </span>
      </div>

      {/* Admin Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        <div className="px-3 mb-2 text-[10px] font-mono text-slate-500 uppercase tracking-widest font-bold">
          Admin Operations
        </div>
        {adminNavItems.map((item) => {
          const isActive = activeTab === item.tab
          const Icon = item.icon
          return (
            <Link
              key={item.tab}
              href={`/admin?tab=${item.tab}`}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-150 ${
                isActive
                  ? "bg-rose-500/10 text-rose-400 border-l-2 border-rose-500"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/40 border-l-2 border-transparent"
              }`}
            >
              <Icon size={16} />
              {item.label}
            </Link>
          )}
        )}
      </nav>

      {/* Profile & Logout */}
      <div className="px-3 py-3 border-t border-slate-850 space-y-2">
        <div className="bg-slate-900/60 p-2.5 rounded-md border border-slate-800 space-y-2">
          <div className="flex flex-col">
            <span className="text-[10px] text-rose-400 font-mono uppercase tracking-wider font-semibold">
              Admin Access
            </span>
            <span className="text-xs text-slate-200 font-medium truncate">
              {adminName}
            </span>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 px-3 py-1.5 rounded bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 text-xs font-semibold transition-colors duration-150 border border-rose-500/20"
          >
            <LogOut size={13} />
            Logout
          </button>
        </div>

        <div className="pt-2 px-2 flex items-center justify-between border-t border-slate-850/50">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)] animate-pulse" />
            <span className="text-[9px] text-slate-500 uppercase tracking-wider font-mono font-semibold">
              Gateway Secure
            </span>
          </div>
          <span className="text-[9px] text-slate-600 font-mono">v1.0.0</span>
        </div>
      </div>
    </aside>
  )
}

// ---------------------------------------------------------------------------
// CLIENT SIDEBAR
// ---------------------------------------------------------------------------
function ClientSidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const [role, setRole] = useState<string | null>(null)
  const [clientName, setClientName] = useState<string | null>(null)

  useEffect(() => {
    const claims = getVerifiedClaims()
    if (claims) {
      setRole(claims.role)
      setClientName(claims.bank_name || claims.sub)
    } else {
      setRole(null)
      setClientName(null)
    }
  }, [pathname])

  const handleLogout = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("user_role")
    localStorage.removeItem("bank_name")
    localStorage.removeItem("client_id")
    setRole(null)
    setClientName(null)
    router.push("/login")
  }

  const baseNavItems = [
    { href: "/", label: "Dashboard", icon: LayoutDashboard },
    { href: "/scan", label: "Scan", icon: Search },
    { href: "/batch", label: "Batch Analysis", icon: Upload },
    { href: "/federated-learning", label: "Federated Learning", icon: Cpu },
    { href: "/decisions", label: "Decisions", icon: FileText },
    { href: "/system", label: "System", icon: Settings2 },
    { href: "/guide", label: "User Guide", icon: BookOpen },
  ]

  const navItems = [...baseNavItems]

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 bg-[#0d1526] border-r border-[#1e293b] flex flex-col z-50">
      {/* Logo */}
      <div className="h-14 flex items-center gap-3 px-5 border-b border-[#1e293b]">
        <div className="w-2.5 h-2.5 rounded-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.4)] animate-pulse-glow flex-shrink-0" />
        <span className="text-lg font-bold text-cyan-400 tracking-tight">
          TraceAI
        </span>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = pathname === item.href
          const Icon = item.icon
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-150 ${
                isActive
                  ? "bg-cyan-500/10 text-cyan-400 border-l-2 border-cyan-400"
                  : "text-[#64748b] hover:text-[#f1f5f9] hover:bg-[#0a1220] border-l-2 border-transparent"
              }`}
            >
              <Icon size={18} />
              {item.label}
            </Link>
          )
        })}
      </nav>

      {/* Profile & Logout */}
      <div className="px-3 py-3 border-t border-[#1e293b] space-y-2">
        {role && (
          <div className="bg-[#0f1b35] p-2.5 rounded-md border border-[#1e293b] space-y-2">
            <div className="flex flex-col">
              <span className="text-[10px] text-cyan-400 font-mono uppercase tracking-wider font-semibold">
                Client Node
              </span>
              <span className="text-xs text-[#e2e8f0] font-medium truncate">
                {clientName}
              </span>
            </div>
            <button
              onClick={handleLogout}
              className="w-full flex items-center justify-center gap-2 px-3 py-1.5 rounded bg-[#e11d48]/15 hover:bg-[#e11d48]/25 text-[#f43f5e] text-xs font-semibold transition-colors duration-150 border border-[#e11d48]/20"
            >
              <LogOut size={13} />
              Logout
            </button>
          </div>
        )}

        <div className="pt-2 px-2 flex items-center justify-between border-t border-[#1e293b]/50">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)] animate-pulse" />
            <span className="text-[9px] text-[#64748b] uppercase tracking-wider font-semibold">
              System Online
            </span>
          </div>
          <span className="text-[9px] text-[#475569] font-mono">v1.0.0</span>
        </div>
      </div>
    </aside>
  )
}
