"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  LayoutDashboard,
  Search,
  Upload,
  FileText,
  Settings2,
  BookOpen,
} from "lucide-react"

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/scan", label: "Scan", icon: Search },
  { href: "/batch", label: "Batch", icon: Upload },
  { href: "/decisions", label: "Decisions", icon: FileText },
  { href: "/system", label: "System", icon: Settings2 },
  { href: "/guide", label: "User Guide", icon: BookOpen },
]

export function Sidebar() {
  const pathname = usePathname()

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
      <nav className="flex-1 py-4 px-3 space-y-1">
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

      {/* Footer */}
      <div className="px-5 py-4 border-t border-[#1e293b]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]" />
          <span className="text-[10px] text-[#64748b] uppercase tracking-wider font-medium">
            System Online
          </span>
        </div>
        <div className="text-[9px] text-[#475569] mt-1 font-mono">
          v1.0.0
        </div>
      </div>
    </aside>
  )
}
