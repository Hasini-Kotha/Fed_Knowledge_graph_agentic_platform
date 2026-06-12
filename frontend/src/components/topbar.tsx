"use client"

import { useEffect, useState } from "react"

export function Topbar() {
  const [time, setTime] = useState("")

  useEffect(() => {
    const update = () => {
      const now = new Date()
      const y = now.getFullYear()
      const M = String(now.getMonth() + 1).padStart(2, "0")
      const d = String(now.getDate()).padStart(2, "0")
      const h = String(now.getHours()).padStart(2, "0")
      const m = String(now.getMinutes()).padStart(2, "0")
      const s = String(now.getSeconds()).padStart(2, "0")
      setTime(`${y}-${M}-${d} ${h}:${m}:${s}`)
    }
    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="fixed top-0 left-56 right-0 h-14 bg-[#020617]/80 backdrop-blur-xl border-b border-[#1e293b] flex items-center justify-between px-6 z-40">
      <div className="text-xs text-[#64748b] font-mono">
        {/* page title is set per-page via layout */}
      </div>
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[10px] text-emerald-500 font-mono uppercase tracking-wider font-medium">
            All Systems Nominal
          </span>
        </div>
        <span className="text-sm text-[#94a3b8] font-mono">{time}</span>
      </div>
    </header>
  )
}
