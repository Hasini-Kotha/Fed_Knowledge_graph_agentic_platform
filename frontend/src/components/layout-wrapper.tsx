"use client"

import { usePathname } from "next/navigation"
import { Sidebar } from "./sidebar"
import { Topbar } from "./topbar"
import ChatbotWidget from "./chatbot-widget"

export function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const isLoginPage = pathname === "/login"

  if (isLoginPage) {
    return (
      <main className="min-h-screen bg-[#020617] text-slate-100 flex items-center justify-center">
        {children}
      </main>
    )
  }

  return (
    <>
      <Sidebar />
      <Topbar />
      <main className="ml-56 pt-14 min-h-screen">
        <div className="p-6 max-w-7xl mx-auto">{children}</div>
      </main>
      <ChatbotWidget />
    </>
  )
}
