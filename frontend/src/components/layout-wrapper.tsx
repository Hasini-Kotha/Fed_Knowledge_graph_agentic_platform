"use client"

import { usePathname, useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import { Sidebar, getVerifiedClaims } from "./sidebar"
import { Topbar } from "./topbar"
import ChatbotWidget from "./chatbot-widget"

export function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const router = useRouter()
  const [authorized, setAuthorized] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const claims = getVerifiedClaims()
    const token = localStorage.getItem("token")
    const role = localStorage.getItem("user_role")

    if (pathname === "/login") {
      if (token && role) {
        if (role === "admin") {
          router.push("/admin")
        } else {
          router.push("/")
        }
        return
      }
      setAuthorized(true)
      setLoading(false)
      return
    }

    if (!token || !role || !claims) {
      router.push("/login")
      setAuthorized(false)
      setLoading(false)
      return
    }

    // Role-based route guard
    if (pathname.startsWith("/admin") && role !== "admin") {
      router.push("/")
      setAuthorized(false)
      setLoading(false)
      return
    }

    if ((pathname.startsWith("/scan") || pathname.startsWith("/batch") || pathname.startsWith("/federated-learning")) && role !== "fl_client") {
      router.push("/admin")
      setAuthorized(false)
      setLoading(false)
      return
    }

    setAuthorized(true)
    setLoading(false)
  }, [pathname, router])

  if (loading) {
    return (
      <main className="min-h-screen bg-[#020617] text-slate-100 flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
      </main>
    )
  }

  if (!authorized) {
    return null
  }

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
