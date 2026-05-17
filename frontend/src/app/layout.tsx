import type { Metadata } from "next"
import { Sidebar } from "@/components/sidebar"
import { Topbar } from "@/components/topbar"
import "./globals.css"

export const metadata: Metadata = {
  title: "TraceAI — Federated Intelligence Console",
  description: "AI-powered fraud detection and investigation platform",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">
        <Sidebar />
        <Topbar />
        <main className="ml-56 pt-14 min-h-screen">
          <div className="p-6 max-w-7xl mx-auto">{children}</div>
        </main>
      </body>
    </html>
  )
}
