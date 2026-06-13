import type { Metadata } from "next"
import { LayoutWrapper } from "@/components/layout-wrapper"
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
      <body className="antialiased bg-[#020617] text-slate-100">
        <LayoutWrapper>{children}</LayoutWrapper>
      </body>
    </html>
  )
}
