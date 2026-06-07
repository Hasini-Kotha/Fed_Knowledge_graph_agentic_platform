"use client"

import {
  Search,
  Upload,
  FileText,
  BarChart3,
  Shield,
  Network,
  Lightbulb,
  Bot,
  ArrowRight,
} from "lucide-react"

const pipelineSteps = [
  {
    icon: BarChart3,
    title: "1. Scan & Predict",
    desc: "Enter transaction details or upload a CSV. The AI model analyzes patterns across thousands of past transactions to calculate a fraud risk score (0–100%).",
    color: "text-cyan-400",
  },
  {
    icon: Network,
    title: "2. Cross-Reference",
    desc: "The system checks the transaction against a Knowledge Graph — a map of related entities (merchants, IP addresses, accounts) to find hidden connections to known fraud patterns.",
    color: "text-emerald-400",
  },
  {
    icon: Lightbulb,
    title: "3. Explain Why",
    desc: "An explanation engine identifies the top factors that drove the risk score (e.g., unusual location, high amount, suspicious IP). You see exactly why a transaction was flagged.",
    color: "text-amber-400",
  },
  {
    icon: Bot,
    title: "4. Decide & Act",
    desc: "An AI agent reviews all evidence and automatically decides: Allow (safe), Flag (needs human review), or Block (fraudulent). Every decision is logged for audit.",
    color: "text-red-400",
  },
]

const pages = [
  {
    icon: BarChart3,
    title: "Dashboard",
    path: "/",
    desc: "Your command center. View fraud statistics, recent alerts, and system health at a glance. All key numbers update automatically.",
    color: "text-cyan-400",
  },
  {
    icon: Search,
    title: "Scan",
    path: "/scan",
    desc: "Analyze one transaction in detail. Enter the amount, location, IP, and merchant. The system runs all 4 stages and shows you the risk score, explanation, and AI decision.",
    color: "text-cyan-400",
  },
  {
    icon: Upload,
    title: "Batch",
    path: "/batch",
    desc: "Upload a CSV file with multiple transactions. The system analyzes all of them at once and gives you a summary table with decisions for each row. Export results to CSV.",
    color: "text-cyan-400",
  },
  {
    icon: FileText,
    title: "Decisions",
    path: "/decisions",
    desc: "Audit trail of every AI decision. Filter by decision type (Allow / Flag / Block) or minimum risk score. Click any row to see full details and factors.",
    color: "text-cyan-400",
  },
  {
    icon: Shield,
    title: "System",
    path: "/system",
    desc: "Technical overview: see the AI model's training progress, knowledge graph size, pipeline stage health, and system connectivity.",
    color: "text-cyan-400",
  },
]

export default function GuidePage() {
  return (
    <div className="space-y-8 animate-fadeIn max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-[#f1f5f9]">User Guide</h1>
        <p className="text-sm text-[#94a3b8] mt-0.5">
          How TraceAI works and how to use it
        </p>
      </div>

      {/* What is TraceAI */}
      <div className="glass-panel p-5">
        <h2 className="text-base font-semibold text-[#f1f5f9] mb-2">
          What is TraceAI?
        </h2>
        <p className="text-sm text-[#94a3b8] leading-relaxed">
          TraceAI is an intelligent fraud detection platform that analyzes
          financial transactions across multiple stages. It combines AI models,
          relationship mapping, explainable reasoning, and automated decision-making
          to help you identify and block fraudulent activity — fast.
        </p>
      </div>

      {/* How it works — 4 steps */}
      <div className="glass-panel p-5">
        <h2 className="text-base font-semibold text-[#f1f5f9] mb-4">
          How It Works — The 4-Stage Pipeline
        </h2>
        <div className="space-y-4">
          {pipelineSteps.map((step) => (
            <div key={step.title} className="flex gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-[#0a1220] border border-[#1e293b] flex items-center justify-center">
                <step.icon size={18} className={step.color} />
              </div>
              <div>
                <h3 className="text-sm font-medium text-[#f1f5f9]">{step.title}</h3>
                <p className="text-xs text-[#94a3b8] mt-1 leading-relaxed">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pipeline visualization */}
      <div className="glass-panel p-5 border-cyan-500/20">
        <h2 className="text-sm font-medium text-[#f1f5f9] mb-4 text-center">
          End-to-End Flow
        </h2>
        <div className="flex flex-wrap items-center justify-center gap-2 text-[10px] font-mono">
          {["Transaction", "AI Model", "Knowledge Graph", "Explanation", "Agent Decision"].map(
            (step, i, arr) => (
              <div key={step} className="flex items-center gap-2">
                <span className="bg-[#0a1220] border border-[#1e293b] px-3 py-1.5 rounded text-[#94a3b8]">
                  {step}
                </span>
                {i < arr.length - 1 && (
                  <ArrowRight size={12} className="text-cyan-500" />
                )}
              </div>
            )
          )}
        </div>
      </div>

      {/* Page-by-page guide */}
      <div className="space-y-4">
        <h2 className="text-base font-semibold text-[#f1f5f9]">
          Page-by-Page Guide
        </h2>
        {pages.map((page) => (
          <div key={page.title} className="glass-panel p-4">
            <div className="flex items-center gap-3 mb-2">
              <page.icon size={16} className={page.color} />
              <div>
                <span className="text-sm font-medium text-[#f1f5f9]">{page.title}</span>
                <span className="text-[9px] text-[#64748b] font-mono ml-2">{page.path}</span>
              </div>
            </div>
            <p className="text-xs text-[#94a3b8] leading-relaxed">{page.desc}</p>
          </div>
        ))}
      </div>

      {/* Tips */}
      <div className="glass-panel p-5 border-amber-500/20">
        <h2 className="text-sm font-semibold text-[#f1f5f9] mb-3">
          Quick Tips
        </h2>
        <ul className="space-y-2 text-xs text-[#94a3b8]">
          <li className="flex items-start gap-2">
            <span className="text-amber-400 mt-0.5">•</span>
            <span>Start with the <strong className="text-[#cbd5e1]">Scan</strong> page to test a single transaction and understand the full flow.</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-400 mt-0.5">•</span>
            <span>Use <strong className="text-[#cbd5e1]">Batch</strong> when you have a list of transactions to review — upload a CSV with columns: <code className="text-[#06b6d4] font-mono">amount, merchant</code>.</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-400 mt-0.5">•</span>
            <span>Check the <strong className="text-[#cbd5e1]">Decisions</strong> page to review past AI actions and override if needed.</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-amber-400 mt-0.5">•</span>
            <span>Risk scores above <strong className="text-red-400">70%</strong> are automatically blocked. Scores between <strong className="text-amber-400">35–70%</strong> are flagged for review.</span>
          </li>
        </ul>
      </div>
    </div>
  )
}
