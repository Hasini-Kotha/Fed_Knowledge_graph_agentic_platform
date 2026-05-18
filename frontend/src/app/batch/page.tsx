"use client"

import { useState, useRef } from "react"
import { Upload, Download, Loader2, AlertCircle } from "lucide-react"
import { api } from "@/lib/api"
import type { BatchRow } from "@/lib/types"

export default function BatchPage() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [rows, setRows] = useState<{ amount: number; merchant: string }[]>([])
  const [results, setResults] = useState<BatchRow[]>([])
  const [analyzing, setAnalyzing] = useState(false)
  const [fileName, setFileName] = useState("")
  const [batchError, setBatchError] = useState("")

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    setResults([])
    setBatchError("")

    const reader = new FileReader()
    reader.onload = (evt) => {
      const text = evt.target?.result as string
      const lines = text.split("\n").filter((l) => l.trim())
      if (lines.length < 2) return

      const parsed: Record<string, number | string>[] = []
      const headers = lines[0].split(",").map((h) => h.trim())
      const headersLower = headers.map((h) => h.toLowerCase())

      const amountIdx = headersLower.findIndex(
        (h) => h === "amount" || h === "transaction_amount"
      )

      // Detect if CSV has V1-V28 feature columns (full ML mode)
      const featureCols = new Set(["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                                   "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                                   "V21","V22","V23","V24","V25","V26","V27","V28","Amount"])
      const stringCols = new Set(["merchant", "location", "ip", "transaction_id", "id"])
      const hasFeatures = [...featureCols].every((c) => headers.includes(c))

      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(",").map((c) => c.trim())
        if (hasFeatures) {
          const row: Record<string, number | string> = {}
          for (let j = 0; j < headers.length; j++) {
            const key = headers[j]
            const val = cols[j]
            if (stringCols.has(key.toLowerCase())) {
              row[key] = val || "UNKNOWN"
            } else {
              const num = parseFloat(val)
              row[key] = isNaN(num) ? 0 : num
            }
          }
          parsed.push(row)
        } else if (amountIdx >= 0) {
          const amount = parseFloat(cols[amountIdx] || "0")
          if (!isNaN(amount)) {
            const merchantIdx = headersLower.findIndex(
              (h) => h === "merchant" || h === "merchant_name"
            )
            const merchant = merchantIdx >= 0 ? cols[merchantIdx] || "UNKNOWN" : "UNKNOWN"
            parsed.push({ amount, merchant } as Record<string, number | string>)
          }
        }
      }
      setRows(parsed)
    }
    reader.readAsText(file)
  }

  const handleAnalyze = async () => {
    if (!rows.length) return
    setAnalyzing(true)
    setBatchError("")
    try {
      const res = await api.batchAnalyze(rows)
      setResults(res)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Batch analysis failed"
      setBatchError(msg)
    } finally {
      setAnalyzing(false)
    }
  }

  const handleExport = () => {
    if (!results.length) return
    const csv = [
      ["Row", "Transaction ID", "Amount", "Risk Score", "Confidence", "Decision", "Merchant"].join(","),
      ...results.map((r) =>
        [r.rowIndex, r.transactionId, r.amount, r.riskScore, r.confidence, r.decision, r.merchant].join(",")
      ),
    ].join("\n")

    const blob = new Blob([csv], { type: "text/csv" })
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = "traceai-batch-results.csv"
    a.click()
  }

  const summary = {
    total: results.length,
    blocked: results.filter((r) => r.decision === "BLOCK").length,
    flagged: results.filter((r) => r.decision === "FLAG").length,
    allowed: results.filter((r) => r.decision === "ALLOW").length,
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-[#f1f5f9]">
          Batch Analysis
        </h1>
        <p className="text-sm text-[#64748b] mt-0.5">
          Upload a CSV of transactions for bulk analysis
        </p>
        <p className="text-[10px] text-[#475569] mt-1">
          Full ML mode: V1..V28, Amount, merchant &nbsp;|&nbsp; Quick mode: amount, merchant
        </p>
      </div>

      {/* Error */}
      {batchError && (
        <div className="glass-panel p-4 border-red-500/30 bg-red-500/5">
          <p className="text-sm text-red-400">{batchError}</p>
        </div>
      )}

      {/* Upload Zone */}
      <div className="glass-panel p-6">
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-[#1e293b] rounded-lg p-8 text-center cursor-pointer hover:border-cyan-500/30 transition-colors"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFile}
            className="hidden"
          />
          <Upload size={32} className="mx-auto mb-3 text-[#475569]" />
          <p className="text-sm text-[#94a3b8]">
            {fileName ? fileName : "Drop a CSV file here or click to browse"}
          </p>
          <p className="text-[10px] text-[#475569] mt-1">
            Expected columns: amount, merchant (optional)
          </p>
        </div>

        {rows.length > 0 && (
          <div className="mt-4 flex items-center justify-between">
            <span className="text-sm text-[#94a3b8]">
              Parsed <strong className="text-[#f1f5f9]">{rows.length}</strong> transactions
            </span>
            <button
              onClick={handleAnalyze}
              disabled={analyzing}
              className="bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 px-4 py-2 rounded-md text-sm font-semibold uppercase tracking-wider hover:bg-cyan-500/20 transition-all disabled:opacity-50 flex items-center gap-2"
            >
              {analyzing ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Run Analysis"
              )}
            </button>
          </div>
        )}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4 animate-fadeIn">
          {/* Summary Cards */}
          <div className="grid grid-cols-4 gap-4">
            <div className="glass-panel p-3 text-center">
              <span className="text-2xl font-bold font-mono text-[#f1f5f9]">{summary.total}</span>
              <p className="text-[9px] text-[#64748b] uppercase tracking-wider mt-0.5">Total</p>
            </div>
            <div className="glass-panel p-3 text-center border-emerald-500/20">
              <span className="text-2xl font-bold font-mono text-emerald-400">{summary.allowed}</span>
              <p className="text-[9px] text-[#64748b] uppercase tracking-wider mt-0.5">Allowed</p>
            </div>
            <div className="glass-panel p-3 text-center border-amber-500/20">
              <span className="text-2xl font-bold font-mono text-amber-400">{summary.flagged}</span>
              <p className="text-[9px] text-[#64748b] uppercase tracking-wider mt-0.5">Flagged</p>
            </div>
            <div className="glass-panel p-3 text-center border-red-500/20">
              <span className="text-2xl font-bold font-mono text-red-400">{summary.blocked}</span>
              <p className="text-[9px] text-[#64748b] uppercase tracking-wider mt-0.5">Blocked</p>
            </div>
          </div>

          {/* Table */}
          <div className="glass-panel">
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e293b]">
              <h2 className="text-sm font-medium text-[#f1f5f9]">Results</h2>
              <button
                onClick={handleExport}
                className="flex items-center gap-1.5 text-[10px] text-[#64748b] hover:text-cyan-400 transition-colors uppercase tracking-wider font-medium"
              >
                <Download size={12} />
                Export CSV
              </button>
            </div>
            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[10px] text-[#64748b] uppercase tracking-wider border-b border-[#1e293b] sticky top-0 bg-[#0d1526]">
                    <th className="text-left px-4 py-2.5 font-medium">#</th>
                    <th className="text-left px-4 py-2.5 font-medium">Tx ID</th>
                    <th className="text-right px-4 py-2.5 font-medium">Amount</th>
                    <th className="text-right px-4 py-2.5 font-medium">Risk</th>
                    <th className="text-right px-4 py-2.5 font-medium">Confidence</th>
                    <th className="text-center px-4 py-2.5 font-medium">Decision</th>
                    <th className="text-left px-4 py-2.5 font-medium">Merchant</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r) => (
                    <tr
                      key={r.rowIndex}
                      className="border-b border-[#1e293b] hover:bg-[#1e293b]/40 transition-colors"
                    >
                      <td className="px-4 py-2.5 text-xs text-[#64748b] font-mono">{r.rowIndex + 1}</td>
                      <td className="px-4 py-2.5 font-mono text-xs text-[#94a3b8]">{r.transactionId}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs">${r.amount.toFixed(2)}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs">
                        {(r.riskScore * 100).toFixed(0)}%
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs">
                        {(r.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="px-4 py-2.5 text-center">
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider ${
                            r.decision === "BLOCK"
                              ? "bg-red-500/10 text-red-400"
                              : r.decision === "FLAG"
                              ? "bg-amber-500/10 text-amber-400"
                              : "bg-emerald-500/10 text-emerald-400"
                          }`}
                        >
                          {r.decision}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-xs text-[#94a3b8]">{r.merchant}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
