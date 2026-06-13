"use client"

import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Upload, Download, Loader2, AlertCircle, CheckCircle } from "lucide-react"
import { api } from "@/lib/api"
import type { BatchRow } from "@/lib/types"
import { getVerifiedClaims } from "@/components/sidebar"

export default function BatchPage() {
  const router = useRouter()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [rows, setRows] = useState<Record<string, number | string>[]>([])
  const [results, setResults] = useState<BatchRow[]>([])
  const [analyzing, setAnalyzing] = useState(false)
  const [fileName, setFileName] = useState("")
  const [batchError, setBatchError] = useState("")
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [processingTime, setProcessingTime] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)

  // Redirect guard
  useEffect(() => {
    const claims = getVerifiedClaims()
    if (!claims) {
      router.push("/login")
    } else if (claims.role !== "fl_client") {
      router.push("/admin")
    }
  }, [router])

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (file.size === 0) {
      setBatchError("The uploaded CSV file is empty. Please upload a valid CSV file.")
      setUploadSuccess(false)
      setFileName("")
      setRows([])
      return
    }

    setFileName(file.name)
    setResults([])
    setBatchError("")
    setUploadSuccess(false)

    const reader = new FileReader()
    reader.onload = (evt) => {
      const text = evt.target?.result as string
      const lines = text.split("\n").filter((l) => l.trim())
      if (lines.length < 2) {
        setBatchError("CSV file must contain a header row and at least one data row.")
        setRows([])
        return
      }

      const parsed: Record<string, number | string>[] = []
      const headers = lines[0].split(",").map((h) => h.trim())
      const headersLower = headers.map((h) => h.toLowerCase())

      const amountIdx = headersLower.findIndex(
        (h) => h === "amount" || h === "transaction_amount"
      )

      // Detect if CSV has V1-V28 feature columns (full ML mode, case-insensitive)
      const featureCols = ["v1","v2","v3","v4","v5","v6","v7","v8","v9","v10",
                           "v11","v12","v13","v14","v15","v16","v17","v18","v19","v20",
                           "v21","v22","v23","v24","v25","v26","v27","v28","amount"]
      const stringCols = new Set(["merchant", "location", "ip", "transaction_id", "id"])
      const hasFeatures = featureCols.every((c) => headersLower.includes(c))

      if (!hasFeatures && amountIdx < 0) {
        setBatchError("Missing required columns. CSV must contain 'amount' (for Quick mode) or full features V1-V28 + 'Amount' (for ML mode).")
        setRows([])
        return
      }

      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(",").map((c) => c.trim())
        if (cols.length < headers.length) continue

        if (hasFeatures) {
          const row: Record<string, number | string> = {}
          for (let j = 0; j < headers.length; j++) {
            const key = headers[j]
            const keyLower = key.toLowerCase()
            const val = cols[j]
            if (keyLower === "amount") {
              const num = parseFloat(val)
              row["Amount"] = isNaN(num) ? 0 : num
            } else if (keyLower.startsWith("v") && !isNaN(parseInt(keyLower.substring(1)))) {
              const vNum = keyLower.toUpperCase() // e.g. V1
              const num = parseFloat(val)
              row[vNum] = isNaN(num) ? 0 : num
            } else if (stringCols.has(keyLower)) {
              row[keyLower] = val || "UNKNOWN"
            } else {
              row[key] = val
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
            parsed.push({ Amount: amount, merchant } as Record<string, number | string>)
          }
        }
      }

      if (parsed.length === 0) {
        setBatchError("No valid records could be parsed. Check your CSV layout.")
        setRows([])
        return
      }

      setRows(parsed)
      setUploadSuccess(true)
    }
    
    reader.onerror = () => {
      setBatchError("Error reading file. Please try again.")
      setUploadSuccess(false)
    }
    
    reader.readAsText(file)
  }

  const handleAnalyze = async () => {
    if (!rows.length) return
    setAnalyzing(true)
    setBatchError("")
    setResults([])
    setCurrentStep(1) // Step 1: CSV Loaded
    
    const startTime = performance.now()
    
    // Simulate steps progress 
    let step = 1
    const interval = setInterval(() => {
      if (step < 5) {
        step += 1
        setCurrentStep(step)
      }
    }, 450)

    try {
      const res = await api.batchAnalyze(rows)
      clearInterval(interval)
      setCurrentStep(5)
      // Visual feedback wait
      await new Promise((resolve) => setTimeout(resolve, 200))
      
      const endTime = performance.now()
      setProcessingTime((endTime - startTime) / 1000)
      setResults(res)
    } catch (e: unknown) {
      clearInterval(interval)
      const msg = e instanceof Error ? e.message : "Batch analysis failed"
      setBatchError(`Backend prediction failed: ${msg}. Verify backend server is online and latest model weights exist.`)
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
    avgRisk: results.length > 0 ? results.reduce((acc, r) => acc + r.riskScore, 0) / results.length : 0,
    highRisk: results.filter((r) => r.riskScore >= 0.5).length,
  }

  const steps = [
    { label: "CSV Loaded" },
    { label: "Data Preprocessed" },
    { label: "Running LiteFraudNet Inference" },
    { label: "Generating KG Evidence" },
    { label: "Preparing Results" },
  ]

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

      {/* Upload Success Banner */}
      {uploadSuccess && !analyzing && results.length === 0 && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg flex items-center justify-between text-emerald-400 text-xs font-semibold">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span>CSV "{fileName}" uploaded and parsed successfully ({rows.length} rows ready for analysis).</span>
          </div>
        </div>
      )}

      {/* Upload Zone */}
      {!analyzing && results.length === 0 && (
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
                className="bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 px-4 py-2 rounded-md text-sm font-semibold uppercase tracking-wider hover:bg-cyan-500/20 transition-all disabled:opacity-50 flex items-center gap-2"
              >
                Run Analysis
              </button>
            </div>
          )}
        </div>
      )}

      {/* Step-wise Processing Progress */}
      {analyzing && (
        <div className="glass-panel p-6 border-cyan-500/20 bg-cyan-950/5 flex flex-col items-center justify-center space-y-6">
          <div className="flex items-center gap-3">
            <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
            <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Processing CSV Batch...
            </h2>
          </div>
          
          <div className="w-full max-w-md space-y-3">
            {steps.map((s, idx) => {
              const stepNum = idx + 1
              const isDone = stepNum < currentStep
              const isActive = stepNum === currentStep
              
              return (
                <div key={idx} className="flex items-center justify-between text-xs py-1.5 border-b border-[#1e293b] last:border-0">
                  <span className={`font-mono ${isDone ? "text-emerald-400" : isActive ? "text-cyan-400 font-semibold" : "text-slate-500"}`}>
                    {isDone ? "✓ " : isActive ? "⏳ " : "○ "} {s.label}
                  </span>
                  <div>
                    {isDone ? (
                      <span className="text-emerald-400 font-bold font-mono">Completed</span>
                    ) : isActive ? (
                      <span className="text-cyan-400 font-semibold font-mono animate-pulse">Running...</span>
                    ) : (
                      <span className="text-slate-650 font-mono">Pending</span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Results summary stats & list */}
      {results.length > 0 && !analyzing && (
        <div className="space-y-4 animate-fadeIn">
          
          {/* Summary Panel */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="glass-panel p-4 text-center">
              <span className="text-2xl font-bold font-mono text-[#f1f5f9]">{summary.total}</span>
              <p className="text-[9px] text-[#64748b] uppercase tracking-wider mt-1 font-semibold">Total Processed</p>
            </div>
            <div className="glass-panel p-4 text-center border-red-500/20 bg-red-500/5">
              <span className="text-2xl font-bold font-mono text-red-400">{summary.blocked}</span>
              <p className="text-[9px] text-red-400/75 uppercase tracking-wider mt-1 font-semibold">Fraud Predictions</p>
            </div>
            <div className="glass-panel p-4 text-center border-emerald-500/20 bg-emerald-500/5">
              <span className="text-2xl font-bold font-mono text-emerald-400">{summary.allowed}</span>
              <p className="text-[9px] text-emerald-400/75 uppercase tracking-wider mt-1 font-semibold">Safe Predictions</p>
            </div>
            <div className="glass-panel p-4 text-center border-amber-500/20 bg-amber-500/5">
              <span className="text-2xl font-bold font-mono text-amber-400 font-mono">{(summary.avgRisk * 100).toFixed(1)}%</span>
              <p className="text-[9px] text-amber-400/75 uppercase tracking-wider mt-1 font-semibold">Avg Risk Score</p>
            </div>
            <div className="glass-panel p-4 text-center border-purple-500/20 bg-purple-500/5">
              <span className="text-2xl font-bold font-mono text-purple-400">{summary.highRisk}</span>
              <p className="text-[9px] text-purple-400/75 uppercase tracking-wider mt-1 font-semibold">High Risk Cases</p>
            </div>
            <div className="glass-panel p-4 text-center">
              <span className="text-2xl font-bold font-mono text-cyan-400 font-mono">{processingTime.toFixed(2)}s</span>
              <p className="text-[9px] text-cyan-400/75 uppercase tracking-wider mt-1 font-semibold">Processing Time</p>
            </div>
          </div>

          {/* Table */}
          <div className="glass-panel">
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e293b]">
              <h2 className="text-sm font-medium text-[#f1f5f9]">Detailed Results</h2>
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
                        {(r.riskScore * 100).toFixed(2)}%
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
          
          <div className="flex justify-end">
            <button
              onClick={() => {
                setResults([])
                setRows([])
                setFileName("")
                setUploadSuccess(false)
              }}
              className="px-4 py-2 bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 rounded-md text-xs font-semibold uppercase tracking-wider transition-colors"
            >
              Analyze Another Batch
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
