"use client"

import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import {
  Cpu,
  ShieldCheck,
  Clock,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Lock,
  FileText,
  Radio,
  Server,
  Download,
  Upload,
  AlertTriangle,
  FileWarning,
  Check,
  X,
  Info
} from "lucide-react"
import { getVerifiedClaims } from "@/components/sidebar"

const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000"

interface ValidationStep {
  name: string
  status: "pending" | "success" | "failed"
  description: string
}

export default function ClientFLPanel() {
  const router = useRouter()

  // Auth & Client identity
  const [token, setToken] = useState<string | null>(null)
  const [clientId, setClientId] = useState<string | null>(null)
  const [bankName, setBankName] = useState<string | null>(null)

  // Page level states
  const [roundStatus, setRoundStatus] = useState<any>(null)
  const [submissions, setSubmissions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Download state
  const [downloadState, setDownloadState] = useState<"idle" | "downloading" | "done">("idle")
  const [downloadError, setDownloadError] = useState<string | null>(null)

  // Upload states
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [uploadState, setUploadState] = useState<"idle" | "uploading" | "validating" | "accepted" | "rejected">("idle")
  const [submissionId, setSubmissionId] = useState<string | null>(null)
  const [rejectionReason, setRejectionReason] = useState<string | null>(null)
  const [alreadySubmittedThisRound, setAlreadySubmittedThisRound] = useState(false)
  const [validationSteps, setValidationSteps] = useState<ValidationStep[]>([
    { name: "HMAC Signature Check", status: "pending", description: "Verifies payload authenticity and integrity." },
    { name: "Fernet Decryption Check", status: "pending", description: "Decrypts weights using gateway secret key." },
    { name: "LiteFraudNet Tensor Shape Check", status: "pending", description: "Validates layer shapes match client model." },
    { name: "Data Integrity (NaN/Inf) Check", status: "pending", description: "Checks weights are clean and unpoisoned." }
  ])

  // Local Training states
  const [isCsv, setIsCsv] = useState(false)
  const [trainState, setTrainState] = useState<"idle" | "training" | "done">("idle")

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // 1. Role & Token Validation
  useEffect(() => {
    const claims = getVerifiedClaims()
    if (!claims || claims.role !== "fl_client") {
      router.push("/")
      return
    }
    setToken(localStorage.getItem("token"))
    setClientId(claims.sub || "")
    setBankName(claims.bank_name || claims.sub || "Partner Bank")
  }, [router])

  // 2. Fetch Round State and Submissions
  useEffect(() => {
    if (!token) return
    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [token])

  // 3. Persistent Validation State Restoration
  useEffect(() => {
    if (!token || !roundStatus) return

    const savedSubId = localStorage.getItem("fl_last_submission_id")
    const savedRound = localStorage.getItem("fl_last_submission_round")

    if (savedSubId && savedRound) {
      // If a new round has started, clear the old submission state
      if (parseInt(savedRound, 10) < roundStatus.current_round) {
        localStorage.removeItem("fl_last_submission_id")
        localStorage.removeItem("fl_last_submission_round")
      } else {
        setSubmissionId(savedSubId)
        setUploadState("validating")
        startPolling(savedSubId)
      }
    }
  }, [token, roundStatus])

  // Clean up poll on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const fetchData = async () => {
    if (!token) return
    setLoading(true)
    setErrorMsg(null)
    try {
      const headers = { Authorization: `Bearer ${token}` }
      const [roundRes, subRes] = await Promise.all([
        fetch(`${GATEWAY_URL}/fl/round-status`, { headers }),
        fetch(`${GATEWAY_URL}/fl/my-submissions`, { headers }),
      ])
      let fetchedRoundStatus = roundStatus
      if (roundRes.ok) {
        fetchedRoundStatus = await roundRes.json()
        setRoundStatus(fetchedRoundStatus)
      } else {
        setErrorMsg("Could not fetch active round status from secure gateway.")
      }
      if (subRes.ok) {
        const subData = await subRes.json()
        const subs: any[] = subData.submissions || []
        setSubmissions(subs)
        
        if (fetchedRoundStatus) {
          const already = subs.some(
            (s) => s.round_number === fetchedRoundStatus.current_round && s.validation_status !== "REJECTED"
          )
          setAlreadySubmittedThisRound(already)
        }
      }
    } catch (err: any) {
      setErrorMsg("Network connection error: " + err.message)
    } finally {
      setLoading(false)
    }
  }

  // Panel 2 — Download Global Model
  const handleDownload = async () => {
    if (!token) return
    setDownloadState("downloading")
    setDownloadError(null)
    try {
      const res = await fetch(`${GATEWAY_URL}/fl/global-model`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error(`Server returned status: ${res.status}`)
      const blob = await res.blob()
      
      const contentDisp = res.headers.get("content-disposition") || ""
      const fnMatch = contentDisp.match(/filename="?([^"]+)"?/)
      const filename = fnMatch ? fnMatch[1] : `global_model_round_${roundStatus?.current_round ?? 1}.bin`
      
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setDownloadState("done")
    } catch (err: any) {
      setDownloadError("Download failed: " + err.message)
      setDownloadState("idle")
    }
  }

  // Panel 3 — File Selection validation
  const ALLOWED_EXTENSIONS = [".csv", ".bin", ".pt", ".npz"]
  const handleFileSelect = (file: File) => {
    if (file.size > 500 * 1024 * 1024) {
      setFileError("File size exceeds the maximum limit of 500MB.")
      setSelectedFile(null)
      return
    }
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setFileError(`File type "${ext}" not supported. Accepted: ${ALLOWED_EXTENSIONS.join(", ")}`)
      setSelectedFile(null)
      return
    }
    setFileError(null)
    setSelectedFile(file)
    setIsCsv(ext === ".csv")
    setTrainState("idle")
    setUploadState("idle")
    setRejectionReason(null)
    resetSteps()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  const resetSteps = () => {
    setValidationSteps([
      { name: "HMAC Signature Check", status: "pending", description: "Verifies payload authenticity and integrity." },
      { name: "Fernet Decryption Check", status: "pending", description: "Decrypts weights using gateway secret key." },
      { name: "LiteFraudNet Tensor Shape Check", status: "pending", description: "Validates layer shapes match client model." },
      { name: "Data Integrity (NaN/Inf) Check", status: "pending", description: "Checks weights are clean and unpoisoned." }
    ])
  }

  // Panel 3 — Train Local Model
  const handleTrainLocal = async () => {
    if (!selectedFile || !token) return
    setTrainState("training")
    setFileError(null)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      const res = await fetch(`${GATEWAY_URL}/fl/client/train-local`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || "Local training failed.")
      }

      const blob = await res.blob()
      
      const contentDisp = res.headers.get("content-disposition") || ""
      const fnMatch = contentDisp.match(/filename="?([^"]+)"?/)
      const filename = fnMatch ? fnMatch[1] : `local_trained_weights_${clientId}_R${roundStatus?.current_round ?? 1}.bin`

      const newFile = new File([blob], filename, { type: "application/octet-stream" })
      setSelectedFile(newFile)
      setIsCsv(false)
      setTrainState("done")
    } catch (err: any) {
      setFileError("Training error: " + err.message)
      setTrainState("idle")
    }
  }

  // Panel 3 — Submit Weight Update
  const handleUpload = async () => {
    if (!selectedFile || !token || alreadySubmittedThisRound) return
    setUploadState("uploading")
    setRejectionReason(null)
    resetSteps()

    try {
      // Read signature from the file prefix to send in the custom header for compliance
      let signatureHeader = ""
      try {
        const text = await selectedFile.slice(0, 64).text()
        if (text.length === 64 && /^[0-9a-fA-F]+$/.test(text)) {
          signatureHeader = text
        }
      } catch (e) {}

      const formData = new FormData()
      formData.append("file", selectedFile)
      if (roundStatus) {
        formData.append("round_num", String(roundStatus.current_round))
      }

      const headers: Record<string, string> = {
        Authorization: `Bearer ${token}`
      }
      if (signatureHeader) {
        headers["HMAC-Signature"] = signatureHeader
      }

      const res = await fetch(`${GATEWAY_URL}/fl/submit-weights`, {
        method: "POST",
        headers,
        body: formData,
      })

      const data = await res.json()

      if (res.status === 429) {
        setAlreadySubmittedThisRound(true)
        setUploadState("accepted")
        return
      }

      if (!res.ok) {
        const msg = data?.message || (typeof data?.detail === 'string' ? data.detail : data?.detail?.message) || "Upload rejected."
        setRejectionReason(msg)
        setUploadState("rejected")
        return
      }

      const sid = data.submission_id
      setSubmissionId(sid)
      setUploadState("validating")
      
      // Save state in localStorage to survive page refreshes
      localStorage.setItem("fl_last_submission_id", sid)
      if (roundStatus) {
        localStorage.setItem("fl_last_submission_round", String(roundStatus.current_round))
      }

      startPolling(sid)
    } catch (err: any) {
      setRejectionReason("Upload error: " + err.message)
      setUploadState("rejected")
    }
  }

  const startPolling = (sid: string) => {
    if (pollRef.current) clearInterval(pollRef.current)
    
    pollRef.current = setInterval(async () => {
      try {
        const pollRes = await fetch(`${GATEWAY_URL}/fl/submission-status/${sid}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (!pollRes.ok) {
          if (pollRes.status === 404 || pollRes.status === 403) {
            if (pollRef.current) clearInterval(pollRef.current)
            localStorage.removeItem("fl_last_submission_id")
            localStorage.removeItem("fl_last_submission_round")
            setUploadState("idle")
            setSubmissionId(null)
          }
          return
        }
        const pollData = await pollRes.json()

        // Update step-by-step progress cards
        const stepsCopy = [
          { name: "HMAC Signature Check", status: pollData.hmac_verified ? "success" : pollData.validation_status === "REJECTED" && !pollData.hmac_verified ? "failed" : "pending", description: "Verifies payload authenticity and integrity." },
          { name: "Fernet Decryption Check", status: pollData.hmac_verified ? "success" : pollData.validation_status === "REJECTED" && !pollData.hmac_verified ? "failed" : "pending", description: "Decrypts weights using gateway secret key." },
          { name: "LiteFraudNet Tensor Shape Check", status: pollData.shape_valid ? "success" : pollData.validation_status === "REJECTED" && !pollData.shape_valid && pollData.hmac_verified ? "failed" : "pending", description: "Validates layer shapes match client model." },
          { name: "Data Integrity (NaN/Inf) Check", status: pollData.nan_inf_clean ? "success" : pollData.validation_status === "REJECTED" && !pollData.nan_inf_clean && pollData.shape_valid ? "failed" : "pending", description: "Checks weights are clean and unpoisoned." }
        ] as ValidationStep[]
        
        setValidationSteps(stepsCopy)

        if (pollData.validation_status === "VALID") {
          setUploadState("accepted")
          setAlreadySubmittedThisRound(true)
          if (pollRef.current) clearInterval(pollRef.current)
          fetchData()
        } else if (pollData.validation_status === "REJECTED") {
          setUploadState("rejected")
          setRejectionReason(pollData.rejection_reason || "Weights rejected by validation checks.")
          if (pollRef.current) clearInterval(pollRef.current)
        }
      } catch { /* ignore connection errors */ }
    }, 3000)
  }

  if (!token) return null

  // Get last submission metadata for Panel 1
  const lastSub = submissions.length > 0 ? submissions[0] : null

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-800 pb-5">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs uppercase tracking-wider font-semibold">
            <Radio className="w-3.5 h-3.5 animate-pulse" />
            FL Node Connection Active
          </div>
          <h1 className="text-3xl font-extrabold text-slate-100 tracking-tight">Federated Node Console</h1>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 text-sm text-slate-300 font-semibold transition-colors disabled:opacity-50 self-start"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Refresh Node
        </button>
      </div>

      {errorMsg && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-sm flex items-start gap-2.5">
          <XCircle className="w-5 h-5 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* PANEL 1: Node Identity & Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="p-6 bg-slate-900/40 border border-slate-800/80 rounded-2xl space-y-4">
          <h2 className="text-sm font-bold text-slate-300 font-mono uppercase tracking-wider flex items-center gap-2">
            <Server className="w-4 h-4 text-cyan-400" />
            Node Identity
          </h2>
          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between border-b border-slate-800/60 pb-2">
              <span className="text-slate-500">Bank Partner:</span>
              <span className="text-slate-200 font-semibold">{bankName}</span>
            </div>
            <div className="flex justify-between border-b border-slate-800/60 pb-2">
              <span className="text-slate-500">Client ID:</span>
              <span className="text-slate-300 select-all">{clientId}</span>
            </div>
            <div className="flex justify-between pb-1">
              <span className="text-slate-500">Global Model Version:</span>
              <span className="text-cyan-400 font-semibold">LiteFraudNet-v1</span>
            </div>
          </div>
        </div>

        {/* Status panel */}
        {roundStatus && (
          <div className="lg:col-span-2 p-6 bg-slate-900/40 border border-slate-800/80 rounded-2xl space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-md font-bold text-slate-100 flex items-center gap-2">
                <Cpu className="w-5 h-5 text-cyan-400" />
                Active FL Round Information
              </h2>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                roundStatus.round_open
                  ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
                  : "bg-rose-500/10 text-rose-400 border-rose-500/20"
              }`}>
                ROUND {roundStatus.round_open ? "OPEN" : "CLOSED"}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="p-4 bg-slate-950/40 border border-slate-800 rounded-xl">
                <div className="text-[10px] text-slate-500 uppercase font-mono font-semibold">Active Round</div>
                <div className="text-lg font-extrabold text-slate-100 mt-1">Round {roundStatus.current_round}</div>
              </div>
              <div className="p-4 bg-slate-950/40 border border-slate-800 rounded-xl">
                <div className="text-[10px] text-slate-500 uppercase font-mono font-semibold">Updates</div>
                <div className="text-lg font-extrabold text-slate-100 mt-1">
                  {roundStatus.received_updates} / {roundStatus.expected_clients}
                </div>
              </div>
              <div className="p-4 bg-slate-950/40 border border-slate-800 rounded-xl">
                <div className="text-[10px] text-slate-500 uppercase font-mono font-semibold">Last Status</div>
                <div className="mt-1">
                  {lastSub ? (
                    <span className={`inline-flex items-center gap-1 font-bold text-xs uppercase px-2 py-0.5 rounded border ${
                      lastSub.validation_status === "VALID"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        : lastSub.validation_status === "REJECTED"
                        ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                        : "bg-amber-500/10 text-amber-400 border-amber-500/20 animate-pulse"
                    }`}>
                      {lastSub.validation_status}
                    </span>
                  ) : (
                    <span className="text-slate-500 text-xs font-mono">No History</span>
                  )}
                </div>
              </div>
            </div>
            {lastSub && (
              <div className="p-3 bg-[#090d16]/65 border border-slate-850 rounded-xl font-mono text-[10px] text-slate-500 flex justify-between">
                <span>Last Submission ID: <strong className="text-slate-300">{lastSub.submission_id}</strong></span>
                <span>Submitted At: <strong className="text-slate-300">{new Date(lastSub.submitted_at).toLocaleString()}</strong></span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* PANEL 2: Download Model */}
      <div className="p-6 bg-slate-900/40 border border-slate-800/80 rounded-2xl space-y-4">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Download className="w-5 h-5 text-indigo-400" />
          Download Global Model Weights
        </h2>
        
        {/* ENCRYPTION SCHEME WARNING */}
        <div className="p-3.5 bg-indigo-500/10 border border-indigo-500/25 rounded-lg flex items-start gap-2.5">
          <Lock className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5 animate-pulse" />
          <p className="text-[11px] text-indigo-300 leading-relaxed font-mono">
            <strong>Encryption Notice:</strong> Global weights are distributed in an encrypted format. Decryption occurs automatically using the secure credentials derived from the <code>FL_GATEWAY_SECRET</code> during local training runs. Keep weights secured within local sandboxes.
          </p>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={handleDownload}
            disabled={downloadState === "downloading"}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-bold transition-all"
          >
            <Download className="w-4 h-4" />
            {downloadState === "downloading" ? "Downloading…" : "Download Binary Weights (.bin)"}
          </button>
          {roundStatus && (
            <span className="text-xs text-slate-400 font-mono">
              Round Checkpoint: #{roundStatus.current_round}
            </span>
          )}
        </div>

        {downloadState === "done" && (
          <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-emerald-400 text-xs font-medium font-mono">
            Download complete. Loaded encrypted binary weights into browser memory and saved to disk.
          </div>
        )}
        {downloadError && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-xs font-medium font-mono">
            {downloadError}
          </div>
        )}
      </div>

      {/* PANEL 3: Local Training & Upload Weights */}
      <div className="p-6 bg-slate-900/40 border border-slate-800/80 rounded-2xl space-y-5">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Upload className="w-5 h-5 text-cyan-400" />
          Local Training & Submission
        </h2>

        {alreadySubmittedThisRound && uploadState !== "rejected" ? (
          <div className="space-y-4">
            <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-400 text-sm font-medium flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5" />
              Consensus updates submitted and validated for Round #{roundStatus?.current_round}.
            </div>
            
            {/* Show completed progress cards when accepted */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {validationSteps.map((step) => (
                <div key={step.name} className="p-4 bg-slate-950/65 border border-emerald-500/20 rounded-xl flex items-start gap-3">
                  <div className="p-1 bg-emerald-500/20 text-emerald-400 rounded-full">
                    <Check className="w-3.5 h-3.5" />
                  </div>
                  <div>
                    <div className="text-xs font-bold text-slate-200">{step.name}</div>
                    <div className="text-[10px] text-slate-500 font-mono mt-0.5">{step.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="text-xs text-slate-400 font-mono">
              Consensus Round: <span className="text-cyan-400 font-bold">Round {roundStatus?.current_round ?? "—"}</span>
            </div>

            {/* Drag and Drop Zone */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-800 hover:border-cyan-500/30 rounded-xl p-8 text-center cursor-pointer transition-colors bg-slate-950/30"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.bin,.pt,.npz"
                className="hidden"
                onChange={(e) => { if (e.target.files?.[0]) handleFileSelect(e.target.files[0]) }}
              />
              {selectedFile ? (
                <div className="space-y-1">
                  <div className="text-cyan-400 font-bold text-sm">{selectedFile.name}</div>
                  <div className="text-slate-500 text-xs">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</div>
                  {trainState === "done" && <div className="text-emerald-400 text-xs font-bold mt-2">✅ Local model trained successfully! Ready to submit.</div>}
                </div>
              ) : (
                <div className="space-y-2">
                  <FileWarning className="w-8 h-8 text-slate-700 mx-auto" />
                  <p className="text-xs text-slate-400">Drag & drop raw dataset (.csv) or weights binary (.bin)</p>
                  <p className="text-[10px] text-slate-600 font-mono">Max size: 500MB (Accepted formats: .csv, .bin, .pt, .npz)</p>
                </div>
              )}
            </div>

            {fileError && (
              <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-xs font-mono">
                {fileError}
              </div>
            )}

            {/* Submit / Train button */}
            {isCsv ? (
              <button
                onClick={handleTrainLocal}
                disabled={trainState === "training"}
                className="w-full py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-400 disabled:opacity-40 text-white font-extrabold text-sm transition-all flex justify-center items-center gap-2"
              >
                {trainState === "training" ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Cpu className="w-4 h-4" />}
                {trainState === "training" ? "Training Local Model... (Please wait)" : "Train Local Model"}
              </button>
            ) : (
              <button
                onClick={handleUpload}
                disabled={!selectedFile || uploadState === "uploading" || uploadState === "validating"}
                className="w-full py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 text-slate-950 font-extrabold text-sm transition-all flex justify-center items-center gap-2"
              >
                {uploadState === "uploading" ? "Uploading Payload…"
                  : uploadState === "validating" ? "Validating Update..."
                  : "Submit Update Package"}
              </button>
            )}

            {/* Polling step-by-step progress checks */}
            {(uploadState === "validating" || uploadState === "rejected") && (
              <div className="space-y-4">
                <div className="flex items-center justify-between border-t border-slate-850 pt-3">
                  <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest font-bold flex items-center gap-1.5">
                    <Radio className="w-3 h-3 text-cyan-400 animate-pulse" />
                    Live Weight Validation Status
                  </span>
                  {uploadState === "validating" && (
                    <span className="text-[9px] text-cyan-400 font-mono animate-pulse bg-cyan-500/10 px-2 py-0.5 rounded">
                      Polling API...
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {validationSteps.map((step) => {
                    const isSuccess = step.status === "success"
                    const isFailed = step.status === "failed"
                    return (
                      <div 
                        key={step.name} 
                        className={`p-4 bg-slate-950/50 rounded-xl border transition-all ${
                          isSuccess ? "border-emerald-500/25 bg-emerald-500/5"
                            : isFailed ? "border-rose-500/25 bg-rose-500/5"
                            : "border-slate-850"
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <span className="text-xs font-bold text-slate-200">{step.name}</span>
                          {isSuccess ? (
                            <span className="p-0.5 bg-emerald-500/25 text-emerald-400 rounded-full">
                              <Check className="w-3 h-3" />
                            </span>
                          ) : isFailed ? (
                            <span className="p-0.5 bg-rose-500/25 text-rose-400 rounded-full">
                              <X className="w-3 h-3" />
                            </span>
                          ) : (
                            <span className="p-0.5 bg-slate-800 text-slate-500 rounded-full animate-spin">
                              <RefreshCw className="w-3 h-3" />
                            </span>
                          )}
                        </div>
                        <p className="text-[10px] text-slate-500 font-mono mt-1.5 leading-relaxed">{step.description}</p>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Error logs */}
            {uploadState === "rejected" && rejectionReason && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/25 rounded-xl space-y-2 max-w-3xl">
                <div className="text-xs font-bold text-rose-400 flex items-center gap-1.5">
                  <XCircle className="w-4 h-4" /> Validation Pipeline Halted
                </div>
                <p className="text-xs font-mono text-rose-300 leading-relaxed bg-slate-950/80 p-3 rounded border border-rose-500/10 select-all">
                  {rejectionReason}
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* PANEL 4: Submission History Log */}
      <div className="p-6 bg-slate-900/40 border border-slate-800/80 rounded-2xl space-y-4">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <FileText className="w-5 h-5 text-indigo-400" />
          Submission Audit Log (Last 10 updates)
        </h2>
        <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-950/20">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/40 text-[10px] font-mono uppercase text-slate-400 tracking-wider">
                <th className="p-4">Round</th>
                <th className="p-4">Submission ID</th>
                <th className="p-4">Submitted At</th>
                <th className="p-4">Status</th>
                <th className="p-4">Rejection Reason</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-xs font-mono">
              {submissions.slice(0, 10).length > 0 ? (
                submissions.slice(0, 10).map((s) => (
                  <tr key={s.submission_id} className="hover:bg-slate-900/10">
                    <td className="p-4 font-bold text-cyan-400">Round {s.round_number}</td>
                    <td className="p-4 text-slate-300 select-all">{s.submission_id}</td>
                    <td className="p-4 text-slate-400">{new Date(s.submitted_at).toLocaleString()}</td>
                    <td className="p-4">
                      {s.validation_status === "VALID" ? (
                        <span className="inline-flex items-center gap-1 text-emerald-400 font-bold text-[10px] bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                          <CheckCircle2 className="w-3 h-3" /> Validated
                        </span>
                      ) : s.validation_status === "REJECTED" ? (
                        <span className="inline-flex items-center gap-1 text-rose-400 font-bold text-[10px] bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20">
                          <XCircle className="w-3 h-3" /> Rejected
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-amber-400 font-bold text-[10px] bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                          <Clock className="w-3 h-3" /> Processing
                        </span>
                      )}
                    </td>
                    <td className="p-4 text-slate-400 max-w-[200px] truncate" title={s.rejection_reason || ""}>
                      {s.rejection_reason || "—"}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="p-6 text-center text-slate-600">
                    No submissions recorded for this client yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
