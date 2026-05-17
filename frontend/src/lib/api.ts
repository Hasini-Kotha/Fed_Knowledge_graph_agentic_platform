import { API_BASE_URL } from "./config"
import type {
  TransactionResult,
  Decision,
  SystemStats,
  FLStatus,
  Alert,
  BatchRow,
} from "./types"

// ─── Fetch helper ─────────────────────────────────────────────────────
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  })
  if (!res.ok) throw new Error(`Backend error: ${res.status} ${res.statusText}`)
  return res.json()
}

// ─── Public API — all functions call real backend endpoints ────────────

export const api = {
  /** POST /api/scan — single transaction through the 5-stage pipeline. */
  async scanTransaction(data: {
    transactionId?: string
    amount: number
    location?: string
    ip?: string
    merchant?: string
  }): Promise<TransactionResult> {
    return apiFetch("/api/scan", {
      method: "POST",
      body: JSON.stringify(data),
    })
  },

  /** POST /api/batch — analyze an array of { amount, merchant } rows. */
  async batchAnalyze(rows: { amount: number; merchant?: string }[]): Promise<BatchRow[]> {
    return apiFetch("/api/batch", {
      method: "POST",
      body: JSON.stringify(rows),
    })
  },

  /** GET /api/decisions — audit log with optional filters. */
  async getDecisions(filters?: {
    decision?: string
    minRisk?: number
    dateFrom?: string
    dateTo?: string
  }): Promise<Decision[]> {
    const params = new URLSearchParams()
    if (filters?.decision && filters.decision !== "all") {
      params.set("type", filters.decision)
    }
    if (filters?.minRisk) {
      params.set("minRisk", String(filters.minRisk))
    }
    const qs = params.toString()
    return apiFetch(`/api/decisions${qs ? `?${qs}` : ""}`)
  },

  /** GET /api/dashboard/stats — KPI cards + approval breakdown. */
  async getSystemStats(): Promise<SystemStats> {
    return apiFetch("/api/dashboard/stats")
  },

  /** GET /api/dashboard/alerts — recent alerts table. */
  async getRecentAlerts(): Promise<Alert[]> {
    return apiFetch("/api/dashboard/alerts")
  },

  /** GET /api/system/fl — FL training status with accuracy history. */
  async getFLStatus(): Promise<FLStatus> {
    return apiFetch("/api/system/fl")
  },

  /** GET /api/system/mcp — MCP bridge health statuses. */
  async getMCPStatus(): Promise<
    { name: string; status: "ONLINE" | "SYNCING" | "OFFLINE"; latency: string }[]
  > {
    return apiFetch("/api/system/mcp")
  },

  /** GET /api/system/kg — knowledge graph node/edge/community counts. */
  async getKGStats(): Promise<{
    nodes: number
    edges: number
    communities: number
    lastUpdated: string
  }> {
    return apiFetch("/api/system/kg")
  },
}
