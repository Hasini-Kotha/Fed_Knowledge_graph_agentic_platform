export interface TransactionResult {
  id: string
  riskScore: number
  confidence: number
  decision: "ALLOW" | "FLAG" | "BLOCK"
  factors: Factor[]
  graph: GraphData
  rationale: string[]
  timestamp: string
}

export interface Factor {
  name: string
  contribution: number
  description: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GraphNode {
  id: string
  label: string
  type: string
  risk: number
  cluster: number
}

export interface GraphEdge {
  source: string
  target: string
  type: string
  weight: number
}

export interface Decision {
  id: string
  transactionId: string
  riskScore: number
  confidence: number
  decision: "ALLOW" | "FLAG" | "BLOCK"
  timestamp: string
  merchant: string
  amount: number
  factors: Factor[]
}

export interface FLHistoryPoint {
  round: number
  accuracy: number
}

export interface ApprovalBreakdown {
  approvedPercent: number
  flaggedPercent: number
  blockedPercent: number
}

export interface SystemStats {
  totalTransactions: number
  fraudRate: number
  blockedToday: number
  pendingReview: number
  modelAccuracy: number
  approvalBreakdown: ApprovalBreakdown
}

export interface FLStatus {
  round: number
  clients: number
  accuracy: number
  loss: number
  history: FLHistoryPoint[]
}

export interface Alert {
  id: string
  transactionId: string
  riskScore: number
  decision: "ALLOW" | "FLAG" | "BLOCK"
  timestamp: string
  merchant: string
  amount: number
}

export interface BatchRow {
  rowIndex: number
  transactionId: string
  amount: number
  riskScore: number
  confidence: number
  decision: "ALLOW" | "FLAG" | "BLOCK"
  merchant: string
}
