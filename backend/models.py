"""Pydantic models — JSON field names match frontend/src/lib/types.ts exactly."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Factor(BaseModel):
    name: str
    contribution: float = Field(ge=0.0, le=1.0)
    description: str


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    risk: float = Field(ge=0.0, le=1.0)
    cluster: int


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    weight: float = Field(ge=0.0, le=1.0)


class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class TransactionResult(BaseModel):
    id: str
    riskScore: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    decision: str
    factors: List[Factor]
    graph: GraphData
    rationale: List[str]
    timestamp: str


class ScanRequest(BaseModel):
    transactionId: Optional[str] = None
    amount: float
    location: Optional[str] = None
    ip: Optional[str] = None
    merchant: Optional[str] = None


class BatchRowResult(BaseModel):
    rowIndex: int
    transactionId: str
    amount: float
    riskScore: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    decision: str
    merchant: str


class DecisionResult(BaseModel):
    id: str
    transactionId: str
    riskScore: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    decision: str
    timestamp: str
    merchant: str
    amount: float
    factors: List[Factor]


class ApprovalBreakdown(BaseModel):
    approvedPercent: float
    flaggedPercent: float
    blockedPercent: float


class SystemStats(BaseModel):
    totalTransactions: int
    fraudRate: float
    blockedToday: int
    pendingReview: int
    modelAccuracy: float
    approvalBreakdown: ApprovalBreakdown


class FLHistoryPoint(BaseModel):
    round: int
    accuracy: float


class FLStatus(BaseModel):
    round: int
    clients: int
    accuracy: float
    loss: float
    history: List[FLHistoryPoint]


class AlertItem(BaseModel):
    id: str
    transactionId: str
    riskScore: float
    decision: str
    timestamp: str
    merchant: str
    amount: float


class ActivityItem(BaseModel):
    time: str
    action: str
    description: str
    status: str


class KGStats(BaseModel):
    nodes: int
    edges: int
    communities: int
    lastUpdated: str


class MCPStatus(BaseModel):
    name: str
    status: str
    latency: str


class PipelineStage(BaseModel):
    name: str
    status: str
    duration: str
