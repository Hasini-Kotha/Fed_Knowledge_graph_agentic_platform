export interface OverrideEventDetail {
  transactionId: string
  decision: string
  timestamp: string
}

let _pending: OverrideEventDetail | null = null

export function dispatchOverride(detail: OverrideEventDetail) {
  _pending = detail
  try {
    localStorage.setItem("traceai:override", JSON.stringify({
      ...detail,
      _eventTime: Date.now(),
    }))
  } catch {}
}

export function consumePendingOverride(): OverrideEventDetail | null {
  const val = _pending
  if (val) {
    _pending = null
    return val
  }
  try {
    const raw = localStorage.getItem("traceai:override")
    if (raw) {
      localStorage.removeItem("traceai:override")
      return JSON.parse(raw) as OverrideEventDetail
    }
  } catch {}
  return null
}
