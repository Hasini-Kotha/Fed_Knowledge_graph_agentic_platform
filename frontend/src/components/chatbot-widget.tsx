"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { API_BASE_URL } from "@/lib/config";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatbotWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Let me know how I can help you. I'll do my best to help.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [mounted, setMounted] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (open) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      setTimeout(() => inputRef.current?.focus(), 100);
    } else {
      setMessages([
        {
          role: "assistant",
          content:
            "Let me know how I can help you. I'll do my best to help.",
        },
      ]);
      setInput("");
    }
  }, [open]);

  const send = useCallback(
    async (overrideText?: string) => {
      const text = (overrideText ?? input).trim();
      if (!text || loading) return;
      setInput("");

      const updated: Message[] = [
        ...messages,
        { role: "user", content: text },
      ];
      setMessages(updated);
      setLoading(true);

      try {
        const res = await fetch(`${API_BASE_URL}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: updated.map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          const detail = errData.detail ?? `HTTP ${res.status}`;
          throw new Error(detail);
        }

        const data = await res.json();
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.reply },
        ]);
      } catch (err: unknown) {
        const msg =
          err instanceof Error ? err.message : "Unknown error";
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Backend error: ${msg}. Make sure the backend server is running.`,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, messages]
  );

  const widget = (
    <>
      {/* Chat panel — opens above the button */}
      {open && (
        <div
          style={{
            position: "fixed",
            bottom: "88px",
            right: "24px",
            zIndex: 99999,
            width: "480px",
            height: "700px",
            background: "#0b1220",
            border: "1px solid rgba(6,182,212,0.3)",
            borderRadius: "16px",
            display: "flex",
            flexDirection: "column",
            boxShadow: "0 12px 48px rgba(0,0,0,0.7), 0 0 0 1px rgba(6,182,212,0.1)",
            overflow: "hidden",
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: "12px 16px",
              borderBottom: "1px solid rgba(255,255,255,0.06)",
              background: "#070d1a",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: "#22c55e",
                boxShadow: "0 0 6px #22c55e",
              }}
            />
            <span
              style={{
                color: "#f1f5f9",
                fontSize: "13px",
                fontWeight: 600,
                letterSpacing: "0.01em",
              }}
            >
              TraceAI Assistant
            </span>
            </div>

          {/* Messages area */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "12px",
              display: "flex",
              flexDirection: "column",
              gap: "8px",
              scrollbarWidth: "thin",
              scrollbarColor: "rgba(255,255,255,0.1) transparent",
            }}
          >
            {messages.map((m, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  justifyContent:
                    m.role === "user" ? "flex-end" : "flex-start",
                  alignItems: "flex-end",
                  gap: "6px",
                }}
              >
                {m.role === "assistant" && (
                  <div
                    style={{
                      width: "22px",
                      height: "22px",
                      borderRadius: "50%",
                      background: "rgba(6,182,212,0.15)",
                      border: "1px solid rgba(6,182,212,0.3)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                      fontSize: "9px",
                      color: "#06b6d4",
                      fontWeight: 700,
                    }}
                  >
                    AI
                  </div>
                )}
                <div
                  style={{
                    maxWidth: "84%",
                    padding: "8px 11px",
                    borderRadius:
                      m.role === "user"
                        ? "12px 3px 12px 12px"
                        : "3px 12px 12px 12px",
                    background:
                      m.role === "user"
                        ? "linear-gradient(135deg, #0891b2, #0e7490)"
                        : "rgba(255,255,255,0.05)",
                    color: m.role === "user" ? "#fff" : "#cbd5e1",
                    fontSize: "12px",
                    lineHeight: "1.6",
                    border:
                      m.role === "assistant"
                        ? "1px solid rgba(255,255,255,0.06)"
                        : "none",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                  }}
                >
                  {m.content}
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  padding: "4px 2px",
                }}
              >
                <div
                  style={{
                    width: "22px",
                    height: "22px",
                    borderRadius: "50%",
                    background: "rgba(6,182,212,0.15)",
                    border: "1px solid rgba(6,182,212,0.3)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "9px",
                    color: "#06b6d4",
                    fontWeight: 700,
                  }}
                >
                  AI
                </div>
                <div
                  style={{
                    display: "flex",
                    gap: "3px",
                    background: "rgba(255,255,255,0.05)",
                    padding: "8px 12px",
                    borderRadius: "3px 12px 12px 12px",
                    border: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      style={{
                        width: "5px",
                        height: "5px",
                        borderRadius: "50%",
                        background: "#06b6d4",
                        animation: `traceai-dot 1.2s ${i * 0.2}s infinite`,
                      }}
                    />
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input row */}
          <div
            style={{
              padding: "10px 12px",
              borderTop: "1px solid rgba(255,255,255,0.06)",
              display: "flex",
              gap: "8px",
              flexShrink: 0,
              background: "#070d1a",
            }}
          >
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !loading) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Ask about your transactions…"
              disabled={loading}
              style={{
                flex: 1,
                padding: "8px 11px",
                borderRadius: "8px",
                fontSize: "12px",
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.09)",
                color: "#e2e8f0",
                outline: "none",
                transition: "border-color 0.15s",
              }}
              onFocus={(e) =>
                (e.target.style.borderColor = "rgba(6,182,212,0.5)")
              }
              onBlur={(e) =>
                (e.target.style.borderColor = "rgba(255,255,255,0.09)")
              }
            />
            <button
              onClick={() => send()}
              disabled={loading || !input.trim()}
              style={{
                padding: "8px 14px",
                borderRadius: "8px",
                fontSize: "12px",
                fontWeight: 500,
                background:
                  loading || !input.trim()
                    ? "rgba(6,182,212,0.15)"
                    : "#06b6d4",
                border: "none",
                color:
                  loading || !input.trim() ? "#164e63" : "#fff",
                cursor:
                  loading || !input.trim()
                    ? "not-allowed"
                    : "pointer",
                transition: "all 0.15s",
                flexShrink: 0,
              }}
            >
              {loading ? "…" : "Send"}
            </button>
          </div>
        </div>
      )}

      {/* The round floating button — always on top, always visible */}
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? "Close assistant" : "Open TraceAI Assistant"}
        style={{
          position: "fixed",
          bottom: "24px",
          right: "24px",
          zIndex: 99999,
          width: "52px",
          height: "52px",
          borderRadius: "50%",
          background: open
            ? "#1e293b"
            : "linear-gradient(135deg, #06b6d4 0%, #0284c7 100%)",
          border: open
            ? "1px solid rgba(6,182,212,0.4)"
            : "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: open
            ? "0 2px 12px rgba(0,0,0,0.4)"
            : "0 4px 20px rgba(6,182,212,0.5), 0 2px 8px rgba(0,0,0,0.3)",
          transition: "all 0.2s ease",
        }}
        onMouseEnter={(e) => {
          const b = e.currentTarget as HTMLButtonElement;
          b.style.transform = "scale(1.1)";
        }}
        onMouseLeave={(e) => {
          const b = e.currentTarget as HTMLButtonElement;
          b.style.transform = "scale(1.0)";
        }}
      >
        {open ? (
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#06b6d4"
            strokeWidth="2.5"
            strokeLinecap="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <svg
            width="22"
            height="22"
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        )}
      </button>

      <style>{`
        @keyframes traceai-dot {
          0%, 80%, 100% { opacity: 0.15; transform: scale(0.8); }
          40% { opacity: 1; transform: scale(1.1); }
        }
      `}</style>
    </>
  );

  return mounted ? createPortal(widget, document.body) : null;
}
