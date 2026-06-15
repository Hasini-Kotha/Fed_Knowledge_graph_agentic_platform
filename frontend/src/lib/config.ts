/** Dynamically compute backend URL based on frontend port */
export const getApiBaseUrl = () => {
  if (typeof window !== "undefined") {
    const port = window.location.port;
    if (port === "3000") return "http://127.0.0.1:8000";
    if (port === "3001") return "http://127.0.0.1:8001";
    if (port === "3002") return "http://127.0.0.1:8002";
    if (port === "3003") return "http://127.0.0.1:8003";
  }
  return process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
};

export const API_BASE_URL = getApiBaseUrl();

/** When true, API functions return mock data instead of calling the backend */
export const USE_MOCK = false;
