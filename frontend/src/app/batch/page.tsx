// Step 7 — Batch page removed from navigation.
// Redirect all direct URL access to the dashboard.
import { redirect } from "next/navigation"
export default function Page() { redirect("/") }
