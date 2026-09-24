import { isIP } from "node:net";
import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Enable only during the ANPR-specific edge check, while inference is disabled.
// Never return the actual address or the caller-supplied value.
export function GET(request: NextRequest): Response {
  if (process.env.ANPR_PROXY_PROOF_ENABLED !== "1" ||
      process.env.ANPR_TRUSTED_PROXY_IP_HEADER === "x-real-ip") {
    return new Response(null, { status: 404, headers: { "Cache-Control": "no-store" } });
  }
  const value = request.headers.get("x-real-ip") ?? "";
  return Response.json({
    header_present: isIP(value) !== 0,
    forged_ipv4_replaced: isIP(value) !== 0 && value !== "198.51.100.77",
    forged_ipv6_replaced: isIP(value) !== 0 && value !== "2001:db8::77"
  }, { headers: { "Cache-Control": "no-store" } });
}
