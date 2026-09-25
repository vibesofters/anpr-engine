import type { NextRequest } from "next/server";
import { isIP } from "node:net";
import { admission } from "@/server/admission";
import { callPrivate } from "@/server/bridge";
import { consumeDailyQuota } from "@/server/daily-quota";
import { requestId, safeError } from "@/server/errors";
import { ImageError, mediaType, readBounded, validateImage } from "@/server/image";
import { isRateLimitExempt } from "@/server/rate-limit-exemption";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const SESSION_PATTERN = /^[a-f0-9]{64}$/;

function locale(request: NextRequest, origin: string): "en" | "tr" {
  try {
    const referer = new URL(request.headers.get("referer") ?? "");
    if (referer.origin === origin && (referer.pathname === "/tr" || referer.pathname.startsWith("/tr/"))) return "tr";
  } catch { /* Default English without accepting a client-supplied metadata field. */ }
  return "en";
}

function clientIp(request: NextRequest): string | null {
  if (process.env.NODE_ENV !== "production") return "local-development";
  if (process.env.ANPR_LOCAL_DIRECT_MODE === "1" &&
      /^http:\/\/(?:127\.0\.0\.1|localhost):[0-9]{1,5}$/.test(process.env.ANPR_PUBLIC_ORIGIN ?? "")) {
    // The local Docker port is loopback-only and has no trusted edge proxy.
    return "local-container-loopback";
  }
  if (process.env.ANPR_TRUSTED_PROXY_IP_HEADER !== "x-real-ip") return null;
  const ip = request.headers.get("x-real-ip");
  if (!ip || ip.length > 64 || ip.includes(",") || isIP(ip) === 0) return null;
  return ip;
}

export async function POST(request: NextRequest): Promise<Response> {
  const id = requestId();
  const expectedOrigin = process.env.ANPR_PUBLIC_ORIGIN ?? "http://127.0.0.1:3000";
  if (request.headers.get("origin") !== expectedOrigin ||
      ![null, "same-origin", "none"].includes(request.headers.get("sec-fetch-site"))) {
    return safeError("ORIGIN_REJECTED", 403, id);
  }
  const session = request.cookies.get("anpr_session")?.value;
  const ip = clientIp(request);
  if (!session || !SESSION_PATTERN.test(session)) return safeError("ORIGIN_REJECTED", 403, id);
  if (!ip) return safeError("MODEL_NOT_READY", 503, id);
  const rateLimitExempt = isRateLimitExempt(ip);
  if (request.headers.get("x-privacy-acknowledged") !== "true") {
    return safeError("PRIVACY_ACKNOWLEDGEMENT_REQUIRED", 400, id);
  }
  let type: ReturnType<typeof mediaType>;
  try { type = mediaType(request.headers.get("content-type")); }
  catch { return safeError("UNSUPPORTED_MEDIA_TYPE", 415, id); }
  const ticket = admission.acquire(session, ip, rateLimitExempt);
  if (ticket === "session") return safeError("SESSION_INFERENCE_ACTIVE", 409, id);
  if (ticket === "capacity") return safeError("CAPACITY_UNAVAILABLE", 503, id);
  if (ticket === "rate") return safeError("RATE_LIMITED", 429, id);
  try {
    const bytes = await readBounded(request.body, request.headers.get("content-length"));
    await validateImage(bytes, type);
    try {
      if (!rateLimitExempt && !consumeDailyQuota(ip)) return safeError("RATE_LIMITED", 429, id);
    } catch { return safeError("CAPACITY_UNAVAILABLE", 503, id); }
    const result = await callPrivate(bytes, type, locale(request, expectedOrigin), id);
    return Response.json(result, { headers: { "Cache-Control": "private, no-store" } });
  } catch (error) {
    if (error instanceof ImageError) {
      const status: Record<string, number> = {
        IMAGE_TOO_LARGE: 413, IMAGE_DIMENSIONS_EXCEEDED: 422, IMAGE_DECODE_FAILED: 422,
        IMAGE_MULTIFRAME_REJECTED: 422, IMAGE_TRUNCATED: 422, UNSUPPORTED_MEDIA_TYPE: 415,
        SESSION_INFERENCE_ACTIVE: 409, RATE_LIMITED: 429, CAPACITY_UNAVAILABLE: 503,
        MODEL_NOT_READY: 503, INFERENCE_DEADLINE_EXCEEDED: 504, INVALID_REQUEST: 400
      };
      return safeError(error.code, status[error.code] ?? 500, id);
    }
    return safeError("INFERENCE_FAILED", 500, id);
  } finally { ticket.release(); }
}
