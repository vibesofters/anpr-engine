import { randomBytes } from "node:crypto";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

const SESSION = "anpr_session";
const SESSION_PATTERN = /^[a-f0-9]{64}$/;

export function proxy(request: NextRequest) {
  const nonce = randomBytes(16).toString("base64");
  const development = process.env.NODE_ENV === "development";
  const csp = `default-src 'self'; script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${development ? " 'unsafe-eval'" : ""}; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'${development ? " ws:" : ""}; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'`;
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("Content-Security-Policy", csp);
  const response = NextResponse.next({ request: { headers: requestHeaders } });
  const existing = request.cookies.get(SESSION)?.value;
  if (!existing || !SESSION_PATTERN.test(existing)) {
    response.cookies.set(SESSION, randomBytes(32).toString("hex"), {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      path: "/",
      maxAge: 60 * 60
    });
  }
  response.headers.set("Cache-Control", "private, no-store");
  response.headers.set("Content-Security-Policy", csp);
  return response;
}

export const config = { matcher: ["/", "/information/:path*", "/en/:path*", "/tr/:path*"] };
