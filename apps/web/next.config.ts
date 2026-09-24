import type { NextConfig } from "next";
import { resolve } from "node:path";

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: resolve(process.cwd(), "../.."),
  reactStrictMode: true,
  agentRules: false,
  turbopack: { root: resolve(process.cwd(), "../..") },
  poweredByHeader: false,
  serverExternalPackages: ["sharp"],
  async headers() {
    return [{
      source: "/:path*",
      headers: [
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "Referrer-Policy", value: "no-referrer" },
        { key: "X-Frame-Options", value: "DENY" },
        { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        { key: "Content-Security-Policy", value: "default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'" }
      ]
    }];
  }
};

export default nextConfig;
