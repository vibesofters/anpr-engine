import { isIP } from "node:net";

/** Exact trusted-client-IP exemptions; never inspect caller-supplied forwarding chains. */
export function isRateLimitExempt(ip: string): boolean {
  if (isIP(ip) === 0) return false;
  const configured = process.env.ANPR_RATE_LIMIT_EXEMPT_IPS ?? "";
  return configured.split(",").some((value) => {
    const candidate = value.trim();
    return isIP(candidate) !== 0 && candidate === ip;
  });
}
