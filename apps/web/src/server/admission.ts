import { createHash } from "node:crypto";

type Entry = { hits: number[] };
const WINDOW_MS = 5 * 60_000;
const MAX_KEYS = 4096;
// Covers malformed requests too; the persistent daily limit below covers
// validated requests. Keep enough headroom for a few user-side mistakes.
const SESSION_LIMIT = 12;
const IP_LIMIT = 24;

class Admission {
  private active = false;
  private activeSessions = new Set<string>();
  private recent = new Map<string, Entry>();

  acquire(session: string, ip: string, rateLimitExempt = false): { release: () => void } | "session" | "capacity" | "rate" {
    const sessionKey = `s:${this.digest(session)}`;
    const ipKey = `i:${this.digest(ip)}`;
    if (this.activeSessions.has(sessionKey)) return "session";
    if (this.active) return "capacity";
    const now = Date.now();
    this.prune(now);
    if (!rateLimitExempt) {
      if (this.count(sessionKey, now) >= SESSION_LIMIT || this.count(ipKey, now) >= IP_LIMIT) return "rate";
      this.record(sessionKey, now);
      this.record(ipKey, now);
    }
    this.active = true;
    this.activeSessions.add(sessionKey);
    return {
      release: () => {
        this.activeSessions.delete(sessionKey);
        this.active = false;
      }
    };
  }

  private digest(value: string): string { return createHash("sha256").update(value).digest("hex"); }
  private count(key: string, now: number): number { return this.recent.get(key)?.hits.filter((hit) => now - hit < WINDOW_MS).length ?? 0; }
  private record(key: string, now: number): void {
    const hits = this.recent.get(key)?.hits.filter((hit) => now - hit < WINDOW_MS) ?? [];
    hits.push(now);
    this.recent.delete(key);
    this.recent.set(key, { hits });
  }
  private prune(now: number): void {
    for (const [key, entry] of this.recent) {
      if (entry.hits.every((hit) => now - hit >= WINDOW_MS)) this.recent.delete(key);
    }
    while (this.recent.size > MAX_KEYS) this.recent.delete(this.recent.keys().next().value!);
  }
}

const globalAdmission = globalThis as typeof globalThis & { __anprAdmission?: Admission };
export const admission = globalAdmission.__anprAdmission ??= new Admission();
