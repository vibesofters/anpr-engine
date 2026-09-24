import { createHmac } from "node:crypto";
import { DatabaseSync } from "node:sqlite";

const DAILY_LIMIT = 5;

function istanbulDay(now: Date): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Europe/Istanbul", year: "numeric", month: "2-digit", day: "2-digit"
  }).formatToParts(now);
  const value = (type: string) => parts.find((part) => part.type === type)?.value ?? "";
  return `${value("year")}-${value("month")}-${value("day")}`;
}

/** Count validated, admitted processing attempts; never store an IP or image. */
export function consumeDailyQuota(ip: string, now = new Date()): boolean {
  const path = process.env.ANPR_DAILY_QUOTA_PATH;
  const secret = process.env.ANPR_INTERNAL_SERVICE_CREDENTIAL;
  if (!path || !secret || secret.length < 32) throw new Error("daily quota unavailable");
  const day = istanbulDay(now);
  const key = createHmac("sha256", secret).update(`${day}\0${ip}`).digest("hex");
  const db = new DatabaseSync(path, { timeout: 1000 });
  try {
    db.exec("PRAGMA max_page_count = 4096");
    db.exec("CREATE TABLE IF NOT EXISTS quota (day TEXT NOT NULL, key TEXT NOT NULL, used INTEGER NOT NULL CHECK (used BETWEEN 1 AND 5), PRIMARY KEY (day, key)) STRICT");
    db.exec("BEGIN IMMEDIATE");
    try {
      const row = db.prepare("SELECT used FROM quota WHERE day = ? AND key = ?").get(day, key) as { used: number } | undefined;
      if (row && row.used >= DAILY_LIMIT) { db.exec("ROLLBACK"); return false; }
      db.prepare("INSERT INTO quota(day, key, used) VALUES (?, ?, 1) ON CONFLICT(day, key) DO UPDATE SET used = used + 1").run(day, key);
      db.prepare("DELETE FROM quota WHERE day < ?").run(day);
      db.exec("COMMIT");
      return true;
    } catch (error) { db.exec("ROLLBACK"); throw error; }
  } finally { db.close(); }
}
