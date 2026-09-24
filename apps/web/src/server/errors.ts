import { randomBytes } from "node:crypto";
import type { AnprApiErrorV1, ErrorCode } from "../../../../packages/api-contract/generated/typescript/contracts";

export function requestId(): string {
  // 26 Crockford characters; independent of user input.
  const alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
  const bytes = randomBytes(26);
  return Array.from(bytes, (value) => alphabet[value & 31]).join("");
}

export function safeError(code: ErrorCode, status: number, id: string): Response {
  const body: AnprApiErrorV1 = {
    schema_version: "anpr-api-error-v1",
    request_id: id,
    error: { code, message_key: `error.${code.toLowerCase()}`, retryable: [429, 503, 504].includes(status) }
  };
  return Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
}
