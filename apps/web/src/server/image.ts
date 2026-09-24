import sharp from "sharp";
import type { ErrorCode } from "../../../../packages/api-contract/generated/typescript/contracts";

export const MAX_BYTES = 10 * 1024 * 1024;
const MAX_PIXELS = 20_000_000;
const MAX_SIDE = 8192;
export type MediaType = "image/jpeg" | "image/png";

export class ImageError extends Error {
  constructor(readonly code: ErrorCode) { super(code); }
}

export function mediaType(value: string | null): MediaType {
  const type = value?.split(";", 1)[0]?.trim().toLowerCase();
  if (type !== "image/jpeg" && type !== "image/png") throw new ImageError("UNSUPPORTED_MEDIA_TYPE");
  return type;
}

export async function readBounded(stream: ReadableStream<Uint8Array> | null, length: string | null): Promise<Buffer> {
  if (!stream) throw new ImageError("INVALID_REQUEST");
  if (length !== null) {
    if (!/^\d+$/.test(length)) throw new ImageError("INVALID_REQUEST");
    if (Number(length) > MAX_BYTES) throw new ImageError("IMAGE_TOO_LARGE");
    if (Number(length) < 1) throw new ImageError("INVALID_REQUEST");
  }
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  let expired = false;
  const timer = setTimeout(() => {
    expired = true;
    void reader.cancel().catch(() => undefined);
  }, 15_000);
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (expired) throw new ImageError("INFERENCE_DEADLINE_EXCEEDED");
      if (done) break;
      total += value.byteLength;
      if (total > MAX_BYTES) throw new ImageError("IMAGE_TOO_LARGE");
      chunks.push(value);
    }
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    throw error;
  } finally { clearTimeout(timer); reader.releaseLock(); }
  if (!total) throw new ImageError("INVALID_REQUEST");
  return Buffer.concat(chunks, total);
}

function pngBoundary(bytes: Buffer): void {
  let offset = 8;
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset);
    const kind = bytes.toString("ascii", offset + 4, offset + 8);
    offset += 12 + length;
    if (offset > bytes.length) throw new ImageError("IMAGE_TRUNCATED");
    if (kind === "IEND") {
      if (length !== 0) throw new ImageError("IMAGE_DECODE_FAILED");
      if (offset !== bytes.length) throw new ImageError("IMAGE_MULTIFRAME_REJECTED");
      return;
    }
  }
  throw new ImageError("IMAGE_TRUNCATED");
}

function jpegBoundary(bytes: Buffer): void {
  let offset = 2;
  let inScan = false;
  while (offset < bytes.length) {
    if (bytes[offset] !== 0xff) {
      if (!inScan) throw new ImageError("IMAGE_DECODE_FAILED");
      offset++;
      continue;
    }
    while (offset < bytes.length && bytes[offset] === 0xff) offset++;
    if (offset >= bytes.length) throw new ImageError("IMAGE_TRUNCATED");
    const marker = bytes[offset++];
    if (inScan && (marker === 0 || (marker >= 0xd0 && marker <= 0xd7))) continue;
    if (marker === 0xd9) {
      if (offset !== bytes.length) throw new ImageError("IMAGE_MULTIFRAME_REJECTED");
      return;
    }
    if (marker === 0xd8 || marker === 0) throw new ImageError("IMAGE_DECODE_FAILED");
    if (marker === 1) continue;
    if (offset + 2 > bytes.length) throw new ImageError("IMAGE_TRUNCATED");
    const segment = bytes.readUInt16BE(offset);
    if (segment < 2 || offset + segment > bytes.length) throw new ImageError("IMAGE_TRUNCATED");
    offset += segment;
    inScan = marker === 0xda;
  }
  throw new ImageError("IMAGE_TRUNCATED");
}

export async function validateImage(bytes: Buffer, declared: MediaType): Promise<void> {
  const png = bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  const jpeg = bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  if ((declared === "image/png" && !png) || (declared === "image/jpeg" && !jpeg)) {
    throw new ImageError("UNSUPPORTED_MEDIA_TYPE");
  }
  if (png) pngBoundary(bytes); else jpegBoundary(bytes);
  try {
    const decoder = sharp(bytes, { failOn: "error", limitInputPixels: MAX_PIXELS, animated: false });
    const info = await decoder.metadata();
    const expected = declared === "image/png" ? "png" : "jpeg";
    if (info.format !== expected) throw new ImageError("UNSUPPORTED_MEDIA_TYPE");
    if ((info.pages ?? 1) !== 1) throw new ImageError("IMAGE_MULTIFRAME_REJECTED");
    if (!info.width || !info.height || info.width > MAX_SIDE || info.height > MAX_SIDE || info.width * info.height > MAX_PIXELS) {
      throw new ImageError("IMAGE_DIMENSIONS_EXCEEDED");
    }
    // Force full decoding, not only header inspection. The decoded pixels are discarded.
    await decoder.rotate().raw().toBuffer();
  } catch (error) {
    if (error instanceof ImageError) throw error;
    throw new ImageError("IMAGE_DECODE_FAILED");
  }
}
