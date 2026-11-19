/**
 * Incremental SHA-256 for hashing large files in chunks.
 *
 * Web Crypto (`crypto.subtle.digest`) only hashes one buffer at once, so a multi-gigabyte video
 * would have to be read into memory whole. This hasher keeps a 64-byte block buffer and accepts
 * the file slice by slice, so memory stays flat whatever the file size.
 */

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

export class Sha256 {
  private readonly h = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  private readonly w = new Uint32Array(64);
  private readonly block = new Uint8Array(64);
  private blockLen = 0;
  private totalBytes = 0;
  private finished = false;

  update(data: Uint8Array): this {
    if (this.finished) throw new Error('Sha256: update() after digest()');
    this.totalBytes += data.length;
    let offset = 0;
    if (this.blockLen > 0) {
      const take = Math.min(64 - this.blockLen, data.length);
      this.block.set(data.subarray(0, take), this.blockLen);
      this.blockLen += take;
      offset = take;
      if (this.blockLen < 64) return this;
      this.compress(this.block, 0);
      this.blockLen = 0;
    }
    while (offset + 64 <= data.length) {
      this.compress(data, offset);
      offset += 64;
    }
    if (offset < data.length) {
      this.block.set(data.subarray(offset), 0);
      this.blockLen = data.length - offset;
    }
    return this;
  }

  /** Finish and return the lowercase hex digest. */
  digestHex(): string {
    if (this.finished) throw new Error('Sha256: digest() called twice');
    this.finished = true;
    const bitLenHi = Math.floor(this.totalBytes / 0x20000000);
    const bitLenLo = (this.totalBytes * 8) >>> 0;
    this.block[this.blockLen++] = 0x80;
    if (this.blockLen > 56) {
      this.block.fill(0, this.blockLen);
      this.compress(this.block, 0);
      this.blockLen = 0;
    }
    this.block.fill(0, this.blockLen, 56);
    const view = new DataView(this.block.buffer, this.block.byteOffset, 64);
    view.setUint32(56, bitLenHi);
    view.setUint32(60, bitLenLo);
    this.compress(this.block, 0);
    let hex = '';
    for (let i = 0; i < 8; i++) hex += this.h[i].toString(16).padStart(8, '0');
    return hex;
  }

  private compress(bytes: Uint8Array, offset: number): void {
    const w = this.w;
    for (let i = 0; i < 16; i++) {
      const j = offset + i * 4;
      w[i] = (bytes[j] << 24) | (bytes[j + 1] << 16) | (bytes[j + 2] << 8) | bytes[j + 3];
    }
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15];
      const y = w[i - 2];
      const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
      const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let a = this.h[0];
    let b = this.h[1];
    let c = this.h[2];
    let d = this.h[3];
    let e = this.h[4];
    let f = this.h[5];
    let g = this.h[6];
    let h = this.h[7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + K[i] + w[i]) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) | 0;
      h = g;
      g = f;
      f = e;
      e = (d + t1) | 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) | 0;
    }
    this.h[0] = (this.h[0] + a) | 0;
    this.h[1] = (this.h[1] + b) | 0;
    this.h[2] = (this.h[2] + c) | 0;
    this.h[3] = (this.h[3] + d) | 0;
    this.h[4] = (this.h[4] + e) | 0;
    this.h[5] = (this.h[5] + f) | 0;
    this.h[6] = (this.h[6] + g) | 0;
    this.h[7] = (this.h[7] + h) | 0;
  }
}

/** Files up to this size are hashed in one native Web Crypto call; larger ones are streamed. */
const ONE_SHOT_LIMIT = 64 * 1024 * 1024;
const CHUNK_BYTES = 8 * 1024 * 1024;

const toHex = (buf: ArrayBuffer) =>
  Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');

/**
 * SHA-256 of a file or blob as lowercase hex, reading it slice by slice so memory use stays at
 * one chunk. Returns null if the file cannot be read.
 */
export async function hashBlobSha256(
  blob: Blob,
  options: { signal?: AbortSignal; onProgress?: (fraction: number) => void } = {}
): Promise<string | null> {
  const { signal, onProgress } = options;
  try {
    if (blob.size <= ONE_SHOT_LIMIT && globalThis.crypto?.subtle) {
      const digest = toHex(await globalThis.crypto.subtle.digest('SHA-256', await blob.arrayBuffer()));
      onProgress?.(1);
      return signal?.aborted ? null : digest;
    }
    const hasher = new Sha256();
    for (let start = 0; start < blob.size; start += CHUNK_BYTES) {
      if (signal?.aborted) return null;
      const chunk = await blob.slice(start, Math.min(start + CHUNK_BYTES, blob.size)).arrayBuffer();
      hasher.update(new Uint8Array(chunk));
      onProgress?.(Math.min(1, (start + chunk.byteLength) / blob.size));
    }
    return hasher.digestHex();
  } catch {
    return null;
  }
}
