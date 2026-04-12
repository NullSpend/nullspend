/**
 * Constant-time string comparison using Web Crypto API.
 * Used for auth token and webhook signature verification.
 */
export function timingSafeStringEqual(a: string, b: string): boolean {
  const encoder = new TextEncoder();
  const bufA = encoder.encode(a);
  const bufB = encoder.encode(b);
  const lengthsMatch = bufA.byteLength === bufB.byteLength;
  return lengthsMatch
    ? crypto.subtle.timingSafeEqual(bufA, bufB)
    : !crypto.subtle.timingSafeEqual(bufA, bufA);
}
