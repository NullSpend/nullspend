/**
 * Sliding-window loop detection for repeated LLM calls.
 *
 * Mirror of Python implementation. Uses Map<string, Array<{ts, hash}>>.
 * No lock needed (single-threaded Node.js).
 */

export interface LoopCheck {
  isLoop: boolean;
  isWarning: boolean;
  callCount: number;
  detectionType: "per_key" | "aggregate";
}

export interface LoopDetectionOptions {
  maxCalls?: number;      // default 50
  windowSeconds?: number; // default 60
  aggregateMaxKeys?: number; // default 5
}

export class LoopDetector {
  private readonly _maxCalls: number;
  private readonly _window: number; // seconds
  private readonly _aggregateMaxKeys: number;
  private readonly _aggregateMinRepeats = 3;
  private readonly _warningRatio = 0.8;

  /** key → list of { ts (ms monotonic), hash } */
  private readonly _callLog = new Map<string, Array<{ ts: number; hash: string }>>();
  private readonly _warningsFired = new Set<string>();

  constructor(options?: LoopDetectionOptions) {
    const maxCalls = options?.maxCalls ?? 50;
    const windowSeconds = options?.windowSeconds ?? 60;
    const aggregateMaxKeys = options?.aggregateMaxKeys ?? 5;

    if (maxCalls < 0) throw new RangeError(`maxCalls must be >= 0 (got ${maxCalls})`);
    if (windowSeconds <= 0) throw new RangeError(`windowSeconds must be > 0 (got ${windowSeconds})`);
    if (aggregateMaxKeys < 0) throw new RangeError(`aggregateMaxKeys must be >= 0 (got ${aggregateMaxKeys})`);

    this._maxCalls = maxCalls;
    this._window = windowSeconds;
    this._aggregateMaxKeys = aggregateMaxKeys;
  }

  /**
   * Hash full request body to 8 hex chars. Cap at 8KB for large payloads.
   * Uses SHA-256 when available (Web Crypto), falls back to simple hash.
   */
  static contentHashSync(body: string | null | undefined): string {
    if (!body) return "empty";
    const slice = body.slice(0, 8192);
    // Simple but deterministic hash for synchronous path (FNV-1a 32-bit → 8 hex)
    let hash = 0x811c9dc5;
    for (let i = 0; i < slice.length; i++) {
      hash ^= slice.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    // Convert to unsigned 32-bit, then 8 hex chars
    return (hash >>> 0).toString(16).padStart(8, "0");
  }

  /**
   * Async SHA-256 content hash matching the proxy implementation.
   * Use this when Web Crypto is available and you can await.
   */
  static async contentHash(body: string | null | undefined): Promise<string> {
    if (!body) return "empty";
    const slice = body.slice(0, 8192);
    if (typeof globalThis.crypto?.subtle?.digest === "function") {
      const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(slice));
      const arr = new Uint8Array(buf);
      return Array.from(arr.slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");
    }
    // Fallback to sync hash
    return LoopDetector.contentHashSync(body);
  }

  /**
   * Record a call and check for loops.
   */
  check(key: string, contentHash: string): LoopCheck {
    const now = performance.now();
    const cutoffMs = now - this._window * 1000;
    const warningCount = Math.floor(this._maxCalls * this._warningRatio);
    let detectionType: "per_key" | "aggregate" = "per_key";

    // Per-key: prune old entries, append new, and cap to bound memory under
    // burst load with many distinct content hashes (SEC-2). Cap at 10× maxCalls
    // — enough headroom for legitimate detection (counting by hash, see below)
    // while preventing unbounded growth from a high-QPS attacker.
    let entries = this._callLog.get(key) ?? [];
    entries = entries.filter((e) => e.ts > cutoffMs);
    entries.push({ ts: now, hash: contentHash });
    const entryCap = Math.max(this._maxCalls * 10, this._maxCalls);
    if (entries.length > entryCap) {
      entries = entries.slice(entries.length - entryCap);
    }
    this._callLog.set(key, entries);

    // Count matching content hash
    let count = 0;
    for (const e of entries) {
      if (e.hash === contentHash) count++;
    }

    // Per-key loop
    let isLoop = count >= this._maxCalls;

    // Aggregate: count distinct keys with 3+ same-content repeats
    // Prune all keys lazily to avoid counting stale entries
    if (!isLoop) {
      let qualifyingKeys = 0;
      const staleKeys: string[] = [];
      for (const [k, ents] of this._callLog) {
        let filtered = ents;
        if (k !== key) {
          filtered = ents.filter((e) => e.ts > cutoffMs);
          this._callLog.set(k, filtered);
        }
        if (filtered.length === 0) {
          staleKeys.push(k);
          continue;
        }
        const hashCounts = new Map<string, number>();
        for (const e of filtered) {
          hashCounts.set(e.hash, (hashCounts.get(e.hash) ?? 0) + 1);
        }
        for (const c of hashCounts.values()) {
          if (c >= this._aggregateMinRepeats) {
            qualifyingKeys++;
            break;
          }
        }
      }
      for (const k of staleKeys) {
        this._callLog.delete(k);
      }
      if (qualifyingKeys >= this._aggregateMaxKeys) {
        isLoop = true;
        count = qualifyingKeys;
        detectionType = "aggregate";
      }
    }

    // Warning (fires once per window per composite key)
    const composite = `${key}:${contentHash}`;
    let isWarning = false;
    if (count >= warningCount && !this._warningsFired.has(composite)) {
      isWarning = true;
      this._warningsFired.add(composite);
    }
    if (count < warningCount) {
      this._warningsFired.delete(composite);
    }

    return { isLoop, isWarning, callCount: count, detectionType };
  }

  /** Reset call history (all keys or a specific key). */
  reset(key?: string): void {
    if (key === undefined) {
      this._callLog.clear();
      this._warningsFired.clear();
    } else {
      this._callLog.delete(key);
      const prefix = `${key}:`;
      for (const k of this._warningsFired) {
        if (k.startsWith(prefix)) this._warningsFired.delete(k);
      }
    }
  }

  get maxCalls(): number { return this._maxCalls; }
  get windowSeconds(): number { return this._window; }
}
