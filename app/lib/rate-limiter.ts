/**
 * Simple sliding-window rate limiter (in-memory, per-process).
 */
export function createRateLimiter(maxRequests: number, windowMs: number) {
  const timestamps: number[] = [];

  return function isRateLimited(): boolean {
    const now = Date.now();
    while (timestamps.length > 0 && timestamps[0] <= now - windowMs) {
      timestamps.shift();
    }
    if (timestamps.length >= maxRequests) {
      return true;
    }
    timestamps.push(now);
    return false;
  };
}
