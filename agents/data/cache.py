"""
Cache — unified interface over Redis with an in-memory fallback.

Why a cache? Data providers (yfinance, NSEPython) are either rate-limited,
slow, or both. Without caching, a single agent run would hammer the same
endpoints hundreds of times (each agent asks about overlapping tickers).
With caching, each ticker is fetched once per freshness window.

Why Redis? It's:
  - External to the process, so multiple agents (or test runs) share it
  - Has built-in TTL support (key auto-expires after N seconds)
  - Survives across process restarts

Why the in-memory fallback? So the project runs on day 1 without forcing
the user to install Redis. The fallback mimics Redis's GET/SETEX/DEL
semantics using a dict. Production should always use Redis.

Serialization: we JSON-encode every value. This keeps the cache language-
agnostic (anything could inspect it) and avoids pickle's security risks.
Callers that want to cache pandas DataFrames should convert via
`.to_dict(orient="records")` before storing.
"""

import json
import time


class Cache:
    """
    Unified cache interface. Tries Redis first; if Redis isn't reachable,
    silently falls back to an in-process dict. Callers use the same API
    either way, so dev machines without Redis don't need special handling.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        namespace: str = "herd-logic",
    ):
        self.namespace = namespace
        self._client = None
        self._memory: dict[str, dict] = {}

        try:
            import redis

            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            self._client = client
            self.backend = "redis"
        except Exception as e:
            # Redis not installed, not running, or unreachable — fall back.
            # We still run; we just lose cross-process persistence.
            print(f"[cache] Redis unavailable ({type(e).__name__}), using in-memory fallback.")
            self.backend = "memory"

    def _nskey(self, key: str) -> str:
        """Prefix every key with the namespace to avoid collisions across projects."""
        return f"{self.namespace}:{key}"

    def get(self, key: str):
        """Return the cached value, or None on miss / expired / error."""
        nskey = self._nskey(key)

        if self._client is not None:
            try:
                raw = self._client.get(nskey)
                return json.loads(raw) if raw is not None else None
            except Exception:
                # Cache failures must NEVER break the calling agent. Treat as miss.
                return None

        entry = self._memory.get(nskey)
        if entry is None:
            return None
        if entry["expires_at"] is not None and time.time() >= entry["expires_at"]:
            del self._memory[nskey]
            return None
        return entry["value"]

    def set(self, key: str, value, ttl: int | None = None) -> None:
        """
        Store a value with an optional TTL (seconds). If ttl is None, the
        entry lives until explicitly deleted — use sparingly.
        """
        nskey = self._nskey(key)

        if self._client is not None:
            try:
                raw = json.dumps(value, default=str)
                if ttl is not None:
                    self._client.setex(nskey, ttl, raw)
                else:
                    self._client.set(nskey, raw)
            except Exception:
                # Swallow — cache writes are best-effort.
                pass
            return

        self._memory[nskey] = {
            "value": value,
            "expires_at": time.time() + ttl if ttl else None,
        }

    def delete(self, key: str) -> None:
        nskey = self._nskey(key)
        if self._client is not None:
            try:
                self._client.delete(nskey)
            except Exception:
                pass
            return
        self._memory.pop(nskey, None)

    def clear_namespace(self) -> int:
        """Delete every key under our namespace. Returns count deleted."""
        pattern = f"{self.namespace}:*"
        if self._client is not None:
            try:
                keys = list(self._client.scan_iter(match=pattern))
                if keys:
                    return self._client.delete(*keys)
                return 0
            except Exception:
                return 0
        prefix = f"{self.namespace}:"
        to_delete = [k for k in self._memory if k.startswith(prefix)]
        for k in to_delete:
            del self._memory[k]
        return len(to_delete)

    def health(self) -> dict:
        """Diagnostic — useful from tests and the dashboard."""
        return {
            "backend": self.backend,
            "namespace": self.namespace,
            "memory_keys": len(self._memory) if self.backend == "memory" else None,
        }
