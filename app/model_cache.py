"""
Thread-safe in-memory model cache.

The model is loaded once on first use and held for the lifetime of the
process.  Call `model_cache.reload()` (or hit POST /model/reload) to swap
in a freshly-trained version without restarting the server.
"""

import logging
import threading
from datetime import datetime, timezone

from pipelines.inference import get_production_model

logger = logging.getLogger(__name__)


class _ModelCache:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._loaded_at: str | None = None
        self._version: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self):
        """Return the cached model, loading it on first call."""
        with self._lock:
            if self._model is None:
                self._load_locked()
            return self._model

    def reload(self) -> dict:
        """
        Force a fresh load from the MLflow registry.
        Returns metadata about the newly loaded model.
        """
        with self._lock:
            logger.info("Reloading model from MLflow registry...")
            self._load_locked()
            logger.info("Model reloaded successfully (loaded_at=%s)", self._loaded_at)
            return self.info()

    def info(self) -> dict:
        """Return cache metadata (safe to call without holding the lock)."""
        with self._lock:
            return {
                "loaded": self._model is not None,
                "loaded_at_utc": self._loaded_at,
                "model_version": self._version,
            }

    # ------------------------------------------------------------------
    # Internal helpers  (must be called while holding self._lock)
    # ------------------------------------------------------------------

    def _load_locked(self):
        model = get_production_model()

        # Best-effort: try to extract the MLflow model version from metadata
        version = None
        try:
            version = str(model.metadata.run_id[:8])
        except Exception:
            pass

        self._model = model
        self._loaded_at = datetime.now(timezone.utc).isoformat()
        self._version = version


# Module-level singleton — import this everywhere
model_cache = _ModelCache()
