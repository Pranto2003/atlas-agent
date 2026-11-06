import random
import time


def retry(fn, *, attempts: int = 3, base: float = 0.6, on_retry=None, no_retry=(PermissionError,)):
    """Exponential backoff with jitter. Re-raises the last error after `attempts`."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except no_retry:
            raise
        except Exception as e:  # noqa: BLE001 - boundary: any tool/LLM failure is retryable
            if i == attempts:
                raise
            delay = base * (2 ** (i - 1)) + random.uniform(0, base / 2)
            if on_retry:
                on_retry(i, e, delay)
            time.sleep(delay)
