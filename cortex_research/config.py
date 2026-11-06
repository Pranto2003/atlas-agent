import os
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Real environment variables win."""
    f = Path(__file__).resolve().parent.parent / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    anthropic_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("CORTEX_ANTHROPIC_MODEL", "claude-sonnet-5-5")
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("CORTEX_OPENAI_MODEL", "gpt-4o-mini")
    max_concurrent_runs: int = int(os.getenv("CORTEX_MAX_RUNS", "3"))
    http_timeout: float = float(os.getenv("CORTEX_HTTP_TIMEOUT", "12"))
    runs_dir: str = os.getenv("CORTEX_RUNS_DIR", "runs")
    user_agent: str = os.getenv("CORTEX_USER_AGENT", "CortexResearch/1.0 (educational research assistant; set CORTEX_USER_AGENT with your contact)")


def settings() -> Settings:
    return Settings()
