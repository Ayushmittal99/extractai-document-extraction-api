"""Configuration loaded from environment variables (.env supported)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _num(name: str, default, cast):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    google_api_key: str = ""
    model: str = "gemini-3.5-flash"
    max_input_chars: int = 20_000
    request_timeout_seconds: float = 60.0
    max_retries: int = 3  # retries AFTER the first attempt -> at most 4 calls
    retry_base_delay_seconds: float = 1.0  # waits 1s, 2s, 4s ...


def get_settings() -> Settings:
    d = Settings()
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if key.lower() in {"your_gemini_api_key_here", "changeme"}:
        key = ""  # the .env.example placeholder counts as "not set"
    return Settings(
        google_api_key=key,
        model=os.getenv("GEMINI_MODEL", d.model),
        max_input_chars=_num("MAX_INPUT_CHARS", d.max_input_chars, int),
        request_timeout_seconds=_num("REQUEST_TIMEOUT_SECONDS", d.request_timeout_seconds, float),
        max_retries=_num("MAX_RETRIES", d.max_retries, int),
        retry_base_delay_seconds=_num("RETRY_BASE_DELAY_SECONDS", d.retry_base_delay_seconds, float),
    )
