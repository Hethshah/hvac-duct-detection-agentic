from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> str:
    current = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = current / ".env"
        if candidate.exists():
            return str(candidate)
        current = current.parent
    return ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")

    anthropic_api_key: str = ""
    vision_model: str = "claude-opus-4-7"
    orchestrator_model: str = "claude-sonnet-4-6"
    ingestion_model: str = "claude-haiku-4-5-20251001"
    measurement_model: str = "claude-sonnet-4-6"
    review_model: str = "claude-sonnet-4-6"
    confidence_threshold: float = 0.85
    max_retries: int = 3
    dpi: int = 300
    output_dir: str = "outputs"


settings = Settings()
