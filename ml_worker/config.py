"""config"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"


class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASS: str
    DB_NAME: str

    RABBITMQ_URL: str
    RABBITMQ_USER: str | None = None
    RABBITMQ_PASS: str | None = None

    MINIO_ENDPOINT: str
    MINIO_PUBLIC_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str

    # ML Worker Configuration
    FRAME_SKIP: int = 15  # Initial frame skip (adaptive skip overrides at runtime)
    BATCH_SIZE: int = 32
    MOTION_THRESHOLD: int = 100
    YOLO_CONF: float = 0.25
    MODEL_PATH: str = "yolo26s.pt"
    SIGLIP_MODEL_ID: str = "google/siglip2-base-patch16-224"
    RAW_SEARCH_LIMIT: int = 5000
    SQL_MIN_SIMILARITY: float = 0.05

    # Backward-compatible aliases
    YOLO_MODEL: str = "yolo26s.pt"
    CLIP_MODEL: str = "google/siglip2-base-patch16-224"

    @property
    def DATABASE_URL(self):
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(env_file=ENV_FILE)


settings = Settings()

if __name__ == "__main__":
    print(ENV_FILE)
