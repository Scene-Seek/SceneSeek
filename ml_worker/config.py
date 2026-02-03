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
    FRAME_SKIP: int = 15  # Extract frames every N frames
    YOLO_BATCH_SIZE: int = 16  # Batch size for YOLO inference
    FLORENCE_BATCH_SIZE: int = 4  # Batch size for Florence caption generation
    MOTION_THRESHOLD: int = 1000  # Threshold for motion detection
    YOLO_CONF: float = 0.25  # YOLO confidence threshold
    MODEL_PATH: str = "yolov8n.pt"  # YOLO model path
    FLORENCE_MODEL_ID: str = "microsoft/Florence-2-base-ft"  # Florence model ID
    EMBEDDER_MODEL: str = "all-MiniLM-L6-v2"  # Sentence transformer model

    @property
    def DATABASE_URL(self):
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(env_file=ENV_FILE)


settings = Settings()

if __name__ == "__main__":
    print(ENV_FILE)
