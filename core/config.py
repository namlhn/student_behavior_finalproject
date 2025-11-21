from pydantic_settings import BaseSettings
from pathlib import Path
from typing import ClassVar, Dict, Any


class Settings(BaseSettings):
    DATABASE_HOST: str
    DATABASE_PORT: int
    DATABASE_NAME: str
    DATABASE_USERNAME: str
    DATABASE_PASSWORD: str
    DATABASE_URL: str = ""

    def model_post_init(self, __context):
        if not self.DATABASE_URL and self.DATABASE_HOST:
            self.DATABASE_URL = f"mysql+pymysql://{self.DATABASE_USERNAME}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"

    EMBEDDING_DIM: int = 512
    # Resolve absolute paths relative to the project root
    _BASE_DIR: Path = Path(__file__).resolve().parents[1]
    FAISS_INDEX_FILE: str = str(_BASE_DIR / "assets" / "faiss_index.bin")
    METADATA_FILE: str = str(_BASE_DIR / "assets" / "metadata.json")
    UPLOAD_DIR: str = str(_BASE_DIR / "assets" / "uploads")
    FAISS_THRESHOLD_COSINE: float = 0.6
    LOGGER_CONFIG: ClassVar[Dict[str, Any]] = {
        'log_dir': 'log'
    }

    # Logging/detail toggles used by fastapi_util
    log_request_body: bool = True
    max_response_length: int = 500
    max_request_body_length: int = 500
    log_response: bool = True

    # SQLAlchemy engine tuning
    SQLALCHEMY_ECHO: bool = False
    DB_POOL_PRE_PING: bool = True
    DB_POOL_RECYCLE: int = 1800
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # CORS and Request ID
    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    REQUEST_ID_HEADER: str = "X-Request-ID"

    class Config:
        env_file = ".env"


settings = Settings()
