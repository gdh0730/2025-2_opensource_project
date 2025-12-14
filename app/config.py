# app/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ---- 기본 ----
    PROJECT_NAME: str = "MetroVision AI API"
    API_V1_PREFIX: str = "/api"

    # ---- DB 설정 ----
    # 예) postgresql+psycopg2://metro:metro@db:5432/metrovision
    DATABASE_URL: str = "postgresql+psycopg2://metro:metro@db:5432/metrovision"

    # ---- TOPIS OpenAPI ----
    TOPIS_API_KEY: str
    TOPIS_BASE_URL: str = "http://openapi.seoul.go.kr:8088"

    # ---- 모델/데이터 경로 (컨테이너 내부 경로 기준) ----
    BASE_DIR: Path = Path("/workspace")  # Dockerfile에서 작업 디렉토리
    SPEED_CKPT: Path = BASE_DIR / "models" / "speed_seq2seq_gcn_lstm_scaled.pt"
    VOL_CKPT: Path = BASE_DIR / "models" / "volume_seq2seq_gcn_lstm_scaled.pt"

    # GeoJSON (도로 라인 시각화용, 선택)
    ROADS_GEOJSON: Path = BASE_DIR / "data" / "roads.geojson"

    # ★ pydantic v2 방식 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",        # 환경변수 많을 때 편의용 (선택)
    )


settings = Settings()
