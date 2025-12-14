# app/schemas.py
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal
from datetime import datetime

InOutDirection = Literal["유입", "유출"]

# ----- 공통 응답 -----
class HorizonPoint(BaseModel):
    offset_h: int = Field(..., description="현재 시점으로부터 몇 시간 뒤인지 (1~H)")
    timestamp: datetime
    value: float


class SpeedForecast(BaseModel):
    link_id: str
    road_name: str
    free_flow_speed: float
    predictions: List[HorizonPoint]
    congestion_index: List[float]  # 0~1


class VolumeForecast(BaseModel):
    spot_num: str
    io_direction: InOutDirection     # ✅ direction -> io_direction
    road_name: str
    max_vol: float
    predictions: List[HorizonPoint]
    load_index: List[float]          # 0~1


class CombinedCongestion(BaseModel):
    road_name: str
    link_id: str
    spot_num: Optional[str]
    io_direction: Optional[InOutDirection]   # ✅
    horizon: int
    combined_index: List[float]


class SegmentForecastResponse(BaseModel):
    speed: SpeedForecast
    volume: Optional[VolumeForecast] = None
    combined: CombinedCongestion


# ==============================
# Road 단위 예측 응답
# ==============================
class RoadIndexPoint(BaseModel):
    offset_h: int
    timestamp: datetime
    speed_index: float
    volume_index: Optional[float] = None
    combined_index: float


class RoadForecastResponse(BaseModel):
    road_name: str
    anchor_ts: datetime
    horizon: int
    speed_links: List[SpeedForecast]
    volume_spots: Optional[List[VolumeForecast]] = None
    road_index: List[RoadIndexPoint]
    missing_speed_links: List[str] = []
    missing_volume_spots: List[str] = []


# ----- 요청 스키마 -----
class SegmentPredictRequest(BaseModel):
    link_id: str = Field(..., description="속도 모델에서 사용한 링크아이디")
    spot_num: Optional[str] = Field(None, description="교통량 지점번호 (A-01 등)")

    # ✅ direction -> io_direction
    io_direction: Optional[InOutDirection] = Field(None, description="교통량 유입/유출")

    horizon: int = Field(6, ge=1, le=24)

    # ✅ spot_num이 있으면 io_direction 필수 강제
    @model_validator(mode="after")
    def _validate_volume_selector(self):
        if self.spot_num and not self.io_direction:
            raise ValueError("spot_num을 주면 io_direction('유입'/'유출')이 반드시 필요합니다.")
        return self


class RoadPredictRequest(BaseModel):
    road_name: str = Field(..., description="도로명(예: 올림픽대로)")
    horizon: int = Field(6, ge=1, le=24)

    include_volume: bool = True

    # ✅ directions -> io_directions
    io_directions: Optional[List[InOutDirection]] = None  # 예: ["유입", "유출"]


# ---- 관리용(OpenAPI fetch) ----
class FetchSpeedRequest(BaseModel):
    link_id: str


class FetchVolumeRequest(BaseModel):
    spot_num: str
    ymd: Optional[str] = Field(None, description="YYYYMMDD, 없으면 오늘")
    hh: Optional[str] = Field(None, description="HH, 없으면 현재 시각 기준 시간")
    io_type: Optional[int] = Field(None, description="1: 유입, 2: 유출, None이면 유입+유출 둘 다 수집")


class SeedSpeedDummyRequest(BaseModel):
    link_id: str
    hours: int = Field(72, ge=48, le=24 * 60, description="과거 몇 시간치를 채울지 (최소 48 권장, 기본 72)")
    end_ts: Optional[datetime] = Field(None, description="기준 종료 시각(없으면 현재 KST 시간을 정시로 내림)")
    base_speed: Optional[float] = Field(None, description="더미 속도 베이스(없으면 free_flow_speed 기반 자동 산정)")
    noise_std: float = Field(2.0, ge=0.0, le=30.0, description="랜덤 노이즈 표준편차")


class BackfillVolumeRequest(BaseModel):
    spot_num: str
    hours: int = Field(72, ge=48, le=24 * 14, description="과거 몇 시간치를 TOPIS로 역수집할지(최대 14일 권장)")
    end_ts: Optional[datetime] = Field(None, description="기준 종료 시각(없으면 현재 KST 시간을 정시로 내림)")
    io_type: Optional[int] = Field(None, description="1: 유입, 2: 유출, None이면 둘 다 수집")


class SeedRoadSpeedDummyRequest(BaseModel):
    road_name: str
    hours: int = Field(72, ge=48, le=24 * 60)
    end_ts: Optional[datetime] = None
    noise_std: float = Field(2.0, ge=0.0, le=30.0)


class BackfillRoadVolumeRequest(BaseModel):
    road_name: str
    hours: int = Field(72, ge=48, le=24 * 14)
    end_ts: Optional[datetime] = None

    # ✅ io_type 제거: road 단위는 io_directions만 사용
    io_directions: Optional[List[InOutDirection]] = None
