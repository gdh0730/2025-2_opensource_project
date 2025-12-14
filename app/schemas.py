# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

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
    direction: str
    road_name: str
    max_vol: float
    predictions: List[HorizonPoint]
    load_index: List[float]        # 0~1


class CombinedCongestion(BaseModel):
    road_name: str
    link_id: str
    spot_num: Optional[str]
    direction: Optional[str]
    horizon: int
    combined_index: List[float]


class SegmentForecastResponse(BaseModel):
    speed: SpeedForecast
    volume: Optional[VolumeForecast] = None
    combined: CombinedCongestion


# ==============================
# ✅ Road 단위 예측 응답 추가
# ==============================
class RoadIndexPoint(BaseModel):
    offset_h: int
    timestamp: datetime
    speed_index: float                 # road-level speed congestion (0~1)
    volume_index: Optional[float] = None  # road-level volume load (0~1)
    combined_index: float              # fused (0~1)


class RoadForecastResponse(BaseModel):
    road_name: str
    anchor_ts: datetime                # road 기준 공통 anchor(마지막 관측시간)
    horizon: int

    # road 구성요소별 예측
    speed_links: List[SpeedForecast]
    volume_spots: Optional[List[VolumeForecast]] = None

    # road-level 합성 결과 (시간대별)
    road_index: List[RoadIndexPoint]

    # 일부 link/spot 히스토리 부족 등으로 스킵된 목록
    missing_speed_links: List[str] = []
    missing_volume_spots: List[str] = []


# ----- 요청 스키마 -----
class SegmentPredictRequest(BaseModel):
    link_id: str = Field(..., description="속도 모델에서 사용한 링크아이디")
    spot_num: Optional[str] = Field(None, description="교통량 지점번호 (A-01 등)")
    direction: Optional[str] = Field(None, description="유입/유출")
    horizon: int = Field(6, ge=1, le=24)


class RoadPredictRequest(BaseModel):
    road_name: str = Field(..., description="도로명(예: 올림픽대로)")
    horizon: int = Field(6, ge=1, le=24)

    # volume까지 합성할지 (도로에 volume 메타가 없으면 자동으로 speed만)
    include_volume: bool = True

    # 특정 방향만 보고 싶을 때(기본: 해당 도로에 존재하는 방향 모두)
    directions: Optional[List[str]] = None  # 예: ["유입", "유출"]


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
    hours: int = Field(72, ge=48, le=24*60, description="과거 몇 시간치를 채울지 (최소 48 권장, 기본 72)")
    end_ts: Optional[datetime] = Field(None, description="기준 종료 시각(없으면 현재 KST 시간을 정시로 내림)")
    base_speed: Optional[float] = Field(None, description="더미 속도 베이스(없으면 free_flow_speed 기반 자동 산정)")
    noise_std: float = Field(2.0, ge=0.0, le=30.0, description="랜덤 노이즈 표준편차")

class BackfillVolumeRequest(BaseModel):
    spot_num: str
    hours: int = Field(72, ge=48, le=24*14, description="과거 몇 시간치를 TOPIS로 역수집할지(최대 14일 권장)")
    end_ts: Optional[datetime] = Field(None, description="기준 종료 시각(없으면 현재 KST 시간을 정시로 내림)")
    io_type: Optional[int] = Field(None, description="1: 유입, 2: 유출, None이면 둘 다 수집")

class SeedRoadSpeedDummyRequest(BaseModel):
    road_name: str
    hours: int = Field(72, ge=48, le=24*60)
    end_ts: Optional[datetime] = None
    noise_std: float = Field(2.0, ge=0.0, le=30.0)

class BackfillRoadVolumeRequest(BaseModel):
    road_name: str
    hours: int = Field(72, ge=48, le=24*14)
    end_ts: Optional[datetime] = None
    io_type: Optional[int] = Field(None, description="1:유입, 2:유출, None:둘다")
    directions: Optional[List[str]] = None  # ["유입","유출"] 필터
