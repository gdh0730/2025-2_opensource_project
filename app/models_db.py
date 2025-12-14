# app/models_db.py
from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    UniqueConstraint
)
from sqlalchemy.sql import func
from .db import Base

class SpeedHist(Base):
    __tablename__ = "speed_hist"
    id = Column(Integer, primary_key=True, index=True)
    link_id = Column(String(20), index=True, nullable=False)
    ts = Column(DateTime(timezone=True), index=True, nullable=False)
    speed = Column(Float, nullable=False)        # PRCS_SPD
    travel_time = Column(Float, nullable=True)   # PRCS_TRV_TIME
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("link_id", "ts", name="uq_speed_link_ts"),
    )


class VolumeHist(Base):
    __tablename__ = "volume_hist"
    id = Column(Integer, primary_key=True, index=True)
    spot_num = Column(String(10), index=True, nullable=False)     # A-01
    direction = Column(String(10), nullable=False)                # '유입' / '유출'
    ts = Column(DateTime(timezone=True), index=True, nullable=False)
    vol = Column(Integer, nullable=False)                         # 차로 합산 교통량
    lane_cnt = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("spot_num", "direction", "ts", name="uq_vol_spot_dir_ts"),
    )
