# app/services.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import httpx
import torch
from zoneinfo import ZoneInfo
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
import math
import asyncio
import random
import unicodedata as ud

from .config import settings
from .models_db import SpeedHist, VolumeHist
from .schemas import (
    SegmentPredictRequest, SegmentForecastResponse,
    RoadPredictRequest, RoadForecastResponse, RoadIndexPoint,
    SpeedForecast, VolumeForecast, HorizonPoint, CombinedCongestion,
)
from .ml_models import Seq2SeqGCNLSTM

KST = ZoneInfo("Asia/Seoul")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
number_pattern = r"([0-9]+(?:\.[0-9]+)?)"

# road 합성 가중치(필요하면 설정으로 뺄 수 있음)
ROAD_FUSION_ALPHA = 0.5  # 0.5면 speed/volume 동등


# ------------------------------------------------
# 0) torch.load 호환 래퍼
# ------------------------------------------------
def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

# ----------------------------
# io_type <-> io_direction 매핑 (교통량 전용)
# ----------------------------
IO_TYPE_TO_DIR = {1: "유입", 2: "유출"}
DIR_TO_IO_TYPE = {"유입": 1, "유출": 2}

def _ymd_hh_from_ts(ts_kst: datetime) -> Tuple[str, str]:
    return ts_kst.strftime("%Y%m%d"), ts_kst.strftime("%H")

def _kst_hour_floor(dt: Optional[datetime]) -> datetime:
    """
    dt가 None이면 now(KST) 정시로 내림.
    dt가 tz-aware면 KST로 변환 후 정시 내림.
    dt가 naive면 KST로 가정 후 정시 내림.
    """
    if dt is None:
        now = datetime.now(KST)
        return now.replace(minute=0, second=0, microsecond=0)

    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize(KST)
    else:
        ts = ts.tz_convert(KST)
    ts = ts.floor("h")
    return ts.to_pydatetime()

def _norm(s: str) -> str:
    return ud.normalize("NFC", str(s)).strip().replace(" ", "")


# ------------------------------------------------
# 1) ckpt에서 정적 메타 로딩 (lazy)
# ------------------------------------------------
class StaticMeta:
    def __init__(self):
        speed_ckpt = safe_torch_load(settings.SPEED_CKPT, map_location="cpu")
        vol_ckpt = safe_torch_load(settings.VOL_CKPT, map_location="cpu")

        self.speed_meta: Dict[str, dict] = speed_ckpt.get("speed_meta", {})
        self.vol_meta: Dict[str, dict] = vol_ckpt.get("vol_meta", {})

        if not self.speed_meta:
            raise RuntimeError("SPEED_CKPT 안에 speed_meta 가 없습니다. (학습 저장 로직 확인)")
        if self.vol_meta is None:
            self.vol_meta = {}

    def get_speed_meta(self, link_id: str) -> dict:
        if link_id not in self.speed_meta:
            raise KeyError(f"speed_meta에 없는 link_id: {link_id}")
        return self.speed_meta[link_id]

    def get_vol_meta(self, spot_num: str, io_direction: str) -> dict:
        sid = f"{spot_num}_{io_direction}"
        if sid not in self.vol_meta:
            raise KeyError(f"vol_meta에 없는 spot_dir_id: {sid}")
        return self.vol_meta[sid]

    def find_links_by_road(self, road_name: str) -> List[str]:
        rn = _norm(road_name)
        return [lid for lid, m in self.speed_meta.items() if _norm(m.get("road_name", "")) == rn]

    def find_spot_dirs_by_road(self, road_name: str, io_directions=None) -> List[str]:
        rn = _norm(road_name)
        out = []
        for sid, m in self.vol_meta.items():
            if _norm(m.get("road_name", "")) != rn:
                continue
            if "_" not in sid:
                continue
            spot_num, io_dir = sid.split("_", 1)
            if io_directions and io_dir not in io_directions:
                continue
            out.append(sid)
        return out

@lru_cache(maxsize=1)
def get_static_meta() -> StaticMeta:
    return StaticMeta()


# ------------------------------------------------
# 2) 시계열을 "시간당 1개"로 정규화 + (선택) anchor_end로 자르기
# ------------------------------------------------
def normalize_hourly_series(
    df: pd.DataFrame,
    value_col: str,
    min_points: int,
    anchor_end_ts_kst: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    df: columns ['ts', value_col]
    - ts를 KST 기준 tz-aware로 변환
    - 1H resample로 시간당 1개로 정규화
    - 결측은 interpolate + ffill/bfill
    - anchor_end_ts_kst가 있으면 그 시각(포함)까지만 사용(road-level 시간축 정렬용)
    """
    if df.empty:
        raise ValueError("히스토리 데이터가 없습니다.")

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts")

    # tz 처리: tz-aware면 KST로 convert, naive면 KST로 localize(로컬/SQLite fallback 대비)
    ts = out["ts"]
    if getattr(ts.dt, "tz", None) is None:
        out["ts_kst"] = ts.dt.tz_localize(KST)
    else:
        out["ts_kst"] = ts.dt.tz_convert(KST)

    s = out.set_index("ts_kst")[value_col].astype(float)

    # 시간당 1개
    s = s.resample("1h").mean()

    # 결측 보정
    s = s.interpolate(method="time").ffill().bfill()

    norm = s.reset_index().rename(columns={value_col: value_col})

    if anchor_end_ts_kst is not None:
        # anchor는 tz-aware KST여야 함
        anchor = pd.Timestamp(anchor_end_ts_kst).tz_convert(KST) if pd.Timestamp(anchor_end_ts_kst).tzinfo else pd.Timestamp(anchor_end_ts_kst).tz_localize(KST)
        norm = norm[norm["ts_kst"] <= anchor].copy()

    if len(norm) < min_points:
        raise ValueError(f"정규화 후 데이터 포인트가 부족합니다. 필요:{min_points}, 현재:{len(norm)}")
    return norm


def get_series_max_ts_kst(df: pd.DataFrame, value_col: str) -> datetime:
    tmp = normalize_hourly_series(df, value_col=value_col, min_points=2)
    return tmp["ts_kst"].max().to_pydatetime()


# ------------------------------------------------
# 3) 모델 래퍼 (lazy) + anchor_end 기반 예측 지원
# ------------------------------------------------
class SpeedModelWrapper:
    def __init__(self):
        ckpt = safe_torch_load(settings.SPEED_CKPT, map_location="cpu")
        cfg = ckpt["config"]

        raw = ckpt["link_id_to_idx"]
        self.link_id_to_idx = {str(k): int(v) for k, v in raw.items()}

        self.scaler_dynamic = StandardScaler()
        self.scaler_dynamic.__dict__.update(ckpt["scaler_dynamic_state"])
        self.scaler_static = StandardScaler()
        self.scaler_static.__dict__.update(ckpt["scaler_static_state"])

        self.target_mean = float(ckpt.get("target_mean", self.scaler_dynamic.mean_[0]))
        self.target_std = float(ckpt.get("target_std", self.scaler_dynamic.scale_[0] + 1e-8))

        X0 = ckpt["X0"].to(torch.float32)
        A_hat = ckpt["A_hat"].to(torch.float32)

        self.T = int(cfg["T"])
        self.H = int(cfg["H"])

        self.model = Seq2SeqGCNLSTM(
            dynamic_dim=int(cfg["dynamic_dim"]),
            static_dim=int(cfg["static_dim"]),
            horizon=self.H,
            time_dim=int(cfg["time_dim"]),
            gcn_in_dim=int(cfg["gcn_in_dim"]),
            gcn_hidden_dim=int(cfg.get("gcn_hidden_dim", 64)),
            gcn_out_dim=int(cfg["gcn_out_dim"]),
            X0=X0.to(device),
            A_hat=A_hat.to(device),
            lstm_hidden_dim=int(cfg.get("lstm_hidden_dim", 128)),
            lstm_layers=int(cfg.get("lstm_layers", 2)),
            fc_hidden_dim=int(cfg.get("fc_hidden_dim", 128)),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
        ).to(device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _build_dyn_features(
        self,
        hist_df: pd.DataFrame,
        anchor_end_ts_kst: Optional[datetime],
    ) -> Tuple[np.ndarray, float, datetime]:
        df = normalize_hourly_series(
            hist_df, "speed",
            min_points=self.T + 24,
            anchor_end_ts_kst=anchor_end_ts_kst,
        )

        df["hour"] = df["ts_kst"].dt.hour
        df["day_of_week"] = df["ts_kst"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["lag_1h"] = df["speed"].shift(1)
        df["lag_24h"] = df["speed"].shift(24)
        df["roll_mean_3h"] = df["speed"].rolling(3, min_periods=1).mean()
        df["roll_std_3h"] = df["speed"].rolling(3, min_periods=1).std().fillna(0.0)
        df["roll_mean_6h"] = df["speed"].rolling(6, min_periods=1).mean()
        df["roll_std_6h"] = df["speed"].rolling(6, min_periods=1).std().fillna(0.0)

        df = df.dropna(subset=["lag_1h", "lag_24h"])
        if len(df) < self.T:
            raise ValueError("속도 히스토리가 부족합니다. 최소 (T+24)시간 이상 필요합니다.")

        df_T = df.iloc[-self.T:].copy()
        last_speed = float(df_T["speed"].iloc[-1])
        last_ts_kst = df_T["ts_kst"].iloc[-1].to_pydatetime()

        X_dyn = df_T[
            ["speed", "hour", "day_of_week", "is_weekend",
             "sin_hour", "cos_hour",
             "lag_1h", "lag_24h",
             "roll_mean_3h", "roll_std_3h",
             "roll_mean_6h", "roll_std_6h"]
        ].values.astype(float)

        return X_dyn, last_speed, last_ts_kst

    def predict(
        self,
        link_id: str,
        hist_df: pd.DataFrame,
        horizon: int,
        anchor_end_ts_kst: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, List[datetime], datetime]:
        if link_id not in self.link_id_to_idx:
            raise ValueError(f"학습에 사용되지 않은 link_id: {link_id}")

        H_use = min(horizon, self.H)

        X_dyn, last_speed, last_ts_kst = self._build_dyn_features(hist_df, anchor_end_ts_kst)
        meta = get_static_meta().get_speed_meta(link_id)
        x_static = np.array([meta["distance"], meta["lanes"], meta["free_flow_speed"]], dtype=float)

        X_dyn_scaled = self.scaler_dynamic.transform(X_dyn).reshape(1, self.T, -1)
        X_stat_scaled = self.scaler_static.transform(x_static.reshape(1, -1))
        last_scaled = (last_speed - self.target_mean) / self.target_std

        future_ts = [last_ts_kst + timedelta(hours=h) for h in range(1, H_use + 1)]
        ft = pd.DataFrame({"ts": pd.to_datetime(future_ts)})
        ft["hour"] = ft["ts"].dt.hour
        ft["day_of_week"] = ft["ts"].dt.dayofweek
        ft["is_weekend"] = (ft["day_of_week"] >= 5).astype(int)
        ft["sin_hour"] = np.sin(2 * np.pi * ft["hour"] / 24)
        ft["cos_hour"] = np.cos(2 * np.pi * ft["hour"] / 24)
        future_time = ft[["hour", "day_of_week", "is_weekend", "sin_hour", "cos_hour"]].values.astype(float)

        x_dyn_t = torch.tensor(X_dyn_scaled, dtype=torch.float32, device=device)
        idx_t = torch.tensor([self.link_id_to_idx[link_id]], dtype=torch.long, device=device)
        x_stat_t = torch.tensor(X_stat_scaled, dtype=torch.float32, device=device)
        last_t = torch.tensor([last_scaled], dtype=torch.float32, device=device)
        future_t = torch.tensor(future_time.reshape(1, H_use, -1), dtype=torch.float32, device=device)

        with torch.no_grad():
            y_hat_scaled = self.model(
                x_dyn_t, idx_t, x_stat_t,
                future_time=future_t,
                y=None,
                last_value=last_t,
                teacher_forcing=False,
            )

        y_hat_scaled = y_hat_scaled.cpu().numpy().reshape(-1)
        y_hat = y_hat_scaled * self.target_std + self.target_mean
        return y_hat, future_ts, last_ts_kst


class VolumeModelWrapper:
    def __init__(self):
        ckpt = safe_torch_load(settings.VOL_CKPT, map_location="cpu")
        cfg = ckpt["config"]

        raw = ckpt["spot_id_to_idx"]
        self.spot_id_to_idx = {str(k): int(v) for k, v in raw.items()}


        self.scaler_dynamic = StandardScaler()
        self.scaler_dynamic.__dict__.update(ckpt["scaler_dynamic_state"])
        self.scaler_static = StandardScaler()
        self.scaler_static.__dict__.update(ckpt["scaler_static_state"])

        self.vol_mean = float(ckpt.get("target_mean", self.scaler_dynamic.mean_[0]))
        self.vol_std = float(ckpt.get("target_std", self.scaler_dynamic.scale_[0] + 1e-8))

        X0 = ckpt["X0"].to(torch.float32)
        A_hat = ckpt["A_hat"].to(torch.float32)

        self.T = int(cfg["T"])
        self.H = int(cfg["H"])

        self.model = Seq2SeqGCNLSTM(
            dynamic_dim=int(cfg["dynamic_dim"]),
            static_dim=int(cfg["static_dim"]),
            horizon=self.H,
            time_dim=int(cfg["time_dim"]),
            gcn_in_dim=int(cfg["gcn_in_dim"]),
            gcn_hidden_dim=int(cfg.get("gcn_hidden_dim", 64)),
            gcn_out_dim=int(cfg["gcn_out_dim"]),
            X0=X0.to(device),
            A_hat=A_hat.to(device),
            lstm_hidden_dim=int(cfg.get("lstm_hidden_dim", 128)),
            lstm_layers=int(cfg.get("lstm_layers", 2)),
            fc_hidden_dim=int(cfg.get("fc_hidden_dim", 128)),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
        ).to(device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _build_dyn_features(
        self,
        hist_df: pd.DataFrame,
        anchor_end_ts_kst: Optional[datetime],
    ) -> Tuple[np.ndarray, float, datetime]:
        df = normalize_hourly_series(
            hist_df, "vol",
            min_points=self.T + 24,
            anchor_end_ts_kst=anchor_end_ts_kst,
        )

        df["hour"] = df["ts_kst"].dt.hour
        df["day_of_week"] = df["ts_kst"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["lag_1h"] = df["vol"].shift(1)
        df["lag_24h"] = df["vol"].shift(24)
        df["roll_mean_3h"] = df["vol"].rolling(3, min_periods=1).mean()
        df["roll_std_3h"] = df["vol"].rolling(3, min_periods=1).std().fillna(0.0)
        df["roll_mean_6h"] = df["vol"].rolling(6, min_periods=1).mean()
        df["roll_std_6h"] = df["vol"].rolling(6, min_periods=1).std().fillna(0.0)

        df = df.dropna(subset=["lag_1h", "lag_24h"])
        if len(df) < self.T:
            raise ValueError("교통량 히스토리가 부족합니다. 최소 (T+24)시간 이상 필요합니다.")

        df_T = df.iloc[-self.T:].copy()
        last_vol = float(df_T["vol"].iloc[-1])
        last_ts_kst = df_T["ts_kst"].iloc[-1].to_pydatetime()

        X_dyn = df_T[
            ["vol", "hour", "day_of_week", "is_weekend",
             "sin_hour", "cos_hour",
             "lag_1h", "lag_24h",
             "roll_mean_3h", "roll_std_3h",
             "roll_mean_6h", "roll_std_6h"]
        ].values.astype(float)

        return X_dyn, last_vol, last_ts_kst

    def predict(
    self,
    spot_num: str,
    io_direction: str,
    hist_df: pd.DataFrame,
    horizon: int,
    anchor_end_ts_kst: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, List[datetime], datetime]:
        sid = f"{spot_num}_{io_direction}"
        if sid not in self.spot_id_to_idx:
            raise ValueError(f"학습에 사용되지 않은 spot_dir_id: {sid}")

        H_use = min(horizon, self.H)

        X_dyn, last_vol, last_ts_kst = self._build_dyn_features(hist_df, anchor_end_ts_kst)
        meta = get_static_meta().get_vol_meta(spot_num, io_direction)
        x_static = np.array([meta["spot_mean_vol"], meta["spot_max_vol"]], dtype=float)

        X_dyn_scaled = self.scaler_dynamic.transform(X_dyn).reshape(1, self.T, -1)
        X_stat_scaled = self.scaler_static.transform(x_static.reshape(1, -1))
        last_scaled = (last_vol - self.vol_mean) / self.vol_std

        future_ts = [last_ts_kst + timedelta(hours=h) for h in range(1, H_use + 1)]
        ft = pd.DataFrame({"ts": pd.to_datetime(future_ts)})
        ft["hour"] = ft["ts"].dt.hour
        ft["day_of_week"] = ft["ts"].dt.dayofweek
        ft["is_weekend"] = (ft["day_of_week"] >= 5).astype(int)
        ft["sin_hour"] = np.sin(2 * np.pi * ft["hour"] / 24)
        ft["cos_hour"] = np.cos(2 * np.pi * ft["hour"] / 24)
        future_time = ft[["hour", "day_of_week", "is_weekend", "sin_hour", "cos_hour"]].values.astype(float)

        x_dyn_t = torch.tensor(X_dyn_scaled, dtype=torch.float32, device=device)
        idx_t = torch.tensor([self.spot_id_to_idx[sid]], dtype=torch.long, device=device)
        x_stat_t = torch.tensor(X_stat_scaled, dtype=torch.float32, device=device)
        last_t = torch.tensor([last_scaled], dtype=torch.float32, device=device)
        future_t = torch.tensor(future_time.reshape(1, H_use, -1), dtype=torch.float32, device=device)

        with torch.no_grad():
            y_hat_scaled = self.model(
                x_dyn_t, idx_t, x_stat_t,
                future_time=future_t,
                y=None,
                last_value=last_t,
                teacher_forcing=False,
            )

        y_hat_scaled = y_hat_scaled.cpu().numpy().reshape(-1)
        y_hat = y_hat_scaled * self.vol_std + self.vol_mean
        return y_hat, future_ts, last_ts_kst


@lru_cache(maxsize=1)
def get_speed_model() -> SpeedModelWrapper:
    return SpeedModelWrapper()


@lru_cache(maxsize=1)
def get_vol_model() -> VolumeModelWrapper:
    return VolumeModelWrapper()


def warm_up_models():
    _ = get_static_meta()
    _ = get_speed_model()
    _ = get_vol_model()


# ------------------------------------------------
# 4) TOPIS OpenAPI
# ------------------------------------------------
async def fetch_speed_from_topis(link_id: str) -> Tuple[float, Optional[float]]:
    url = f"{settings.TOPIS_BASE_URL}/{settings.TOPIS_API_KEY}/xml/TrafficInfo/1/5/{link_id}"
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        text = r.text

        import re
        m_spd = re.search(rf"<prcs_spd>{number_pattern}</prcs_spd>", text, flags=re.IGNORECASE)
        m_trv = re.search(rf"<prcs_trv_time>{number_pattern}</prcs_trv_time>", text, flags=re.IGNORECASE)

        if not m_spd:
            m_code = re.search(r"<CODE>([^<]+)</CODE>", text, flags=re.IGNORECASE)
            m_msg = re.search(r"<MESSAGE>([^<]+)</MESSAGE>", text, flags=re.IGNORECASE)
            code = m_code.group(1) if m_code else "N/A"
            msg = m_msg.group(1) if m_msg else "N/A"
            raise RuntimeError(f"TOPIS speed parse fail. status={r.status_code}, code={code}, msg={msg}")

        spd = float(m_spd.group(1))
        trv = float(m_trv.group(1)) if m_trv else None
        return spd, trv

# ----------------------------
# ✅ TOPIS VolInfo: "항상 1회 호출" → 응답에서 io_type별 합산 정규화
# ----------------------------
async def fetch_volume_from_topis_all(
    spot_num: str,
    ymd: str,
    hh: str,
    start_index: int = 1,
    end_index: int = 200,
) -> Dict[int, Tuple[int, int]]:
    """
    return: { io_type: (total_vol, lane_cnt) }
      - io_type별로 lane별 vol 합산
      - lane_cnt는 (해당 io_type의 고유 lane_num 개수)
    """
    url = f"{settings.TOPIS_BASE_URL}/{settings.TOPIS_API_KEY}/xml/VolInfo/{start_index}/{end_index}/{spot_num}/{ymd}/{hh}/"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        text = r.text

    # XML/정규식 혼용 가능하지만, 기존 코드 스타일(정규식) 유지하면서 안전하게 그룹핑
    import re
    rows = re.findall(
        rf"<row>.*?<io_type>(\d+)</io_type>.*?<lane_num>(\d+)</lane_num>.*?<vol>{number_pattern}</vol>.*?</row>",
        text,
        flags=re.DOTALL,
    )
    if not rows:
        return {}

    sums: Dict[int, int] = {}
    lanes: Dict[int, set] = {}

    for io, lane, vol in rows:
        io_t = int(io)
        if io_t not in (1, 2):
            continue
        ln = int(lane)
        v = int(float(vol))

        sums[io_t] = sums.get(io_t, 0) + v
        lanes.setdefault(io_t, set()).add(ln)

    return {t: (sums[t], len(lanes.get(t, set()))) for t in sums}

# ✅ 기존 시그니처 호환용(필요한 io_type만 뽑는 래퍼)
async def fetch_volume_from_topis(
    spot_num: str,
    ymd: str,
    hh: str,
    io_type: Optional[int] = None,
) -> Tuple[int, int]:
    all_map = await fetch_volume_from_topis_all(spot_num, ymd, hh)
    if io_type is None:
        # (유입+유출) 합산이 아니라, 여기서는 기존 호환용이므로 "둘 다 합친 total"을 반환하지 않음.
        # 호출부에서 반드시 io_type을 지정하거나 all_map을 직접 써야 한다.
        return (0, 0)
    return all_map.get(io_type, (0, 0))


# ------------------------------------------------
# 5) DB upsert (PostgreSQL ON CONFLICT)
# ------------------------------------------------
def _is_postgres(db: Session) -> bool:
    return db.bind.dialect.name == "postgresql"

def upsert_speed_hist(db: Session, link_id: str, ts: datetime, speed: float, travel_time: Optional[float]):
    try:
        if _is_postgres(db):
            stmt = pg_insert(SpeedHist).values(
                link_id=link_id, ts=ts, speed=speed, travel_time=travel_time
            ).on_conflict_do_update(
                # constraint 대신 index_elements가 더 안전함(아래 2번 참고)
                index_elements=[SpeedHist.link_id, SpeedHist.ts],
                set_={"speed": speed, "travel_time": travel_time},
            )
            db.execute(stmt)
            db.commit()
            return

        rec = SpeedHist(link_id=link_id, ts=ts, speed=speed, travel_time=travel_time)
        db.add(rec)
        db.commit()

    except Exception:
        db.rollback()
        raise

def upsert_volume_hist(db: Session, spot_num: str, io_direction: str, ts: datetime, vol: int, lane_cnt: Optional[int]):
    try:
        if _is_postgres(db):
            stmt = pg_insert(VolumeHist).values(
                spot_num=spot_num,
                io_direction=io_direction,
                ts=ts,
                vol=vol,
                lane_cnt=lane_cnt,
            ).on_conflict_do_update(
                index_elements=[VolumeHist.spot_num, VolumeHist.io_direction, VolumeHist.ts],
                set_={"vol": vol, "lane_cnt": lane_cnt},
            )
            db.execute(stmt)
            db.commit()
            return

        rec = VolumeHist(spot_num=spot_num, io_direction=io_direction, ts=ts, vol=vol, lane_cnt=lane_cnt)
        db.add(rec)
        db.commit()

    except Exception:
        db.rollback()
        raise


def bulk_upsert_speed_hist(
    db: Session,
    rows: List[Dict],
):
    """
    rows: [{"link_id":..., "ts":..., "speed":..., "travel_time":...}, ...]
    """
    if not rows:
        return 0

    if _is_postgres(db):
        stmt = pg_insert(SpeedHist).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_speed_link_ts",
            set_={
                "speed": stmt.excluded.speed,
                "travel_time": stmt.excluded.travel_time,
            },
        )
        db.execute(stmt)
        db.commit()
        return len(rows)

    # sqlite 등 fallback
    n = 0
    for r in rows:
        upsert_speed_hist(db, r["link_id"], r["ts"], r["speed"], r.get("travel_time"))
        n += 1
    return n

def bulk_upsert_volume_hist(db: Session, rows: List[Dict]):
    """
    rows: [{"spot_num":..., "io_direction":..., "ts":..., "vol":..., "lane_cnt":...}, ...]
    """
    if not rows:
        return 0

    if _is_postgres(db):
        stmt = pg_insert(VolumeHist).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_vol_spot_dir_ts",
            set_={
                "vol": stmt.excluded.vol,
                "lane_cnt": stmt.excluded.lane_cnt,
            },
        )
        db.execute(stmt)
        db.commit()
        return len(rows)

    n = 0
    for r in rows:
        upsert_volume_hist(db, r["spot_num"], r["io_direction"], r["ts"], r["vol"], r.get("lane_cnt"))
        n += 1
    return n

def seed_speed_dummy(
    db: Session,
    link_id: str,
    hours: int = 72,
    end_ts: Optional[datetime] = None,
    base_speed: Optional[float] = None,
    noise_std: float = 2.0,
) -> Dict:
    """
    TOPIS 없이 speed_hist를 즉시 채움.
    - free_flow_speed 기반으로 (주/야 + 러시아워 dip) 형태를 만들어줌
    - 최소 hours는 T+24 이상이어야 예측이 안정적으로 됨 (기본 72)
    """
    static_meta = get_static_meta()
    meta = static_meta.get_speed_meta(link_id)

    end_kst = _kst_hour_floor(end_ts)
    start_kst = end_kst - timedelta(hours=hours - 1)

    ff = float(meta["free_flow_speed"])
    dist = float(meta["distance"])

    # base_speed 자동 산정
    if base_speed is None:
        base_speed = max(10.0, min(ff * 0.75, ff - 3.0))

    rng = random.Random(42)  # 재현 가능하게 고정(원하면 제거)
    rows = []
    for i in range(hours):
        ts = start_kst + timedelta(hours=i)
        h = ts.hour

        # 일변동(밤 빠름/낮 느림) + 러시아워 dip(8시/18시 부근)
        diurnal = 1.0 + 0.10 * math.sin(2 * math.pi * (h - 4) / 24.0)
        rush1 = math.exp(-((h - 8.0) ** 2) / (2 * 2.0 ** 2))
        rush2 = math.exp(-((h - 18.0) ** 2) / (2 * 2.5 ** 2))
        rush_penalty = 0.35 * (rush1 + rush2)

        spd = base_speed * diurnal * (1.0 - rush_penalty)
        spd += rng.gauss(0.0, noise_std)

        # clamp
        spd = float(np.clip(spd, 3.0, ff * 1.05))

        # travel_time(분) = 거리(km)/속도(km/h)*60
        trv = float((dist / max(spd, 1e-6)) * 60.0)

        rows.append({
            "link_id": link_id,
            "ts": ts,
            "speed": spd,
            "travel_time": trv,
        })

    inserted = bulk_upsert_speed_hist(db, rows)

    return {
        "link_id": link_id,
        "hours": hours,
        "start_ts_kst": start_kst,
        "end_ts_kst": end_kst,
        "inserted": inserted,
        "free_flow_speed": ff,
        "base_speed": base_speed,
    }

async def backfill_volume(
    db: Session,
    spot_num: str,
    hours: int = 72,
    end_ts: Optional[datetime] = None,
    io_type: Optional[int] = None,
    sleep_sec: float = 0.05,
) -> Dict:
    """
    TOPIS VolInfo를 시간 단위로 호출해서 volume_hist를 과거 N시간만큼 채움.
    io_type:
      - None: 유입(1), 유출(2) 둘 다 저장
      - 1/2: 해당 방향만 저장
    """
    end_kst = _kst_hour_floor(end_ts)
    start_kst = end_kst - timedelta(hours=hours - 1)

    target_types = [1, 2] if io_type is None else [io_type]
    if any(t not in (1, 2) for t in target_types):
        raise ValueError("io_type은 1(유입), 2(유출), None 중 하나여야 합니다.")

    rows = []
    errors = []

    for i in range(hours):
        ts = start_kst + timedelta(hours=i)
        ymd, hh = _ymd_hh_from_ts(ts)

        try:
            all_map = await fetch_volume_from_topis_all(spot_num=spot_num, ymd=ymd, hh=hh)
            for t in target_types:
                if t not in all_map:
                    continue
                total_vol, lane_cnt = all_map[t]
                rows.append({
                    "spot_num": spot_num,
                    "io_direction": IO_TYPE_TO_DIR[t],
                    "ts": ts,
                    "vol": int(total_vol),
                    "lane_cnt": int(lane_cnt),
                })
        except Exception as e:
            errors.append({"ts": ts, "spot_num": spot_num, "error": str(e)})

        if sleep_sec > 0:
            await asyncio.sleep(sleep_sec)

    inserted = bulk_upsert_volume_hist(db, rows)

    return {
        "spot_num": spot_num,
        "hours": hours,
        "start_ts_kst": start_kst,
        "end_ts_kst": end_kst,
        "io_type": io_type,
        "requested_rows": len(rows),
        "inserted": inserted,
        "errors": errors[:20],
        "error_count": len(errors),
    }


def seed_speed_dummy_for_road(
    db: Session,
    road_name: str,
    hours: int = 72,
    end_ts: Optional[datetime] = None,
    noise_std: float = 2.0,
) -> dict:
    static = get_static_meta()
    link_ids = static.find_links_by_road(road_name)
    if not link_ids:
        raise ValueError(f"해당 도로명에 속한 링크가 없습니다: {road_name}")

    end_kst = _kst_hour_floor(end_ts)
    ts_list = [end_kst - timedelta(hours=i) for i in range(hours-1, -1, -1)]

    rows = []
    for lid in link_ids:
        meta = static.get_speed_meta(lid)
        ff = float(meta["free_flow_speed"])
        base = ff * 0.8  # 적당한 기본값(원하면 파라미터로 뺄 것)

        for t in ts_list:
            # 아주 단순한 일변화 + 노이즈 (원하면 더 정교하게)
            hour = t.hour
            diurnal = 5.0 * math.sin(2 * math.pi * hour / 24)
            spd = max(1.0, base + diurnal + random.gauss(0, noise_std))
            rows.append({"link_id": lid, "ts": t, "speed": float(spd), "travel_time": None})

    n = bulk_upsert_speed_hist(db, rows)
    return {
        "road_name": road_name,
        "links": len(link_ids),
        "hours": hours,
        "inserted": n,
        "end_ts": end_kst,
    }

async def backfill_volume_for_road(
    db: Session,
    road_name: str,
    hours: int = 72,
    end_ts: Optional[datetime] = None,
    io_directions: Optional[List[str]] = None,
    sleep_sec: float = 0.05,
) -> dict:
    static = get_static_meta()

    dirs = io_directions or ["유입", "유출"]
    for d in dirs:
        if d not in ("유입", "유출"):
            raise ValueError("io_directions는 ['유입','유출']만 허용합니다.")

    spot_dir_ids = static.find_spot_dirs_by_road(road_name, io_directions=dirs)
    if not spot_dir_ids:
        return {"road_name": road_name, "spots": 0, "hours": hours, "message": "해당 도로의 volume spot이 없습니다."}

    # spot_num -> 필요한 io_type set 구성 (시간당 spot_num당 1회 호출)
    need: Dict[str, set] = {}
    for sid in spot_dir_ids:
        spot_num, io_dir = sid.split("_", 1)
        need.setdefault(spot_num, set()).add(DIR_TO_IO_TYPE[io_dir])

    end_kst = _kst_hour_floor(end_ts)
    start_kst = end_kst - timedelta(hours=hours - 1)

    inserted = 0
    skipped = 0
    errors = []

    for i in range(hours):
        ts = start_kst + timedelta(hours=i)
        ymd, hh = _ymd_hh_from_ts(ts)

        for spot_num, need_types in need.items():
            try:
                all_map = await fetch_volume_from_topis_all(spot_num=spot_num, ymd=ymd, hh=hh)
                for t in need_types:
                    if t not in all_map:
                        skipped += 1
                        continue
                    total_vol, lane_cnt = all_map[t]
                    upsert_volume_hist(db, spot_num, IO_TYPE_TO_DIR[t], ts, int(total_vol), int(lane_cnt))
                    inserted += 1
            except Exception as e:
                errors.append({"ts": ts, "spot_num": spot_num, "error": str(e)})

            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)

    return {
        "road_name": road_name,
        "hours": hours,
        "start_ts_kst": start_kst,
        "end_ts_kst": end_kst,
        "io_directions": dirs,
        "spots": list(need.keys()),
        "inserted": inserted,
        "skipped": skipped,
        "error_count": len(errors),
        "errors": errors[:20],
    }


# ------------------------------------------------
# 6) DB window fetch
# ------------------------------------------------
def get_speed_hist_window(db: Session, link_id: str, T: int) -> pd.DataFrame:
    need = T + 24
    rows = (
        db.query(SpeedHist)
        .filter(SpeedHist.link_id == link_id)
        .order_by(SpeedHist.ts.desc())
        .limit(T + 96)
        .all()
    )
    rows = list(reversed(rows))
    if len(rows) < need:
        raise ValueError(f"속도 히스토리가 부족합니다. 필요:{need} 현재:{len(rows)}")
    return pd.DataFrame([{"ts": r.ts, "speed": r.speed} for r in rows])

def get_volume_hist_window(db: Session, spot_num: str, io_direction: str, T: int) -> pd.DataFrame:
    need = T + 24
    rows = (
        db.query(VolumeHist)
        .filter(VolumeHist.spot_num == spot_num, VolumeHist.io_direction == io_direction)
        .order_by(VolumeHist.ts.desc())
        .limit(T + 96)
        .all()
    )
    rows = list(reversed(rows))
    if len(rows) < need:
        raise ValueError(f"교통량 히스토리가 부족합니다. 필요:{need} 현재:{len(rows)}")
    return pd.DataFrame([{"ts": r.ts, "vol": r.vol} for r in rows])


# ------------------------------------------------
# 7) 기존 Segment predict (유지)
# ------------------------------------------------
def predict_segment(db: Session, req: SegmentPredictRequest) -> SegmentForecastResponse:
    speed_model = get_speed_model()
    static_meta = get_static_meta()

    H_use = min(req.horizon, speed_model.H)

    hist_speed = get_speed_hist_window(db, req.link_id, speed_model.T)
    speed_pred, speed_ts, _ = speed_model.predict(req.link_id, hist_speed, horizon=H_use)
    meta_speed = static_meta.get_speed_meta(req.link_id)

    ff = float(meta_speed["free_flow_speed"])
    speed_ci = np.clip(1.0 - speed_pred / (ff + 1e-6), 0.0, 1.0)

    speed_forecast = SpeedForecast(
        link_id=req.link_id,
        road_name=meta_speed["road_name"],
        free_flow_speed=ff,
        predictions=[
            HorizonPoint(offset_h=i + 1, timestamp=speed_ts[i], value=float(speed_pred[i]))
            for i in range(H_use)
        ],
        congestion_index=[float(x) for x in speed_ci[:H_use]],
    )

    vol_forecast = None
    vol_load = None

    if req.spot_num and req.io_direction:
        vol_model = get_vol_model()

        hist_vol = get_volume_hist_window(db, req.spot_num, req.io_direction, vol_model.T)
        vol_pred, vol_ts, _ = vol_model.predict(req.spot_num, req.io_direction, hist_vol, horizon=H_use)

        meta_vol = static_meta.get_vol_meta(req.spot_num, req.io_direction)
        max_vol = float(meta_vol["spot_max_vol"])
        vol_load = np.clip(vol_pred / (max_vol + 1e-6), 0.0, 1.0)

        vol_forecast = VolumeForecast(
            spot_num=req.spot_num,
            io_direction=req.io_direction,
            road_name=meta_vol["road_name"],
            max_vol=max_vol,
            predictions=[
                HorizonPoint(offset_h=i + 1, timestamp=vol_ts[i], value=float(vol_pred[i]))
                for i in range(H_use)
            ],
            load_index=[float(x) for x in vol_load[:H_use]],
        )

    combined = []
    for i in range(H_use):
        s = float(speed_ci[i])
        if vol_load is not None:
            v = float(vol_load[i])
            combined.append(0.5 * s + 0.5 * v)
        else:
            combined.append(s)

    combined_resp = CombinedCongestion(
        road_name=speed_forecast.road_name,
        link_id=req.link_id,
        spot_num=req.spot_num,
        io_direction=req.io_direction,
        horizon=H_use,
        combined_index=combined,
    )

    return SegmentForecastResponse(speed=speed_forecast, volume=vol_forecast, combined=combined_resp)


# ------------------------------------------------
# 8) ✅ Road predict
# ------------------------------------------------
def predict_road(db: Session, req: RoadPredictRequest) -> RoadForecastResponse:
    static_meta = get_static_meta()
    speed_model = get_speed_model()

    link_ids = static_meta.find_links_by_road(req.road_name)
    if not link_ids:
        raise ValueError(f"해당 도로명에 속하는 링크가 없습니다: {req.road_name}")

    H_use = min(req.horizon, speed_model.H)

    spot_dir_ids: List[str] = []
    if req.include_volume and static_meta.vol_meta:
        spot_dir_ids = static_meta.find_spot_dirs_by_road(req.road_name, io_directions=req.io_directions)

    # --- 1) 공통 anchor_ts 계산 (speed/volume 모두 시간축 정렬) ---
    speed_end_list: List[datetime] = []
    missing_speed_links: List[str] = []
    for lid in link_ids:
        try:
            hist = get_speed_hist_window(db, lid, speed_model.T)
            speed_end_list.append(get_series_max_ts_kst(hist, "speed"))
        except Exception:
            missing_speed_links.append(lid)

    vol_end_list: List[datetime] = []
    missing_volume_spots: List[str] = []
    vol_model = None
    if spot_dir_ids:
        vol_model = get_vol_model()
        for sid in spot_dir_ids:
            try:
                spot_num, direction = sid.split("_", 1)
                hist = get_volume_hist_window(db, spot_num, direction, vol_model.T)
                vol_end_list.append(get_series_max_ts_kst(hist, "vol"))
            except Exception:
                missing_volume_spots.append(sid)

    if not speed_end_list:
        raise ValueError("해당 도로에 대해 speed 히스토리가 충분한 링크가 하나도 없습니다. (seed-speed-dummy 먼저 실행)")

    anchor_ts = min(speed_end_list + (vol_end_list if vol_end_list else speed_end_list))
    anchor_ts = _kst_hour_floor(anchor_ts)

    # --- 2) link별 speed 예측 ---
    speed_links: List[SpeedForecast] = []
    for lid in link_ids:
        if lid in missing_speed_links:
            continue
        hist_speed = get_speed_hist_window(db, lid, speed_model.T)

        y_hat, future_ts, _last_ts = speed_model.predict(
            lid, hist_speed, horizon=H_use, anchor_end_ts_kst=anchor_ts
        )
        meta = static_meta.get_speed_meta(lid)
        ff = float(meta["free_flow_speed"])
        speed_ci = np.clip(1.0 - y_hat / (ff + 1e-6), 0.0, 1.0)

        speed_links.append(
            SpeedForecast(
                link_id=lid,
                road_name=meta["road_name"],
                free_flow_speed=ff,
                predictions=[
                    HorizonPoint(offset_h=i + 1, timestamp=future_ts[i], value=float(y_hat[i]))
                    for i in range(H_use)
                ],
                congestion_index=[float(x) for x in speed_ci[:H_use]],
            )
        )

    if not speed_links:
        raise ValueError("speed 예측 가능한 링크가 없습니다. (seed-speed-dummy / DB 상태 확인)")

    # --- 3) spot별 volume 예측(옵션) ---
    volume_spots: Optional[List[VolumeForecast]] = None
    if spot_dir_ids and vol_model is not None:
        volume_spots = []
        for sid in spot_dir_ids:
            if sid in missing_volume_spots:
                continue
            spot_num, io_dir = sid.split("_", 1)
            hist_vol = get_volume_hist_window(db, spot_num, io_dir, vol_model.T)

            v_hat, v_future_ts, _ = vol_model.predict(
                spot_num, io_dir, hist_vol, horizon=H_use, anchor_end_ts_kst=anchor_ts
            )
            meta_v = static_meta.get_vol_meta(spot_num, io_dir)
            max_vol = float(meta_v["spot_max_vol"])
            load_idx = np.clip(v_hat / (max_vol + 1e-6), 0.0, 1.0)

            volume_spots.append(
                VolumeForecast(
                    spot_num=spot_num,
                    io_direction=io_dir,
                    road_name=meta_v["road_name"],
                    max_vol=max_vol,
                    predictions=[
                        HorizonPoint(offset_h=i + 1, timestamp=v_future_ts[i], value=float(v_hat[i]))
                        for i in range(H_use)
                    ],
                    load_index=[float(x) for x in load_idx[:H_use]],
                )
            )

    # --- 4) road-level index 합성 ---
    road_index: List[RoadIndexPoint] = []

    # speed road index = 링크 congestion_index 평균
    speed_mat = np.array([sl.congestion_index for sl in speed_links], dtype=float)  # (L, H)
    speed_road = speed_mat.mean(axis=0)  # (H,)

    volume_road = None
    if volume_spots:
        vol_mat = np.array([vs.load_index for vs in volume_spots], dtype=float)  # (S, H)
        volume_road = vol_mat.mean(axis=0)

    for i in range(H_use):
        ts = anchor_ts + timedelta(hours=i + 1)
        s = float(speed_road[i])
        if volume_road is not None:
            v = float(volume_road[i])
            comb = ROAD_FUSION_ALPHA * s + (1.0 - ROAD_FUSION_ALPHA) * v
            road_index.append(
                RoadIndexPoint(
                    offset_h=i + 1,
                    timestamp=ts,
                    speed_index=s,
                    volume_index=v,
                    combined_index=float(np.clip(comb, 0.0, 1.0)),
                )
            )
        else:
            road_index.append(
                RoadIndexPoint(
                    offset_h=i + 1,
                    timestamp=ts,
                    speed_index=s,
                    volume_index=None,
                    combined_index=s,
                )
            )

    return RoadForecastResponse(
        road_name=req.road_name,
        anchor_ts=anchor_ts,
        horizon=H_use,
        speed_links=speed_links,
        volume_spots=volume_spots,
        road_index=road_index,
        missing_speed_links=missing_speed_links,
        missing_volume_spots=missing_volume_spots,
    )
