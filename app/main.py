# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .config import settings
from .db import Base, engine, get_db
from .schemas import (
    SegmentPredictRequest, SegmentForecastResponse,
    RoadPredictRequest, RoadForecastResponse,
    FetchSpeedRequest, FetchVolumeRequest,
    SeedSpeedDummyRequest, BackfillVolumeRequest,
    SeedRoadSpeedDummyRequest, BackfillRoadVolumeRequest, 
)
from .services import (
    fetch_speed_from_topis, fetch_volume_from_topis,
    upsert_speed_hist, upsert_volume_hist,
    predict_segment, predict_road,
    warm_up_models,
    get_static_meta,
    seed_speed_dummy, backfill_volume,
    seed_speed_dummy_for_road, backfill_volume_for_road, fetch_volume_from_topis_all, 
)

from datetime import datetime, timezone, timedelta


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    warm_up_models()
    yield


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "MetroVision AI API running"}


# ------------------------------------------------
# 1) 도로/지점 메타 정보 조회
# ------------------------------------------------
@app.get("/api/roads")
def list_roads():
    static_meta = get_static_meta()
    road_from_speed = {m["road_name"] for m in static_meta.speed_meta.values()}
    road_from_vol = {m["road_name"] for m in static_meta.vol_meta.values()} if static_meta.vol_meta else set()

    roads = []
    for rn in sorted(road_from_speed):
        roads.append({
            "road_name": rn,
            "has_speed": True,
            "has_volume": rn in road_from_vol
        })
    return roads


@app.get("/api/roads/{road_name}/links")
def list_links_of_road(road_name: str):
    static_meta = get_static_meta()
    links = []
    for lid, meta in static_meta.speed_meta.items():
        if meta["road_name"] == road_name:
            links.append({
                "link_id": lid,
                "start_name": meta["start_name"],
                "end_name": meta["end_name"],
                "direction": meta["direction"],
                "distance": meta["distance"],
                "lanes": meta["lanes"],
            })
    if not links:
        raise HTTPException(status_code=404, detail="해당 도로의 링크 정보가 없습니다.")
    return {"road_name": road_name, "links": links}


@app.get("/api/roads/{road_name}/spots")
def list_spots_of_road(road_name: str):
    static_meta = get_static_meta()
    spots = []
    for sid, meta in (static_meta.vol_meta or {}).items():
        if meta["road_name"] == road_name and "_" in sid:
            spot_num, io_dir = sid.split("_", 1)
            spots.append({"spot_num": spot_num, "io_direction": io_dir, "spot_name": meta["spot_name"]})
    if not spots:
        raise HTTPException(status_code=404, detail="해당 도로의 교통량 지점이 없습니다.")
    return {"road_name": road_name, "spots": spots}


# ------------------------------------------------
# 2) TOPIS에서 실시간 값 가져와서 히스토리 적재 (관리용)
# ------------------------------------------------
@app.post("/api/admin/fetch-speed")
async def api_fetch_speed(req: FetchSpeedRequest, db: Session = Depends(get_db)):
    spd, trv = await fetch_speed_from_topis(req.link_id)

    kst_now = datetime.now(timezone(timedelta(hours=9)))
    ts_kst = kst_now.replace(minute=0, second=0, microsecond=0)

    upsert_speed_hist(db, req.link_id, ts_kst, spd, trv)
    return {"link_id": req.link_id, "ts": ts_kst, "speed": spd, "travel_time": trv}


@app.post("/api/admin/fetch-volume")
async def api_fetch_volume(req: FetchVolumeRequest, db: Session = Depends(get_db)):
    kst = datetime.now(timezone(timedelta(hours=9)))
    ymd = req.ymd or kst.strftime("%Y%m%d")
    hh = req.hh or kst.strftime("%H")

    ts_kst = kst.replace(minute=0, second=0, microsecond=0)

    all_map = await fetch_volume_from_topis_all(req.spot_num, ymd=ymd, hh=hh)

    target_io_types = [1, 2] if req.io_type is None else [req.io_type]
    if any(t not in (1, 2) for t in target_io_types):
        raise HTTPException(status_code=400, detail="io_type은 1/2 또는 None")

    results = []
    for t in target_io_types:
        if t not in all_map:
            continue
        total_vol, lane_cnt = all_map[t]
        io_dir = "유입" if t == 1 else "유출"
        upsert_volume_hist(db, req.spot_num, io_dir, ts_kst, int(total_vol), int(lane_cnt))
        results.append({
            "spot_num": req.spot_num,
            "io_direction": io_dir,
            "ts": ts_kst,
            "vol": int(total_vol),
            "lane_cnt": int(lane_cnt),
        })

    return {"spot_num": req.spot_num, "results": results}


@app.post("/api/admin/seed-speed-dummy")
def api_seed_speed_dummy(req: SeedSpeedDummyRequest, db: Session = Depends(get_db)):
    """
    TOPIS 없이 speed_hist를 더미로 채워서 즉시 speed 추론 가능하게 함.
    """
    result = seed_speed_dummy(
        db=db,
        link_id=req.link_id,
        hours=req.hours,
        end_ts=req.end_ts,
        base_speed=req.base_speed,
        noise_std=req.noise_std,
    )
    return result

@app.post("/api/admin/backfill-volume")
async def api_backfill_volume(req: BackfillVolumeRequest, db: Session = Depends(get_db)):
    """
    TOPIS VolInfo를 시간단위로 역수집해서 volume_hist를 채움.
    """
    result = await backfill_volume(
        db=db,
        spot_num=req.spot_num,
        hours=req.hours,
        end_ts=req.end_ts,
        io_type=req.io_type,
        sleep_sec=0.05,
    )
    return result

@app.post("/api/admin/seed-road-speed-dummy")
def api_seed_road_speed_dummy(req: SeedRoadSpeedDummyRequest, db: Session = Depends(get_db)):
    return seed_speed_dummy_for_road(
        db=db,
        road_name=req.road_name,
        hours=req.hours,
        end_ts=req.end_ts,
        noise_std=req.noise_std,
    )

@app.post("/api/admin/backfill-road-volume")
async def api_backfill_road_volume(req: BackfillRoadVolumeRequest, db: Session = Depends(get_db)):
    return await backfill_volume_for_road(
        db=db,
        road_name=req.road_name,
        hours=req.hours,
        end_ts=req.end_ts,
        io_directions=req.io_directions,
    )


# ------------------------------------------------
# 3) 예측 API
# ------------------------------------------------
@app.post("/api/predict/segment", response_model=SegmentForecastResponse)
def api_predict_segment(req: SegmentPredictRequest, db: Session = Depends(get_db)):
    try:
        return predict_segment(db, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")


@app.post("/api/predict/road", response_model=RoadForecastResponse)
def api_predict_road(req: RoadPredictRequest, db: Session = Depends(get_db)):
    try:
        return predict_road(db, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")
