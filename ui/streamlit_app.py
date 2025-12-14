# streamlit_app.py
import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# -------------------------
# Config
# -------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TIMEOUT = 20

st.set_page_config(page_title="MetroVision Traffic Forecast", layout="wide")

# -------------------------
# Helpers
# -------------------------
def _safe_get(url: str):
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _safe_post(url: str, payload: dict):
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def _parse_ts(x):
    # FastAPI가 tz-aware ISO를 주므로 pandas가 잘 파싱함
    return pd.to_datetime(x)

def _level_from_index(x: float) -> str:
    # 0~1: 높을수록 혼잡 (backend에서 congestion_index/load_index/combined_index)
    if x < 0.25:
        return "원활"
    if x < 0.50:
        return "보통"
    if x < 0.75:
        return "혼잡"
    return "매우혼잡"

def _format_dt(dt) -> str:
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt)

@st.cache_data(ttl=60)
def fetch_roads():
    return _safe_get(f"{API_BASE}/api/roads")

@st.cache_data(ttl=60)
def fetch_links_meta(road_name: str):
    return _safe_get(f"{API_BASE}/api/roads/{road_name}/links")

@st.cache_data(ttl=60)
def fetch_spots_meta(road_name: str):
    # volume 없는 도로면 404 날 수 있음
    try:
        return _safe_get(f"{API_BASE}/api/roads/{road_name}/spots")
    except requests.HTTPError:
        return {"road_name": road_name, "spots": []}

def build_link_label_map(links_meta: dict):
    # link_id -> "시점명→종점명 (방향) [link_id]"
    m = {}
    for x in links_meta.get("links", []):
        lid = x["link_id"]
        s = x.get("start_name", "")
        e = x.get("end_name", "")
        d = x.get("direction", "")
        m[lid] = f"{s} → {e} ({d}) [{lid}]"
    return m

def build_spot_label_map(spots_meta: dict):
    # (spot_num, io_direction) -> "지점명 - (유입/유출) [spot]"
    m = {}
    for x in spots_meta.get("spots", []):
        spot = x["spot_num"]
        io = x["io_direction"]
        name = x.get("spot_name", "")
        m[(spot, io)] = f"{name} - {io} [{spot}]"
    return m

def road_index_df(resp: dict) -> pd.DataFrame:
    rows = []
    for p in resp.get("road_index", []):
        rows.append({
            "offset_h": p["offset_h"],
            "timestamp": _parse_ts(p["timestamp"]),
            "speed_index": float(p["speed_index"]),
            "volume_index": float(p["volume_index"]) if p.get("volume_index") is not None else None,
            "combined_index": float(p["combined_index"]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["level"] = df["combined_index"].apply(_level_from_index)
    return df

def speed_links_long_df(resp: dict, link_label_map: dict) -> pd.DataFrame:
    rows = []
    for link in resp.get("speed_links", []):
        lid = link["link_id"]
        ff = float(link["free_flow_speed"])
        ci_list = link.get("congestion_index", [])
        preds = link.get("predictions", [])
        for i, p in enumerate(preds):
            spd = float(p["value"])
            ci = float(ci_list[i]) if i < len(ci_list) else None
            rows.append({
                "link_id": lid,
                "label": link_label_map.get(lid, lid),
                "free_flow_speed": ff,
                "offset_h": int(p["offset_h"]),
                "timestamp": _parse_ts(p["timestamp"]),
                "speed": spd,
                "congestion_index": ci,
                "level": _level_from_index(ci) if ci is not None else None,
            })
    return pd.DataFrame(rows)

def volume_spots_long_df(resp: dict, spot_label_map: dict) -> pd.DataFrame:
    rows = []
    for spot in (resp.get("volume_spots") or []):
        spot_num = spot["spot_num"]
        io = spot["io_direction"]
        maxv = float(spot["max_vol"])
        li_list = spot.get("load_index", [])
        preds = spot.get("predictions", [])
        for i, p in enumerate(preds):
            vol = float(p["value"])
            li = float(li_list[i]) if i < len(li_list) else None
            rows.append({
                "spot_num": spot_num,
                "io_direction": io,
                "label": spot_label_map.get((spot_num, io), f"{spot_num} {io}"),
                "max_vol": maxv,
                "offset_h": int(p["offset_h"]),
                "timestamp": _parse_ts(p["timestamp"]),
                "volume": vol,
                "load_index": li,
                "level": _level_from_index(li) if li is not None else None,
            })
    return pd.DataFrame(rows)

def detect_anomaly_speed(df_speed: pd.DataFrame) -> str | None:
    # 지금 네 샘플 응답처럼 speed가 음수면 사용자에게 즉시 경고
    if df_speed.empty:
        return None
    neg = (df_speed["speed"] < 0).any()
    if neg:
        return "⚠️ 속도 예측값에 음수가 포함되어 있습니다. (스케일 역변환/모델 출력 처리 문제 가능) UI는 그대로 표시하지만, 지표(congestion_index)가 1로 포화될 수 있습니다."
    return None


# -------------------------
# UI
# -------------------------
st.title("MetroVision AI · 도로 단위 교통 예측 대시보드 (Streamlit)")

with st.sidebar:
    st.subheader("연결 설정")
    api_base_input = st.text_input("API Base", API_BASE)
    if api_base_input != API_BASE:
        API_BASE = api_base_input

    st.divider()
    st.subheader("도로 선택")

    roads = fetch_roads()
    df_roads = pd.DataFrame(roads)

    # 카테고리: 교통량 포함 가능/불가
    cat = st.radio(
        "카테고리",
        ["전체", "교통량 포함 가능", "교통량 없음(속도만)"],
        index=0
    )

    search = st.text_input("검색(부분 문자열)", "")

    if not df_roads.empty:
        if cat == "교통량 포함 가능":
            df_roads = df_roads[df_roads["has_volume"] == True]
        elif cat == "교통량 없음(속도만)":
            df_roads = df_roads[df_roads["has_volume"] == False]

        if search.strip():
            df_roads = df_roads[df_roads["road_name"].str.contains(search.strip(), case=False, na=False)]

        road_candidates = df_roads["road_name"].tolist()
    else:
        road_candidates = []

    road_name = st.selectbox("도로명", road_candidates if road_candidates else ["(도로 없음)"])

    st.divider()
    st.subheader("예측 옵션")
    horizon = st.slider("horizon (시간)", min_value=1, max_value=24, value=6, step=1)
    include_volume = st.checkbox("include_volume", value=True)

    io_dirs = None
    if include_volume:
        io_dirs_sel = st.multiselect("교통량 방향(io_directions)", ["유입", "유출"], default=["유입", "유출"])
        io_dirs = io_dirs_sel if io_dirs_sel else None

    run = st.button("예측 실행", type="primary")

# -------------------------
# Run Prediction
# -------------------------
if run and road_name and road_name != "(도로 없음)":
    # 메타 로딩(라벨링용)
    links_meta = fetch_links_meta(road_name)
    link_label_map = build_link_label_map(links_meta)

    spots_meta = fetch_spots_meta(road_name) if include_volume else {"spots": []}
    spot_label_map = build_spot_label_map(spots_meta)

    payload = {
        "road_name": road_name,
        "horizon": horizon,
        "include_volume": include_volume,
    }
    if io_dirs is not None:
        payload["io_directions"] = io_dirs

    try:
        resp = _safe_post(f"{API_BASE}/api/predict/road", payload)
    except Exception as e:
        st.error(f"예측 호출 실패: {e}")
        st.stop()

    # 데이터프레임 구성
    df_road = road_index_df(resp)
    df_speed = speed_links_long_df(resp, link_label_map)
    df_vol = volume_spots_long_df(resp, spot_label_map)

    warn = detect_anomaly_speed(df_speed)
    if warn:
        st.warning(warn)

    # -------------------------
    # Summary
    # -------------------------
    anchor_ts = resp.get("anchor_ts")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("도로", resp.get("road_name", "-"))
    c2.metric("anchor_ts", _format_dt(anchor_ts))
    c3.metric("horizon", str(resp.get("horizon", "-")))
    c4.metric("링크/지점", f"{len(resp.get('speed_links', []))} / {len(resp.get('volume_spots') or [])}")

    tab1, tab2, tab3, tab4 = st.tabs(["요약", "링크 속도", "지점 교통량(유입/유출)", "원본 JSON"])

    with tab1:
        st.subheader("도로 전체 혼잡도(road_index)")

        if df_road.empty:
            st.info("road_index 데이터가 없습니다.")
        else:
            now_row = df_road.sort_values("offset_h").iloc[0]
            avg_comb = float(df_road["combined_index"].mean())
            max_comb = float(df_road["combined_index"].max())

            c1, c2, c3 = st.columns(3)
            c1.metric("1시간 뒤 혼잡도", f"{now_row['combined_index']:.3f} ({now_row['level']})")
            c2.metric("평균 혼잡도", f"{avg_comb:.3f} ({_level_from_index(avg_comb)})")
            c3.metric("최대 혼잡도", f"{max_comb:.3f} ({_level_from_index(max_comb)})")

            plot_cols = ["speed_index", "combined_index"]
            if df_road["volume_index"].notna().any():
                plot_cols = ["speed_index", "volume_index", "combined_index"]

            fig = px.line(
                df_road,
                x="timestamp",
                y=plot_cols,
                markers=True,
                title="Road Index (speed/volume/combined)"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df_road[["offset_h", "timestamp", "speed_index", "volume_index", "combined_index", "level"]],
                use_container_width=True
            )

        st.subheader("결측/누락 상태")
        ms = resp.get("missing_speed_links", [])
        mv = resp.get("missing_volume_spots", [])
        colA, colB = st.columns(2)
        with colA:
            st.write(f"missing_speed_links: {len(ms)}")
            if ms:
                st.code("\n".join(ms[:50]))
        with colB:
            st.write(f"missing_volume_spots: {len(mv)}")
            if mv:
                st.code("\n".join(mv[:50]))

    with tab2:
        st.subheader("링크별 속도 예측 + 혼잡도")

        if df_speed.empty:
            st.info("speed_links 데이터가 없습니다.")
        else:
            # 링크 선택
            link_labels = sorted(df_speed["label"].unique().tolist())
            sel_links = st.multiselect("표시할 링크", link_labels, default=link_labels[: min(5, len(link_labels))])

            dff = df_speed[df_speed["label"].isin(sel_links)].copy()
            dff = dff.sort_values(["label", "timestamp"])

            # 그래프: speed
            fig1 = px.line(dff, x="timestamp", y="speed", color="label", markers=True, title="Predicted Speed by Link")
            st.plotly_chart(fig1, use_container_width=True)

            # 그래프: congestion_index
            fig2 = px.line(dff, x="timestamp", y="congestion_index", color="label", markers=True, title="Congestion Index by Link (0~1)")
            st.plotly_chart(fig2, use_container_width=True)

            # 테이블
            show_cols = ["label", "link_id", "offset_h", "timestamp", "speed", "free_flow_speed", "congestion_index", "level"]
            st.dataframe(dff[show_cols], use_container_width=True)

            # 링크 전체 평균(시간별)
            agg = df_speed.groupby("timestamp", as_index=False).agg(
                avg_speed=("speed", "mean"),
                avg_cong=("congestion_index", "mean"),
            )
            fig3 = px.line(agg, x="timestamp", y=["avg_speed"], markers=True, title="Average Speed (All Links)")
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("지점별 교통량 예측 (유입/유출 포함)")

        if df_vol.empty:
            st.info("volume_spots 데이터가 없습니다. (도로가 교통량 대상이 아니거나 include_volume=false)")
        else:
            # 방향 탭
            t_in, t_out = st.tabs(["유입", "유출"])

            for dir_name, tab in [("유입", t_in), ("유출", t_out)]:
                with tab:
                    ddir = df_vol[df_vol["io_direction"] == dir_name].copy()
                    if ddir.empty:
                        st.info(f"{dir_name} 데이터가 없습니다.")
                        continue

                    spot_labels = sorted(ddir["label"].unique().tolist())
                    sel_spots = st.multiselect(
                        f"{dir_name} 표시할 지점",
                        spot_labels,
                        default=spot_labels[: min(5, len(spot_labels))],
                        key=f"spots_{dir_name}",
                    )

                    dff = ddir[ddir["label"].isin(sel_spots)].sort_values(["label", "timestamp"])

                    fig1 = px.line(dff, x="timestamp", y="volume", color="label", markers=True, title=f"Predicted Volume ({dir_name})")
                    st.plotly_chart(fig1, use_container_width=True)

                    fig2 = px.line(dff, x="timestamp", y="load_index", color="label", markers=True, title=f"Load Index ({dir_name}, 0~1)")
                    st.plotly_chart(fig2, use_container_width=True)

                    show_cols = ["label", "spot_num", "io_direction", "offset_h", "timestamp", "volume", "max_vol", "load_index", "level"]
                    st.dataframe(dff[show_cols], use_container_width=True)

    with tab4:
        st.subheader("원본 응답(JSON)")
        st.json(resp)

else:
    st.info("왼쪽에서 도로를 선택하고 **예측 실행**을 눌러 주세요.")
