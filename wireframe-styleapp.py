import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_autorefresh import st_autorefresh
import threading

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SeagrassLive Pro", page_icon="🪸", layout="wide")

# --- SESSION STATE ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'detection'

is_running = st.session_state.is_running
active_tab = st.session_state.active_tab

dot_icon     = "●" if is_running else "○"
btn_title    = "⏹  Kamera Aktif — Deteksi Berjalan"  if is_running else "▶  Aktifkan Kamera Lamun"
btn_sub      = "Klik untuk menonaktifkan kamera"      if is_running else "Klik untuk memulai deteksi real-time"
status_color = "#1a1a1a" if is_running else "#888888"
status_text  = "AKTIF"   if is_running else "NONAKTIF"
status_bg    = "#e0e0e0" if is_running else "#f0f0f0"

# ── TAB STATE via query param ──
params = st.query_params
if "tab" in params:
    st.session_state.active_tab = params["tab"]
    active_tab = st.session_state.active_tab

tab_det_cls = "tab-active" if active_tab == "detection"   else ""
tab_enc_cls = "tab-active" if active_tab == "encyclopedia" else ""

# --- CSS WIREFRAME HITAM PUTIH (ORIGINAL) + TAB ADDITIONS ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    header {{visibility: hidden;}}

    /* Sembunyikan autorefresh iframe */
    div[data-testid="stElementContainer"]:has(iframe[title="streamlit_autorefresh.st_autorefresh"]) {{
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }}

    /* BODY & BACKGROUND */
    div.block-container, .stApp {{
        background-color: #f5f5f5 !important;
        padding-top: 0px !important;
    }}
    div.block-container {{
        padding: 1rem 1.5rem !important;
        max-width: 100% !important;
    }}

    /* NAVBAR */
    .nav-container {{
        background: #ffffff;
        padding: 14px 28px;
        border-radius: 0px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 2px solid #1a1a1a;
        margin-bottom: 0px;
        position: relative;
        overflow: hidden;
    }}
    .nav-container::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: repeating-linear-gradient(
            90deg,
            #1a1a1a 0px,
            #1a1a1a 10px,
            transparent 10px,
            transparent 20px
        );
    }}
    .nav-badge {{
        background: #1a1a1a;
        color: #ffffff;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        padding: 4px 12px;
        border-radius: 0px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: fit-content;
        white-space: nowrap;
        font-family: 'IBM Plex Mono', monospace;
        border: 1.5px solid #1a1a1a;
    }}

    /* ── TAB NAVIGATION ── */
    .tab-nav {{
        display: flex;
        gap: 0;
        border: 2px solid #1a1a1a;
        border-top: none;
        background: #ffffff;
        margin-bottom: 20px;
    }}
    .tab-btn {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 12px 22px;
        background: #f5f5f5;
        color: #888888;
        border: none;
        border-right: 1.5px solid #cccccc;
        cursor: pointer;
        text-decoration: none;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: background 0.15s, color 0.15s;
        flex: 1;
        min-width: 0;
        overflow: hidden;
    }}
    .tab-btn:last-child {{ border-right: none; }}
    .tab-btn:hover {{
        background: #ebebeb;
        color: #444444;
    }}
    .tab-active {{
        background: #1a1a1a !important;
        color: #ffffff !important;
    }}
    .tab-active:hover {{
        background: #333333 !important;
        color: #ffffff !important;
    }}

    /* WEBRTC VIDEO */
    div[data-testid="stVerticalBlock"] > div:has(> div.stWebRtcStreamer) {{
        padding: 0 !important;
        margin: 0 !important;
    }}
    .stWebRtcStreamer > div {{
        padding: 0 !important;
        margin: 0 !important;
    }}
    .stWebRtcStreamer video {{
        width: 100% !important;
        border-radius: 0px !important;
        background: #e8e8e8 !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 2px solid #1a1a1a !important;
    }}
    .stWebRtcStreamer div[class*="style__mediaPlayer"] {{
        border-radius: 0px !important;
        overflow: hidden !important;
        border: 2px solid #1a1a1a !important;
    }}
    .stWebRtcStreamer {{
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* TOMBOL RADAR */
    div[data-testid="stButton"] {{
        margin-top: 10px !important;
        margin-bottom: 0 !important;
        width: 100% !important;
    }}
    div[data-testid="stButton"] > button {{
        width: 100% !important;
        min-height: 68px !important;
        background: #d8d8d8 !important;
        border: 2px solid #1a1a1a !important;
        border-radius: 0px !important;
        padding: 14px 20px !important;
        box-shadow: 4px 4px 0px #1a1a1a !important;
        cursor: pointer !important;
        transition: box-shadow 0.15s, transform 0.15s !important;
        font-size: 0 !important;
        color: transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        position: relative !important;
        overflow: visible !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        background: #c8c8c8 !important;
        box-shadow: 6px 6px 0px #1a1a1a !important;
        transform: translate(-1px, -1px) !important;
        cursor: pointer !important;
    }}
    div[data-testid="stButton"] > button:focus,
    div[data-testid="stButton"] > button:active {{
        outline: none !important;
        box-shadow: 2px 2px 0px #1a1a1a !important;
        transform: translate(2px, 2px) !important;
        color: transparent !important;
    }}
    div[data-testid="stButton"] > button::before {{
        content: "{dot_icon}  {btn_title}\\A {btn_sub}";
        white-space: pre;
        font-size: 13px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: #1a1a1a;
        text-align: left;
        line-height: 1.7;
        pointer-events: none;
        flex: 1;
    }}
    div[data-testid="stButton"] > button::after {{
        content: "STATUS\\A {status_text}";
        white-space: pre;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        color: {status_color};
        text-align: center;
        line-height: 2;
        pointer-events: none;
        padding-left: 14px;
        border-left: 2px solid #1a1a1a;
        margin-left: 14px;
    }}

    /* CAMERA PLACEHOLDER — ikon kamera statis */
    .cam-placeholder {{
        width: 100%;
        aspect-ratio: 16/9;
        background: #ffffff;
        border: 2px solid #1a1a1a;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
        position: relative;
        overflow: hidden;
        min-height: 240px;
    }}
    .cam-placeholder::before {{
        content: '';
        position: absolute;
        inset: 0;
        background-image: radial-gradient(circle, #cccccc 1px, transparent 1px);
        background-size: 20px 20px;
        opacity: 0.6;
    }}
    .cam-placeholder::after {{
        content: '';
        position: absolute;
        inset: 10px;
        border: 1px dashed #999999;
        pointer-events: none;
    }}

    /* Ikon kamera statis (mengganti radar berputar) */
    .cam-icon-wrap {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 2px solid #1a1a1a;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #ffffff;
        z-index: 1;
        flex-shrink: 0;
    }}

    .cam-status-text {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        font-weight: 600;
        color: #555555;
        text-align: center;
        line-height: 1.5;
        z-index: 1;
        background: #ffffff;
        padding: 4px 12px;
        border: 1.5px solid #cccccc;
    }}
    .cam-status-text span {{
        color: #1a1a1a;
        font-weight: 700;
    }}
    .cam-hint {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: #888888;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 500;
        text-align: center;
        z-index: 1;
    }}

    /* STATUS BAR */
    .cam-statusbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 2px solid #1a1a1a;
        border-top: none;
        padding: 10px 18px;
    }}
    .cam-statusbar-item {{
        display: flex;
        align-items: center;
        gap: 7px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 500;
        color: #888888;
    }}
    .cam-statusbar-item.ready {{ color: #1a1a1a; font-weight: 700; }}
    .status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #cccccc;
        border: 1.5px solid #888888;
    }}
    .status-dot.ready {{
        background: #1a1a1a;
        border-color: #1a1a1a;
    }}

    /* STAT CARDS */
    .stat-card {{
        background: #ffffff;
        padding: 18px;
        border: 2px solid #1a1a1a;
        margin-bottom: 12px;
        position: relative;
    }}
    .stat-card::before {{
        content: '';
        position: absolute;
        top: 4px; left: 4px; right: -4px; bottom: -4px;
        background: #1a1a1a;
        z-index: -1;
    }}
    .stat-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1.5px solid #e0e0e0;
    }}
    .stat-model-name {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 15px;
        font-weight: 700;
        color: #1a1a1a;
    }}
    .stat-badge {{
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 1.5px;
        padding: 3px 9px;
        text-transform: uppercase;
        font-family: 'IBM Plex Mono', monospace;
        border: 1.5px solid #1a1a1a;
        background: #f0f0f0;
        color: #1a1a1a;
    }}
    .metric-row {{ display: flex; gap: 10px; }}
    .metric-box {{
        flex: 1;
        background: #f5f5f5;
        border: 1.5px solid #cccccc;
        padding: 12px;
        text-align: center;
    }}
    .metric-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        display: block;
        line-height: 1;
        margin-bottom: 5px;
        color: #1a1a1a;
    }}
    .metric-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        font-weight: 600;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}

    /* INFO BOX */
    .info-box {{
        background: #ffffff;
        border: 2px solid #1a1a1a;
        padding: 18px 20px;
        position: relative;
    }}
    .info-box::before {{
        content: '';
        position: absolute;
        top: 4px; left: 4px; right: -4px; bottom: -4px;
        background: #888888;
        z-index: -1;
    }}
    .info-box::after {{
        content: '[ SEAGRASS ]';
        position: absolute;
        bottom: 10px; right: 14px;
        font-size: 20px;
        opacity: 0.08;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 900;
        color: #1a1a1a;
        letter-spacing: 2px;
    }}
    .info-box-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        color: #1a1a1a;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 7px;
        padding-bottom: 8px;
        border-bottom: 1.5px dashed #cccccc;
    }}
    .info-box-body {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 12px;
        color: #444444;
        line-height: 1.7;
    }}

    /* ══════════════════════════════════
       ENCYCLOPEDIA TAB STYLES
    ══════════════════════════════════ */

    /* Ekologi paragraph block */
    .enc-section {{
        background: #ffffff;
        border: 2px solid #1a1a1a;
        padding: 20px 22px;
        margin-bottom: 18px;
        position: relative;
    }}
    .enc-section::before {{
        content: '';
        position: absolute;
        top: 4px; left: 4px; right: -4px; bottom: -4px;
        background: #888888;
        z-index: -1;
    }}
    .enc-section-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        color: #1a1a1a;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1.5px dashed #cccccc;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .enc-section-body {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 12.5px;
        color: #444444;
        line-height: 1.8;
    }}

    /* Eco fact pills row */
    .enc-facts-row {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 14px;
    }}
    .enc-fact-pill {{
        background: #f0f0f0;
        border: 1.5px solid #1a1a1a;
        padding: 8px 14px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: 0.5px;
        display: flex;
        flex-direction: column;
        gap: 2px;
        flex: 1;
        min-width: 100px;
    }}
    .enc-fact-val {{
        font-size: 22px;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 3px;
    }}
    .enc-fact-label {{
        font-size: 9px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }}

    /* Species grid */
    .species-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 16px;
        margin-top: 4px;
    }}

    /* Species card */
    .species-card {{
        background: #ffffff;
        border: 2px solid #1a1a1a;
        position: relative;
        overflow: hidden;
    }}
    .species-card::before {{
        content: '';
        position: absolute;
        top: 4px; left: 4px; right: -4px; bottom: -4px;
        background: #1a1a1a;
        z-index: -1;
    }}
    .species-card-img {{
        width: 100%;
        height: 130px;
        background: #f0f0f0;
        border-bottom: 2px solid #1a1a1a;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }}
    /* Crosshatch pattern for image placeholder */
    .species-card-img::before {{
        content: '';
        position: absolute;
        inset: 0;
        background-image:
            repeating-linear-gradient(
                45deg,
                #d8d8d8 0px, #d8d8d8 1px,
                transparent 1px, transparent 14px
            ),
            repeating-linear-gradient(
                -45deg,
                #d8d8d8 0px, #d8d8d8 1px,
                transparent 1px, transparent 14px
            );
    }}
    .species-card-img-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        color: #888888;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: #ffffff;
        border: 1px solid #cccccc;
        padding: 4px 10px;
        z-index: 1;
        position: relative;
    }}
    .species-card-num {{
        position: absolute;
        top: 8px; right: 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: rgba(0,0,0,0.08);
        line-height: 1;
        z-index: 1;
    }}
    .species-card-body {{
        padding: 14px 16px 16px;
    }}
    .species-name-sci {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        color: #1a1a1a;
        font-style: italic;
        margin-bottom: 2px;
    }}
    .species-name-tag {{
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #888888;
        background: #f0f0f0;
        border: 1px solid #cccccc;
        padding: 2px 7px;
        margin-bottom: 10px;
    }}
    .species-divider {{
        height: 1px;
        background: #e0e0e0;
        margin: 10px 0;
    }}
    .morpho-row {{
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-bottom: 10px;
    }}
    .morpho-item {{
        display: flex;
        gap: 8px;
        align-items: flex-start;
    }}
    .morpho-key {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        color: #888888;
        text-transform: uppercase;
        min-width: 56px;
        flex-shrink: 0;
        padding-top: 2px;
        letter-spacing: 0.5px;
    }}
    .morpho-val {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 11.5px;
        color: #444444;
        line-height: 1.55;
    }}
    .detection-cue {{
        background: #f5f5f5;
        border: 1.5px solid #cccccc;
        border-left: 3px solid #1a1a1a;
        padding: 8px 10px;
        margin-top: 10px;
    }}
    .detection-cue-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        color: #1a1a1a;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 3px;
    }}
    .detection-cue-text {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 11px;
        color: #555555;
        line-height: 1.5;
    }}

    /* Section divider label */
    .section-divider {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 20px 0 16px;
    }}
    .section-divider-line {{
        flex: 1;
        height: 1.5px;
        background: #cccccc;
    }}
    .section-divider-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        font-weight: 700;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        white-space: nowrap;
    }}

    /* MOBILE RESPONSIVE */
    @media (max-width: 768px) {{
        div.block-container {{ padding: 0.4rem 0.6rem !important; }}
        .nav-container {{ padding: 10px 14px; margin-bottom: 0; }}
        .nav-badge {{ font-size: 8px; padding: 2px 8px; }}

        .tab-btn {{
            padding: 9px 6px;
            font-size: 9px;
            letter-spacing: 0.3px;
            gap: 4px;
        }}
        .tab-btn svg {{
            width: 10px !important;
            height: 10px !important;
            flex-shrink: 0;
        }}

        .cam-placeholder {{ min-height: 150px; }}
        .cam-icon-wrap {{ width: 56px; height: 56px; }}
        .cam-status-text {{ font-size: 12px; }}
        .cam-statusbar {{ padding: 6px 10px; }}
        .cam-statusbar-item {{ font-size: 10px; }}

        .stat-card {{ padding: 12px; margin-bottom: 10px; }}
        .stat-model-name {{ font-size: 13px; }}
        .metric-val {{ font-size: 18px; }}
        .metric-label {{ font-size: 8px; }}

        .info-box {{ padding: 12px 14px; }}

        div[data-testid="stButton"] > button {{
            min-height: 55px !important;
            padding: 12px 16px !important;
        }}
        div[data-testid="stButton"] > button::before {{
            font-size: 10px;
            line-height: 1.4;
        }}
        div[data-testid="stButton"] > button::after {{
            font-size: 8px;
            padding-left: 10px;
            margin-left: 10px;
        }}

        .species-grid {{ grid-template-columns: 1fr; }}
        .enc-facts-row {{ gap: 8px; }}
        .enc-fact-pill {{ min-width: 80px; }}
    }}

    /* Samsung S8+ */
    @media (max-width: 414px) and (min-height: 736px) {{
        .cam-placeholder {{ min-height: 160px; }}
        .cam-icon-wrap {{ width: 62px; height: 62px; }}
        div[data-testid="stButton"] > button {{ min-height: 60px !important; }}
        div[data-testid="stButton"] > button::before {{ font-size: 11px; }}
        div[data-testid="stButton"] > button::after {{ font-size: 9px; }}

        .tab-btn {{
            padding: 9px 5px;
            font-size: 8.5px;
            letter-spacing: 0.2px;
            gap: 3px;
        }}
        .tab-btn svg {{
            width: 10px !important;
            height: 10px !important;
        }}
    }}

    /* Ultra Compact */
    @media (max-width: 375px) {{
        div.block-container {{ padding: 0.3rem 0.5rem !important; }}
        .stat-card {{ padding: 10px; }}
        .metric-box {{ padding: 6px 3px; }}
        .metric-val {{ font-size: 16px; }}

        .tab-btn {{
            padding: 8px 4px;
            font-size: 8px;
            letter-spacing: 0px;
            gap: 2px;
        }}
        .tab-btn svg {{
            width: 9px !important;
            height: 9px !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

# --- NAVBAR ---
st.markdown("""
    <div class="nav-container">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="background:#1a1a1a;padding:9px;line-height:1;display:flex;align-items:center;justify-content:center;width:42px;height:42px;">
                <svg xmlns="http://www.w3.org/2000/svg" fill="#ffffff" width="24px" height="24px" version="1.1" viewBox="144 144 512 512">
                    <path d="m505.14 580.4c59.309 0 111.45 22.957 141.55 57.691h-493.58c30.094-34.719 82.254-57.691 141.55-57.691 13.242 0 26.117 1.1484 38.465 3.3086-26.754-41.805-28.145-119.6-27.523-160.88 0.95312-64.07-23.426-108.04-23.426-108.04s47.098 28.445 57.133 59.277c10.035 30.848-11.715 76.496 1.4219 139.84 11.562 55.727 52.16 86.094 61.773 92.605 2.9766-1.707 6.0469-3.3555 9.1758-4.9258l14.223-16.898c17.383-9.6719 32.195-24 44.285-39.66 7.0898-9.1758 14.359-20.164 19.723-30.816 16.836-11.246 49.652-30.59 82.828-35.746 0 0-57.133 36.82-69.555 68.605-4.3086 11.004-12.559 23.562-23.215 34.734 8.207-0.92188 16.609-1.4062 25.164-1.4062zm51.844-50.664c11.094 0 20.086 8.9922 20.086 20.086 0 11.078-8.9922 20.07-20.086 20.07-11.078 0-20.07-8.9922-20.07-20.07 0-11.094 8.9922-20.086 20.07-20.086zm31.559-197.92c11.094 0 20.086 8.9766 20.086 20.07s-8.9922 20.086-20.086 20.086c-11.078 0-20.07-8.9922-20.07-20.086s8.9922-20.07 20.07-20.07zm38.254-92.758c11.078 0 20.07 8.9766 20.07 20.07s-8.9922 20.086-20.07 20.086c-11.094 0-20.07-8.9922-20.07-20.086s8.9766-20.07 20.07-20.07zm-346.74 10.156c11.078 0 20.07 8.9922 20.07 20.07 0 11.094-8.9922 20.086-20.07 20.086-11.094 0-20.086-8.9922-20.086-20.086 0-11.078 8.9922-20.07 20.086-20.07zm-38.844 289.73c11.078 0 20.07 8.9766 20.07 20.07 0 11.078-8.9922 20.07-20.07 20.07-11.094 0-20.086-8.9922-20.086-20.07 0-11.094 8.9922-20.07 20.086-20.07zm-67.895-78.883c11.094 0 20.07 8.9766 20.07 20.07s-8.9766 20.07-20.07 20.07-20.086-8.9766-20.086-20.07 8.9922-20.07 20.086-20.07zm336.51-42.547c3.0391-2.5547 6.0156-4.5352 8.9023-5.7422 24.863-10.52 94.18-0.95312 94.18-0.95312s-80.559 18.895-98.242 56.648c-4.4727 9.5508-11.395 23.578-19.832 39.16h-1.3594c5.1836-12.031 9.2656-25.996 11.957-38.406 3.5977-16.582 5.3516-33.75 4.3828-50.707zm-80.953 80.801c8.0273-10.246 17.352-25.711 19.816-44.48 4.5508-34.656-7.1641-105.42-6.9375-134.35 0.24219-28.914 18.895-59.52 43.742-77.93 0 0-21.75 86.773-1.6758 142.71 20.07 55.938-9.793 118.56-9.793 118.56s-17.441 37.395-48.609 59.398c0.83203-13.664 1.6016-27.344 2.3125-41.02 0.39453-7.6016 0.78516-15.25 1.1484-22.898zm-138.48 16.34c-25.695-12.227-80.832-52.312-73.078-114.14 0 0 24.379 58.328 61.199 78.398 3.1602 1.7227 6.0625 3.2812 8.7344 4.6836 0.69531 10.188 1.707 20.645 3.1602 31.059zm205.21-167.71c5.8203-8.3125 10.988-16.777 15.023-25.09 24.379-50.195 40.645-106.13 40.645-106.13s5.6211 93.801-43.785 182.37c-1.3008-6.8906-3.1133-13.68-5.5-20.328-3.2344-9.0234-5.2734-19.559-6.3789-30.816zm-66.473 75.57c0.21094-44.133-0.98438-85.531-12.211-115.96-13.859-37.527-23.426-114.98-14.812-144.63 0 0-28.945 39.402-35.91 95.781 35.898 47.672 57.965 103.14 62.953 164.79zm-48.562 13.723s11.637 68.863-6.5312 111.35c4.7617 8.2383 10.445 15.992 16.93 23.172 3.8398 4.2461 8.707 9.0977 13.77 13.359 2.3594-36.684 6.1953-101.72 6.1953-134.59 0-98.441-44.812-180.24-118.82-240.73 0 0 76.254 136.5 88.434 227.44zm-115.61-109.47c4.4297 8.0547 8.1914 18.695 10.883 27.402 5.8945 19.043 9.25 39.012 10.004 58.977-33.797-30.004-73.258-84.551-58.508-161.92 12.453 26.164 23.684 50.391 37.621 75.559z" fill-rule="evenodd"/>
                </svg>
            </div>
            <div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:17px;font-weight:700;color:#1a1a1a;letter-spacing:-0.5px;">
                    Seagrass<span style="color:#555555;">Live</span>
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#888888;font-weight:500;letter-spacing:2.5px;text-transform:uppercase;">
                    Real-Time Comparative System
                </div>
            </div>
        </div>
        <div class="nav-badge">● Live</div>
    </div>
    """, unsafe_allow_html=True)

# --- TAB NAVIGATION ---
st.markdown(f"""
    <div class="tab-nav">
        <a class="tab-btn {tab_det_cls}" href="?tab=detection" target="_self">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5"
                 stroke-linecap="round" stroke-linejoin="round">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                <circle cx="12" cy="13" r="4"/>
            </svg>
            [ 01 ] Kamera Deteksi
        </a>
        <a class="tab-btn {tab_enc_cls}" href="?tab=encyclopedia" target="_self">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5"
                 stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
            </svg>
            [ 02 ] Ensiklopedia Lamun
        </a>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_yolo():
    try:
        return YOLO('best.pt')
    except Exception as e:
        print(f"[YOLO] Gagal dimuat: {e}")
        return None

@st.cache_resource
def load_efficientdet():
    try:
        from backbone import EfficientDetBackbone
        import re as _re, collections as _col

        raw_sd = torch.load('best_efficientdet_2species.pth', map_location='cpu')
        # PENTING: ratios & scales HARUS SAMA dengan saat training (dari Colab)
        # Tanpa ini anchor box berbeda → posisi bbox kacau total
        model = EfficientDetBackbone(
            compound_coef=0,
            num_classes=2,
            ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
            scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        )

        PAT = r'\.conv\.(weight|bias)'   # akan diganti di sini

        ckpt_lookup = {}
        for k, v in raw_sd.items():
            stripped = _re.sub(r'\.conv\.(weight|bias)', r'.\1', k)
            ckpt_lookup[stripped] = v
            ckpt_lookup[k] = v        # fallback key asli juga disimpan

        model_sd = model.state_dict()
        new_sd   = _col.OrderedDict()
        loaded, missing = 0, []

        for pname, ptensor in model_sd.items():
            if pname in ckpt_lookup:
                ct = ckpt_lookup[pname]
                if ptensor.shape == ct.shape:
                    new_sd[pname] = ct
                    loaded += 1
                else:
                    new_sd[pname] = ptensor
                    print(f"[EffDet] shape mismatch: {pname}")
            else:
                new_sd[pname] = ptensor
                missing.append(pname)

        model.load_state_dict(new_sd, strict=True)
        print(f"[EfficientDet] Loaded {loaded}/{len(model_sd)} | Missing: {len(missing)}")
        if missing:
            print(f"[EfficientDet] Missing contoh: {missing[:3]}")
        model.eval()
        print("[EfficientDet] Siap ✓")
        return model

    except Exception as e:
        print(f"[EfficientDet] Gagal: {e}")
        return None


yolo_model   = load_yolo()
effdet_model = load_efficientdet()

# ============================================================
# HELPER: Gambar bounding box EfficientDet ke frame BGR
# ============================================================
def draw_effdet_boxes(img_bgr, rois, class_ids, scores, species_names,
                      color=(0, 255, 255), thickness=2):
    h, w = img_bgr.shape[:2]

    for i in range(len(rois)):
        box = rois[i].astype(float)
        a, b, c, d = box[0], box[1], box[2], box[3]

        x1 = int(np.clip(min(a, c), 0, w - 1))
        y1 = int(np.clip(min(b, d), 0, h - 1))
        x2 = int(np.clip(max(a, c), 1, w))
        y2 = int(np.clip(max(b, d), 1, h))

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        cls_idx = int(class_ids[i])
        label   = species_names[cls_idx] if cls_idx < len(species_names) else f"cls{cls_idx}"
        conf    = float(scores[i])
        text    = f"{label} {conf:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        label_x1 = x1
        label_y1 = y1 - text_h - 8
        label_x2 = x1 + text_w + 4
        label_y2 = y1

        if label_y1 < 0:
            label_y1 = y2
            label_y2 = y2 + text_h + 8

        cv2.rectangle(img_bgr, (label_x1, label_y1), (label_x2, label_y2),
                      (255, 255, 255), cv2.FILLED)

        text_y = label_y2 - 4 if label_y1 < y1 else label_y1 + text_h + 3
        cv2.putText(img_bgr, text, (label_x1 + 2, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 0, 0), 1, cv2.LINE_AA)

    return img_bgr

# ============================================================
# DUAL VIDEO PROCESSOR
# ============================================================
class DualSeagrassProcessor(VideoProcessorBase):
    SPECIES = ['Cymodocea rotundata', 'Enhalus acoroides']

    def __init__(self):
        self.yolo   = load_yolo()
        self.effdet = load_efficientdet()

        self.regressBoxes  = None
        self.clipBoxes     = None
        self._preprocess   = None
        self._postprocess  = None
        self._invert_affine = None

        if self.effdet is not None:
            try:
                from efficientdet.utils import BBoxTransform, ClipBoxes
                self.regressBoxes = BBoxTransform()
                self.clipBoxes    = ClipBoxes()
                print("[EffDet] BBoxTransform/ClipBoxes: OK")
            except Exception as e:
                print(f"[EffDet] WARNING: BBoxTransform gagal: {e}")

            try:
                from utils.utils import preprocess_video, postprocess, invert_affine
                self._preprocess    = preprocess_video
                self._postprocess   = postprocess
                self._invert_affine = invert_affine
                print("[EffDet] preprocess_video/postprocess/invert_affine: OK")
            except Exception as e:
                print(f"[EffDet] WARNING: utils import gagal: {e}")

            if self.regressBoxes is None:
                print("[EffDet] FATAL: BBoxTransform None")
            if self._preprocess is None:
                print("[EffDet] FATAL: preprocess_video None")

        self._lock = threading.Lock()

        self._yolo_conf = 0
        self._yolo_fps  = 0
        self._eff_conf  = 0
        self._eff_fps   = 0

        self._yolo_annotated = None
        self._eff_rois       = None
        self._eff_class_ids  = None
        self._eff_scores     = None

        self.frame_count = 0
        self.frame_skip  = 2

    @property
    def conf(self):
        with self._lock: return self._yolo_conf
    @property
    def fps(self):
        with self._lock: return self._yolo_fps
    @property
    def eff_conf(self):
        with self._lock: return self._eff_conf
    @property
    def eff_fps(self):
        with self._lock: return self._eff_fps

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.frame_count += 1

        if self.frame_count % self.frame_skip == 0:

            if self.yolo is not None:
                t0 = time.perf_counter()
                results = self.yolo(img_rgb, conf=0.4, verbose=False)
                t1 = time.perf_counter()

                elapsed    = max(t1 - t0, 1e-6)
                boxes      = results[0].boxes
                new_conf   = round(boxes.conf.mean().item() * 100) if len(boxes) > 0 else 0
                new_fps    = round(1 / elapsed)

                annotated_rgb = results[0].plot()
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                with self._lock:
                    self._yolo_conf      = new_conf
                    self._yolo_fps       = new_fps
                    self._yolo_annotated = annotated_bgr

            if (self.effdet is not None
                    and self.regressBoxes is not None
                    and self._postprocess is not None):
                try:
                    t0 = time.perf_counter()

                    gray_brightness = float(cv2.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))[0])
                    if gray_brightness < 40.0:
                        print(f"[EffDet] SKIP — frame terlalu gelap (brightness={gray_brightness:.1f})")
                        with self._lock:
                            self._eff_rois      = None
                            self._eff_class_ids = None
                            self._eff_scores    = None
                            self._eff_conf      = 0
                        t1 = time.perf_counter()
                        elapsed = max(t1 - t0, 1e-6)
                        with self._lock:
                            self._eff_fps = round(1 / elapsed)
                    else:
                        MAX_SIZE = 512
                        old_h, old_w = img_bgr.shape[:2]
                        scale = MAX_SIZE / max(old_h, old_w)
                        new_w = int(old_w * scale)
                        new_h = int(old_h * scale)

                        img_resized = cv2.resize(img_bgr, (new_w, new_h))
                        mean_bgr = np.array([0.406, 0.456, 0.485], dtype=np.float32)
                        std_bgr  = np.array([0.225, 0.224, 0.229], dtype=np.float32)
                        img_norm = (img_resized.astype(np.float32) / 255.0 - mean_bgr) / std_bgr

                        canvas = np.zeros((MAX_SIZE, MAX_SIZE, 3), dtype=np.float32)
                        canvas[:new_h, :new_w] = img_norm
                        x = torch.from_numpy(canvas).unsqueeze(0).permute(0, 3, 1, 2)

                        framed_meta = (new_w, new_h, old_w, old_h,
                                       MAX_SIZE - new_w, MAX_SIZE - new_h)

                        with torch.no_grad():
                            features, regression, classification, anchors = self.effdet(x)
                            out = self._postprocess(
                                x, anchors, regression, classification,
                                self.regressBoxes, self.clipBoxes,
                                threshold=0.2, iou_threshold=0.2
                            )

                        out = self._invert_affine([framed_meta], out)

                        t1 = time.perf_counter()
                        elapsed = max(t1 - t0, 1e-6)

                        results_eff = out[0] if len(out) > 0 else {}
                        rois      = results_eff.get('rois',      np.array([]))
                        class_ids = results_eff.get('class_ids', np.array([]))
                        scores    = results_eff.get('scores',    np.array([]))

                        if not hasattr(rois, '__len__') or len(rois) == 0:
                            rois = class_ids = scores = np.array([])

                        if len(scores) > 0:
                            scores = scores.astype(np.float32)
                            if np.any(scores > 1.0) or np.any(scores < 0.0):
                                scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -88, 88)))

                        CONF_THRESHOLD = 0.55
                        if len(scores) > 0:
                            keep      = scores >= CONF_THRESHOLD
                            rois      = rois[keep]
                            class_ids = class_ids[keep]
                            scores    = scores[keep]

                        if len(rois) > 0:
                            h_frame, w_frame = img_bgr.shape[:2]
                            valid = []
                            for idx, r in enumerate(rois):
                                bw = abs(float(r[2]) - float(r[0]))
                                bh = abs(float(r[3]) - float(r[1]))
                                if bw >= 20 and bh >= 20 and bw < w_frame * 0.9 and bh < h_frame * 0.9:
                                    valid.append(idx)
                            if valid:
                                rois      = rois[valid]
                                class_ids = class_ids[valid]
                                scores    = scores[valid]
                            else:
                                rois = class_ids = scores = np.array([])

                        if len(rois) > 1:
                            try:
                                boxes_xywh = [[float(min(r[0],r[2])), float(min(r[1],r[3])),
                                               float(abs(r[2]-r[0])), float(abs(r[3]-r[1]))]
                                              for r in rois]
                                nms_idx = cv2.dnn.NMSBoxes(
                                    boxes_xywh, scores.tolist(),
                                    score_threshold=CONF_THRESHOLD,
                                    nms_threshold=0.35
                                )
                                if len(nms_idx) > 0:
                                    nms_idx   = nms_idx.flatten()
                                    rois      = rois[nms_idx]
                                    class_ids = class_ids[nms_idx]
                                    scores    = scores[nms_idx]
                                else:
                                    rois = class_ids = scores = np.array([])
                            except Exception as nms_err:
                                print(f"[EffDet] NMS error: {nms_err}")

                        MAX_DET = 8
                        if len(scores) > MAX_DET:
                            top_idx   = np.argsort(scores)[::-1][:MAX_DET]
                            rois      = rois[top_idx]
                            class_ids = class_ids[top_idx]
                            scores    = scores[top_idx]

                        print(f"[EffDet] rois={len(rois)} bright={gray_brightness:.0f} "
                              f"scores={np.round(scores, 2) if len(scores) > 0 else []}")

                        new_eff_conf = round(float(np.mean(scores)) * 100) if len(scores) > 0 else 0
                        new_eff_fps  = round(1 / elapsed)

                        with self._lock:
                            self._eff_conf      = new_eff_conf
                            self._eff_fps       = new_eff_fps
                            self._eff_rois      = rois.copy()      if len(rois) > 0 else None
                            self._eff_class_ids = class_ids.copy() if len(class_ids) > 0 else None
                            self._eff_scores    = scores.copy()    if len(scores) > 0 else None

                except Exception as eff_err:
                    print(f"[EffDet ERROR] {type(eff_err).__name__}: {eff_err}")

        # ── RENDER ──
        with self._lock:
            yolo_frame    = self._yolo_annotated
            eff_rois      = self._eff_rois
            eff_class_ids = self._eff_class_ids
            eff_scores    = self._eff_scores

        if yolo_frame is not None:
            if yolo_frame.shape[:2] != img_bgr.shape[:2]:
                out = cv2.resize(yolo_frame, (img_bgr.shape[1], img_bgr.shape[0]))
            else:
                out = yolo_frame.copy()
        else:
            out = img_bgr.copy()

        if (eff_rois is not None
                and eff_class_ids is not None
                and eff_scores is not None
                and len(eff_rois) > 0):
            out = draw_effdet_boxes(
                out,
                eff_rois, eff_class_ids, eff_scores,
                self.SPECIES,
                color=(0, 255, 255),
                thickness=2
            )

        with self._lock:
            current_fps = self._yolo_fps

        if current_fps > 0:
            fps_text = f"FPS: {current_fps}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_color = (0, 255, 0)
            bg_color = (0, 0, 0)

            (text_w, text_h), baseline = cv2.getTextSize(
                fps_text, font, font_scale, font_thickness
            )

            margin = 8
            x = out.shape[1] - text_w - margin - 6
            y = text_h + margin + 2

            cv2.rectangle(out, (x - 3, y - text_h - 3),
                          (x + text_w + 3, y + baseline + 1), bg_color, cv2.FILLED)
            cv2.rectangle(out, (x - 3, y - text_h - 3),
                          (x + text_w + 3, y + baseline + 1), text_color, 1)
            cv2.putText(out, fps_text, (x, y), font, font_scale,
                        text_color, font_thickness, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

# ============================================================
# STUN CONFIG
# ============================================================
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ============================================================
# ── TAB 1: RADAR DETEKSI ──
# ============================================================
if active_tab == 'detection':

    ctx = None
    if is_running:
        ctx = webrtc_streamer(
            key="seagrass-camera",
            video_processor_factory=DualSeagrassProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": {
                    "width":  {"ideal": 640},
                    "height": {"ideal": 480},
                    "facingMode": "environment",
                },
                "audio": False,
            },
            async_processing=True,
        )
    else:
        # ── CAMERA PLACEHOLDER dengan ikon kamera statis (bukan radar berputar) ──
        st.markdown("""
            <div class="cam-placeholder">
                <div class="cam-icon-wrap">
                    <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36"
                         viewBox="0 0 24 24" fill="none" stroke="#1a1a1a"
                         stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8
                                 a2 2 0 0 1 2-2h4l2-3h6l2 3h4
                                 a2 2 0 0 1 2 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                    </svg>
                </div>
                <div>
                    <div class="cam-status-text">Kamera <span>Tidak Aktif</span></div>
                    <div class="cam-hint" style="margin-top:6px;">Aktifkan tombol untuk memulai</div>
                </div>
            </div>
            <div class="cam-statusbar">
                <div class="cam-statusbar-item ready">
                    <div class="status-dot ready"></div>Kamera Siap
                </div>
                <div class="cam-statusbar-item">
                    <div class="status-dot"></div>Deteksi Nonaktif
                </div>
                <div class="cam-statusbar-item">[ 6 Spesies Target ]</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("KAMERA", key='camera_btn'):
        st.session_state.is_running = not st.session_state.is_running
        st.rerun()

    st.markdown("""
        <div class="info-box">
            <div class="info-box-title">// Lokasi Penelitian</div>
            <div class="info-box-body">
                Perairan <strong style="color:#1a1a1a;">Desa Pengudang</strong>, Pulau Bintan.
                Sistem ini mendeteksi dan mengidentifikasi
                <strong style="color:#1a1a1a;">6 spesies lamun</strong> utama sebagai
                indikator kondisi ekosistem padang lamun.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if is_running:
        st_autorefresh(interval=1000, limit=None, key="live_refresh")


# ============================================================
# ── TAB 2: ENSIKLOPEDIA LAMUN ──
# ============================================================
elif active_tab == 'encyclopedia':

    # ── INFO EKOSISTEM ──
    st.markdown("""
        <div class="enc-section">
            <div class="enc-section-title">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2.5"
                     stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                // 01 — Informasi Ekosistem Lamun
            </div>
            <div class="enc-section-body">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
                Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. 
            </div>
            <div class="enc-facts-row">
                <div class="enc-fact-pill">
                    <span class="enc-fact-val">14</span>
                    <span class="enc-fact-label">Spesies di Indonesia</span>
                </div>
                <div class="enc-fact-pill">
                    <span class="enc-fact-val">11</span>
                    <span class="enc-fact-label">Spesies di Kep. Riau</span>
                </div>
                <div class="enc-fact-pill">
                    <span class="enc-fact-val">9–10</span>
                    <span class="enc-fact-label">Spesies di P. Bintan</span>
                </div>
                <div class="enc-fact-pill">
                    <span class="enc-fact-val">6</span>
                    <span class="enc-fact-label">Target Deteksi Sistem</span>
                </div>
            </div>
        </div>

        <div class="enc-section">
            <div class="enc-section-title">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2.5"
                     stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                    <polyline points="9 22 9 12 15 12 15 22"/>
                </svg>
                // 02 — Peran Ekologis &amp; Fungsi Habitat
            </div>
            <div class="enc-section-body">
                Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit
                anim id est laborum. 
                Curabitur pretium tincidunt lacus. Nulla gravida orci a odio, et tempus feugiat.
                Nullam varius turpis molestie risus auctor gravida. 
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── KATALOG SPESIES ──
    st.markdown("""
        <div class="section-divider">
            <div class="section-divider-line"></div>
            <div class="section-divider-label">Katalog Spesies · 6 Target Deteksi</div>
            <div class="section-divider-line"></div>
        </div>
        """, unsafe_allow_html=True)

    # Data spesies
    species_list = [
        {
            "num": "01",
            "sci": "Enhalus acoroides",
            "tag": "Hydrocharitaceae",
            "daun": "Berbentuk pita, panjang 30–150 cm (hingga 200 cm), lebar 1–2 cm. Ujung daun tumpul, tepi daun menebal, urat daun memanjang yang jelas.",
            "batang": "Pendek dan tegak lurus.",
            "rimpang": "Tebal (ø 1–1,5 cm), tertutup rambut kaku berserat tebal seperti ijuk.",
            "ciri": "Secara visual merupakan lamun dengan dimensi helaian daun paling panjang dan lebar.",
        },
        {
            "num": "02",
            "sci": "Cymodocea rotundata",
            "tag": "Potamogetonaceae",
            "daun": "Pita linear dan pipih, panjang 7–15 cm, lebar 0,2–0,4 cm. Ujung daun membulat halus sempurna tanpa gerigi.",
            "batang": "Pendek, dengan selubung daun yang menyelimuti batang.",
            "rimpang": "Halus, silindris, menjalar rata menembus substrat.",
            "ciri": "Daun lurus pipih rata tanpa corak khusus, sering membaur halus dengan latar belakang pasir.",
        },
        {
            "num": "03",
            "sci": "Thalassia hemprichii",
            "tag": "Hydrocharitaceae",
            "daun": "Pita melengkung, panjang 10–40 cm, lebar 0,4–1 cm. Ujung tumpul, sering terlihat bercak garis-garis coklat (sel tannin).",
            "batang": "Tumbuh tegak dan pendek.",
            "rimpang": "Tebal, tertutup bekas daun berbentuk segitiga.",
            "ciri": "Helai daun tumbuh rapat menyerupai bentuk sabit melengkung yang rimbun.",
        },
        {
            "num": "04",
            "sci": "Syringodium isoetifolium",
            "tag": "Potamogetonaceae",
            "daun": "Berbentuk silindris (menyerupai lidi), menyempit ke ujung. Panjang 7–30 cm, diameter 1–2 mm.",
            "batang": "Berbuku-buku, setiap nodus memiliki 2–3 helaian daun silindris.",
            "rimpang": "Halus, silindris, dan menjalar kuat.",
            "ciri": "Bentuk silindrisnya menghasilkan fitur garis spasial yang sangat berbeda dari lamun pipih.",
        },
        {
            "num": "05",
            "sci": "Halophila sp.",
            "tag": "Hydrocharitaceae",
            "daun": "Berbentuk oval, tumbuh berpasangan. Panjang 1–4 cm, lebar 0,5–2 cm. Tulang daun menyirip 10–25 pasang.",
            "batang": "Sangat tipis dan memanjang.",
            "rimpang": "Tipis, rapuh, menjalar tepat di bawah permukaan pasir.",
            "ciri": "Daun oval sangat kecil menempel substrat — small object detection yang menantang.",
        },
        {
            "num": "06",
            "sci": "Halodule uninervis",
            "tag": "Potamogetonaceae",
            "daun": "Pita linear, panjang 6–15 cm, lebar 0,25–3,5 mm. Ciri utama: ujung daun memiliki tiga gigi (tridentate).",
            "batang": "Tumbuh tegak pada rimpang, berbuku-buku.",
            "rimpang": "Tipis menjalar dengan nodus yang jelas.",
            "ciri": "Helai daun sangat sempit dan tipis menyerupai rumput daratan — ujung tridentate menjadi penciri utama.",
        },
    ]

    # Render grid 3 kolom
    cols = st.columns(3)
    for idx, sp in enumerate(species_list):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="species-card">
                    <div class="species-card-img">
                        <span class="species-card-img-label">[ Gambar Lamun ]</span>
                        <span class="species-card-num">{sp['num']}</span>
                    </div>
                    <div class="species-card-body">
                        <div class="species-name-sci">{sp['sci']}</div>
                        <div class="species-name-tag">{sp['tag']}</div>
                        <div class="species-divider"></div>
                        <div class="morpho-row">
                            <div class="morpho-item">
                                <span class="morpho-key">Daun</span>
                                <span class="morpho-val">{sp['daun']}</span>
                            </div>
                            <div class="morpho-item">
                                <span class="morpho-key">Batang</span>
                                <span class="morpho-val">{sp['batang']}</span>
                            </div>
                            <div class="morpho-item">
                                <span class="morpho-key">Rimpang</span>
                                <span class="morpho-val">{sp['rimpang']}</span>
                            </div>
                        </div>
                        <div class="detection-cue">
                            <div class="detection-cue-label">// Ciri Deteksi</div>
                            <div class="detection-cue-text">{sp['ciri']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)