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

is_running = st.session_state.is_running

dot_icon     = "●" if is_running else "○"
btn_title    = "⏹  Radar Aktif — Deteksi Berjalan"  if is_running else "▶  Aktifkan Radar Lamun"
btn_sub      = "Klik untuk menonaktifkan kamera"      if is_running else "Klik untuk memulai deteksi real-time"
status_color = "#1a1a1a" if is_running else "#888888"
status_text  = "AKTIF"   if is_running else "NONAKTIF"
status_bg    = "#e0e0e0" if is_running else "#f0f0f0"

# --- CSS WIREFRAME HITAM PUTIH ---
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
        margin-bottom: 24px;
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

    /* CAMERA PLACEHOLDER */
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
    /* Grid dots pattern untuk wireframe feel */
    .cam-placeholder::before {{
        content: '';
        position: absolute;
        inset: 0;
        background-image: radial-gradient(circle, #cccccc 1px, transparent 1px);
        background-size: 20px 20px;
        opacity: 0.6;
    }}
    /* Corner markers wireframe */
    .cam-placeholder::after {{
        content: '';
        position: absolute;
        inset: 10px;
        border: 1px dashed #999999;
        pointer-events: none;
    }}

    /* Radar ring wireframe */
    .cam-radar-ring {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 2px solid #1a1a1a;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        animation: radar-spin 3s linear infinite;
        background: #ffffff;
        z-index: 1;
    }}
    .cam-radar-ring::before {{
        content: '';
        position: absolute;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        border: 1.5px solid #888888;
    }}
    .cam-radar-ring::after {{
        content: '';
        position: absolute;
        width: 2px;
        height: 36px;
        background: linear-gradient(180deg, #1a1a1a, transparent);
        transform-origin: bottom center;
        bottom: 50%;
    }}
    @keyframes radar-spin {{
        from {{ transform: rotate(0deg); }}
        to   {{ transform: rotate(360deg); }}
    }}

    .cam-status-text {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 14px;
        font-weight: 600;
        color: #555555;
        text-align: center;
        line-height: 1.5;
        padding: 0 20px;
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

    /* MOBILE RESPONSIVE */
    @media (max-width: 768px) {{
        div.block-container {{ padding: 0.4rem 0.6rem !important; }}
        .nav-container {{ padding: 10px 14px; margin-bottom: 15px; }}
        .nav-badge {{ font-size: 8px; padding: 2px 8px; }}

        .cam-placeholder {{ min-height: 150px; }}
        .cam-radar-ring {{ width: 50px; height: 50px; }}
        .cam-radar-ring::before {{ width: 38px; height: 38px; }}
        .cam-radar-ring::after {{ height: 25px; }}
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
    }}

    /* Samsung S8+ */
    @media (max-width: 414px) and (min-height: 736px) {{
        .cam-placeholder {{ min-height: 160px; }}
        .cam-radar-ring {{ width: 60px; height: 60px; }}
        div[data-testid="stButton"] > button {{
            min-height: 60px !important;
        }}
        div[data-testid="stButton"] > button::before {{
            font-size: 11px;
        }}
        div[data-testid="stButton"] > button::after {{
            font-size: 9px;
        }}
    }}

    /* Ultra Compact */
    @media (max-width: 375px) {{
        div.block-container {{ padding: 0.3rem 0.5rem !important; }}
        .stat-card {{ padding: 10px; }}
        .metric-box {{ padding: 6px 3px; }}
        .metric-val {{ font-size: 16px; }}
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
        model  = EfficientDetBackbone(compound_coef=0, num_classes=2)

        # Checkpoint disimpan dengan Conv2dStaticSamePadding wrapper:
        # key checkpoint : "xxx.conv.weight"  /  "xxx.conv.bias"
        # key model      : "xxx.weight"       /  "xxx.bias"
        # Strip ".conv" sebelum ".weight"/".bias" dari checkpoint key
        # PENTING: pakai string biasa (bukan raw) agar escape benar
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
#   - Warna ORANYE (0, 165, 255) agar beda dari YOLO
#   - Label: nama spesies + confidence (mirroring format YOLO .plot())
# ============================================================
def draw_effdet_boxes(img_bgr, rois, class_ids, scores, species_names,
                      color=(0, 165, 255), thickness=2):
    """
    Gambar bounding box EfficientDet pada img_bgr (in-place).
    Mendukung format koordinat [y1,x1,y2,x2] maupun [x1,y1,x2,y2].
    rois      : array [N, 4]
    class_ids : array [N] int
    scores    : array [N] float  (0–1)
    """
    h, w = img_bgr.shape[:2]

    for i in range(len(rois)):
        box = rois[i].astype(float)
        a, b, c, d = box[0], box[1], box[2], box[3]

        # Heuristik: Yet-Another-EfficientDet output [x1,y1,x2,y2]
        # Namun beberapa fork output [y1,x1,y2,x2].
        # Deteksi otomatis: jika a > c atau b > d kemungkinan [y1,x1,y2,x2]
        # Cara paling aman: gunakan min/max agar selalu valid
        x1 = int(np.clip(min(a, c), 0, w - 1))
        y1 = int(np.clip(min(b, d), 0, h - 1))
        x2 = int(np.clip(max(a, c), 0, w - 1))
        y2 = int(np.clip(max(b, d), 0, h - 1))

        # Skip box yang terlalu kecil (noise)
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue

        cls_idx = int(class_ids[i])
        label   = species_names[cls_idx] if cls_idx < len(species_names) else f"cls{cls_idx}"
        conf    = float(scores[i])
        text    = f"{label} {conf:.2f}"

        # Kotak
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        # Background label
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_y = max(y1, th + 6)
        cv2.rectangle(
            img_bgr,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + baseline),
            color, cv2.FILLED
        )
        cv2.putText(
            img_bgr, text,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0),   # teks hitam di atas box oranye
            1, cv2.LINE_AA
        )
    return img_bgr

# ============================================================
# DUAL VIDEO PROCESSOR
# Logika YOLO mengikuti app_lamun_stream.py (results[0].plot())
# ============================================================
class DualSeagrassProcessor(VideoProcessorBase):
    # Nama spesies EfficientDet — sesuaikan urutan dengan label di dataset kamu
    SPECIES = ['Enhalus acoroides', 'Cymodocea rotundata']

    def __init__(self):
        self.yolo   = load_yolo()
        self.effdet = load_efficientdet()

        # Inisialisasi preprocess/postprocess/invert_affine & BBoxTransform/ClipBoxes
        # Coba beberapa path import yang umum dipakai di berbagai fork repo
        self.regressBoxes  = None
        self.clipBoxes     = None
        self._preprocess   = None
        self._postprocess  = None
        self._invert_affine = None

        if self.effdet is not None:
            try:
                # BBoxTransform & ClipBoxes ada di efficientdet/utils.py
                from efficientdet.utils import BBoxTransform, ClipBoxes
                self.regressBoxes = BBoxTransform()
                self.clipBoxes    = ClipBoxes()
                print("[EffDet] BBoxTransform/ClipBoxes: OK")
            except Exception as e:
                print(f"[EffDet] WARNING: BBoxTransform gagal: {e}")

            try:
                # PENTING: gunakan preprocess_video, BUKAN preprocess
                # preprocess() hanya terima file path string (cv2.imread)
                # preprocess_video() terima numpy array langsung dari frame kamera
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

        # Statistik YOLO
        self._yolo_conf = 0
        self._yolo_fps  = 0

        # Statistik EfficientDet
        self._eff_conf = 0
        self._eff_fps  = 0

        # Cache hasil inferensi (untuk frame-skip)
        self._yolo_annotated = None   # frame BGR sudah dianotasi YOLO
        self._eff_rois       = None   # array rois
        self._eff_class_ids  = None
        self._eff_scores     = None

        self.frame_count = 0
        self.frame_skip  = 5          # inferensi setiap 5 frame

    # ---- getter thread-safe ----
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

    # ---- inference ----
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.frame_count += 1

        # ================================================================
        # TAHAP INFERENSI — hanya setiap frame_skip frame
        # ================================================================
        if self.frame_count % self.frame_skip == 0:

            # ---- 1. YOLOv8 (persis seperti app_lamun_stream.py) ----
            if self.yolo is not None:
                t0 = time.perf_counter()
                results = self.yolo(img_rgb, conf=0.4, verbose=False)
                t1 = time.perf_counter()

                elapsed    = max(t1 - t0, 1e-6)
                boxes      = results[0].boxes
                new_conf   = round(boxes.conf.mean().item() * 100) if len(boxes) > 0 else 0
                new_fps    = round(1 / elapsed)

                # results[0].plot() sudah berisi label spesies + confidence bawaan YOLO
                # Output RGB → convert ke BGR untuk rendering
                annotated_rgb = results[0].plot()
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                with self._lock:
                    self._yolo_conf     = new_conf
                    self._yolo_fps      = new_fps
                    self._yolo_annotated = annotated_bgr  # simpan cache

            # ---- 2. EfficientDet-D0 ----
            if (self.effdet is not None
                    and self.regressBoxes is not None
                    and self._postprocess is not None):
                try:
                    t0 = time.perf_counter()

                    # ── GUARD: Skip frame gelap — tidak ada yang perlu dideteksi ──
                    # Hitung mean brightness frame grayscale (0–255)
                    # Jika rata-rata < 40 → frame terlalu gelap, bersihkan cache & skip
                    gray_brightness = float(cv2.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))[0])
                    if gray_brightness < 40.0:
                        print(f"[EffDet] SKIP — frame terlalu gelap (brightness={gray_brightness:.1f})")
                        with self._lock:
                            self._eff_rois      = None
                            self._eff_class_ids = None
                            self._eff_scores    = None
                            self._eff_conf      = 0
                        # tetap hitung FPS dari elapsed nanti
                        t1 = time.perf_counter()
                        elapsed = max(t1 - t0, 1e-6)
                        with self._lock:
                            self._eff_fps = round(1 / elapsed)
                    else:
                        # ── Preprocessing ────────────────────────────────────────
                        MAX_SIZE = 512
                        mean_img = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std_img  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

                        old_h, old_w = img_bgr.shape[:2]
                        scale = MAX_SIZE / max(old_h, old_w)
                        new_w = int(old_w * scale)
                        new_h = int(old_h * scale)

                        img_rgb_eff  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        img_resized  = cv2.resize(img_rgb_eff, (new_w, new_h)).astype(np.float32) / 255.0
                        img_norm     = (img_resized - mean_img) / std_img

                        canvas = np.zeros((MAX_SIZE, MAX_SIZE, 3), dtype=np.float32)
                        canvas[:new_h, :new_w] = img_norm
                        x = torch.from_numpy(canvas.copy()).unsqueeze(0).permute(0, 3, 1, 2)

                        # ── Inferensi ─────────────────────────────────────────────
                        # Pakai threshold rendah (0.1) karena kita akan filter manual
                        # SETELAH sigmoid diterapkan sendiri. Ini lebih aman karena
                        # beberapa fork tidak apply sigmoid di dalam postprocess.
                        with torch.no_grad():
                            features, regression, classification, anchors = self.effdet(x)
                            out = self._postprocess(
                                x, anchors, regression, classification,
                                self.regressBoxes, self.clipBoxes,
                                threshold=0.1, iou_threshold=0.5
                            )

                        # ── invert_affine ─────────────────────────────────────────
                        padding_w = MAX_SIZE - new_w
                        padding_h = MAX_SIZE - new_h
                        metas = [(new_w, new_h, old_w, old_h, padding_w, padding_h)]
                        out   = self._invert_affine(metas, out)

                        results_eff = out[0] if len(out) > 0 else {}

                        t1 = time.perf_counter()
                        elapsed = max(t1 - t0, 1e-6)

                        # ── Ekstrak hasil ─────────────────────────────────────────
                        rois      = results_eff.get('rois',      np.array(()))
                        class_ids = results_eff.get('class_ids', np.array(()))
                        scores    = results_eff.get('scores',    np.array(()))

                        if rois.size == 0:
                            rois = class_ids = scores = np.array([])

                        # ── FIX UTAMA 1: Normalisasi scores ke range 0–1 ─────────
                        # Beberapa fork Yet-Another-EfficientDet mengembalikan scores
                        # berupa LOGIT (belum sigmoid), bukan probabilitas.
                        # Cirinya: nilai > 1.0 atau < 0.0 → pasti logit.
                        # Solusi: selalu apply sigmoid agar scores jadi 0–1 seperti
                        # output YOLO (misal 0.65, 0.72, bukan 100%).
                        if len(scores) > 0:
                            scores = scores.astype(np.float32)
                            if np.any(scores > 1.0) or np.any(scores < 0.0):
                                scores = 1.0 / (1.0 + np.exp(-scores.clip(-88, 88)))
                                print("[EffDet] Sigmoid applied — scores jadi probabilitas")

                        # ── FIX UTAMA 2: Filter confidence seperti YOLO ──────────
                        # YOLO default conf=0.4 → kita pakai 0.35 agar bbox mudah
                        # muncul tapi juga tidak noise
                        CONF_THRESHOLD = 0.35
                        if len(scores) > 0:
                            keep      = scores >= CONF_THRESHOLD
                            rois      = rois[keep]
                            class_ids = class_ids[keep]
                            scores    = scores[keep]

                        # ── FIX UTAMA 3: NMS ulang di skala frame asli ───────────
                        # Fork ini melakukan NMS di skala 512px, bukan skala asli.
                        # Setelah invert_affine, bbox yang tadinya tidak overlap
                        # bisa jadi overlap banyak → perlu NMS ulang.
                        # cv2.dnn.NMSBoxes = NMS per-kelas seperti YOLO.
                        if len(rois) > 1:
                            try:
                                boxes_xywh = []
                                for r in rois:
                                    bx1 = float(min(r[0], r[2]))
                                    by1 = float(min(r[1], r[3]))
                                    bw  = float(abs(r[2] - r[0]))
                                    bh  = float(abs(r[3] - r[1]))
                                    boxes_xywh.append([bx1, by1, bw, bh])
                                nms_idx = cv2.dnn.NMSBoxes(
                                    boxes_xywh,
                                    scores.tolist(),
                                    score_threshold=CONF_THRESHOLD,
                                    nms_threshold=0.45
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

                        # ── Batasi maks deteksi ───────────────────────────────────
                        MAX_DET = 20
                        if len(scores) > MAX_DET:
                            top_idx   = np.argsort(scores)[::-1][:MAX_DET]
                            rois      = rois[top_idx]
                            class_ids = class_ids[top_idx]
                            scores    = scores[top_idx]

                        print(f"[EffDet] FINAL rois={len(rois)} brightness={gray_brightness:.1f} "
                              f"scores={np.round(scores[:3],2) if len(scores)>0 else []}")

                        # Confidence = rata-rata score (bukan 100%)
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

        # ================================================================
        # TAHAP RENDER — gunakan cache hasil inferensi terakhir
        # ================================================================

        # A. Mulai dari frame YOLO yang sudah dianotasi (cache)
        with self._lock:
            yolo_frame     = self._yolo_annotated
            eff_rois       = self._eff_rois
            eff_class_ids  = self._eff_class_ids
            eff_scores     = self._eff_scores

        if yolo_frame is not None:
            # Pastikan ukuran sama dengan frame saat ini (jika kamera resize)
            if yolo_frame.shape[:2] != img_bgr.shape[:2]:
                out = cv2.resize(yolo_frame, (img_bgr.shape[1], img_bgr.shape[0]))
            else:
                out = yolo_frame.copy()
        else:
            out = img_bgr.copy()

        # B. Overlay bounding box EfficientDet (ORANYE) di atas frame YOLO
        if (eff_rois is not None
                and eff_class_ids is not None
                and eff_scores is not None
                and len(eff_rois) > 0):
            out = draw_effdet_boxes(
                out,
                eff_rois, eff_class_ids, eff_scores,
                self.SPECIES,
                color=(0, 165, 255),  # BGR oranye
                thickness=2
            )

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
# LAYOUT UTAMA
# ============================================================
ctx = None
col_cam, col_stat = st.columns([1.8, 1])

with col_cam:
    if is_running:
        ctx = webrtc_streamer(
            key="seagrass-radar",
            video_processor_factory=DualSeagrassProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": {
                    "width":  {"ideal": 1280},
                    "height": {"ideal": 720},
                    "facingMode": "environment",
                },
                "audio": False,
            },
            async_processing=True,
        )
    else:
        st.markdown("""
            <div class="cam-placeholder">
                <div class="cam-radar-ring"></div>
                <div>
                    <div class="cam-status-text">Radar <span>Tidak Aktif</span></div>
                    <div class="cam-hint" style="margin-top:6px;">Aktifkan toggle untuk memulai</div>
                </div>
            </div>
            <div class="cam-statusbar">
                <div class="cam-statusbar-item ready"><div class="status-dot ready"></div>Kamera Siap</div>
                <div class="cam-statusbar-item"><div class="status-dot"></div>Radar Nonaktif</div>
                <div class="cam-statusbar-item">[ 6 Spesies Target ]</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("RADAR", key='radar_btn'):
        st.session_state.is_running = not st.session_state.is_running
        st.rerun()

with col_stat:
    yolo_placeholder = st.empty()
    eff_placeholder  = st.empty()

    if not is_running:
        yolo_placeholder.markdown("""
            <div class="stat-card">
                <div class="stat-card-header">
                    <span class="stat-model-name">YOLOv8</span>
                    <span class="stat-badge">Standby</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <span class="metric-val" style="color:#888888;">—</span>
                        <span class="metric-label">Confidence</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-val" style="color:#888888;">—</span>
                        <span class="metric-label">FPS Rate</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        eff_placeholder.markdown("""
            <div class="stat-card">
                <div class="stat-card-header">
                    <span class="stat-model-name">EfficientDet-D0</span>
                    <span class="stat-badge">Standby</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <span class="metric-val" style="color:#888888;">—</span>
                        <span class="metric-label">Confidence</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-val" style="color:#888888;">—</span>
                        <span class="metric-label">FPS Rate</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    else:
        yolo_conf = 0
        yolo_fps  = 0
        eff_conf  = 0
        eff_fps   = 0
        try:
            if ctx and ctx.video_processor:
                yolo_conf = ctx.video_processor.conf
                yolo_fps  = ctx.video_processor.fps
                eff_conf  = ctx.video_processor.eff_conf
                eff_fps   = ctx.video_processor.eff_fps
        except Exception:
            pass

        yolo_placeholder.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-header">
                    <span class="stat-model-name">YOLOv8</span>
                    <span class="stat-badge" style="background:#1a1a1a;color:#ffffff;">⚡ Live</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <span class="metric-val">{yolo_conf}%</span>
                        <span class="metric-label">Confidence</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-val">{yolo_fps}</span>
                        <span class="metric-label">FPS Rate</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        eff_placeholder.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-header">
                    <span class="stat-model-name">EfficientDet-D0</span>
                    <span class="stat-badge" style="background:#1a1a1a;color:#ffffff;">⚡ Live</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <span class="metric-val">{eff_conf}%</span>
                        <span class="metric-label">Confidence</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-val">{eff_fps}</span>
                        <span class="metric-label">FPS Rate</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <div class="info-box-title">// Lokasi Penelitian</div>
            <div class="info-box-body">
                Perairan <strong style="color:#1a1a1a;">Desa Pengudang</strong>, Pulau Bintan.
                Sistem ini mendeteksi dan mengidentifikasi
                <strong style="color:#1a1a1a;">6 spesies lamun</span> utama sebagai
                indikator kondisi ekosistem padang lamun.
            </div>
        </div>
        """, unsafe_allow_html=True)

if is_running:
    st_autorefresh(interval=1000, limit=None, key="live_refresh")