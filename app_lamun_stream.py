import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
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

dot_icon     = "🟢" if is_running else "🔴"
btn_title    = "⏹  Radar Aktif — Deteksi Berjalan"  if is_running else "▶  Aktifkan Radar Lamun"
btn_sub      = "Klik untuk menonaktifkan kamera"      if is_running else "Klik untuk memulai deteksi real-time"
status_color = "#06b6d4" if is_running else "#ef4444"
status_text  = "AKTIF"   if is_running else "NONAKTIF"

# --- CSS (sama persis dengan versi kamu, hanya tambah styling webrtc) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;900&family=Inter:wght@300;400;500&display=swap');

    header {{visibility: hidden;}}

    /* ← TAMBAH DI SINI */
    div[data-testid="stElementContainer"]:has(iframe[title="streamlit_autorefresh.st_autorefresh"]) {{
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }}

  div.block-container, .stApp {{
        background-color: #0a1628 !important;
        padding-top: 0px !important;
    }}
    div.block-container {{ padding: 1rem 1.5rem !important; max-width: 100% !important; }}

    /* NAVBAR */
    .nav-container {{
        background: linear-gradient(135deg, #0d1f3c, #0a1628);
        padding: 14px 28px; border-radius: 20px;
        display: flex; justify-content: space-between; align-items: center;
        box-shadow: 0 4px 30px rgba(0,0,0,0.4);
        border: 1px solid rgba(6,182,212,0.2);
        margin-bottom: 24px; position: relative; overflow: hidden;
    }}
    .nav-container::before {{
        content:''; position:absolute; top:0; left:0; right:0; height:2px;
        background: linear-gradient(90deg,#06b6d4,#3b82f6,#06b6d4);
        background-size: 200% 100%; animation: shimmer 3s linear infinite;
    }}

    /* Hilangkan jarak atas dan bawah container webrtc */
    div[data-testid="stVerticalBlock"] > div:has(> div.stWebRtcStreamer) {{
        padding: 0 !important;
        margin: 0 !important;
    }}

    .stWebRtcStreamer > div {{
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* Paksa video mengisi penuh tanpa jarak */
    .stWebRtcStreamer video {{
        width: 100% !important;
        border-radius: 20px !important;
        background: #0a1628 !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* WEBRTC — sembunyikan tombol START/STOP bawaan, styling video */
    .stWebRtcStreamer div[class*="style__mediaPlayer"] {{
        border-radius: 20px !important;
        overflow: hidden !important;
        border: 1.5px solid rgba(6,182,212,0.2) !important;
    }}

    .stWebRtcStreamer {{
        padding: 0 !important;
        margin: 0 !important;
    }}
    
    @keyframes shimmer {{ 0%{{background-position:-200% 0}} 100%{{background-position:200% 0}} }}
    .nav-badge {{
        background: rgba(6,182,212,0.15); border: 1px solid rgba(6,182,212,0.4);
        color: #06b6d4; font-size: 10px; font-weight: 700; letter-spacing: 2px;
        padding: 4px 12px; border-radius: 20px; text-transform: uppercase;
        display: flex; align-items: center; justify-content: center;
        min-width: fit-content; white-space: nowrap;
    }}

    /* RADAR BOX = TOMBOL */
    div[data-testid="stButton"] {{
        margin-top: 10px !important;
        margin-bottom: 0 !important;
        width: 100% !important;
    }}
    div[data-testid="stButton"] > button {{
        width: 100% !important;
        min-height: 68px !important;
        background: linear-gradient(135deg, #0d2040, #0f2a50) !important;
        border: 1.5px solid rgba(6,182,212,0.35) !important;
        border-radius: 16px !important;
        padding: 14px 20px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
        cursor: pointer !important;
        transition: border-color 0.3s, box-shadow 0.3s, background 0.3s !important;
        font-size: 0 !important;
        color: transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        position: relative !important;
        overflow: visible !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        border-color: #06b6d4 !important;
        box-shadow: 0 4px 28px rgba(6,182,212,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
        background: linear-gradient(135deg, #0f2a50, #122d58) !important;
        cursor: pointer !important;
    }}
    div[data-testid="stButton"] > button:focus,
    div[data-testid="stButton"] > button:active {{
        outline: none !important;
        box-shadow: 0 4px 28px rgba(6,182,212,0.3) !important;
        background: linear-gradient(135deg, #0d2040, #0f2a50) !important;
        color: transparent !important;
        border-color: #06b6d4 !important;
    }}
    div[data-testid="stButton"] > button::before {{
        content: "{dot_icon}  {btn_title}\\A {btn_sub}";
        white-space: pre;
        font-size: 14px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #e2e8f0;
        text-align: left;
        line-height: 1.6;
        pointer-events: none;
        flex: 1;
    }}
    div[data-testid="stButton"] > button::after {{
        content: "STATUS\\A {status_text}";
        white-space: pre;
        font-size: 13px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 900;
        color: {status_color};
        text-align: center;
        line-height: 2;
        pointer-events: none;
        padding-left: 14px;
        border-left: 1px solid rgba(6,182,212,0.2);
        margin-left: 14px;
    }}

    /* CAMERA PLACEHOLDER */
    .cam-placeholder {{
        width:100%; aspect-ratio:16/9;
        background: linear-gradient(145deg,#0d1f3c,#0a1628);
        border-radius:20px; border:1.5px solid rgba(6,182,212,0.2);
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:16px; position:relative; overflow:hidden; min-height:240px;
    }}
    .cam-radar-ring {{
        width:80px; height:80px; border-radius:50%;
        border:2px solid rgba(6,182,212,0.3);
        display:flex; align-items:center; justify-content:center;
        position:relative; animation:radar-spin 3s linear infinite;
    }}
    .cam-radar-ring::before {{
        content:''; position:absolute; width:60px; height:60px;
        border-radius:50%; border:1.5px solid rgba(6,182,212,0.2);
    }}
    .cam-radar-ring::after {{
        content:''; position:absolute; width:2px; height:40px;
        background:linear-gradient(180deg,#06b6d4,transparent);
        transform-origin:bottom center; bottom:50%;
    }}
    @keyframes radar-spin {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
    .cam-status-text {{
        font-family:'Space Grotesk',sans-serif; font-size:15px; font-weight:600;
        color:#94a3b8; text-align:center; line-height:1.5; padding:0 20px;
    }}
    .cam-status-text span {{ color:#06b6d4; }}
    .cam-hint {{
        font-family:'Inter',sans-serif; font-size:11px; color:#475569;
        letter-spacing:1px; text-transform:uppercase; font-weight:500; text-align:center;
    }}
    .cam-statusbar {{
        display:flex; align-items:center; justify-content:space-between;
        background:rgba(13,31,60,0.8); border:1px solid rgba(6,182,212,0.15);
        border-radius:12px; padding:10px 18px; margin-top:10px;
    }}
    .cam-statusbar-item {{
        display:flex; align-items:center; gap:7px;
        font-family:'Inter',sans-serif; font-size:11px; font-weight:500; color:#64748b;
    }}
    .cam-statusbar-item.ready {{ color:#06b6d4; }}
    .status-dot {{ width:7px; height:7px; border-radius:50%; background:#ef4444; }}
    .status-dot.ready {{ background:#06b6d4; }}

    /* STAT CARDS */
    .stat-card {{
        background: linear-gradient(135deg,#0d1f3c,#0a1628);
        padding:20px; border-radius:20px;
        border:1px solid rgba(255,255,255,0.06);
        margin-bottom:14px; box-shadow:0 8px 30px rgba(0,0,0,0.3);
    }}
    .stat-card-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }}
    .stat-model-name {{ font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#e2e8f0; }}
    .stat-badge {{ font-size:9px; font-weight:700; letter-spacing:1.5px; padding:3px 9px; border-radius:8px; text-transform:uppercase; }}
    .metric-row {{ display:flex; gap:10px; }}
    .metric-box {{
        flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
        border-radius:12px; padding:12px; text-align:center;
    }}
    .metric-val {{ font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:900; display:block; line-height:1; margin-bottom:5px; }}
    .metric-label {{ font-family:'Inter',sans-serif; font-size:9px; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:1.5px; }}

    /* INFO BOX */
    .info-box {{
        background: linear-gradient(135deg,#0e3460,#1a4a8a);
        border:1px solid rgba(59,130,246,0.3); border-radius:20px;
        padding:20px 22px; position:relative; overflow:hidden;
        box-shadow:0 8px 30px rgba(0,0,0,0.3);
    }}
    .info-box::before {{ content:'🌊'; position:absolute; bottom:-10px; right:10px; font-size:60px; opacity:0.08; }}
    .info-box-title {{
        font-family:'Space Grotesk',sans-serif; font-size:13px; font-weight:700;
        color:#93c5fd; text-transform:uppercase; letter-spacing:1.5px;
        margin-bottom:10px; display:flex; align-items:center; gap:7px;
    }}
    .info-box-body {{ font-family:'Inter',sans-serif; font-size:12px; color:rgba(255,255,255,0.75); line-height:1.7; }}

    /* MOBILE RESPONSIVE - iPhone SE & Similar Devices */
    @media (max-width:768px) {{
        div.block-container {{ padding:0.4rem 0.6rem !important; }}
        .nav-container {{ padding:10px 14px; border-radius:12px; margin-bottom:15px; }}
        .nav-container > div:first-child {{ gap:8px; }}
        .nav-container > div:first-child > div:first-child {{ padding:5px; font-size:14px; }}
        .nav-container > div:first-child > div:last-child > div:first-child {{ font-size:14px; }}
        .nav-container > div:first-child > div:last-child > div:last-child {{ font-size:7px; }}
        .nav-badge {{ font-size:8px; padding:2px 8px; }}
        
        .cam-placeholder {{ min-height:150px; border-radius:14px; }}
        .cam-radar-ring {{ width:50px; height:50px; }}
        .cam-radar-ring::before {{ width:38px; height:38px; }}
        .cam-radar-ring::after {{ height:25px; }}
        .cam-status-text {{ font-size:13px; }}
        .cam-hint {{ font-size:9px; }}
        .cam-statusbar {{ padding:6px 10px; border-radius:10px; margin-top:8px; }}
        .cam-statusbar-item {{ font-size:10px; gap:5px; }}
        .status-dot {{ width:5px; height:5px; }}
        
        .stat-card {{ padding:12px; border-radius:14px; margin-bottom:10px; }}
        .stat-card-header {{ margin-bottom:10px; }}
        .stat-model-name {{ font-size:14px; }}
        .stat-badge {{ font-size:8px; padding:2px 6px; }}
        .metric-row {{ gap:8px; }}
        .metric-box {{ padding:8px 4px; border-radius:10px; }}
        .metric-val {{ font-size:16px; }}
        .metric-label {{ font-size:9px; }}
        
        .info-box {{ padding:14px 16px; border-radius:14px; }}
        .info-box-title {{ font-size:14px; gap:5px; }}
        .info-box-body {{ font-size:12px; }}
        
        div[data-testid="stButton"] > button {{
            min-height: 55px !important;
            padding: 14px 40px !important;
            border-radius: 12px !important;
        }}
        div[data-testid="stButton"] > button::before {{
            font-size: 10px;
            line-height: 1.3;
        }}
        div[data-testid="stButton"] > button::after {{
            font-size: 7px;
            padding-left: 10px;
            margin-left: 10px;
        }}
    }}

    /* Samsung S8+ Specific Optimizations */
    @media (max-width:414px) and (min-height:736px) {{
        div.block-container {{ padding:0.4rem 0.6rem !important; }}
        .nav-container {{ padding:10px 14px; border-radius:12px; margin-bottom:18px; }}
        .nav-container > div:first-child {{ gap:10px; }}
        .nav-container > div:first-child > div:first-child {{ padding:5px; font-size:14px !important; }}
        .nav-container > div:first-child > div:last-child > div:first-child {{ font-size:16px !important; }}
        .nav-container > div:first-child > div:last-child > div:last-child {{ font-size:8px !important; }}
        .nav-badge {{ font-size:9px; padding:3px 10px; }}

        .cam-placeholder {{ min-height:160px; border-radius:16px; }}
        .cam-radar-ring {{ width:60px; height:60px; }}
        .cam-radar-ring::before {{ width:45px; height:45px; }}
        .cam-radar-ring::after {{ height:30px; }}
        .cam-status-text {{ font-size:14px; }}
        .cam-hint {{ font-size:10px; }}
        .cam-statusbar {{ padding:6px 10px; border-radius:10px; margin-top:8px; }}
        .cam-statusbar-item {{ font-size:9px; gap:5px; }}
        .status-dot {{ width:6px; height:6px; }}

        .stat-card {{ padding:12px; border-radius:14px; margin-bottom:10px; }}
        .stat-card-header {{ margin-bottom:10px; }}
        .stat-model-name {{ font-size:14px; }}
        .stat-badge {{ font-size:10px; padding:2px 6px; }}
        .metric-row {{ gap:8px; }}
        .metric-box {{ padding:8px 4px; border-radius:10px; }}
        .metric-val {{ font-size:16px; }}
        .metric-label {{ font-size:8px; }}

        .info-box {{ padding:14px 16px; border-radius:14px; }}
        .info-box-title {{ font-size:14px; gap:5px; }}
        .info-box-body {{ font-size:12px; }}

        div[data-testid="stButton"] > button {{
            min-height: 60px !important;
            padding: 12px 30px !important;
            border-radius: 14px !important;
        }}
        div[data-testid="stButton"] > button::before {{
            font-size: 11px;
            line-height: 1.4;
        }}
        div[data-testid="stButton"] > button::after {{
            font-size: 8px;
            padding-left: 12px;
            margin-left: 12px;
        }}
    }}

    /* Ultra Compact Mobile */
    @media (max-width:375px) {{
        div.block-container {{ padding:0.3rem 0.5rem !important; }}
        .nav-container {{ padding:8px 12px; border-radius:10px; margin-bottom:15px; }}
        .stat-card {{ padding:10px; border-radius:12px; margin-bottom:8px; }}
        .metric-box {{ padding:6px 3px; }}
        .metric-val {{ font-size:16px; }}
        .info-box {{ padding:12px 14px; border-radius:12px; }}
        div[data-testid="stButton"] > button {{
            min-height: 55px !important;
            padding: 14px 33px !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

# --- NAVBAR ---
st.markdown("""
    <div class="nav-container">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="background:linear-gradient(135deg,#06b6d4,#3b82f6);padding:9px;border-radius:12px;line-height:1;display:flex;align-items:center;justify-content:center;width:42px;height:42px;">
                <svg xmlns="http://www.w3.org/2000/svg" fill="#ffffff" width="24px" height="24px" version="1.1" viewBox="144 144 512 512">
                    <path d="m505.14 580.4c59.309 0 111.45 22.957 141.55 57.691h-493.58c30.094-34.719 82.254-57.691 141.55-57.691 13.242 0 26.117 1.1484 38.465 3.3086-26.754-41.805-28.145-119.6-27.523-160.88 0.95312-64.07-23.426-108.04-23.426-108.04s47.098 28.445 57.133 59.277c10.035 30.848-11.715 76.496 1.4219 139.84 11.562 55.727 52.16 86.094 61.773 92.605 2.9766-1.707 6.0469-3.3555 9.1758-4.9258l14.223-16.898c17.383-9.6719 32.195-24 44.285-39.66 7.0898-9.1758 14.359-20.164 19.723-30.816 16.836-11.246 49.652-30.59 82.828-35.746 0 0-57.133 36.82-69.555 68.605-4.3086 11.004-12.559 23.562-23.215 34.734 8.207-0.92188 16.609-1.4062 25.164-1.4062zm51.844-50.664c11.094 0 20.086 8.9922 20.086 20.086 0 11.078-8.9922 20.07-20.086 20.07-11.078 0-20.07-8.9922-20.07-20.07 0-11.094 8.9922-20.086 20.07-20.086zm31.559-197.92c11.094 0 20.086 8.9766 20.086 20.07s-8.9922 20.086-20.086 20.086c-11.078 0-20.07-8.9922-20.07-20.086s8.9922-20.07 20.07-20.07zm38.254-92.758c11.078 0 20.07 8.9766 20.07 20.07s-8.9922 20.086-20.07 20.086c-11.094 0-20.07-8.9922-20.07-20.086s8.9766-20.07 20.07-20.07zm-346.74 10.156c11.078 0 20.07 8.9922 20.07 20.07 0 11.094-8.9922 20.086-20.07 20.086-11.094 0-20.086-8.9922-20.086-20.086 0-11.078 8.9922-20.07 20.086-20.07zm-38.844 289.73c11.078 0 20.07 8.9766 20.07 20.07 0 11.078-8.9922 20.07-20.07 20.07-11.094 0-20.086-8.9922-20.086-20.07 0-11.094 8.9922-20.07 20.086-20.07zm-67.895-78.883c11.094 0 20.07 8.9766 20.07 20.07s-8.9766 20.07-20.07 20.07-20.086-8.9766-20.086-20.07 8.9922-20.07 20.086-20.07zm336.51-42.547c3.0391-2.5547 6.0156-4.5352 8.9023-5.7422 24.863-10.52 94.18-0.95312 94.18-0.95312s-80.559 18.895-98.242 56.648c-4.4727 9.5508-11.395 23.578-19.832 39.16h-1.3594c5.1836-12.031 9.2656-25.996 11.957-38.406 3.5977-16.582 5.3516-33.75 4.3828-50.707zm-80.953 80.801c8.0273-10.246 17.352-25.711 19.816-44.48 4.5508-34.656-7.1641-105.42-6.9375-134.35 0.24219-28.914 18.895-59.52 43.742-77.93 0 0-21.75 86.773-1.6758 142.71 20.07 55.938-9.793 118.56-9.793 118.56s-17.441 37.395-48.609 59.398c0.83203-13.664 1.6016-27.344 2.3125-41.02 0.39453-7.6016 0.78516-15.25 1.1484-22.898zm-138.48 16.34c-25.695-12.227-80.832-52.312-73.078-114.14 0 0 24.379 58.328 61.199 78.398 3.1602 1.7227 6.0625 3.2812 8.7344 4.6836 0.69531 10.188 1.707 20.645 3.1602 31.059zm205.21-167.71c5.8203-8.3125 10.988-16.777 15.023-25.09 24.379-50.195 40.645-106.13 40.645-106.13s5.6211 93.801-43.785 182.37c-1.3008-6.8906-3.1133-13.68-5.5-20.328-3.2344-9.0234-5.2734-19.559-6.3789-30.816zm-66.473 75.57c0.21094-44.133-0.98438-85.531-12.211-115.96-13.859-37.527-23.426-114.98-14.812-144.63 0 0-28.945 39.402-35.91 95.781 35.898 47.672 57.965 103.14 62.953 164.79zm-48.562 13.723s11.637 68.863-6.5312 111.35c4.7617 8.2383 10.445 15.992 16.93 23.172 3.8398 4.2461 8.707 9.0977 13.77 13.359 2.3594-36.684 6.1953-101.72 6.1953-134.59 0-98.441-44.812-180.24-118.82-240.73 0 0 76.254 136.5 88.434 227.44zm-115.61-109.47c4.4297 8.0547 8.1914 18.695 10.883 27.402 5.8945 19.043 9.25 39.012 10.004 58.977-33.797-30.004-73.258-84.551-58.508-161.92 12.453 26.164 23.684 50.391 37.621 75.559z" fill-rule="evenodd"/>
                </svg>
            </div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:900;color:#e2e8f0;letter-spacing:-0.5px;">
                    Seagrass<span style="color:#06b6d4;">Live</span>
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:9px;color:#475569;font-weight:600;letter-spacing:2px;text-transform:uppercase;">
                    Real-Time Comparative System
                </div>
            </div>
        </div>
        <div class="nav-badge">● Live</div>
    </div>
    """, unsafe_allow_html=True)

# --- MODEL ---
@st.cache_resource
def load_models():
    try:
        return YOLO('best.pt')
    except:
        return None

yolo_model = load_models()

# ============================================================
# YOLO VIDEO PROCESSOR — DIPERBAIKI (tanpa st_autorefresh)
# - Thread-safe dengan threading.Lock()
# - Mengukur FPS & inferensi secara akurat (Persamaan 24 & 25)
# - Tidak menyebabkan kedap-kedip karena tidak pakai autorefresh
# ============================================================
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model  = load_models()
        self._lock  = threading.Lock()
        self._conf  = 0    # confidence rata-rata deteksi (%)
        self._fps   = 0    # FPS inferensi (Persamaan 24)
        self._infer = 0.0  # waktu inferensi ms (Persamaan 25)

    # --- getter thread-safe ---
    @property
    def conf(self):
        with self._lock:
            return self._conf

    @property
    def fps(self):
        with self._lock:
            return self._fps

    @property
    def infer_ms(self):
        with self._lock:
            return self._infer

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # ← konversi ke RGB dulu

        if self.model:
            t_start = time.perf_counter()
            results = self.model(img_rgb, conf=0.4, verbose=False)  # ← masukkan RGB
            t_end   = time.perf_counter()

            elapsed_s  = max(t_end - t_start, 1e-6)
            elapsed_ms = elapsed_s * 1000

            boxes = results[0].boxes
            new_conf  = round(boxes.conf.mean().item() * 100) if len(boxes) > 0 else 0
            new_fps   = round(1 / elapsed_s)
            new_infer = round(elapsed_ms, 2)

            with self._lock:
                self._conf  = new_conf
                self._fps   = new_fps
                self._infer = new_infer

            annotated = results[0].plot()              # ← output RGB
            out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)  # ← balik ke BGR untuk WebRTC
        else:
            out = img_bgr

        return av.VideoFrame.from_ndarray(out, format="bgr24")

# STUN server agar bisa diakses dari luar jaringan lokal
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# --- LAYOUT UTAMA ---

ctx = None
col_cam, col_stat = st.columns([1.8, 1])

with col_cam:
    if is_running:
        ctx = webrtc_streamer(
            key="seagrass-radar",
            video_processor_factory=YOLOProcessor,
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
                <div class="cam-statusbar-item">🎯 6 Spesies Target</div>
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
            <div class="stat-card" style="border-left:3px solid #3b82f6;">
                <div class="stat-card-header">
                    <span class="stat-model-name">YOLOv8</span>
                    <span class="stat-badge" style="background:rgba(59,130,246,0.15);color:#60a5fa;">Standby</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box"><span class="metric-val" style="color:#3b82f6;">—</span><span class="metric-label">Confidence</span></div>
                    <div class="metric-box"><span class="metric-val" style="color:#64748b;">—</span><span class="metric-label">FPS Rate</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

        eff_placeholder.markdown("""
            <div class="stat-card" style="border-left:3px solid #10b981;">
                <div class="stat-card-header">
                    <span class="stat-model-name">EfficientDet-D0</span>
                    <span class="stat-badge" style="background:rgba(16,185,129,0.15);color:#34d399;">Standby</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box"><span class="metric-val" style="color:#10b981;">—</span><span class="metric-label">Confidence</span></div>
                    <div class="metric-box"><span class="metric-val" style="color:#64748b;">—</span><span class="metric-label">FPS Rate</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

    else:
        # Baca langsung dari processor — tanpa autorefresh, tanpa kedap-kedip
        conf_val = 0
        fps_val  = 0
        try:
            if ctx and ctx.video_processor:
                conf_val = ctx.video_processor.conf
                fps_val  = ctx.video_processor.fps
        except:
            pass

        yolo_placeholder.markdown(f"""
            <div class="stat-card" style="border-left:3px solid #3b82f6;">
                <div class="stat-card-header">
                    <span class="stat-model-name">YOLOv8</span>
                    <span class="stat-badge" style="background:rgba(59,130,246,0.15);color:#60a5fa;">⚡ Live</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box"><span class="metric-val" style="color:#3b82f6;">{conf_val}%</span><span class="metric-label">Confidence</span></div>
                    <div class="metric-box"><span class="metric-val" style="color:#e2e8f0;">{fps_val}</span><span class="metric-label">FPS Rate</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

        eff_placeholder.markdown("""
            <div class="stat-card" style="border-left:3px solid #10b981;">
                <div class="stat-card-header">
                    <span class="stat-model-name">EfficientDet-D0</span>
                    <span class="stat-badge" style="background:rgba(16,185,129,0.15);color:#34d399;">⚡ Live</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box"><span class="metric-val" style="color:#10b981;">0%</span><span class="metric-label">Confidence</span></div>
                    <div class="metric-box"><span class="metric-val" style="color:#e2e8f0;">—</span><span class="metric-label">FPS Rate</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <div class="info-box-title">📍 Lokasi Penelitian</div>
            <div class="info-box-body">
                Perairan <strong style="color:white;">Desa Pengudang</strong>, Pulau Bintan.
                Sistem ini mendeteksi dan mengidentifikasi
                <strong style="color:#93c5fd;">6 spesies lamun</strong> utama sebagai
                indikator kondisi ekosistem padang lamun.
            </div>
        </div>
        """, unsafe_allow_html=True)

if is_running:
    st_autorefresh(interval=1000, limit=None, key="live_refresh")