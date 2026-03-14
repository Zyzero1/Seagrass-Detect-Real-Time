import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SeagrassLive Pro", page_icon="🌊", layout="wide")

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
    .main, div.block-container, .stApp {{
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
        font-size: 13px;
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
        font-size: 9px;
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

    /* WEBRTC — sembunyikan tombol START/STOP bawaan, styling video */
    .stWebRtcStreamer div[class*="style__mediaPlayer"] {{
        border-radius: 20px !important;
        overflow: hidden !important;
        border: 1.5px solid rgba(6,182,212,0.2) !important;
    }}
    .stWebRtcStreamer video {{
        width: 100% !important;
        border-radius: 20px !important;
        background: #0a1628 !important;
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
    .stat-model-name {{ font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:700; color:#e2e8f0; }}
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
    .info-coord {{
        display:inline-flex; align-items:center; gap:5px;
        background:rgba(255,255,255,0.08); border-radius:8px; padding:4px 10px;
        font-size:10px; font-weight:600; color:#93c5fd; margin-top:10px;
        font-family:'Space Grotesk',sans-serif; letter-spacing:0.5px;
    }}

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
        .cam-status-text {{ font-size:11px; }}
        .cam-hint {{ font-size:9px; }}
        .cam-statusbar {{ padding:6px 10px; border-radius:10px; margin-top:8px; }}
        .cam-statusbar-item {{ font-size:9px; gap:5px; }}
        .status-dot {{ width:5px; height:5px; }}
        
        .stat-card {{ padding:12px; border-radius:14px; margin-bottom:10px; }}
        .stat-card-header {{ margin-bottom:10px; }}
        .stat-model-name {{ font-size:11px; }}
        .stat-badge {{ font-size:7px; padding:2px 6px; }}
        .metric-row {{ gap:8px; }}
        .metric-box {{ padding:8px 4px; border-radius:10px; }}
        .metric-val {{ font-size:15px; }}
        .metric-label {{ font-size:7px; }}
        
        .info-box {{ padding:14px 16px; border-radius:14px; }}
        .info-box-title {{ font-size:10px; gap:5px; }}
        .info-box-body {{ font-size:10px; }}
        .info-coord {{ font-size:8px; padding:3px 8px; }}
        
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
        .cam-status-text {{ font-size:12px; }}
        .cam-hint {{ font-size:10px; }}
        .cam-statusbar {{ padding:6px 10px; border-radius:10px; margin-top:8px; }}
        .cam-statusbar-item {{ font-size:9px; gap:5px; }}
        .status-dot {{ width:6px; height:6px; }}

        .stat-card {{ padding:12px; border-radius:14px; margin-bottom:10px; }}
        .stat-card-header {{ margin-bottom:10px; }}
        .stat-model-name {{ font-size:12px; }}
        .stat-badge {{ font-size:8px; padding:2px 6px; }}
        .metric-row {{ gap:8px; }}
        .metric-box {{ padding:8px 4px; border-radius:10px; }}
        .metric-val {{ font-size:16px; }}
        .metric-label {{ font-size:8px; }}

        .info-box {{ padding:14px 16px; border-radius:14px; }}
        .info-box-title {{ font-size:11px; gap:5px; }}
        .info-box-body {{ font-size:11px; }}
        .info-coord {{ font-size:9px; padding:3px 8px; }}

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
        .metric-val {{ font-size:14px; }}
        .info-box {{ padding:12px 14px; border-radius:12px; }}
        div[data-testid="stButton"] > button {{
            min-height: 55px !important;
            padding: 14px 30px !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

# --- NAVBAR ---
st.markdown("""
    <div class="nav-container">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="background:linear-gradient(135deg,#06b6d4,#3b82f6);padding:9px;border-radius:12px;font-size:20px;line-height:1;">🌊</div>
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

# --- YOLO VIDEO PROCESSOR untuk webrtc ---
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_models()
        self.conf  = 0
        self.fps   = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img     = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t1 = time.time()
        if self.model:
            results   = self.model(img_rgb, conf=0.4, verbose=False)
            elapsed   = max(time.time() - t1, 1e-6)
            self.fps  = round(1 / elapsed)
            self.conf = round(
                results[0].boxes.conf.mean().item() * 100
                if len(results[0].boxes) > 0 else 0
            )
            annotated = results[0].plot()
            out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        else:
            out = img

        return av.VideoFrame.from_ndarray(out, format="bgr24")

# STUN server agar bisa diakses dari luar jaringan lokal
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# --- LAYOUT UTAMA ---
col_cam, col_stat = st.columns([1.8, 1])

with col_cam:
    # Kamera: webrtc saat aktif, placeholder saat nonaktif
    if is_running:
        ctx = webrtc_streamer(
            key="seagrass-radar",
            video_processor_factory=YOLOProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": {
                    "width":  {"ideal": 1280},
                    "height": {"ideal": 720},
                    "facingMode": "environment",  # kamera belakang HP
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

    # Tombol toggle — pindah ke bawah
    if st.button("RADAR", key='radar_btn'):
        st.session_state.is_running = not st.session_state.is_running
        st.rerun()

with col_stat:
    yolo_placeholder = st.empty()
    eff_placeholder  = st.empty()

    if not is_running:
        # Standby cards
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
        # Live cards — ambil conf & fps dari processor jika sudah tersambung
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
            <div class="info-coord">📌 1.1543° N, 104.4017° E</div>
        </div>
        """, unsafe_allow_html=True)