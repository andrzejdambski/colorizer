import streamlit as st
import requests
from PIL import Image
import base64, io, os


st.set_page_config(
    page_title="CATch Me in Color — Premium Edition",
    page_icon="🐈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "show_zoom" not in st.session_state:
    st.session_state.show_zoom = False

if "zoom_image" not in st.session_state:
    st.session_state.zoom_image = None


hero_image_path = "./cool-cat-wearing-round-sunglasses-600nw-2580540819.webp"

if os.path.exists(hero_image_path):
    with open(hero_image_path, "rb") as f:
        hero_bg = base64.b64encode(f.read()).decode()
else:
    hero_bg = None


dark_mode = st.session_state.dark_mode

# Theme colors - keeping original white editorial for light mode
if dark_mode:
    bg_gradient = "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)"
    card_bg = "rgba(255, 255, 255, 0.05)"
    card_border = "rgba(255, 255, 255, 0.1)"
    text_primary = "#ffffff"
    text_secondary = "#b8b8d1"
    accent_color = "#667eea"
    glow_color = "rgba(102, 126, 234, 0.4)"
    shadow = "0 8px 32px rgba(0, 0, 0, 0.4)"
    hero_overlay = "rgba(15, 12, 41, 0.7)"
    step_bg = "rgba(255, 255, 255, 0.08)"
    uploader_bg = "rgba(255, 255, 255, 0.03)"
    uploader_border = "rgba(255, 255, 255, 0.2)"
    button_bg = "#667eea"
    button_border = "#667eea"
else:
    # Original white editorial colors
    bg_gradient = "#ffffff"
    card_bg = "#ffffff"
    card_border = "#eaeaea"
    text_primary = "#1a1a1a"
    text_secondary = "#666"
    accent_color = "#111"
    glow_color = "rgba(0, 0, 0, 0.04)"
    shadow = "0 4px 18px rgba(0,0,0,0.04)"
    hero_overlay = "rgba(255, 255, 255, 0.3)"
    step_bg = "#ffffff"
    uploader_bg = "#fafafa"
    uploader_border = "#dcdcdc"
    button_bg = "#ffffff"
    button_border = "#dcdcdc"

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;600&family=Playfair+Display:wght@400;600;700&display=swap');

    /* ===================== BACKGROUND ===================== */
    .stApp {{
        background: {bg_gradient};
        {'background-size: 400% 400%; animation: gradientShift 15s ease infinite;' if dark_mode else ''}
        font-family: 'Inter', sans-serif;
        color: {text_primary};
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* ===================== FLOATING PARTICLES (subtle in light mode) ===================== */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            radial-gradient(circle at 20% 50%, {glow_color} 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, {glow_color} 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, {glow_color} 0%, transparent 50%);
        opacity: {'0.6' if dark_mode else '0.3'};
        animation: particleFloat 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }}

    @keyframes particleFloat {{
        0%, 100% {{ transform: translate(0, 0) scale(1); }}
        33% {{ transform: translate(30px, -30px) scale(1.1); }}
        66% {{ transform: translate(-20px, 20px) scale(0.9); }}
    }}

    /* ===================== THEME TOGGLE BUTTON ===================== */
    .theme-toggle {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        background: {card_bg};
        {'backdrop-filter: blur(20px);' if dark_mode else ''}
        border: 1px solid {card_border};
        border-radius: 50px;
        padding: 12px 24px;
        cursor: pointer;
        box-shadow: {shadow};
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1.2rem;
    }}

    .theme-toggle:hover {{
        transform: scale(1.05);
        box-shadow: 0 12px 40px {glow_color};
    }}

    /* ===================== TYPOGRAPHY (original style) ===================== */
    h1, h2, h3 {{
        font-family: 'Playfair Display', serif;
        color: {accent_color} !important;
        letter-spacing: {'2px' if dark_mode else '1px'};
        font-weight: {'700' if dark_mode else '600'};
        text-transform: uppercase;
        {'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;' if dark_mode else ''}
    }}

    header[data-testid="stHeader"], footer, #MainMenu {{
        visibility: hidden;
    }}

    .block-container {{
        padding-top: 1rem !important;
        max-width: 1600px !important;
        position: relative;
        z-index: 1;
    }}

    /* ===================== HERO SECTION WITH GRADIENT ===================== */
    .hero-box {{
        background: {'url(data:image/webp;base64,' + hero_bg + ')' if hero_bg else 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)' if dark_mode else 'linear-gradient(135deg, #a8edea 0%, #fed6e3 50%, #fbc2eb 100%)'};
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: {'120px 50px' if dark_mode else '150px 40px'};
        {'border-radius: 24px;' if dark_mode else 'border-bottom: 1px solid #eaeaea;'}
        margin-bottom: {'40px' if dark_mode else '45px'};
        position: relative;
        overflow: hidden;
        {'border: 1px solid ' + card_border + ';' if dark_mode else ''}
        box-shadow: {shadow};
        animation: heroEntrance 1s cubic-bezier(0.4, 0, 0.2, 1);
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
        filter: contrast(1.1) brightness(1.05);
    }}

    @keyframes heroEntrance {{
        from {{
            opacity: 0;
            transform: translateY(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .hero-box::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {'rgba(255, 255, 255, 0.3)' if dark_mode else 'rgba(255, 255, 255, 0.5)'} 0%, transparent 70%);
        animation: heroGlow 8s ease-in-out infinite;
        opacity: 1;
    }}

    @keyframes heroGlow {{
        0%, 100% {{ transform: translate(0, 0); }}
        50% {{ transform: translate(30px, 30px); }}
    }}

    .hero-title {{
        font-size: {'4rem' if dark_mode else '3rem'};
        margin-bottom: {'15px' if dark_mode else '10px'};
        position: relative;
        z-index: 2;
        color: {'#ffffff' if dark_mode else '#1a1a1a'} !important;
        text-shadow: {'0 4px 30px rgba(0, 0, 0, 0.8), 0 2px 10px rgba(0, 0, 0, 0.9)' if dark_mode else '0 2px 20px rgba(255, 255, 255, 0.9), 0 4px 30px rgba(255, 255, 255, 0.8), 0 1px 3px rgba(0, 0, 0, 0.3)'};
        animation: titleFloat 3s ease-in-out infinite;
        font-weight: 700;
    }}

    @keyframes titleFloat {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}

    .hero-sub {{
        font-size: {'1.1rem' if dark_mode else '1rem'};
        color: {'rgba(255, 255, 255, 0.95)' if dark_mode else '#222'};
        position: relative;
        z-index: 2;
        font-weight: {'400' if dark_mode else '500'};
        letter-spacing: {'3px' if dark_mode else '2px'};
        text-shadow: {'0 2px 15px rgba(0, 0, 0, 0.7), 0 1px 5px rgba(0, 0, 0, 0.8)' if dark_mode else '0 1px 10px rgba(255, 255, 255, 0.9), 0 2px 20px rgba(255, 255, 255, 0.7)'};
    }}

    /* ===================== CARDS (original white in light mode) ===================== */
    .card {{
        background: {card_bg};
        {'backdrop-filter: blur(20px);' if dark_mode else ''}
        padding: {'30px' if dark_mode else '22px'};
        border-radius: {'20px' if dark_mode else '6px'};
        border: 1px solid {card_border};
        box-shadow: {shadow};
        margin-bottom: {'30px' if dark_mode else '22px'};
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: cardSlideIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    @keyframes cardSlideIn {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, {glow_color}, transparent);
        transition: left 0.5s;
    }}

    .card:hover::before {{
        left: 100%;
    }}

    .card:hover {{
        transform: translateY(-5px) scale(1.01);
        box-shadow: {'0 16px 48px ' + glow_color if dark_mode else '0 8px 24px rgba(0,0,0,0.08)'};
        {'border-color: ' + glow_color + ';' if dark_mode else ''}
    }}

    .card-title {{
        font-family: 'Playfair Display', serif;
        font-size: {'1.8rem' if dark_mode else '1.25rem'};
        margin-bottom: {'20px' if dark_mode else '14px'};
        text-transform: uppercase;
        letter-spacing: {'2px' if dark_mode else '1px'};
        position: relative;
        z-index: 1;
    }}

    /* ===================== STEP CARDS WITH STAGGER ===================== */
    .steps-row {{
        display: flex;
        gap: {'20px' if dark_mode else '16px'};
        flex-wrap: wrap;
    }}

    .step-card {{
        flex: 1;
        min-width: 250px;
        border: 1px solid {card_border};
        border-radius: {'16px' if dark_mode else '6px'};
        padding: {'25px' if dark_mode else '18px'};
        background: {step_bg};
        {'backdrop-filter: blur(10px);' if dark_mode else ''}
        box-shadow: {shadow};
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    .step-card:nth-child(1) {{ animation: stepSlide 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.1s backwards; }}
    .step-card:nth-child(2) {{ animation: stepSlide 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.2s backwards; }}
    .step-card:nth-child(3) {{ animation: stepSlide 0.6s cubic-bezier(0.4, 0, 0.2, 1) 0.3s backwards; }}

    @keyframes stepSlide {{
        from {{
            opacity: 0;
            transform: translateX(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}

    .step-card:hover {{
        transform: translateY(-8px) scale(1.03);
        box-shadow: {'0 12px 40px ' + glow_color if dark_mode else '0 8px 20px rgba(0,0,0,0.08)'};
        {'border-color: ' + glow_color + ';' if dark_mode else ''}
    }}

    .step-card::after {{
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {glow_color} 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s;
    }}

    .step-card:hover::after {{
        opacity: {'1' if dark_mode else '0.3'};
    }}

    .step-num-h {{
        font-family: 'Playfair Display', serif;
        font-size: {'2.5rem' if dark_mode else '1.3rem'};
        color: {accent_color};
        {'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;' if dark_mode else ''}
        margin-bottom: {'10px' if dark_mode else '6px'};
        font-weight: {'900' if dark_mode else '600'};
        position: relative;
        z-index: 1;
    }}

    .step-text-h {{
        font-size: {'1rem' if dark_mode else '0.9rem'};
        color: {text_secondary};
        line-height: {'1.6' if dark_mode else '1.35'};
        position: relative;
        z-index: 1;
    }}

    /* ===================== FILE UPLOADER ===================== */
    [data-testid="stFileUploader"] {{
        border: {'2px' if dark_mode else '1px'} dashed {uploader_border} !important;
        background: {uploader_bg} !important;
        {'backdrop-filter: blur(10px) !important;' if dark_mode else ''}
        padding: {'30px' if dark_mode else '20px'} !important;
        border-radius: {'16px' if dark_mode else '6px'} !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    [data-testid="stFileUploader"]:hover {{
        {'border-color: ' + glow_color + ' !important;' if dark_mode else ''}
        background: {card_bg} !important;
        box-shadow: {'0 8px 30px ' + glow_color if dark_mode else '0 4px 12px rgba(0,0,0,0.06)'} !important;
        transform: scale(1.01);
    }}

    [data-testid="stFileUploader"] button {{
        background: {button_bg} !important;
        color: {accent_color} !important;
        border: {'none' if dark_mode else '1px solid ' + button_border} !important;
        border-radius: {'12px' if dark_mode else '4px'} !important;
        padding: 12px 28px !important;
        font-weight: {'600' if dark_mode else '400'} !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        {'box-shadow: 0 4px 15px ' + glow_color + ' !important;' if dark_mode else ''}
    }}

    [data-testid="stFileUploader"] button:hover {{
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: {'0 8px 25px ' + glow_color if dark_mode else '0 4px 12px rgba(0,0,0,0.1)'} !important;
    }}

    /* ===================== RESULT FRAMES ===================== */
    .result-frame {{
        border: 1px solid {card_border};
        padding: {'10px' if dark_mode else '6px'};
        border-radius: {'16px' if dark_mode else '6px'};
        background: {card_bg};
        {'backdrop-filter: blur(20px);' if dark_mode else ''}
        box-shadow: {shadow};
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }}

    .result-frame::before {{
        content: '🔍 Click to zoom';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        opacity: 0;
        transition: opacity 0.3s;
        z-index: 10;
        font-size: 0.9rem;
        pointer-events: none;
    }}

    .result-frame:hover::before {{
        opacity: 1;
    }}

    .result-frame:hover {{
        transform: scale(1.02);
        box-shadow: {'0 16px 48px ' + glow_color if dark_mode else '0 8px 24px rgba(0,0,0,0.08)'};
        {'border-color: ' + glow_color + ';' if dark_mode else ''}
    }}

    .result-label {{
        text-align: center;
        font-size: {'0.9rem' if dark_mode else '0.85rem'};
        color: {text_secondary};
        margin-bottom: {'10px' if dark_mode else '6px'};
        letter-spacing: {'2px' if dark_mode else '1px'};
        text-transform: uppercase;
        font-weight: 600;
    }}

    /* ===================== DOWNLOAD BUTTON ===================== */
    .stDownloadButton button {{
        background: {button_bg} !important;
        color: {accent_color} !important;
        border-radius: {'12px' if dark_mode else '6px'} !important;
        border: {'none' if dark_mode else '1px solid ' + button_border} !important;
        padding: {'16px 40px' if dark_mode else '14px 30px'} !important;
        letter-spacing: {'2px' if dark_mode else '1px'} !important;
        font-weight: {'600' if dark_mode else '400'} !important;
        font-size: 1rem !important;
        {'box-shadow: 0 6px 20px ' + glow_color + ' !important;' if dark_mode else ''}
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }}

    .stDownloadButton button::before {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: {'rgba(255, 255, 255, 0.3)' if dark_mode else 'rgba(0, 0, 0, 0.05)'};
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }}

    .stDownloadButton button:hover::before {{
        width: 300px;
        height: 300px;
    }}

    .stDownloadButton button:hover {{
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: {'0 12px 35px ' + glow_color if dark_mode else '0 6px 18px rgba(0,0,0,0.1)'} !important;
    }}

    /* ===================== LOADING ANIMATION ===================== */
    .loading-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(20px);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}

    .loading-spinner {{
        width: 80px;
        height: 80px;
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 20px;
    }}

    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}

    .loading-text {{
        font-family: 'Playfair Display', serif;
        font-size: {'2rem' if dark_mode else '1.6rem'};
        color: {accent_color};
        {'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;' if dark_mode else ''}
        animation: pulse 1.5s ease-in-out infinite;
        font-weight: 700;
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 0.4; transform: scale(1); }}
        50% {{ opacity: 1; transform: scale(1.05); }}
    }}

    .loading-particles {{
        position: absolute;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }}

    .particle {{
        position: absolute;
        width: 4px;
        height: 4px;
        background: {glow_color};
        border-radius: 50%;
        animation: particleRise 3s ease-in infinite;
    }}

    @keyframes particleRise {{
        0% {{
            transform: translateY(100vh) scale(0);
            opacity: 0;
        }}
        50% {{
            opacity: 1;
        }}
        100% {{
            transform: translateY(-100vh) scale(1);
            opacity: 0;
        }}
    }}

    /* ===================== SUCCESS ANIMATION ===================== */
    .success-checkmark {{
        width: 80px;
        height: 80px;
        margin: 0 auto 20px;
    }}

    .check-icon {{
        width: 80px;
        height: 80px;
        position: relative;
        border-radius: 50%;
        box-sizing: content-box;
        border: 4px solid #4CAF50;
    }}

    .check-icon::before {{
        top: 3px;
        left: -2px;
        width: 30px;
        transform-origin: 100% 50%;
        border-radius: 100px 0 0 100px;
    }}

    .check-icon::after {{
        top: 0;
        left: 30px;
        width: 60px;
        transform-origin: 0 50%;
        border-radius: 0 100px 100px 0;
        animation: rotateCircle 4.25s ease-in;
    }}

    .icon-line {{
        height: 5px;
        background-color: #4CAF50;
        display: block;
        border-radius: 2px;
        position: absolute;
        z-index: 10;
    }}

    .icon-line.line-tip {{
        top: 46px;
        left: 14px;
        width: 25px;
        transform: rotate(45deg);
        animation: iconLineTip 0.75s;
    }}

    .icon-line.line-long {{
        top: 38px;
        right: 8px;
        width: 47px;
        transform: rotate(-45deg);
        animation: iconLineLong 0.75s;
    }}

    @keyframes iconLineTip {{
        0% {{ width: 0; left: 1px; top: 19px; }}
        54% {{ width: 0; left: 1px; top: 19px; }}
        70% {{ width: 50px; left: -8px; top: 37px; }}
        84% {{ width: 17px; left: 21px; top: 48px; }}
        100% {{ width: 25px; left: 14px; top: 45px; }}
    }}

    @keyframes iconLineLong {{
        0% {{ width: 0; right: 46px; top: 54px; }}
        65% {{ width: 0; right: 46px; top: 54px; }}
        84% {{ width: 55px; right: 0px; top: 35px; }}
        100% {{ width: 47px; right: 8px; top: 38px; }}
    }}

    /* ===================== ZOOM MODAL ===================== */
    .zoom-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.95);
        backdrop-filter: blur(10px);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.3s;
        cursor: pointer;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}

    .zoom-content {{
        max-width: 90%;
        max-height: 90%;
        animation: zoomIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    @keyframes zoomIn {{
        from {{
            transform: scale(0.8);
            opacity: 0;
        }}
        to {{
            transform: scale(1);
            opacity: 1;
        }}
    }}

    /* ===================== RESPONSIVE ===================== */
    @media (max-width: 768px) {{
        .hero-title {{ font-size: 2.5rem; }}
        .steps-row {{ flex-direction: column; }}
        .step-card {{ min-width: 100%; }}
        .theme-toggle {{ top: 10px; right: 10px; padding: 10px 20px; }}
    }}

</style>
""",
    unsafe_allow_html=True,
)


# THEME TOGGLE BUTTON

st.markdown(
    f"""
<div class="theme-toggle" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'toggle'}}, '*')">
    {'🌙' if not dark_mode else '☀️'}
</div>
""",
    unsafe_allow_html=True,
)

# Handle theme toggle
if st.button("", key="theme_toggle_hidden", help="Toggle theme"):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()


# API CONFIGURATION

API_BASE_URL = "https://colorizer-api-847420607839.europe-west1.run.app"
API_COLORIZE = f"{API_BASE_URL}/colorize_montage"
API_DEBUG = f"{API_BASE_URL}/colorize_montage_debug"


# API INTEGRATION


def colorize_via_api(image_file, debug_mode=False):
    """
    Send image to external colorization API and return colorized result.

    Args:
        image_file: PIL Image or file-like object
        debug_mode: If True, use debug endpoint that returns [Input L | GAN Output | Original] montage

    Returns:
        PIL Image: Colorized image (or montage if debug_mode=True)

    Raises:
        Exception: If API call fails
    """
    try:
        # Select endpoint based on mode
        api_url = API_DEBUG if debug_mode else API_COLORIZE

        # Convert PIL Image to bytes if needed
        if isinstance(image_file, Image.Image):
            buf = io.BytesIO()
            image_file.save(buf, format="PNG")
            buf.seek(0)
            files = {"file": ("image.png", buf, "image/png")}
        else:
            # If it's already a file-like object
            image_file.seek(0)
            files = {"file": ("image.png", image_file, "image/png")}

        # Make API request
        response = requests.post(api_url, files=files, timeout=30)
        response.raise_for_status()

        # Convert response to PIL Image
        colorized_image = Image.open(io.BytesIO(response.content))
        return colorized_image

    except requests.exceptions.Timeout:
        raise Exception("⏱️ API request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise Exception(
            "🌐 Cannot connect to colorization API. Please check your internet connection."
        )
    except requests.exceptions.HTTPError as e:
        raise Exception(f"❌ API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"❌ Unexpected error: {str(e)}")


def img_to_base64(img):
    """Convert PIL Image to base64 for HTML embedding"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def to_png(img):
    """Convert PIL Image to PNG bytes for download"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# HERO HEADER

st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">COLORIZER</div>
    <div class="hero-sub">#BATCH2130PARIS</div>
</div>
""",
    unsafe_allow_html=True,
)


# HOW IT WORKS — HORIZONTAL 3-STEP BAR

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">How it Works</div>', unsafe_allow_html=True)

# Create three columns for the step cards
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        """
    <div class="step-card">
        <div class="step-num-h">01</div>
        <div class="step-text-h">Upload a grayscale cat photo and watch the magic happen.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="step-card">
        <div class="step-num-h">02</div>
        <div class="step-text-h">Our advanced ML model colorizes it with stunning accuracy.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div class="step-card">
        <div class="step-num-h">03</div>
        <div class="step-text-h">Download your masterpiece in high quality PNG format.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# UPLOAD CARD

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Upload Your Photo</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop your image here or click to browse",
    type=["jpg", "jpeg", "png", "webp", "avif", "bmp", "tiff", "tif", "gif", "svg"],
    label_visibility="collapsed",
)

# Debug mode toggle
debug_mode = st.checkbox(
    "🔍 Debug Mode (Show montage: Input L-channel | GAN Output | Original)",
    value=False,
    help="Enable to see a side-by-side comparison montage showing the input L-channel, GAN colorized output, and original image",
)

st.markdown("</div>", unsafe_allow_html=True)


# RESULTS + PREMIUM LOADING
if uploaded:
    try:
        img = Image.open(uploaded)
    except Exception as e:
        st.error(
            f"""
        ❌ **Cannot open this image format.**

        The file `{uploaded.name}` appears to be in a format that requires additional support.

        **For AVIF images**, you need to install: `pip install pillow-avif-plugin`

        **Supported formats without extra plugins**: JPG, PNG, WebP, BMP, TIFF, GIF

        Error details: {str(e)}
        """
        )
        st.stop()

    overlay = st.empty()

    # Premium loading animation with particles
    import random

    particles_html = "".join(
        [
            f'<div class="particle" style="left: {random.randint(0, 100)}%; animation-delay: {random.random() * 2}s;"></div>'
            for _ in range(20)
        ]
    )

    overlay.markdown(
        f"""
    <div class="loading-overlay">
        <div class="loading-particles">
            {particles_html}
        </div>
        <div class="loading-spinner"></div>
        <div class="loading-text">Colorizing Your Cat...</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    try:
        # Call API to colorize the image
        result = colorize_via_api(uploaded, debug_mode=debug_mode)
    except Exception as e:
        overlay.empty()
        st.error(str(e))
        st.stop()

    overlay.empty()

    # Show success animation briefly
    success_placeholder = st.empty()
    success_placeholder.markdown(
        """
    <div class="loading-overlay" style="animation: fadeIn 0.3s;">
        <div class="success-checkmark">
            <div class="check-icon">
                <span class="icon-line line-tip"></span>
                <span class="icon-line line-long"></span>
            </div>
        </div>
        <div class="loading-text">Success!</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    import time

    time.sleep(1)
    success_placeholder.empty()

    # Convert images to base64 for comparison slider
    img_resized = img.resize((256, 256))  # Resize to match API output
    img_base64 = img_to_base64(img_resized)
    result_base64 = img_to_base64(result)

    # Before/After Comparison Slider
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-title">Before & After Comparison</div>',
        unsafe_allow_html=True,
    )

    comparison_html = f"""
    <div style="position: relative; width: 100%; max-width: 800px; margin: 0 auto;">
        <div id="comparison-container" style="position: relative; width: 100%; padding-top: 100%; overflow: hidden; border-radius: 12px; box-shadow: {shadow};">
            <img src="data:image/png;base64,{result_base64}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;">
            <div id="before-image" style="position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden;">
                <img src="data:image/png;base64,{img_base64}" style="width: 200%; height: 100%; object-fit: cover;">
            </div>
            <div id="slider" style="position: absolute; top: 0; left: 50%; width: 4px; height: 100%; background: white; cursor: ew-resize; box-shadow: 0 0 10px rgba(0,0,0,0.5);">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 40px; height: 40px; background: white; border-radius: 50%; box-shadow: 0 0 10px rgba(0,0,0,0.3); display: flex; align-items: center; justify-content: center; font-size: 20px;">
                    ⟷
                </div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.9rem; color: {text_secondary};">
            <span>← ORIGINAL</span>
            <span>COLORIZED →</span>
        </div>
    </div>

    <script>
        const container = document.getElementById('comparison-container');
        const slider = document.getElementById('slider');
        const beforeImage = document.getElementById('before-image');
        let isDragging = false;

        function updateSlider(e) {{
            const rect = container.getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));

            slider.style.left = percentage + '%';
            beforeImage.style.width = percentage + '%';
        }}

        slider.addEventListener('mousedown', () => isDragging = true);
        document.addEventListener('mouseup', () => isDragging = false);
        document.addEventListener('mousemove', (e) => {{
            if (isDragging) updateSlider(e);
        }});

        slider.addEventListener('touchstart', () => isDragging = true);
        document.addEventListener('touchend', () => isDragging = false);
        document.addEventListener('touchmove', (e) => {{
            if (isDragging) updateSlider(e);
        }});

        container.addEventListener('click', updateSlider);
    </script>
    """

    st.components.v1.html(comparison_html, height=650)
    st.markdown("</div>", unsafe_allow_html=True)

    # Individual results with zoom
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Results Gallery</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="result-frame">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Original</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-frame">', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-label">Colorized </div>', unsafe_allow_html=True
        )
        st.image(result, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Download button
    st.download_button(
        "✨ Download Photo",
        data=to_png(result),
        file_name="COLORIZED_PHOTO.png",
        mime="image/png",
    )
