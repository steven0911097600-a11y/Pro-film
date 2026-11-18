import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ======================================================================
# 核心演算法：AI 光影與無損色彩引擎
# ======================================================================
class Engine:
    @staticmethod
    def smoothstep(e0, e1, x):
        """產生平滑的 S 曲線過渡，用於無損對比度調整"""
        t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def luma(img):
        return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722

    @staticmethod
    def soft_clip(img):
        """電影級高光/陰影平滑壓縮，徹底解決數值溢位與破圖色塊"""
        img = np.maximum(img, 0.0)
        # 高光平滑過渡 (Soft Shoulder)
        return np.where(img > 0.8, 0.8 + 0.2 * np.tanh((img - 0.8) / 0.2), img)

    @staticmethod
    def ai_tone_mapping(img_f32):
        """AI 智慧光影平衡：使用 CLAHE 空間自適應色調映射，不改變物體邊緣"""
        img_u8 = (np.clip(img_f32, 0.0, 1.0) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        res = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        return res.astype(np.float32) / 255.0

    @staticmethod
    def apply_clarity(img, amount):
        """Lightroom 清晰度：增強中頻細節 (微對比)"""
        if amount == 0: return img
        blur = cv2.GaussianBlur(img, (0, 0), 3.0)
        detail = img - blur
        return img + detail * amount

    @staticmethod
    def adjust_hsl_channel(img, target_hue, hue_range, sat_shift, lum_shift):
        """獨立色彩控制：針對特定色相 (如膚色橘、風景綠) 進行無損調整"""
        if sat_shift == 0 and lum_shift == 0: return img
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 計算與目標色相的距離 (環狀)
        dist = np.abs((h - target_hue + 180.0) % 360.0 - 180.0)
        mask = np.exp(-0.5 * (dist / hue_range) ** 2).astype(np.float32)
        
        s = s * (1.0 + sat_shift * mask)
        v = v * (1.0 + lum_shift * mask)
        
        hsv_new = cv2.merge([h, np.clip(s, 0.0, 1.0), np.clip(v, 0.0, 1.0)])
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)

    @staticmethod
    def three_way_color_grade(img, h_tint, m_tint, s_tint):
        """三向色調分離：高光、中階調、陰影獨立染色"""
        lum = Engine.luma(img)[..., None]
        
        # 計算權重遮罩
        w_s = np.clip(1.0 - (lum / 0.4), 0.0, 1.0)
        w_h = np.clip((lum - 0.6) / 0.4, 0.0, 1.0)
        w_m = 1.0 - w_s - w_h
        
        # 套用染色 (微量偏移)
        tinted = img.copy()
        if any(s_tint): tinted += w_s * np.array(s_tint) * 0.15
        if any(m_tint): tinted += w_m * np.array(m_tint) * 0.15
        if any(h_tint): tinted += w_h * np.array(h_tint) * 0.15
        
        # 亮度補償，確保染色不影響原本曝光
        new_lum = np.maximum(Engine.luma(tinted)[..., None], 1e-4)
        return tinted * (lum / new_lum)

# ======================================================================
# Web App 介面佈局
# ======================================================================
st.set_page_config(page_title="Pro Film Grader", layout="wide")
st.title("Pro Film Grader 雲端底片調色工具")
st.markdown("搭載 AI 智慧光影平衡、無損平滑壓縮與 Lightroom 級獨立色彩控制。")

# 側邊欄控制面板
st.sidebar.header("✨ 核心演算法")
use_ai = st.sidebar.toggle("啟用 AI 智慧光影平衡", value=False, help="自動偵測明暗分佈，利用空間映射技術還原背光與過曝細節。")

st.sidebar.header("1. 底片預設組")
preset_options = ["無 (Original)", "富士經典底片 (Classic Film)", "富士 CLASSIC CHROME", "富士 Classic Neg.", "徠卡質感 (Leica Pop)"]
preset = st.sidebar.selectbox("選擇相機色彩", preset_options)

st.sidebar.header("2. 基礎曝光 (Basic)")
exposure = st.sidebar.slider("曝光度 Exposure", -2.0, 2.0, 0.0, 0.05)
contrast = st.sidebar.slider("對比度 Contrast", -0.5, 0.5, 0.0, 0.05)
clarity = st.sidebar.slider("清晰度 Clarity", -0.5, 1.0, 0.0, 0.05)

st.sidebar.header("3. 獨立色彩 (HSL Color Mix)")
st.sidebar.caption("精準控制特定物件顏色，不干擾全圖。")
skin_sat = st.sidebar.slider("膚色/橘色 飽和度", -1.0, 1.0, 0.0, 0.05)
skin_lum = st.sidebar.slider("膚色/橘色 明亮度", -0.5, 0.5, 0.0, 0.05)
nature_sat = st.sidebar.slider("風景/綠色 飽和度", -1.0, 1.0, 0.0, 0.05)
sky_sat = st.sidebar.slider("天空/藍色 飽和度", -1.0, 1.0, 0.0, 0.05)

st.sidebar.header("4. 三向色調分離 (Color Grading)")
shadow_tint = st.sidebar.color_picker("陰影色調 (Shadows)", "#000000")
mid_tint = st.sidebar.color_picker("中階色調 (Midtones)", "#000000")
high_tint = st.sidebar.color_picker("高光色調 (Highlights)", "#000000")

def hex_to_rgb_shift(hex_color):
    """將 Hex 轉為 RGB 偏移量 (-1.0 ~ 1.0)"""
    if hex_color == "#000000": return (0.0, 0.0, 0.0)
    h = hex_color.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
    return (rgb[0]-0.5, rgb[1]-0.5, rgb[2]-0.5)

# 主畫面
uploaded_file = st.file_uploader("📸 上傳高畫質照片 (支援 JPG, PNG, WEBP)", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    # 讀取高畫質影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 圖片特徵自動判斷
    mean_lum = np.mean(Engine.luma(img_rgb))
    
    with st.spinner('專業演算法渲染中...'):
        out = img_rgb.copy()
        
        # 1. AI 光影平衡
        if use_ai:
            out = Engine.ai_tone_mapping(out)
            
        # 2. 基礎曝光與對比
        out = out * (2.0 ** exposure)
        if contrast != 0:
            out = out + (Engine.smoothstep(0.0, 1.0, out) - out) * contrast
            
        # 3. 清晰度 (Clarity)
        out = Engine.apply_clarity(out, clarity)
        
        # 4. 底片預設組矩陣 (精簡化的高效特徵)
        if preset == "富士經典底片 (Classic Film)":
            out = Engine.adjust_hsl_channel(out, target_hue=110, hue_range=35, sat_shift=-0.1, lum_shift=0.0) # 綠色偏低飽和
            out = Engine.three_way_color_grade(out, (0.1, 0.05, 0.0), (0.0, 0.0, 0.0), (-0.1, 0.0, 0.1)) # 高光暖，陰影微青
        elif preset == "富士 CLASSIC CHROME":
            out = Engine.adjust_hsl_channel(out, target_hue=225, hue_range=30, sat_shift=-0.25, lum_shift=-0.05) # 藍色深沉
            out = Engine.adjust_hsl_channel(out, target_hue=0, hue_range=20, sat_shift=-0.1, lum_shift=0.0) # 紅色收斂
            out = Engine.three_way_color_grade(out, (0.05, 0.05, -0.05), (0.0, 0.0, 0.0), (-0.05, -0.05, 0.05))
        elif preset == "富士 Classic Neg.":
            out = Engine.adjust_hsl_channel(out, target_hue=115, hue_range=30, sat_shift=0.1, lum_shift=-0.05) # 綠色厚重
            out = Engine.three_way_color_grade(out, (0.1, 0.0, -0.1), (0.0, 0.0, 0.0), (-0.2, 0.1, 0.1)) # 強烈青綠陰影
        elif preset == "徠卡質感 (Leica Pop)":
            out = out + (Engine.smoothstep(0.0, 1.0, out) - out) * 0.15 # 強化微對比
            out = Engine.adjust_hsl_channel(out, target_hue=5, hue_range=20, sat_shift=0.2, lum_shift=-0.05) # 徠卡紅
            
        # 5. 獨立色彩微調 (HSL)
        out = Engine.adjust_hsl_channel(out, 25, 20, skin_sat, skin_lum)    # 橘色/膚色
        out = Engine.adjust_hsl_channel(out, 110, 40, nature_sat, 0.0)      # 綠色/植物
        out = Engine.adjust_hsl_channel(out, 215, 30, sky_sat, 0.0)         # 藍色/天空
        
        # 6. 三向色調分離自訂
        out = Engine.three_way_color_grade(out, hex_to_rgb_shift(high_tint), hex_to_rgb_shift(mid_tint), hex_to_rgb_shift(shadow_tint))
        
        # 7. 最終防破圖柔和壓縮 (Soft Shoulder Clipping)
        out = Engine.soft_clip(out)
        
    # UI 排版顯示對比
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("處理前")
        if mean_lum < 0.3: st.info("💡 系統偵測到這張照片偏暗，建議開啟左側的「AI 智慧光影平衡」。")
        elif mean_lum > 0.7: st.info("💡 系統偵測到這張照片偏亮，可以稍微降低「曝光度」。")
        st.image(img_rgb, use_container_width=True)
    with col2:
        st.subheader("處理後")
        st.image(out, use_container_width=True)

    # 匯出 100% 品質 JPEG
    out_u8 = (out * 255.0).astype(np.uint8)
    out_pil = Image.fromarray(out_u8)
    buf = io.BytesIO()
    out_pil.save(buf, format="JPEG", quality=100)
    
    st.download_button(
        label="📥 下載最高畫質成品",
        data=buf.getvalue(),
        file_name="pro_graded_photo.jpg",
        mime="image/jpeg",
        use_container_width=True
    )