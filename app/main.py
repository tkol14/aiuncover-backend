from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageChops
import io, re, json
import numpy as np
import exifread
import os
import itertools
from collections import Counter
import cv2 # Додаємо OpenCV для розширеного аналізу шуму, якщо доступно

# Якщо cv2 недоступний, використовуємо емуляцію
try:
    _ = cv2.Laplacian
except NameError:
    print("Warning: OpenCV (cv2) not found. Falling back to NumPy/PIL for image processing.")
    
# --- Налаштування FastAPI ---

app = FastAPI(title="AIUncover API", version="1.4.0 - Ultimate", debug=os.environ.get("DEBUG", "False").lower() == "true")

ALLOWED_ORIGINS = [
    "https://aiuncover.net",
    "https://www.aiuncover.net",
    "https://aiuncover-backend-production.up.railway.app",
    "https://api.aiuncover.net",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    prob_ai: float
    explanations: list[str]
    checks: dict

# --- Константи ---

AI_TOOL_HINTS = [
    "stable diffusion", "stablediffusion", "sdxl", "novelai",
    "midjourney", "dall-e", "dalle", "leonardo", "flux", "playground",
    "firefly", "adobe firefly", "cf ai", "canva ai", "ideogram", "pollinations",
    "dreamstudio", "getimg.ai", "artbreeder", "pika labs", "runway", "gen-1", "gen-2",
    "aigenerated", "ai generated"
]

COMMON_AI_SIZES = {256, 384, 448, 512, 576, 640, 704, 768, 896, 960, 1024, 1152, 1280, 1536, 1792, 2048}
UNCOMMON_RATIOS = {
    (1, 1), (4, 3), (3, 4), (16, 9), (9, 16), (3, 2), (2, 3), (5, 4), (4, 5)
}

# --- Утиліти ---

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def load_image(raw: bytes):
    bio = io.BytesIO(raw)
    img = Image.open(bio)
    img.load()
    return img

# --- Модулі Перевірок ---

def read_exif_hints(raw: bytes):
    # (Функція залишається як у попередньому варіанті)
    hints = []
    has_exif = False
    ai_tool_found = False
    try:
        tags = exifread.process_file(io.BytesIO(raw), details=False)
        if tags:
            has_exif = True
            text = " ".join([str(v) for v in tags.values()]).lower()
            if any(h in text for h in AI_TOOL_HINTS):
                hints.append("🚩 EXIF/XMP містить згадки генераторів ШІ")
                ai_tool_found = True
        else:
            hints.append("📝 EXIF відсутній (частий випадок для ШІ)")
    except Exception:
        hints.append("❌ Не вдалося прочитати EXIF")
    return has_exif, ai_tool_found, hints

def png_metadata_check(img: Image.Image, fmt: str):
    # (Функція залишається як у попередньому варіанті)
    reasons = []
    ai_prompt_found = False
    if fmt == "PNG":
        try:
            metadata = img.info
            text = json.dumps(metadata).lower()
            
            if any(h in text for h in AI_TOOL_HINTS) or "prompt" in text or "parameters" in text:
                reasons.append("🚩 PNG метадані (tEXt) містять підказки/промпт ШІ")
                ai_prompt_found = True
            elif "software" in metadata and any(h in metadata["software"].lower() for h in AI_TOOL_HINTS):
                reasons.append(f"🚩 PNG метадані: у полі 'Software' знайдено ШІ-інструмент")

        except Exception:
            pass # Ігноруємо помилки
    return ai_prompt_found, reasons

def size_checks(img: Image.Image):
    # (Функція залишається як у попередньому варіанті)
    w, h = img.size
    reasons = []
    flags = {}
    
    mul_64 = (w % 64 == 0) and (h % 64 == 0)
    mul_8 = (w % 8 == 0) and (h % 8 == 0)
    square_common = (w == h) and (w in COMMON_AI_SIZES)
    
    g = gcd(w, h)
    ratio = (w // g, h // g)
    is_uncommon_ratio = ratio not in UNCOMMON_RATIOS and max(w, h) >= 512
    
    if mul_64:
        reasons.append("📏 Розміри кратні 64 (сильна ознака Stable Diffusion, SDXL)")
    elif mul_8:
        reasons.append("📏 Розміри кратні 8 (слабка ознака)")
        
    if square_common:
        reasons.append(f"🖼️ Квадратний формат {w}×{h}, типовий для ШІ-моделей")
        
    if is_uncommon_ratio and (w not in COMMON_AI_SIZES and h not in COMMON_AI_SIZES):
        reasons.append(f"📐 Нетипова чи дуже висока/низька пропорція {ratio[0]}:{ratio[1]} для фото")
        
    flags["mul64"] = mul_64
    flags["square_common"] = square_common
    flags["is_uncommon_ratio"] = is_uncommon_ratio
    flags["size"] = (w, h)
    return flags, reasons

def alpha_channel_weird(img: Image.Image, fmt: str):
    # (Функція залишається як у попередньому варіанті)
    has_alpha = img.mode in ("LA", "RGBA", "P", "PA")
    if fmt == "JPEG" and has_alpha:
        return True, "⚠️ JPEG з альфа-каналом — нетипово для фотографії"
    return False, None

def high_freq_heuristic(img: Image.Image):
    # (Функція залишається як у попередньому варіанті)
    gray = ImageOps.grayscale(img)
    gray_small = gray.resize((256, int(256 * gray.height / gray.width)), Image.Resampling.LANCZOS) if gray.width > 256 else gray
    arr = np.asarray(gray_small, dtype=np.float32) / 255.0

    k = np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], dtype=np.float32)
    
    pad_width = 1
    arr_padded = np.pad(arr, pad_width, mode='edge')
    lap = np.abs(np.array([[np.sum(arr_padded[i:i+3, j:j+3] * k) for j in range(arr.shape[1])] for i in range(arr.shape[0])]))

    lap_var = float(np.var(lap))

    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h//2, w//2
    r = min(cy, cx)
    
    low = mag[cy-r//4:cy+r//4, cx-r//4:cx+r//4].sum() + 1e-6
    high = mag.sum() - low
    high_ratio = float(high / (high + low))

    ai_like = (lap_var < 0.0003 and high_ratio > 0.55)
    expl = f"🔬 Локальна різкість (var Laplacian={lap_var:.4f}), частотне насичення={high_ratio:.2f}"
    
    if ai_like:
        expl = "🧠 Синтетична текстура: дуже низька локальна різкість (Laplacian Var) та висока частотна енергія ('пластмасовість')"
        
    return ai_like, expl, {"lap_var": lap_var, "high_ratio": high_ratio}

def jpeg_quant_hint(img: Image.Image, fmt: str):
    # (Функція залишається як у попередньому варіанті)
    try:
        if fmt == "JPEG" and hasattr(img, "quantization") and img.quantization:
            q = img.quantization
            if len(q) <= 2 and all(len(v) == 64 for v in q.values()):
                return False, None
            else:
                return True, "💡 Нестандартні JPEG quantization tables (слабка ознака, може бути ре-компресія)"
    except Exception:
        pass
    return False, None

def jpeg_artifact_hint(img: Image.Image, fmt: str):
    # (Функція залишається як у попередньому варіанті)
    if fmt == "JPEG" and img.mode in ("RGB", "L"):
        try:
            temp_io = io.BytesIO()
            img.save(temp_io, format="JPEG", quality=95)
            re_compressed = Image.open(temp_io)
            
            diff = ImageChops.difference(img, re_compressed)
            diff_gray = ImageOps.grayscale(diff)
            arr = np.asarray(diff_gray, dtype=np.float32)
            
            std_dev = np.std(arr)
            mean_abs_dev = np.mean(np.abs(arr))
            
            is_low_artifact = (std_dev < 10.0 and mean_abs_dev < 5.0)
            
            if is_low_artifact:
                return True, f"🖼️ Дуже низький рівень артефактів (STD={std_dev:.2f}) після ре-компресії (може бути перше збереження ШІ)"
                
        except Exception:
            pass
    return False, None

def noise_analysis(img: Image.Image):
    """
    Аналіз шуму:
    1. Перевірка на *відсутність* шуму, типову для ідеальних ШІ-зображень.
    2. Якщо є OpenCV: Виділення шуму за допомогою вейвлет-фільтрів або високочастотних фільтрів.
    """
    
    # 1. Спрощений аналіз шуму (якщо немає CV2):
    # Шукаємо однорідні ділянки і перевіряємо їхнє стандартне відхилення.
    gray = ImageOps.grayscale(img)
    arr = np.asarray(gray, dtype=np.float32)
    # Використовуємо високочастотний фільтр (наприклад, Лапласіан)
    # Знову використовуємо LapVar як проксі.
    try:
        if 'cv2' in globals() and hasattr(cv2, 'Laplacian'):
            arr_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(arr_cv, cv2.CV_64F)
            noise_std = np.std(laplacian)
        else:
            # Якщо немає cv2, використовуємо LapVar з попередньої функції
            _, _, freq_vals = high_freq_heuristic(img)
            noise_std = np.sqrt(freq_vals["lap_var"]) * 100 # Просто масштабуємо

    except Exception:
        # У випадку помилки при обробці
        return False, "❌ Помилка в аналізі шуму", {"noise_std": -1.0}


    # Евристика:
    # 1. Надто низьке STD шуму (нижче 0.05) - може вказувати на агресивне згладжування/відсутність природного шуму.
    # 2. Нетипові "блокові" шуми. (Складно без повноцінного PRNU, але можна спростити).
    
    ai_too_smooth = noise_std < 5.0 if 'cv2' in globals() and hasattr(cv2, 'Laplacian') else noise_std < 0.015
    
    if ai_too_smooth:
        expl = f"✨ Надто гладке зображення: Надзвичайно низький рівень шуму/текстури (STD={noise_std:.2f}) — може бути ознакою ШІ-генерації або агресивного Denoising."
        return True, expl, {"noise_std": float(noise_std)}
        
    return False, f"🔬 Аналіз шуму: STD={noise_std:.2f} (в межах норми)", {"noise_std": float(noise_std)}


def color_statistic_check(img: Image.Image):
    """
    Аналіз колірної статистики:
    1. Перенасиченість (типова для деяких ШІ).
    2. Обмежена/неприродна палітра.
    """
    
    # Конвертуємо в HSV (відтінок, насиченість, яскравість)
    try:
        hsv_img = img.convert("HSV")
        hsv_arr = np.asarray(hsv_img, dtype=np.float32) / 255.0
    except Exception:
        return False, "❌ Помилка конвертації в HSV", {}
        
    # Аналіз насиченості (Saturation)
    S = hsv_arr[:, :, 1]
    mean_S = np.mean(S)
    
    # 1. Перенасиченість: ШІ часто генерує занадто яскраві/насичені зображення.
    is_oversaturated = mean_S > 0.60
    
    # 2. Перевірка на "чисті" кольори (якщо більшість пікселів має S~1.0)
    high_S_count = np.sum(S > 0.95)
    total_pixels = S.size
    high_S_ratio = high_S_count / total_pixels
    
    is_cartoon_like = high_S_ratio > 0.05
    
    reasons = []
    
    if is_oversaturated:
        reasons.append(f"🌈 Висока середня насиченість ({mean_S:.2f}) — типово для деяких ШІ")
        
    if is_cartoon_like:
        reasons.append(f"🎨 Багато 'чистих' кольорів ({high_S_ratio:.2%}) — може вказувати на синтетичну/мультяшну гаму")

    ai_like = is_oversaturated or is_cartoon_like
    
    return ai_like, reasons, {"mean_s": float(mean_S), "high_s_ratio": float(high_S_ratio)}


# --- Ендпоінти FastAPI ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze/image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Порожній файл")

    explanations = []
    checks = {}

    # 1) EXIF/XMP
    has_exif, ai_tool_found_exif, hints = read_exif_hints(raw)
    explanations += hints
    checks["has_exif"] = has_exif
    checks["ai_tool_found_exif"] = ai_tool_found_exif
    
    # 2) Відкриваємо зображення
    try:
        img = load_image(raw)
    except Exception:
        explanations.append("❌ Не вдалося відкрити зображення")
        return AnalyzeResponse(prob_ai=0.8, explanations=explanations, checks={"open_error": True})

    fmt = (img.format or "").upper()
    checks["format"] = fmt

    # 3) PNG метадані
    ai_found_png, png_reasons = png_metadata_check(img, fmt)
    checks["ai_found_png_metadata"] = ai_found_png
    explanations += png_reasons
    
    # 4) Розміри та пропорції
    size_flags, size_reasons = size_checks(img)
    checks.update(size_flags)
    explanations += size_reasons

    # 5) Альфа-канал
    weird_alpha, alpha_reason = alpha_channel_weird(img, fmt)
    checks["weird_alpha"] = weird_alpha
    if alpha_reason:
        explanations.append(alpha_reason)

    # 6) Проста частотна евристика (пластмасовість)
    ai_like_freq, freq_expl, freq_vals = high_freq_heuristic(img)
    checks["ai_like_freq"] = ai_like_freq
    checks["freq"] = freq_vals
    explanations.append(freq_expl)

    # 7) JPEG quantization
    q_hint, q_reason = jpeg_quant_hint(img, fmt)
    checks["jpeg_quant_weird"] = q_hint
    if q_reason:
        explanations.append(q_reason)

    # 8) Артефакти JPEG (евристика)
    low_artifact_hint, artifact_reason = jpeg_artifact_hint(img, fmt)
    checks["low_artifact_hint"] = low_artifact_hint
    if artifact_reason:
        explanations.append(artifact_reason)
        
    # 9) Аналіз шуму (NEW)
    ai_too_smooth, noise_expl, noise_vals = noise_analysis(img)
    checks["ai_too_smooth"] = ai_too_smooth
    checks["noise"] = noise_vals
    explanations.append(noise_expl)
    
    # 10) Аналіз колірної статистики (NEW)
    ai_color_weird, color_reasons, color_vals = color_statistic_check(img)
    checks["ai_color_weird"] = ai_color_weird
    checks["color_stats"] = color_vals
    explanations += color_reasons
    
    # ---- Зважування ознак ----
    score = 0.0
    
    # Найсильніші ознаки
    if ai_tool_found_exif or ai_found_png: score += 0.40
    
    # Сильні ознаки
    if not has_exif and not ai_found_png and fmt == "JPEG": score += 0.15
    if size_flags["mul64"]:               score += 0.20
    if size_flags["square_common"]:       score += 0.15
    
    # Помірні/Синтетичні ознаки
    if weird_alpha:                       score += 0.10
    if ai_like_freq:                      score += 0.20 # Пластмасовість
    if low_artifact_hint:                 score += 0.10 # Чисте перше збереження
    if ai_too_smooth:                     score += 0.15 # Відсутність шуму
    if ai_color_weird:                    score += 0.10 # Неприродна гама/насиченість

    # Слабкі ознаки
    if size_flags["is_uncommon_ratio"]:   score += 0.05
    if q_hint:                            score += 0.05
    
    # Потолок і підлога
    prob_ai = float(max(0.0, min(1.0, score)))

    # Нормалізація для розмірів (якщо зображення дуже мале, довіра до аналізу нижча)
    w, h = size_flags["size"]
    max_dim = max(w, h)
    
    if max_dim < 384:
        prob_ai = min(1.0, prob_ai + 0.05)

    if max_dim > 2048 and not ai_tool_found_exif:
        prob_ai = max(0.0, prob_ai - 0.10) 


    return AnalyzeResponse(
        prob_ai=round(prob_ai, 2),
        explanations=explanations or ["✅ Ознак ШІ не виявлено явних (зображення виглядає як звичайне фото)"],
        checks=checks
    )