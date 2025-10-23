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

app = FastAPI(title="AIUncover API", version="1.3.0 - Expanded", debug=os.environ.get("DEBUG", "False").lower() == "true")

ALLOWED_ORIGINS = [
    "https://aiuncover.net",
    "https://www.aiuncover.net",
    "https://aiuncover-backend-production.up.railway.app",
    "https://api.aiuncover.net",
    "http://localhost:3000" # Додав для локального тестування, якщо потрібно
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

# Оновлений список підказок для ШІ-інструментів
AI_TOOL_HINTS = [
    "stable diffusion", "stablediffusion", "sdxl", "novelai",
    "midjourney", "dall-e", "dalle", "leonardo", "flux", "playground",
    "firefly", "adobe firefly", "cf ai", "canva ai", "ideogram", "pollinations",
    "dreamstudio", "getimg.ai", "artbreeder", "pika labs", "runway", "gen-1", "gen-2",
    "aigenerated", "ai generated"
]

COMMON_AI_SIZES = {256, 384, 448, 512, 576, 640, 704, 768, 896, 960, 1024, 1152, 1280, 1536, 1792, 2048}
UNCOMMON_RATIOS = {
    (1, 1), (4, 3), (3, 4), (16, 9), (9, 16),
} # Додамо типові фото-пропорції

def read_exif_hints(raw: bytes):
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
    reasons = []
    ai_prompt_found = False
    if fmt == "PNG":
        try:
            # DALL-E, Midjourney та інші часто зберігають метадані у PNG tEXt chunks
            metadata = img.info
            text = json.dumps(metadata).lower()
            
            # Пошук конкретних ключів або підказок ШІ
            if any(h in text for h in AI_TOOL_HINTS) or "prompt" in text or "parameters" in text:
                reasons.append("🚩 PNG метадані (tEXt) містять підказки/промпт ШІ")
                ai_prompt_found = True
            elif "software" in metadata and any(h in metadata["software"].lower() for h in AI_TOOL_HINTS):
                reasons.append(f"🚩 PNG метадані: у полі 'Software' знайдено ШІ-інструмент")

        except Exception:
            reasons.append("❌ Не вдалося прочитати PNG метадані")
    return ai_prompt_found, reasons

def load_image(raw: bytes):
    bio = io.BytesIO(raw)
    img = Image.open(bio)
    img.load()
    return img

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def size_checks(img: Image.Image):
    w, h = img.size
    reasons = []
    flags = {}
    
    # Кратність 64/8
    mul_64 = (w % 64 == 0) and (h % 64 == 0)
    mul_8 = (w % 8 == 0) and (h % 8 == 0)
    
    # Типові квадратні ШІ-розміри
    square_common = (w == h) and (w in COMMON_AI_SIZES)
    
    # Нетипові пропорції
    g = gcd(w, h)
    ratio = (w // g, h // g)
    is_uncommon_ratio = ratio not in UNCOMMON_RATIOS and max(w, h) >= 512
    
    if mul_64:
        reasons.append("📏 Розміри кратні 64 (сильна ознака генераторів Stable Diffusion, SDXL)")
    elif mul_8:
        reasons.append("📏 Розміри кратні 8 (слабка ознака генераторів/оптимізації)")
        
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
    has_alpha = img.mode in ("LA", "RGBA", "P", "PA")
    if fmt == "JPEG" and has_alpha:
        return True, "⚠️ JPEG з альфа-каналом — нетипово для фотографії, може вказувати на конвертацію"
    return False, None

def high_freq_heuristic(img: Image.Image):
    # Легка евристика: варіація Лапласіана + частотна енергія (не змінено)
    gray = ImageOps.grayscale(img)
    gray_small = gray.resize((256, int(256 * gray.height / gray.width)), Image.Resampling.LANCZOS) if gray.width > 256 else gray
    arr = np.asarray(gray_small, dtype=np.float32) / 255.0

    # Лапласіан через ядро
    k = np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], dtype=np.float32)
    
    # Використовуємо згортку NumPy
    pad_width = 1
    arr_padded = np.pad(arr, pad_width, mode='edge')
    lap = np.abs(np.array([[np.sum(arr_padded[i:i+3, j:j+3] * k) for j in range(arr.shape[1])] for i in range(arr.shape[0])]))

    lap_var = float(np.var(lap))

    # FFT енергія високих частот
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h//2, w//2
    r = min(cy, cx)
    
    # Низькі частоти - центральна чверть
    low = mag[cy-r//4:cy+r//4, cx-r//4:cx+r//4].sum() + 1e-6
    high = mag.sum() - low
    high_ratio = float(high / (high + low))

    # Евристика: дуже низький LapVar і водночас помірно високий high_ratio -> "пластмасовість"
    ai_like = (lap_var < 0.0003 and high_ratio > 0.55)
    expl = f"🔬 Локальна різкість (var Laplacian={lap_var:.4f}), частотне насичення={high_ratio:.2f}"
    
    # Якщо показники вказують на "пластмасовість"
    if ai_like:
        expl = "🧠 Синтетична текстура: дуже низька локальна різкість і помірно висока частотна енергія (ознака 'пластмасовості' ШІ)"
        
    return ai_like, expl, {"lap_var": lap_var, "high_ratio": high_ratio}

def jpeg_quant_hint(img: Image.Image, fmt: str):
    # Не змінюємо, залишаємо як є
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
    # Евристика для артефактів подвійного стиснення (JPEG Ghost/ELA)
    if fmt == "JPEG" and img.mode in ("RGB", "L"):
        try:
            # Ідея: стиснути зображення з високою якістю (наприклад, Q=95) і порівняти з оригіналом.
            # Якщо зображення вже стиснуто, то різниця (залишкові артефакти) має бути відносно рівномірною.
            # Якщо зображення згенеровано ШІ, артефакти можуть бути нетиповими.
            
            # Створюємо тимчасовий буфер і стискаємо з Q=95
            temp_io = io.BytesIO()
            img.save(temp_io, format="JPEG", quality=95)
            re_compressed = Image.open(temp_io)
            
            # Обчислюємо різницю (ELA-подібний ефект)
            diff = ImageChops.difference(img, re_compressed)
            # Перетворюємо в чорно-біле і знаходимо середнє відхилення
            diff_gray = ImageOps.grayscale(diff)
            arr = np.asarray(diff_gray, dtype=np.float32)
            
            # Стандартне відхилення різниці (яка має бути низькою для "чистого" ШІ)
            std_dev = np.std(arr)
            
            # Середнє відхилення (яке має бути вищим для пере-стиснутих)
            mean_abs_dev = np.mean(np.abs(arr))
            
            # Емпірична евристика
#            Чисті ШІ-зображення (без подальшого стиснення) можуть мати дуже низький std_dev.
            is_low_artifact = (std_dev < 10.0 and mean_abs_dev < 5.0) # Залежить від якості оригіналу
            
            if is_low_artifact:
                # Дуже низьке відхилення після ре-компресії може вказувати на "чисте" зображення
                return True, "🖼️ Дуже низький рівень артефактів (STD={std_dev:.2f}, Mean={mean_abs_dev:.2f}) після ре-компресії (може бути перше збереження ШІ)"
                
        except Exception:
            pass # Ігноруємо помилки
    return False, None

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
        explanations.append("❌ Не вдалося відкрити зображення (непідтримуваний або пошкоджений файл)")
        return AnalyzeResponse(prob_ai=0.8, explanations=explanations, checks={"open_error": True})

    fmt = (img.format or "").upper()
    checks["format"] = fmt

    # 3) PNG метадані (якщо PNG)
    ai_found_png, png_reasons = png_metadata_check(img, fmt)
    checks["ai_found_png_metadata"] = ai_found_png
    explanations += png_reasons
    
    # 4) Розміри та пропорції
    size_flags, size_reasons = size_checks(img)
    checks.update(size_flags)
    explanations += size_reasons

    # 5) Альфа-канал (дивні комбінації)
    weird_alpha, alpha_reason = alpha_channel_weird(img, fmt)
    checks["weird_alpha"] = weird_alpha
    if alpha_reason:
        explanations.append(alpha_reason)

    # 6) Проста частотна евристика (пластмасовість/надто синтетичні краї)
    ai_like_freq, freq_expl, freq_vals = high_freq_heuristic(img)
    checks["ai_like_freq"] = ai_like_freq
    checks["freq"] = freq_vals
    explanations.append(freq_expl)

    # 7) JPEG quantization (слабка ознака)
    q_hint, q_reason = jpeg_quant_hint(img, fmt)
    checks["jpeg_quant_weird"] = q_hint
    if q_reason:
        explanations.append(q_reason)

    # 8) Артефакти JPEG (евристика)
    low_artifact_hint, artifact_reason = jpeg_artifact_hint(img, fmt)
    checks["low_artifact_hint"] = low_artifact_hint
    if artifact_reason:
        explanations.append(artifact_reason)
    
    # ---- Зважування ознак (легка лінійна комбінація) ----
    score = 0.0
    
    # Сильні ознаки
    if ai_tool_found_exif or ai_found_png: score += 0.40 # Найсильніший доказ
    if not has_exif and not ai_found_png and fmt == "JPEG": score += 0.15 # Відсутність EXIF у JPEG
    if size_flags["mul64"]:               score += 0.20 # Кратність 64
    if size_flags["square_common"]:       score += 0.15 # Типовий квадрат
    
    # Помірні ознаки
    if weird_alpha:                       score += 0.10
    if ai_like_freq:                      score += 0.20 # Синтетична текстура
    if size_flags["is_uncommon_ratio"]:   score += 0.05
    if low_artifact_hint:                 score += 0.10 # Схоже на чисте перше збереження

    # Слабкі ознаки
    if q_hint:                            score += 0.05
    
    # Потолок і підлога
    prob_ai = float(max(0.0, min(1.0, score)))

    # Нормалізація для дуже малих або дуже великих зображень
    w, h = size_flags["size"]
    max_dim = max(w, h)
    
    if max_dim < 384:
        prob_ai = min(1.0, prob_ai + 0.05) # Невеликі ШІ-зображення

    if max_dim > 2048 and not ai_tool_found_exif:
        # Дуже великі розміри менш типові для стандартних публічних ШІ-моделей
        prob_ai = max(0.0, prob_ai - 0.10) 


    return AnalyzeResponse(
        prob_ai=round(prob_ai, 2),
        explanations=explanations or ["✅ Ознак ШІ не виявлено явних (або зображення добре оброблено)"],
        checks=checks
    )