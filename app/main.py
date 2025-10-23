from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import io, re, json
import numpy as np
import exifread

app = FastAPI(title="AIUncover API", version="1.2.0")

ALLOWED_ORIGINS = [
    "https://aiuncover.net",
    "https://www.aiuncover.net",
    "https://aiuncover-backend-production.up.railway.app",
    "https://api.aiuncover.net",
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

AI_TOOL_HINTS = [
    "stable diffusion", "stablediffusion", "sdxl", "novelai",
    "midjourney", "dall-e", "dalle", "leonardo", "flux", "playground",
    "firefly", "adobe firefly", "cf ai", "canva ai", "ideogram", "pollinations"
]

COMMON_AI_SIZES = {256, 384, 448, 512, 576, 640, 704, 768, 896, 960, 1024, 1152, 1280}

def read_exif_hints(raw: bytes):
    hints = []
    has_exif = False
    try:
        tags = exifread.process_file(io.BytesIO(raw), details=False)
        if tags:
            has_exif = True
            text = " ".join([str(v) for v in tags.values()]).lower()
            if any(h in text for h in AI_TOOL_HINTS):
                hints.append("EXIF/XMP містить згадки генераторів ШІ")
        else:
            hints.append("EXIF відсутній")
    except Exception:
        hints.append("Не вдалося прочитати EXIF")
    return has_exif, hints

def load_image(raw: bytes):
    bio = io.BytesIO(raw)
    img = Image.open(bio)
    img.load()
    return img

def size_checks(img: Image.Image):
    w, h = img.size
    reasons = []
    flags = {}
    mul64 = (w % 64 == 0) and (h % 64 == 0)
    square_common = (w == h) and (w in COMMON_AI_SIZES)
    if mul64:
        reasons.append("Розміри кратні 64 (ознака генераторів)")
    if square_common:
        reasons.append(f"Квадратний формат {w}×{h}, типовий для ШІ")
    flags["mul64"] = mul64
    flags["square_common"] = square_common
    flags["size"] = (w, h)
    return flags, reasons

def alpha_channel_weird(img: Image.Image, fmt: str):
    # Альфа для JPEG – дивно; для WEBP/PNG – ок
    has_alpha = img.mode in ("LA", "RGBA")
    if fmt == "JPEG" and has_alpha:
        return True, "JPEG з альфа-каналом — нетипово для фото"
    return False, None

def high_freq_heuristic(img: Image.Image):
    # Легка евристика: варіація Лапласіана + частотна енергія
    gray = ImageOps.grayscale(img)
    gray_small = gray.resize((256, int(256 * gray.height / gray.width))) if gray.width > 256 else gray
    arr = np.asarray(gray_small, dtype=np.float32) / 255.0

    # Лапласіан через ядро
    k = np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], dtype=np.float32)
    lap = np.abs(
        np.convolve(arr.ravel(), k.ravel(), mode="same").reshape(arr.shape)
    )
    lap_var = float(np.var(lap))

    # FFT енергія високих частот
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h//2, w//2
    r = min(cy, cx)
    low = mag[cy-r//4:cy+r//4, cx-r//4:cx+r//4].sum() + 1e-6
    high = mag.sum() - low
    high_ratio = float(high / (high + low))

    # Евристика: дуже низький LapVar і водночас помірно високий high_ratio -> "пластмасовість"
    ai_like = (lap_var < 0.0003 and high_ratio > 0.55)
    expl = f"Локальна різкість (var Laplacian={lap_var:.4f}), частотне насичення={high_ratio:.2f}"
    return ai_like, expl, {"lap_var": lap_var, "high_ratio": high_ratio}

def jpeg_quant_hint(img: Image.Image, fmt: str):
    try:
        if fmt == "JPEG" and hasattr(img, "quantization") and img.quantization:
            # Якщо таблиць дуже мало/нестандартні — +слабка ознака
            q = img.quantization
            if len(q) <= 2 and all(len(v) == 64 for v in q.values()):
                return False, None  # норм
            else:
                return True, "Нестандартні JPEG quantization tables"
    except Exception:
        pass
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
    has_exif, hints = read_exif_hints(raw)
    explanations += hints
    checks["has_exif"] = has_exif

    # 2) Відкриваємо зображення
    try:
        img = load_image(raw)
    except Exception:
        explanations.append("Не вдалося відкрити зображення (непідтримуваний або пошкоджений файл)")
        return AnalyzeResponse(prob_ai=0.8, explanations=explanations, checks={"open_error": True})

    fmt = (img.format or "").upper()
    checks["format"] = fmt

    # 3) Розміри
    size_flags, size_reasons = size_checks(img)
    checks.update(size_flags)
    explanations += size_reasons

    # 4) Альфа-канал (дивні комбінації)
    weird_alpha, alpha_reason = alpha_channel_weird(img, fmt)
    checks["weird_alpha"] = weird_alpha
    if alpha_reason:
        explanations.append(alpha_reason)

    # 5) Проста частотна евристика (пластмасовість/надто синтетичні краї)
    ai_like_freq, freq_expl, freq_vals = high_freq_heuristic(img)
    checks["ai_like_freq"] = ai_like_freq
    checks["freq"] = freq_vals
    explanations.append(freq_expl)

    # 6) JPEG quantization (слабка ознака)
    q_hint, q_reason = jpeg_quant_hint(img, fmt)
    checks["jpeg_quant_weird"] = q_hint
    if q_reason:
        explanations.append(q_reason)

    # ---- Зважування ознак (легка лінійна комбінація) ----
    score = 0.0
    if not has_exif:                      score += 0.25
    if size_flags["mul64"]:               score += 0.15
    if size_flags["square_common"]:       score += 0.15
    if weird_alpha:                       score += 0.10
    if ai_like_freq:                      score += 0.20
    if q_hint:                            score += 0.10
    # Потолок і підлога
    prob_ai = float(max(0.0, min(1.0, score)))

    # Невелика нормалізація для дуже малих або дуже великих зображень
    w, h = size_flags["size"]
    if max(w, h) < 384:
        prob_ai = min(1.0, prob_ai + 0.05)

    return AnalyzeResponse(
        prob_ai=round(prob_ai, 2),
        explanations=explanations or ["Ознак ШІ не виявлено явних"],
        checks=checks
    )