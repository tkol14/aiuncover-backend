from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageChops
import io, re, json, os
import numpy as np
import exifread
from typing import Tuple, Dict, Any

# ---------------- FastAPI ----------------

app = FastAPI(
    title="AIUncover API",
    version="1.4.2-perf",
    debug=os.environ.get("DEBUG", "False").lower() == "true",
)

ALLOWED_ORIGINS = [
    "https://aiuncover.net",
    "https://www.aiuncover.net",
    "https://aiuncover-backend-production.up.railway.app",
    "https://api.aiuncover.net",
    "http://localhost:3000",
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

# ---------------- Constants ----------------

AI_TOOL_HINTS = [
    "stable diffusion", "stablediffusion", "sdxl", "novelai",
    "midjourney", "dall-e", "dalle", "leonardo", "flux", "playground",
    "firefly", "adobe firefly", "cf ai", "canva ai", "ideogram", "pollinations",
    "dreamstudio", "getimg.ai", "artbreeder", "pika labs", "runway", "gen-1", "gen-2",
    "aigenerated", "ai generated"
]

COMMON_AI_SIZES = {256, 384, 448, 512, 576, 640, 704, 768, 896, 960, 1024, 1152, 1280, 1536, 1792, 2048}
UNCOMMON_RATIOS = {(1,1), (4,3), (3,4), (16,9), (9,16), (3,2), (2,3), (5,4), (4,5)}

# ---------------- Utilities ----------------

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def load_image(raw: bytes) -> Image.Image:
    bio = io.BytesIO(raw)
    img = Image.open(bio)
    img.load()
    return img

def _sanitize(obj: Any):
    """Convert NumPy scalars/bools to native Python types recursively."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.float32, np.float64, np.float16, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj

def _cap_long_edge(img: Image.Image, max_edge: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_edge:
        return img
    scale = max_edge / float(m)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

def _fft_conv2_same(arr: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Fast 2D convolution via FFT, 'same' output size.
    Assumes float32 arr, float32 kernel (small).
    """
    ah, aw = arr.shape
    kh, kw = k.shape
    H = 1 << (ah + kh - 1 - 1).bit_length()
    W = 1 << (aw + kw - 1 - 1).bit_length()
    F_arr = np.fft.rfft2(arr, s=(H, W))
    # Flip kernel for convolution
    kflip = np.flipud(np.fliplr(k))
    F_k = np.fft.rfft2(kflip, s=(H, W))
    conv = np.fft.irfft2(F_arr * F_k, s=(H, W)).real
    # center crop to 'same'
    y0 = (kh - 1) // 2
    x0 = (kw - 1) // 2
    return conv[y0:y0+ah, x0:x0+aw]

# ---------------- Shared Analysis Context ----------------

class ImgCtx:
    """
    Caches common transforms once per request.
    """
    __slots__ = (
        "img", "fmt", "w", "h",
        "rgb", "gray_u8", "gray_f32",
        "gray_small_f32",
    )
    def __init__(self, img: Image.Image):
        self.img = img
        self.fmt = (img.format or "").upper()
        self.w, self.h = img.size

        # Unify mode early; keeps PNG alpha semantics for separate check
        if img.mode not in ("RGB", "RGBA", "L", "LA", "P", "PA"):
            img = img.convert("RGB")
        self.rgb = img.convert("RGB")

        # Shared grayscale
        gray = ImageOps.grayscale(self.rgb)
        self.gray_u8 = np.asarray(gray, dtype=np.uint8)
        self.gray_f32 = self.gray_u8.astype(np.float32) / 255.0

        # Small grayscale for frequency checks (width=256)
        if gray.width > 256:
            new_h = int(256 * gray.height / gray.width)
            gray_small = gray.resize((256, max(1, new_h)), Image.Resampling.LANCZOS)
            self.gray_small_f32 = np.asarray(gray_small, dtype=np.float32) / 255.0
        else:
            self.gray_small_f32 = self.gray_f32

# ---------------- Checks ----------------

def read_exif_hints(raw: bytes, fmt: str):
    reasons = []
    has_exif = False
    ai_tool_found = False

    # PNG/WebP often carry no EXIF; skip for speed
    if fmt in {"PNG", "WEBP"}:
        return has_exif, ai_tool_found, reasons

    try:
        tags = exifread.process_file(io.BytesIO(raw), details=False)
        if tags:
            has_exif = True
            # quick scan: check values separately to avoid huge string concat
            for v in tags.values():
                low = str(v).lower()
                # cheap early-exit
                if any(h in low for h in AI_TOOL_HINTS):
                    reasons.append("🚩 EXIF/XMP містить згадки генераторів ШІ")
                    ai_tool_found = True
                    break
        else:
            reasons.append("📝 EXIF відсутній (частий випадок для ШІ)")
    except Exception:
        reasons.append("❌ Не вдалося прочитати EXIF")
    return has_exif, ai_tool_found, reasons

def png_metadata_check(img: Image.Image, fmt: str):
    reasons = []
    ai_prompt_found = False
    if fmt == "PNG":
        try:
            metadata = img.info or {}
            # examine only a few common keys to avoid dumping big blobs
            to_scan = []
            if "parameters" in metadata: to_scan.append(str(metadata["parameters"]))
            if "prompt" in metadata: to_scan.append(str(metadata["prompt"]))
            if "description" in metadata: to_scan.append(str(metadata["description"]))
            if "software" in metadata: to_scan.append(str(metadata["software"]))
            # fallback: limited JSON dump
            if not to_scan and metadata:
                to_scan.append(json.dumps({k: metadata[k] for k in list(metadata)[:6]}))

            low = " ".join(to_scan).lower()
            if any(h in low for h in AI_TOOL_HINTS) or ("prompt" in low) or ("parameters" in low):
                reasons.append("🚩 PNG метадані (tEXt) містять підказки/промпт ШІ")
                ai_prompt_found = True
            elif "software" in metadata and any(h in str(metadata["software"]).lower() for h in AI_TOOL_HINTS):
                reasons.append("🚩 PNG метадані: у полі 'Software' знайдено ШІ-інструмент")
        except Exception:
            pass
    return ai_prompt_found, reasons

def size_checks(img: Image.Image):
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
    has_alpha = img.mode in ("LA", "RGBA", "P", "PA")
    if fmt == "JPEG" and has_alpha:
        return True, "⚠️ JPEG з альфа-каналом — нетипово для фотографії"
    return False, None

def high_freq_heuristic(ctx: ImgCtx):
    arr = ctx.gray_small_f32

    # 3x3 Laplacian via FFT conv (fast)
    k = np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], dtype=np.float32)
    lap = np.abs(_fft_conv2_same(arr, k))
    lap_var = float(np.var(lap))

    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(cy, cx)
    low = mag[cy - r // 4: cy + r // 4, cx - r // 4: cx + r // 4].sum() + 1e-6
    high = mag.sum() - low
    high_ratio = float(high / (high + low))

    ai_like = (lap_var < 0.006 and high_ratio > 0.50)
    expl = f"🔬 Локальна різкість (var Laplacian={lap_var:.4f}), частотне насичення={high_ratio:.2f}"
    if ai_like:
        expl = "🧠 Синтетична текстура: дуже низька локальна різкість (Laplacian Var) та висока частотна енергія ('пластмасовість')"

    return ai_like, expl, {"lap_var": lap_var, "high_ratio": high_ratio}

def jpeg_quant_hint(img: Image.Image, fmt: str):
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

def jpeg_artifact_hint(ctx: ImgCtx):
    # recompress on a capped RGB to keep it fast
    if ctx.fmt != "JPEG":
        return False, None
    try:
        small = _cap_long_edge(ctx.rgb, 1280)
        bio = io.BytesIO()
        small.save(bio, format="JPEG", quality=95)
        bio.seek(0)
        re_comp = Image.open(bio).convert("RGB")

        diff = ImageChops.difference(small, re_comp)
        arr = np.asarray(ImageOps.grayscale(diff), dtype=np.float32)

        std_dev = float(np.std(arr))
        mean_abs_dev = float(np.mean(np.abs(arr)))
        is_low_artifact = (std_dev < 10.0 and mean_abs_dev < 5.0)
        if is_low_artifact:
            return True, f"🖼️ Дуже низький рівень артефактів (STD={std_dev:.2f}) після ре-компресії (може бути перше збереження ШІ)"
    except Exception:
        pass
    return False, None

def noise_analysis(ctx: ImgCtx):
    """
    Laplacian STD on a capped grayscale (fast FFT conv).
    """
    try:
        # cap to limit FFT cost but keep structure
        if max(ctx.w, ctx.h) > 1536:
            work_img = _cap_long_edge(ctx.rgb, 1536)
            gray = ImageOps.grayscale(work_img)
            arr = np.asarray(gray, dtype=np.float32) / 255.0
        else:
            arr = ctx.gray_f32

        k = np.array([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]], dtype=np.float32)
        lap = np.abs(_fft_conv2_same(arr, k))
        noise_std = float(np.std(lap))
    except Exception:
        return False, "❌ Помилка в аналізі шуму (NumPy/PIL)", {"noise_std": -1.0}

    # keep your original logic — note: your code comment said "<0.005" but used >14
    ai_too_smooth = noise_std > 14

    if ai_too_smooth:
        expl = (f"✨ Надто гладке зображення: Надзвичайно низький рівень шуму/текстури "
                f"(Laplacian STD={noise_std:.4f}) — може бути ознакою ШІ-генерації або агресивного Denoising.")
        return True, expl, {"noise_std": noise_std}
    return False, f"🔬 Аналіз шуму: Laplacian STD={noise_std:.4f} (в межах норми)", {"noise_std": noise_std}

def color_statistic_check(ctx: ImgCtx):
    try:
        hsv_img = ctx.rgb.convert("HSV")
        hsv_arr = np.asarray(hsv_img, dtype=np.float32) / 255.0
    except Exception:
        return False, "❌ Помилка конвертації в HSV", {}

    S = hsv_arr[:, :, 1]
    mean_S = float(np.mean(S))
    is_oversaturated = mean_S > 0.60

    high_S_ratio = float(np.mean(S > 0.95))
    is_cartoon_like = high_S_ratio > 0.05

    reasons = []
    if is_oversaturated:
        reasons.append(f"🌈 Висока середня насиченість ({mean_S:.2f}) — типово для деяких ШІ")
    if is_cartoon_like:
        reasons.append(f"🎨 Багато 'чистих' кольорів ({high_S_ratio:.2%}) — може вказувати на синтетичну/мультяшну гаму")

    return (is_oversaturated or is_cartoon_like), reasons, {"mean_s": mean_S, "high_s_ratio": high_S_ratio}

def noise_inconsistency_check(ctx: ImgCtx):
    w, h = ctx.w, ctx.h
    if w < 64 or h < 64:
        return False, None, {"inconsistency_ratio": 0.0}

    # use capped image for speed
    base = ctx.rgb if max(w, h) <= 1536 else _cap_long_edge(ctx.rgb, 1536)
    bw, bh = base.size
    gray = ImageOps.grayscale(base)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    # quarters
    midx, midy = bw // 2, bh // 2
    quads = [
        arr[0:midy, 0:midx],
        arr[0:midy, midx:bh*0+midx+ (bw-midx)],
        arr[midy:bh, 0:midx],
        arr[midy:bh, midx:bw],
    ]

    k = np.array([[0, 1, 0],
                  [1,-4, 1],
                  [0, 1, 0]], dtype=np.float32)

    noise_stds = []
    for q in quads:
        if q.size == 0 or q.shape[0] < 8 or q.shape[1] < 8:
            continue
        lap = np.abs(_fft_conv2_same(q, k))
        noise_stds.append(float(np.std(lap)))

    if len(noise_stds) < 4:
        return False, None, {"inconsistency_ratio": 0.0}

    mean_std = float(np.mean(noise_stds))
    std_std = float(np.std(noise_stds))
    inconsistency_ratio = std_std / (mean_std + 1e-6)
    is_inconsistent = inconsistency_ratio > 0.30
    if is_inconsistent:
        return True, (f"✂️ Неоднорідність шуму/текстури (Коеф. варіації={inconsistency_ratio:.2f}) — "
                      f"сильна ознака монтажу"), {"inconsistency_ratio": inconsistency_ratio}
    return False, None, {"inconsistency_ratio": inconsistency_ratio}

# ---------------- Endpoints ----------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze/image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Порожній файл")

    explanations = []
    checks: Dict[str, Any] = {}

    # 1) Try to open & build context early (lets us know format for EXIF choice)
    try:
        img = load_image(raw)
    except Exception:
        explanations.append("❌ Не вдалося відкрити зображення")
        return AnalyzeResponse(prob_ai=0.8, explanations=explanations, checks={"open_error": True})

    ctx = ImgCtx(img)
    checks["format"] = ctx.fmt

    # 2) EXIF/XMP (skip for PNG/WebP to save time)
    has_exif, ai_tool_found_exif, hints = read_exif_hints(raw, ctx.fmt)
    explanations += hints
    checks["has_exif"] = has_exif
    checks["ai_tool_found_exif"] = ai_tool_found_exif

    # 3) PNG metadata (tEXt)
    ai_found_png, png_reasons = png_metadata_check(ctx.img, ctx.fmt)
    checks["ai_found_png_metadata"] = ai_found_png
    explanations += png_reasons

    # 4) Size & ratio
    size_flags, size_reasons = size_checks(ctx.img)
    checks.update(size_flags)
    explanations += size_reasons

    # 5) Alpha channel weirdness
    weird_alpha, alpha_reason = alpha_channel_weird(ctx.img, ctx.fmt)
    checks["weird_alpha"] = weird_alpha
    if alpha_reason:
        explanations.append(alpha_reason)

    # 6) Frequency heuristic
    ai_like_freq, freq_expl, freq_vals = high_freq_heuristic(ctx)
    checks["ai_like_freq"] = ai_like_freq
    checks["freq"] = freq_vals
    explanations.append(freq_expl)

    # 7) JPEG quantization
    q_hint, q_reason = jpeg_quant_hint(ctx.img, ctx.fmt)
    checks["jpeg_quant_weird"] = q_hint
    if q_reason:
        explanations.append(q_reason)

    # 8) JPEG re-compress artifact hint (capped size)
    low_artifact_hint, artifact_reason = jpeg_artifact_hint(ctx)
    checks["low_artifact_hint"] = low_artifact_hint
    if artifact_reason:
        explanations.append(artifact_reason)

    # 9) Noise analysis (FFT Laplacian; capped size inside)
    ai_too_smooth, noise_expl, noise_vals = noise_analysis(ctx)
    checks["ai_too_smooth"] = ai_too_smooth
    checks["noise"] = noise_vals
    explanations.append(noise_expl)

    # 10) Color stats
    ai_color_weird, color_reasons, color_vals = color_statistic_check(ctx)
    checks["ai_color_weird"] = ai_color_weird
    checks["color_stats"] = color_vals
    explanations += color_reasons

    # 11) Noise inconsistency
    ai_inconsistent, inconsistency_reason, inconsistency_vals = noise_inconsistency_check(ctx)
    checks["ai_inconsistent"] = ai_inconsistent
    checks["noise_inconsistency"] = inconsistency_vals
    if inconsistency_reason:
        explanations.append(inconsistency_reason)

    # ---- Scoring (unchanged) ----
    score = 0.0
    if ai_tool_found_exif or ai_found_png: score += 0.40
    if not has_exif and not ai_found_png and ctx.fmt == "JPEG": score += 0.20
    if size_flags["mul64"]:               score += 0.20
    if size_flags["square_common"]:       score += 0.15
    if weird_alpha:                       score += 0.10
    if ai_like_freq:                      score += 0.30
    if low_artifact_hint:                 score += 0.10
    if ai_too_smooth:                     score += 0.25
    if ai_color_weird:                    score += 0.15
    if size_flags["is_uncommon_ratio"]:   score += 0.05
    if q_hint:                            score += 0.05
    if ai_inconsistent:                   score += 0.40

    prob_ai = float(max(0.0, min(1.0, score)))
    w, h = size_flags["size"]
    max_dim = max(w, h)
    if max_dim < 384:
        prob_ai = min(1.0, prob_ai + 0.05)
    if max_dim > 2048 and not ai_tool_found_exif:
        prob_ai = max(0.0, prob_ai - 0.10)

    # Final sanitize for JSON
    final_checks = _sanitize(checks)

    return AnalyzeResponse(
        prob_ai=round(prob_ai, 2),
        explanations=explanations or ["✅ Ознак ШІ не виявлено явних (зображення виглядає як звичайне фото)"],
        checks=final_checks,
    )
