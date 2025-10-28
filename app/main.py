from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageChops
import io, json, os
import numpy as np
import exifread
from typing import Tuple, Dict, Any

# --- Optional dependencies ---
try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

# Paths can be set with env vars; if models missing, checks will auto-skip.
ULTRAFACE_ONNX = os.environ.get("ULTRAFACE_ONNX", "models/ultraface-RFB-320.onnx")  # 320x240 variant
ULTRAFACE_INPUT_SIZE = (320, 240)  # (W,H)
AIUNCOVER_ONNX_MODEL = os.environ.get("AIUNCOVER_ONNX_MODEL", "models/aiuncover_features_classifier.onnx")

# ---------------- FastAPI ----------------

app = FastAPI(
    title="AIUncover API",
    version="1.5.0-perf-ocr-face-onnx",
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
    if isinstance(obj, (np.float16, np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize(x) for x in obj)
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
            for v in tags.values():
                low = str(v).lower()
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
            to_scan = []
            if "parameters" in metadata: to_scan.append(str(metadata["parameters"]))
            if "prompt" in metadata: to_scan.append(str(metadata["prompt"]))
            if "description" in metadata: to_scan.append(str(metadata["description"]))
            if "software" in metadata: to_scan.append(str(metadata["software"]))
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
    expl = (f"🔬 Локальна різкість (var Laplacian={lap_var:.4f}), "
            f"частотне насичення={high_ratio:.2f}")
    if ai_like:
        expl = ("🧠 Синтетична текстура: дуже низька локальна різкість (Laplacian Var) "
                "та висока частотна енергія ('пластмасовість')")

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
            msg = f"🖼️ Дуже низький рівень артефактів (STD={std_dev:.2f}) після ре-компресії (може бути перше збереження ШІ)"
            return True, msg
    except Exception:
        pass
    return False, None

def noise_analysis(ctx: ImgCtx):
    """
    Laplacian STD on a capped grayscale (fast FFT conv).
    """
    try:
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

    # Preserve your original threshold behavior
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

    base = ctx.rgb if max(w, h) <= 1536 else _cap_long_edge(ctx.rgb, 1536)
    bw, bh = base.size
    gray = ImageOps.grayscale(base)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    midx, midy = bw // 2, bh // 2
    quads = [
        arr[0:midy, 0:midx],
        arr[0:midy, midx:bw],
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
        msg = f"✂️ Неоднорідність шуму/текстури (Коеф. варіації={inconsistency_ratio:.2f}) — сильна ознака монтажу"
        return True, msg, {"inconsistency_ratio": inconsistency_ratio}
    return False, None, {"inconsistency_ratio": inconsistency_ratio}

# ---------- New lightweight checkers ----------

def ela_check(ctx: ImgCtx):
    if ctx.fmt != "JPEG":
        return False, None, {"ela_mean": 0.0, "ela_std": 0.0, "ela_p95": 0.0}
    try:
        base = _cap_long_edge(ctx.rgb, 1280)
        bio = io.BytesIO()
        base.save(bio, format="JPEG", quality=90)
        bio.seek(0)
        recompressed = Image.open(bio).convert("RGB")

        diff = ImageChops.difference(base, recompressed)
        arr = np.asarray(ImageOps.grayscale(diff), dtype=np.float32)

        ela_mean = float(np.mean(arr))
        ela_std  = float(np.std(arr))
        ela_p95  = float(np.percentile(arr, 95))

        ai_like = (ela_std < 3.0 and ela_p95 < 12.0)
        if ai_like:
            reason = (f"🧪 Дуже низький/однорідний ELA (mean={ela_mean:.2f}, "
                      f"std={ela_std:.2f}, p95={ela_p95:.1f}) — схоже на одноразове JPEG-збереження")
            return True, reason, {"ela_mean": ela_mean, "ela_std": ela_std, "ela_p95": ela_p95}
        return False, f"🔬 ELA: mean={ela_mean:.2f}, std={ela_std:.2f}, p95={ela_p95:.1f}", \
               {"ela_mean": ela_mean, "ela_std": ela_std, "ela_p95": ela_p95}
    except Exception:
        return False, "❌ Помилка ELA", {"ela_mean": -1.0, "ela_std": -1.0, "ela_p95": -1.0}

def periodicity_check(ctx: ImgCtx):
    a = ctx.gray_small_f32
    f = np.fft.fft2(a)
    mag = np.abs(np.fft.fftshift(f))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy)**2 + (X - cx)**2)

    rmin, rmax = 0.20 * min(cy, cx), 0.45 * min(cy, cx)
    ann = (R >= rmin) & (R <= rmax)
    ring = mag[ann]
    if ring.size == 0:
        return False, None, {"periodic_peak_ratio": 0.0}

    med = float(np.median(ring))
    if med <= 0:
        return False, None, {"periodic_peak_ratio": 0.0}

    topk = np.sort(ring.flatten())[-256:] if ring.size > 256 else ring
    peak_ratio = float(np.max(topk) / med)

    ai_like = peak_ratio > 6.0
    if ai_like:
        reason = f"🧩 Яскраві періодичні піки у спектрі (peak/median={peak_ratio:.1f}) — апскейл/‘checkerboard’"
        return True, reason, {"periodic_peak_ratio": peak_ratio}
    return False, None, {"periodic_peak_ratio": peak_ratio}

def banding_check(ctx: ImgCtx):
    g = (ctx.gray_u8).astype(np.int32)
    hist, _ = np.histogram(g, bins=256, range=(0, 255))
    zeros = int(np.sum(hist == 0))

    gy, gx = np.gradient(ctx.gray_f32)
    grad = np.hypot(gx, gy)
    ghist, _ = np.histogram(grad, bins=64, range=(0.0, float(np.max(grad) + 1e-6)))
    gzeros = int(np.sum(ghist == 0))

    zeros_ratio = zeros / 256.0
    gzeros_ratio = gzeros / 64.0

    ai_like = (zeros_ratio > 0.30 and gzeros_ratio > 0.25 and float(np.std(grad)) < 0.06)
    if ai_like:
        reason = (f"🪄 Сильна постеризація/бендінг (0-бінів: {zeros_ratio:.0%}, "
                  f"градієнтні 0-біни: {gzeros_ratio:.0%}) — синтетичні переходи")
        return True, reason, {"zeros_ratio": float(zeros_ratio), "grad_zeros_ratio": float(gzeros_ratio)}
    return False, None, {"zeros_ratio": float(zeros_ratio), "grad_zeros_ratio": float(gzeros_ratio)}

def symmetry_check(ctx: ImgCtx):
    arr = ctx.gray_small_f32
    h, w = arr.shape
    if w < 32:
        return False, None, {"sym_corr": 0.0}

    mid = w // 2
    left = arr[:, :mid]
    right = arr[:, w - mid:]
    right_flipped = np.flip(right, axis=1)

    l = (left - left.mean()) / (left.std() + 1e-6)
    r = (right_flipped - right_flipped.mean()) / (right_flipped.std() + 1e-6)

    corr = float(np.mean(l * r))
    ai_like = corr > 0.92
    if ai_like:
        return True, f"🪞 Ненормально висока двостороння симетрія (corr={corr:.2f})", {"sym_corr": corr}
    return False, None, {"sym_corr": corr}

def palette_compactness_check(ctx: ImgCtx):
    tiny = ctx.rgb.resize((128, max(1, int(128 * ctx.h / max(ctx.w, 1)))), Image.Resampling.BILINEAR)
    arr = np.asarray(tiny, dtype=np.uint8)
    flat = arr.reshape(-1, 3)
    packed = (flat[:,0].astype(np.uint32) << 16) | (flat[:,1].astype(np.uint32) << 8) | flat[:,2].astype(np.uint32)
    uniq = int(np.unique(packed).size)

    ai_like = (max(ctx.w, ctx.h) >= 1024 and uniq < 1800)
    if ai_like:
        return True, f"🎨 Дуже компактна палітра ({uniq} унікальних кольорів @128px) — непритаманно для фото", {"unique_colors_128": uniq}
    return False, None, {"unique_colors_128": uniq}

# ---------- OCR consistency ----------

def _binarize_for_ocr(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    arr = np.asarray(g, dtype=np.uint8)
    thr = max(120, int(np.mean(arr) * 0.9))
    mask = (arr > thr).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")

def ocr_consistency_check(ctx: ImgCtx):
    if not _HAS_TESS:
        return False, None, {"available": False}

    base = _cap_long_edge(ctx.rgb, 1600)
    imgs = [base, _binarize_for_ocr(base)]
    best_mean = None
    best_data = None

    for im in imgs:
        try:
            data = pytesseract.image_to_data(im, lang="eng+ukr", output_type=pytesseract.Output.DICT)
            confs_raw = data.get("conf", [])
            confs = []
            for c in confs_raw:
                try:
                    confs.append(float(c))
                except Exception:
                    pass
            if not confs:
                continue
            mean_conf = float(np.mean(confs))
            if best_mean is None or mean_conf > best_mean:
                best_mean = mean_conf
                best_data = data
        except Exception:
            continue

    if not best_data:
        return False, "🔎 Текст не виявлено або OCR не спрацював", {"available": True, "has_text": False}

    words = best_data.get("text", []) or []
    confs = []
    for c in best_data.get("conf", []):
        try:
            confs.append(float(c))
        except Exception:
            pass
    mean_conf = float(np.mean(confs)) if confs else 0.0
    low_conf_ratio = float(np.mean([c < 45.0 for c in confs])) if confs else 1.0

    text_join = " ".join([w for w in words if isinstance(w, str) and w.strip()])
    weird = 0
    for c in text_join:
        if not (("0" <= c <= "9") or ("A" <= c <= "Z") or ("a" <= c <= "z") or ("\u0400" <= c <= "\u04FF") or c in " -_.,:;!?()[]\"'"):
            weird += 1
    weird_ratio = float(weird / max(1, len(text_join)))

    ys = best_data.get("top", []) or []
    hs = best_data.get("height", []) or []
    lines = {}
    for y, h in zip(ys, hs):
        try:
            y = int(y); h = int(h)
        except Exception:
            continue
        key = int(round(y / 12))
        lines.setdefault(key, []).append((y, h))
    baselines = [int(np.median([y + h for (y, h) in v])) for v in lines.values() if v]
    spacing = np.diff(sorted(set(baselines))) if len(baselines) > 1 else np.array([0])
    spacing_cv = float(np.std(spacing) / (np.mean(spacing) + 1e-6)) if spacing.size > 0 else 0.0

    ai_like = (mean_conf < 55.0 and low_conf_ratio > 0.50) or (weird_ratio > 0.12 and spacing_cv > 0.45)
    if ai_like:
        reason = (f"🔤 Неконсистентний текст/гліфи: низька якість OCR (mean={mean_conf:.1f}, "
                  f"low≥50%={low_conf_ratio:.0%}), дивні символи={weird_ratio:.0%}, нерівномірні інтервали (CV={spacing_cv:.2f})")
        return True, reason, {
            "available": True, "has_text": True,
            "ocr_mean_conf": mean_conf,
            "ocr_low_conf_ratio": low_conf_ratio,
            "ocr_weird_char_ratio": weird_ratio,
            "ocr_spacing_cv": spacing_cv,
        }

    return False, None, {
        "available": True, "has_text": True,
        "ocr_mean_conf": mean_conf,
        "ocr_low_conf_ratio": low_conf_ratio,
        "ocr_weird_char_ratio": weird_ratio,
        "ocr_spacing_cv": spacing_cv,
    }

# ---------- Face-region focused (Ultraface) ----------

_ULTRA_SESSION = None

def _ultraface_load():
    global _ULTRA_SESSION
    if _ULTRA_SESSION is not None:
        return _ULTRA_SESSION
    if not _HAS_ORT:
        return None
    if not os.path.exists(ULTRAFACE_ONNX):
        return None
    providers = ["CPUExecutionProvider"]
    _ULTRA_SESSION = ort.InferenceSession(ULTRAFACE_ONNX, providers=providers)
    return _ULTRA_SESSION

def _ultraface_detect(rgb: Image.Image):
    """
    Returns (x1,y1,x2,y2,score) in original image coords or None.
    """
    sess = _ultraface_load()
    if sess is None:
        return None

    W, H = ULTRAFACE_INPUT_SIZE
    im = rgb.resize((W, H), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    inp = np.transpose(arr, (2, 0, 1))[None, ...] / 255.0
    inputs = {sess.get_inputs()[0].name: inp}

    outs = sess.run(None, inputs)
    # Expecting boxes + scores (N,4) & (N,2). Adjust if your model differs.
    scores = outs[0] if len(outs) > 0 else None
    boxes = outs[1] if len(outs) > 1 else None
    if scores is None or boxes is None or scores.shape[0] == 0:
        return None

    probs = scores[:, 1]
    idx = int(np.argmax(probs))
    if float(probs[idx]) < 0.5:
        return None

    x1, y1, x2, y2 = boxes[idx]
    ow, oh = rgb.size
    x1 = max(0, min(int(x1 * ow), ow - 1))
    y1 = max(0, min(int(y1 * oh), oh - 1))
    x2 = max(0, min(int(x2 * ow), ow - 1))
    y2 = max(0, min(int(y2 * oh), oh - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2, float(probs[idx]))

def _fft_peak_ratio(a: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    f = np.fft.fft2(a)
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy)**2 + (X - cx)**2)
    rmin, rmax = 0.20 * min(cy, cx), 0.45 * min(cy, cx)
    ann = mag[(R >= rmin) & (R <= rmax)]
    if ann.size == 0:
        return 0.0
    med = float(np.median(ann)) or 1e-6
    return float(np.max(ann) / med)

def face_region_checks(ctx: ImgCtx):
    face = _ultraface_detect(ctx.rgb)
    if face is None:
        return False, None, {"available": (_HAS_ORT and os.path.exists(ULTRAFACE_ONNX)), "found_face": False}

    x1, y1, x2, y2, _ = face
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    face_img = ctx.rgb.crop((x1, y1, x2, y2))

    fx, fy = face_img.size
    eye_h = max(4, int(0.18 * h))
    eye_w = max(4, int(0.28 * w))
    eye_y = max(0, int(0.30 * h))
    left_eye_box  = (int(0.18 * w), eye_y, int(0.18 * w) + eye_w, eye_y + eye_h)
    right_eye_box = (int(0.54 * w), eye_y, int(0.54 * w) + eye_w, eye_y + eye_h)

    mouth_h = max(6, int(0.20 * h))
    mouth_w = max(8, int(0.50 * w))
    mouth_x = int(0.25 * w)
    mouth_y = int(0.65 * h)
    mouth_box = (mouth_x, mouth_y, mouth_x + mouth_w, mouth_y + mouth_h)

    gray = ImageOps.grayscale(face_img)
    arr = np.asarray(gray, dtype=np.float32) / 255.0

    lx1, ly1, lx2, ly2 = left_eye_box
    rx1, ry1, rx2, ry2 = right_eye_box
    sym_corr = 0.0
    try:
        left_eye = arr[ly1:ly2, lx1:lx2]
        right_eye = arr[ry1:ry2, rx1:rx2]
        if left_eye.size and right_eye.size and left_eye.shape == right_eye.shape:
            l = (left_eye - left_eye.mean()) / (left_eye.std() + 1e-6)
            r = np.flip(right_eye, axis=1)
            r = (r - r.mean()) / (r.std() + 1e-6)
            sym_corr = float(np.mean(l * r))
    except Exception:
        sym_corr = 0.0

    hsv = face_img.convert("HSV")
    hsv_arr = np.asarray(hsv, dtype=np.float32) / 255.0
    mask_regions = []
    for (x1e, y1e, x2e, y2e) in (left_eye_box, right_eye_box):
        try:
            region = hsv_arr[y1e:y2e, x1e:x2e, :]
            if region.size:
                S = region[:, :, 1]; V = region[:, :, 2]
                sclera = (V > 0.8) & (S < 0.2)
                if np.any(sclera):
                    mask_regions.append(V[sclera])
        except Exception:
            pass
    if mask_regions:
        sclera_V = np.concatenate([m.ravel() for m in mask_regions])
        sclera_mean = float(np.mean(sclera_V))
        sclera_std  = float(np.std(sclera_V))
    else:
        sclera_mean, sclera_std = 0.0, 0.0

    mx1, my1, mx2, my2 = mouth_box
    mouth = arr[my1:my2, mx1:mx2]
    teeth_peak = _fft_peak_ratio(mouth) if mouth.size else 0.0

    eye_sym_ai = sym_corr > 0.93
    sclera_ai  = (sclera_mean > 0.9 and sclera_std < 0.04 and (len(mask_regions) > 0))
    teeth_ai   = teeth_peak > 7.5

    ai_like = eye_sym_ai or sclera_ai or teeth_ai
    parts = []
    if eye_sym_ai: parts.append(f"👁️ Надмірна симетрія очей (corr={sym_corr:.2f})")
    if sclera_ai:  parts.append(f"👀 Однорідні білки очей (meanV={sclera_mean:.2f}, stdV={sclera_std:.3f})")
    if teeth_ai:   parts.append(f"🦷 Періодичність у зоні зубів (FFT peak ratio={teeth_peak:.1f})")
    reason = "; ".join(parts) if parts else None

    return ai_like, reason, {
        "available": True, "found_face": True,
        "eye_sym_corr": sym_corr,
        "sclera_mean_v": sclera_mean,
        "sclera_std_v": sclera_std,
        "teeth_peak_ratio": teeth_peak,
        "face_box": [int(x1), int(y1), int(x2), int(y2)]
    }

# ---------- ONNX fusion model ----------

_ORT_CLS = None
_FEATURE_ORDER = [
    "ai_like_freq", "low_artifact_hint", "ai_too_smooth", "ai_color_weird",
    "ai_inconsistent", "jpeg_quant_weird",
    "mul64", "square_common", "is_uncommon_ratio",
    "freq.lap_var", "freq.high_ratio",
    "noise.noise_std",
    "color_stats.mean_s", "color_stats.high_s_ratio",
    "noise_inconsistency.inconsistency_ratio",
    "ela.ela_mean", "ela.ela_std", "ela.ela_p95",
    "periodicity.periodic_peak_ratio",
    "banding.zeros_ratio", "banding.grad_zeros_ratio",
    "symmetry.sym_corr",
    "palette.unique_colors_128",
    "ocr.ocr_mean_conf", "ocr.ocr_low_conf_ratio", "ocr.ocr_weird_char_ratio", "ocr.ocr_spacing_cv",
    "face.eye_sym_corr", "face.sclera_mean_v", "face.sclera_std_v", "face.teeth_peak_ratio",
    "size.max_dim"
]

def _get_from_checks(checks: dict, path: str, default=0.0):
    if "." not in path:
        v = checks.get(path, default)
        if isinstance(v, bool): return float(v)
        try:
            return float(v)
        except Exception:
            return float(default)
    head, tail = path.split(".", 1)
    d = checks.get(head, {})
    if not isinstance(d, dict):
        return float(default)
    v = d.get(tail, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _build_feature_vector(checks: dict) -> np.ndarray:
    size = checks.get("size", (0, 0))
    try:
        max_dim = float(max(size)) if isinstance(size, (list, tuple)) and size else 0.0
    except Exception:
        max_dim = 0.0

    vec = []
    for key in _FEATURE_ORDER:
        if key == "size.max_dim":
            vec.append(max_dim)
        else:
            vec.append(_get_from_checks(checks, key, 0.0))
    return np.array(vec, dtype=np.float32)[None, :]

def _load_onnx_classifier():
    global _ORT_CLS
    if _ORT_CLS is not None:
        return _ORT_CLS
    if not (_HAS_ORT and os.path.exists(AIUNCOVER_ONNX_MODEL)):
        return None
    _ORT_CLS = ort.InferenceSession(AIUNCOVER_ONNX_MODEL, providers=["CPUExecutionProvider"])
    return _ORT_CLS

def onnx_fusion_predict(checks: dict):
    sess = _load_onnx_classifier()
    if sess is None:
        return False, 0.0, {}
    x = _build_feature_vector(checks)
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: x})
    prob = float(out[0].ravel()[0])
    return True, prob, {"model": os.path.basename(AIUNCOVER_ONNX_MODEL), "prob": prob}

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

    # Open & build ctx
    try:
        img = load_image(raw)
    except Exception:
        explanations.append("❌ Не вдалося відкрити зображення")
        return AnalyzeResponse(prob_ai=0.8, explanations=explanations, checks={"open_error": True})

    ctx = ImgCtx(img)
    checks["format"] = ctx.fmt

    # 1) EXIF/XMP
    has_exif, ai_tool_found_exif, hints = read_exif_hints(raw, ctx.fmt)
    explanations += hints
    checks["has_exif"] = has_exif
    checks["ai_tool_found_exif"] = ai_tool_found_exif

    # 2) PNG metadata
    ai_found_png, png_reasons = png_metadata_check(ctx.img, ctx.fmt)
    checks["ai_found_png_metadata"] = ai_found_png
    explanations += png_reasons

    # 3) size & ratio
    size_flags, size_reasons = size_checks(ctx.img)
    checks.update(size_flags)
    explanations += size_reasons

    # 4) alpha
    weird_alpha, alpha_reason = alpha_channel_weird(ctx.img, ctx.fmt)
    checks["weird_alpha"] = weird_alpha
    if alpha_reason:
        explanations.append(alpha_reason)

    # 5) frequency
    ai_like_freq, freq_expl, freq_vals = high_freq_heuristic(ctx)
    checks["ai_like_freq"] = ai_like_freq
    checks["freq"] = freq_vals
    explanations.append(freq_expl)

    # 6) JPEG quant
    q_hint, q_reason = jpeg_quant_hint(ctx.img, ctx.fmt)
    checks["jpeg_quant_weird"] = q_hint
    if q_reason:
        explanations.append(q_reason)

    # 7) JPEG artifact hint
    low_artifact_hint, artifact_reason = jpeg_artifact_hint(ctx)
    checks["low_artifact_hint"] = low_artifact_hint
    if artifact_reason:
        explanations.append(artifact_reason)

    # 8) noise
    ai_too_smooth, noise_expl, noise_vals = noise_analysis(ctx)
    checks["ai_too_smooth"] = ai_too_smooth
    checks["noise"] = noise_vals
    explanations.append(noise_expl)

    # 9) color stats
    ai_color_weird, color_reasons, color_vals = color_statistic_check(ctx)
    checks["ai_color_weird"] = ai_color_weird
    checks["color_stats"] = color_vals
    explanations += color_reasons

    # 10) noise inconsistency
    ai_inconsistent, inconsistency_reason, inconsistency_vals = noise_inconsistency_check(ctx)
    checks["ai_inconsistent"] = ai_inconsistent
    checks["noise_inconsistency"] = inconsistency_vals
    if inconsistency_reason:
        explanations.append(inconsistency_reason)

    # 11) Additional lightweight checks
    ai_ela, ela_reason, ela_vals = ela_check(ctx)
    checks["ai_ela"] = ai_ela
    checks["ela"] = ela_vals
    if ela_reason: explanations.append(ela_reason)

    ai_periodic, periodic_reason, periodic_vals = periodicity_check(ctx)
    checks["ai_periodic"] = ai_periodic
    checks["periodicity"] = periodic_vals
    if periodic_reason: explanations.append(periodic_reason)

    ai_banding, banding_reason, banding_vals = banding_check(ctx)
    checks["ai_banding"] = ai_banding
    checks["banding"] = banding_vals
    if banding_reason: explanations.append(banding_reason)

    ai_sym, sym_reason, sym_vals = symmetry_check(ctx)
    checks["ai_symmetry"] = ai_sym
    checks["symmetry"] = sym_vals
    if sym_reason: explanations.append(sym_reason)

    ai_palette, palette_reason, palette_vals = palette_compactness_check(ctx)
    checks["ai_palette_compact"] = ai_palette
    checks["palette"] = palette_vals
    if palette_reason: explanations.append(palette_reason)

    # 12) OCR consistency (optional)
    ai_ocr, ocr_reason, ocr_vals = ocr_consistency_check(ctx)
    checks["ai_ocr_inconsistent"] = ai_ocr
    checks["ocr"] = ocr_vals
    if ocr_reason: explanations.append(ocr_reason)

    # 13) Face-region (optional)
    ai_face, face_reason, face_vals = face_region_checks(ctx)
    checks["ai_face_flags"] = ai_face
    checks["face"] = face_vals
    if face_reason: explanations.append(face_reason)

    # ---- Heuristic scoring ----
    score = 0.0
    if ai_tool_found_exif or ai_found_png: score += 0.40
    if (not has_exif) and (not ai_found_png) and ctx.fmt == "JPEG": score += 0.20
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

    # new signals (gentle weights)
    if ai_ela:                            score += 0.12
    if ai_periodic:                       score += 0.18
    if ai_banding:                        score += 0.10
    if ai_sym:                            score += 0.08
    if ai_palette:                        score += 0.08
    if ai_ocr:                            score += 0.15
    if ai_face:                           score += 0.18

    prob_ai = float(max(0.0, min(1.0, score)))
    w, h = size_flags["size"]
    max_dim = max(w, h)
    if max_dim < 384:
        prob_ai = min(1.0, prob_ai + 0.05)
    if max_dim > 2048 and not ai_tool_found_exif:
        prob_ai = max(0.0, prob_ai - 0.10)

    # ONNX fusion (optional)
    ort_avail, ort_prob, ort_info = onnx_fusion_predict(checks)
    checks["onnx_fusion"] = {"available": ort_avail, **ort_info}
    if ort_avail:
        prob_ai = 0.6 * prob_ai + 0.4 * float(max(0.0, min(1.0, ort_prob)))

    final_checks = _sanitize(checks)

    return AnalyzeResponse(
        prob_ai=round(prob_ai, 2),
        explanations=explanations or ["✅ Ознак ШІ не виявлено явних (зображення виглядає як звичайне фото)"],
        checks=final_checks,
    )
