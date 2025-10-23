
from .exif import has_exif
from .c2pa_stub import check_c2pa
from .model_stub import score_model_stub

def analyze_image_stub(image_bytes: bytes, filename: str) -> dict:
    exif_present = has_exif(image_bytes)
    has_c2pa = check_c2pa(image_bytes)
    model_score = score_model_stub(image_bytes)

    explanations = []
    if not has_c2pa:
        explanations.append("No C2PA (stub)")
    explanations.append("EXIF present" if exif_present else "EXIF stripped or unreadable")
    explanations.append("High-frequency artefacts (stub)")  # placeholder for future FFT checks

    # Combine scores (very naive): if EXIF present, slightly reduce prob_ai
    prob_ai = model_score
    if exif_present:
        prob_ai = max(0.0, prob_ai - 0.08)
    if has_c2pa:
        prob_ai = max(0.0, prob_ai - 0.25)

    # Round for nicer UI
    prob_ai = round(prob_ai, 4)

    return {
        "prob_ai": prob_ai,
        "explanations": explanations,
        "checks": {
            "c2pa": has_c2pa,
            "exif_present": exif_present,
            "model_score": model_score
        }
    }
