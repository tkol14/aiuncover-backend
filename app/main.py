from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import exifread
import requests

app = FastAPI(title="AIUncover API", version="1.1.0")

ALLOWED_ORIGINS = [
    "https://aiuncover.net",
    "https://api.aiuncover.net",
    "https://aiuncover-backend-production.up.railway.app"
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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze/image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()

    explanations = []
    checks = {}

    # === ПЕРЕВІРКА 1: EXIF метадані ===
    try:
        tags = exifread.process_file(io.BytesIO(content))
        if not tags:
            explanations.append("Відсутні EXIF метадані – характерно для зображень згенерованих ШІ")
            checks["exif"] = True
        else:
            explanations.append("EXIF метадані знайдені – зображення скоріш за все реальне")
            checks["exif"] = False
    except Exception:
        explanations.append("Помилка при аналізі EXIF")
        checks["exif"] = None

    # === ПЕРЕВІРКА 2: Спотворення зображення ===
    try:
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        if width % 8 != 0 or height % 8 != 0:
            explanations.append("Розміри зображення не кратні 64 – типово для ШІ генераторів")
            checks["size_ai_pattern"] = True
        else:
            explanations.append("Розміри зображення виглядають як фотографія")
            checks["size_ai_pattern"] = False
    except:
        explanations.append("Не вдалося відкрити зображення")
        checks["size_ai_pattern"] = None

    # === ПЕРЕВІРКА 3: Використання зовнішнього API (наприклад OpenAI Vision) ===
    # Можемо зробити на наступному етапі

    # === Фінальний розрахунок AI score ===
    ai_signals = sum(1 for k, v in checks.items() if v is True)
    prob_ai = round(min(1.0, ai_signals * 0.4), 2)

    return AnalyzeResponse(
        prob_ai=prob_ai,
        explanations=explanations,
        checks=checks
    )