
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .services.analyze import analyze_image_stub

app = FastAPI(title="AI Detector MVP", version="0.1.0")

# CORS (allow local dev from 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image/* files are accepted")
    try:
        content = await file.read()
        result = analyze_image_stub(content, filename=file.filename or "uploaded")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
