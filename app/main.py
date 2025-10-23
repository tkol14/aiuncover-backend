
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AIUncover API", version="1.0")

ALLOWED_ORIGINS = ["https://aiuncover.net", "https://api.aiuncover.net",
                   "https://6my6x5ic.up.railway.app", "https://aiuncover-backend-production.up.railway.app"
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
    return {"status":"ok"}

@app.get("/")
def root():
    return {"ok":True}

@app.post("/analyze/image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()
    return {
        "prob_ai":0.5,
        "explanations":["Це демонстраційна відповідь"],
        "checks":{"stub":True}
    }
