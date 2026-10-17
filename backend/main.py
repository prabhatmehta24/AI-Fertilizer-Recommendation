from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from predict import predict, SOIL_CLASSES, CROP_CLASSES
from schemas import ChatRequest, FeedbackRequest, PredictionRequest

FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="AI Smart Agriculture Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_store = []
feedback_store = []


@app.get("/")
def home():
    return {
        "message": "AI Smart Agriculture Assistant API",
        "endpoints": ["/predict", "/chat", "/history", "/feedback"],
    }


@app.post("/predict")
def predict_api(data: PredictionRequest):
    try:
        result = predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history_store.append(
        {
            "fertilizer": result["recommended_fertilizer"],
            "crop": data.Crop_Type,
            "soil": data.Soil_Type,
            "confidence": result["confidence"],
        }
    )

    return result


@app.get("/options")
def options_api():
    return {
        "soil_types": SOIL_CLASSES,
        "crop_types": CROP_CLASSES,
    }


@app.post("/chat")
def chat_api(payload: ChatRequest):
    question = payload.question.lower()
    fertilizer = payload.fertilizer or "the recommended fertilizer"

    if "why" in question:
        answer = (
            f"{fertilizer} was selected because it best matches the current nutrient balance and crop conditions."
        )
    elif "apply" in question or "use" in question:
        answer = (
            "Apply the fertilizer evenly around the root zone, avoid direct contact with the stem, and water the soil lightly after application."
        )
    elif "nitrogen" in question:
        answer = "If nitrogen is low, a nitrogen-rich blend or a balanced supplement can help restore the crop's growth potential."
    else:
        answer = "I can explain the recommendation, suggest safe application practices, or discuss nutrient concerns for your crop."

    return {"answer": answer, "source": "rule-based assistant"}


@app.get("/history")
def history_api():
    return {"history": history_store[-10:][::-1]}


@app.post("/feedback")
def feedback_api(payload: FeedbackRequest):
    feedback_store.append(
        {
            "rating": payload.rating,
            "message": payload.message,
            "fertilizer": payload.fertilizer,
        }
    )
    return {"message": "Thank you for your feedback.", "received": True}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")