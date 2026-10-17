import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "fertilizer_model.pkl"
SOIL_ENCODER_PATH = BASE_DIR / "saved_models" / "soil_encoder.pkl"
CROP_ENCODER_PATH = BASE_DIR / "saved_models" / "crop_encoder.pkl"
TARGET_ENCODER_PATH = BASE_DIR / "saved_models" / "target_encoder.pkl"

model = joblib.load(MODEL_PATH)
soil_encoder = joblib.load(SOIL_ENCODER_PATH)
crop_encoder = joblib.load(CROP_ENCODER_PATH)
target_encoder = joblib.load(TARGET_ENCODER_PATH)

SOIL_CLASSES = soil_encoder.classes_.tolist()
CROP_CLASSES = crop_encoder.classes_.tolist()
FERTILIZER_CLASSES = target_encoder.classes_.tolist()


def _validate_category(value, valid_values, name):
    if value not in valid_values:
        valid_text = ", ".join(valid_values)
        raise ValueError(f"{name} must be one of: {valid_text}.")


def _build_features(data):
    _validate_category(data.Soil_Type, SOIL_CLASSES, "Soil_Type")
    _validate_category(data.Crop_Type, CROP_CLASSES, "Crop_Type")

    soil = soil_encoder.transform([data.Soil_Type])[0]
    crop = crop_encoder.transform([data.Crop_Type])[0]

    nitrogen = data.Nitrogen
    phosphorous = data.Phosphorous
    potassium = data.Potassium

    features = pd.DataFrame([{
        "Temparature": data.Temparature,
        "Humidity": data.Humidity,
        "Moisture": data.Moisture,
        "Soil Type": soil,
        "Crop Type": crop,
        "Nitrogen": nitrogen,
        "Potassium": potassium,
        "Phosphorous": phosphorous,
        "total_npk": nitrogen + phosphorous + potassium,
        "n_p_ratio": nitrogen / (phosphorous + 1),
        "n_k_ratio": nitrogen / (potassium + 1),
        "p_k_ratio": phosphorous / (potassium + 1),
        "low_n": int(nitrogen < 60),
        "low_p": int(phosphorous < 50),
        "low_k": int(potassium < 50),
    }])

    return features[[
        "Temparature",
        "Humidity",
        "Moisture",
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "total_npk",
        "n_p_ratio",
        "n_k_ratio",
        "p_k_ratio",
        "low_n",
        "low_p",
        "low_k",
    ]]


def _build_explanation(data, fertilizer: str) -> str:
    nutrient_focus = []
    if data.Nitrogen < 60:
        nutrient_focus.append("low nitrogen")
    if data.Phosphorous < 50:
        nutrient_focus.append("low phosphorous")
    if data.Potassium < 50:
        nutrient_focus.append("low potassium")

    if nutrient_focus:
        issue_text = " and ".join(nutrient_focus)
        return (
            f"{fertilizer} is recommended because the input profile points to {issue_text}, "
            f"which can benefit from a balanced nutrient boost for {data.Crop_Type} grown in {data.Soil_Type} soil."
        )

    return (
        f"{fertilizer} is recommended as a balanced option for the given soil and crop conditions. "
        f"The model sees this as the most suitable fertilizer for the current nutrient profile."
    )


def predict(data):
    features = _build_features(data)
    prediction = model.predict(features)[0]
    fertilizer = str(target_encoder.inverse_transform([prediction])[0])

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features)[0].max())

    return {
        "recommended_fertilizer": fertilizer,
        "confidence": round(confidence, 3),
        "explanation": _build_explanation(data, fertilizer),
    }