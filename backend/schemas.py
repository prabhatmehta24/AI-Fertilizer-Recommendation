from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Temparature: float = Field(..., ge=0, le=60, description="Temperature in Celsius")
    Humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    Moisture: float = Field(..., ge=0, le=100, description="Soil moisture percentage")
    Soil_Type: str = Field(..., description="Name of the soil type")
    Crop_Type: str = Field(..., description="Name of the crop")
    Nitrogen: float = Field(..., ge=0, le=100, description="Nitrogen level")
    Potassium: float = Field(..., ge=0, le=100, description="Potassium level")
    Phosphorous: float = Field(..., ge=0, le=100, description="Phosphorous level")


class ChatRequest(BaseModel):
    question: str
    fertilizer: str | None = None


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    message: str
    fertilizer: str | None = None