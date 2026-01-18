import os
import pickle
import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL_ID = "davlan/afro-xlmr-small"
ADAPTER_PATH = "models/final_lora_adapter"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads model, tokenizer, and encoder on startup.
    Cleans up on shutdown.
    """
    logger.info("Starting Irembo Intent API...")
    
    # Load Label Encoder
    if not os.path.exists(LABEL_ENCODER_PATH):
        logger.error(f"CRITICAL: Label Encoder not found at {LABEL_ENCODER_PATH}")
        raise FileNotFoundError("Run training first to generate label_encoder.pkl")
        
    with open(LABEL_ENCODER_PATH, "rb") as f:
        ml_resources["le"] = pickle.load(f)
    logger.info(f"Label Encoder loaded. Classes: {ml_resources['le'].classes_}")

    # Load Tokenizer
    ml_resources["tokenizer"] = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    logger.info("Tokenizer loaded.")

    # Load Model (Base + LoRA)
    logger.info("Loading AfroXLMR model... this might take a moment.")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID, 
        num_labels=len(ml_resources["le"].classes_)
    )
    
    # Check if adapter exists, otherwise load base only (fallback)
    if os.path.exists(ADAPTER_PATH):
        ml_resources["model"] = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        logger.info("LoRA Adapter successfully merged.")
    else:
        logger.warning("LoRA Adapter not found! Loading base model only (Performance will be poor).")
        ml_resources["model"] = base_model
        
    ml_resources["model"].to(DEVICE)
    ml_resources["model"].eval() # Set to inference mode
    
    yield # API is now running
    
    # Cleanup
    ml_resources.clear()
    logger.info("🛑 API Shutting down.")

# --- API INITIALIZATION ---
app = FastAPI(title="Irembo Voice AI Intent API", version="1.0.0", lifespan=lifespan)

# --- DATA MODELS (Input/Output Validation) ---
class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Ndashaka kureba niba pasipoti yanjye yabonetse"])

class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    processing_time_ms: float = 0.0

class HealthResponse(BaseModel):
    status: str
    device: str
    model_version: str

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {"message": "Irembo Voice AI Intent API is running. Please visit /docs to test endpoints."}

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Simple probe to check if the API is alive and ready.
    """
    return {
        "status": "ready" if "model" in ml_resources else "loading",
        "device": DEVICE,
        "model_version": "AfroXLMR-Small + LoRA"
    }

@app.get("/info")
def get_metadata():
    """
    Returns supported intents for frontend synchronization.
    """
    if "le" not in ml_resources:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return {
        "supported_intents": list(ml_resources["le"].classes_),
        "total_intents": len(ml_resources["le"].classes_)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_intent(request: QueryRequest):
    """
    Main inference endpoint. Converts text -> intent.
    """
    if "model" not in ml_resources:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Timer for latency tracking
    import time
    start_time = time.time()
    
    try:
        # Tokenize
        inputs = ml_resources["tokenizer"](
            request.text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = ml_resources["model"](**inputs)
            logits = outputs.logits
            
            # Calculate probabilities
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][int(pred_id)].item()
            
        # Decode Intent
        predicted_intent = ml_resources["le"].inverse_transform([pred_id])[0]
        
        duration = (time.time() - start_time) * 1000
        
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "processing_time_ms": round(duration, 2)
        }

    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")