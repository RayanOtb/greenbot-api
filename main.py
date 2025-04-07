from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import os
from dotenv import load_dotenv
from models.plant_disease_model import PlantDiseaseModel
from services.chat_service import ChatService
import tempfile
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GreenBot API",
    description="API for plant disease detection and chatbot assistance",
    version="1.0.0"
)

# Initialize services
try:
    plant_model = PlantDiseaseModel()
    logger.info("Plant disease model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load plant disease model: {str(e)}")
    raise

try:
    chat_service = ChatService()
    if chat_service.enabled:
        logger.info("Chat service initialized successfully")
    else:
        logger.warning("Chat service is disabled - OPENAI_API_KEY not set")
except Exception as e:
    logger.error(f"Failed to initialize chat service: {str(e)}")
    chat_service = None

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {
        "message": "Welcome to GreenBot API",
        "status": "operational",
        "services": {
            "plant_disease_detection": "enabled",
            "chat": "enabled" if chat_service and chat_service.enabled else "disabled"
        }
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded image for plant diseases."""
    try:
        logger.info(f"Received image upload request: {file.filename}")
        
        if not file.content_type.startswith('image/'):
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        try:
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
                logger.info(f"Temporary file created: {temp_path}")
            
            # Analyze the image
            logger.info("Starting image analysis...")
            result = plant_model.analyze_image(temp_path)
            logger.info("Image analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Temporary file removed: {temp_path}")
    except Exception as e:
        logger.error(f"Error in analyze_image endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(message: str, language: str = "en"):
    """Handle chat messages."""
    if not chat_service or not chat_service.enabled:
        raise HTTPException(
            status_code=503,
            detail="Chat service is currently unavailable. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        logger.info(f"Received chat request - Message: {message[:50]}..., Language: {language}")
        response = await chat_service.get_response(message, language)
        logger.info("Successfully got response from chat service")
        return JSONResponse(
            content={
                "status": "success",
                "response": response,
                "language": language
            }
        )
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 