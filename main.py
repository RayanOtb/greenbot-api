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

app = FastAPI(
    title="GreenBot API",
    description="API for plant disease detection and chatbot assistance",
    version="1.0.0"
)

# Initialize services
plant_model = PlantDiseaseModel()
chat_service = ChatService()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return {"message": "Welcome to GreenBot API"}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Create a temporary file to save the uploaded image
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Analyze the image
            result = plant_model.analyze_image(temp_file_path)
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            
            if "error" in result:
                raise Exception(result["error"])
                
            return result
        except Exception as e:
            # Ensure temporary file is cleaned up even if analysis fails
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise e
            
    except Exception as e:
        print(f"Error in analyze_image endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze image: {str(e)}"
        )

@app.post("/chat")
async def chat(message: str = Body(...), language: str = Body("en")):
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
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "response": str(e),
                "language": language
            }
        )

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 