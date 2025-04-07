import os
import openai
from typing import List, Dict, Any
import logging
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        """Initialize the chat service with OpenAI API key."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.warning("OPENAI_API_KEY not set. Chat functionality will be disabled.")
            self.enabled = False
            return
            
        self.enabled = True
        try:
            # Initialize OpenAI client with custom HTTP client
            http_client = httpx.Client(timeout=30.0)
            self.client = openai.OpenAI(
                api_key=self.api_key,
                http_client=http_client
            )
            
            # Set up the system message for plant care advice
            self.system_message = {
                "role": "system",
                "content": """You are an expert in plant care and disease diagnosis. 
                Provide detailed, accurate advice about plant health, disease treatment, and prevention.
                Be specific about treatment methods and timing.
                Consider environmental factors and provide practical recommendations."""
            }
            
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.enabled = False

    async def get_chat_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get a response from the chat model."""
        if not self.enabled:
            return {
                "error": "Chat service is disabled. Please set OPENAI_API_KEY environment variable.",
                "response": "I'm sorry, but the chat functionality is currently unavailable. Please try again later."
            }
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return {
                "response": response.choices[0].message.content,
                "error": None
            }
        except Exception as e:
            logging.error(f"Error getting chat response: {str(e)}")
            return {
                "error": str(e),
                "response": "I'm sorry, but I encountered an error while processing your request. Please try again later."
            }

    async def get_response(self, message: str, language: str = "en") -> str:
        try:
            logger.info(f"Sending message to OpenAI: {message[:50]}...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    self.system_message,
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            if not response.choices or not response.choices[0].message:
                raise Exception("No response received from OpenAI")
                
            return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error in chat service: {error_message}")
            
            if "insufficient_quota" in error_message:
                return "I apologize, but I'm currently unable to process your request due to service limitations. Please try again later or contact the system administrator."
            elif "rate_limit_exceeded" in error_message:
                return "I'm receiving too many requests at the moment. Please wait a few seconds and try again."
            else:
                return "I'm sorry, but I encountered an error while processing your request. Please try again later." 