from PIL import Image
import numpy as np
import os
from typing import Dict, List, Tuple
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import requests
from io import BytesIO
import logging

# Configure TensorFlow for memory optimization
try:
    # Only set memory growth if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Set soft device placement for better compatibility
    tf.config.set_soft_device_placement(True)
except Exception as e:
    logging.warning(f"Could not configure TensorFlow memory settings: {str(e)}")

class PlantDiseaseModel:
    def __init__(self):
        """Initialize the plant disease detection model."""
        self.model = self._load_pretrained_model()
        self.disease_db = self._load_disease_database()
        self.class_names = self._get_class_names()
        
    def _load_pretrained_model(self):
        """Load the pre-trained model."""
        model_path = 'models/plant_disease_model.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Please run setup_model.py to create the model."
            )
            
        try:
            # Load model with memory optimization
            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
            
    def _get_class_names(self):
        """Get the list of class names for the model."""
        return [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Corn_(maize)___Cercospora_leaf_spot', 'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
            # Additional classes to match the model's 38 classes
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Squash___Powdery_mildew', 'Squash___healthy',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Peach___Bacterial_spot', 'Peach___healthy',
            'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Blueberry___healthy',
            'Raspberry___healthy',
            'Soybean___healthy'
        ]
        
    def _preprocess_image(self, image_path):
        """Preprocess the image for model input."""
        try:
            # Open and convert image to RGB
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize((224, 224))
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            logging.info(f"Preprocessed image shape: {img_array.shape}")
            logging.info(f"Preprocessed image dtype: {img_array.dtype}")
            logging.info(f"Preprocessed image min/max: {img_array.min()}/{img_array.max()}")
            
            return img_array
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            raise
            
    def analyze_image(self, image_path):
        """Analyze an image for plant diseases."""
        try:
            logging.info(f"Starting image analysis for: {image_path}")
            
            # Preprocess the image
            logging.info("Preprocessing image...")
            processed_image = self._preprocess_image(image_path)
            logging.info(f"Image preprocessed. Shape: {processed_image.shape}")
            
            # Make prediction
            logging.info("Making prediction...")
            predictions = self.model.predict(processed_image, verbose=0)
            logging.info(f"Prediction shape: {predictions.shape}")
            logging.info(f"Raw predictions: {predictions}")
            
            # Check if predictions are valid
            if len(predictions) == 0 or len(predictions[0]) == 0:
                raise ValueError("Model returned empty predictions")
            
            # Log the number of classes
            num_classes = len(predictions[0])
            logging.info(f"Number of classes in predictions: {num_classes}")
            logging.info(f"Number of class names: {len(self.class_names)}")
            
            # Check if the number of classes matches
            if num_classes != len(self.class_names):
                raise ValueError(f"Number of classes in predictions ({num_classes}) does not match number of class names ({len(self.class_names)})")
            
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            logging.info(f"Predicted class index: {predicted_class}, Confidence: {confidence}")
            
            # Get class name and parse plant type and disease
            class_name = self.class_names[predicted_class]
            plant_type = class_name.split('___')[0].lower()
            disease = class_name.split('___')[1].lower() if '___' in class_name else 'healthy'
            logging.info(f"Class name: {class_name}, Plant type: {plant_type}, Disease: {disease}")
            
            # Get top 3 predictions
            top_k = 3
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            top_predictions = {
                self.class_names[i]: float(predictions[0][i])
                for i in top_indices
            }
            logging.info(f"Top predictions: {top_predictions}")
            
            # Get disease details if applicable
            disease_details = {}
            if disease != 'healthy':
                disease_details = self.get_disease_details(plant_type, disease)
            
            # Format the result
            result = {
                "class": class_name,
                "confidence": confidence,
                "disease": "healthy" not in class_name.lower(),
                "plant_type": plant_type,
                "disease_name": disease,
                "top_predictions": top_predictions,
                "disease_details": disease_details,
                "analysis_quality": self._get_analysis_quality(confidence),
                "recommendations": self._generate_recommendations(plant_type, disease, confidence)
            }
            
            logging.info("Analysis completed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error analyzing image: {str(e)}")
            logging.error(f"Error type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_analysis_quality(self, confidence: float) -> str:
        """Determine the quality of the analysis based on confidence score."""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, plant_type: str, disease: str, confidence: float) -> List[str]:
        """Generate recommendations based on the analysis results."""
        recommendations = []
        
        # Add confidence-based recommendations
        if confidence < 0.7:
            recommendations.append("Consider taking additional photos from different angles for more accurate analysis.")
            recommendations.append("Ensure the photo is well-lit and focused on the affected area.")
        
        # Add disease-specific recommendations
        if disease != 'healthy':
            disease_info = self.get_disease_details(plant_type, disease)
            if disease_info:
                recommendations.extend(disease_info.get('treatment', []))
                recommendations.extend(disease_info.get('prevention', []))
        else:
            recommendations.extend([
                "Continue regular plant care routine",
                "Maintain proper watering schedule",
                "Ensure adequate sunlight exposure",
                "Monitor for any changes in plant health",
                "Keep up with regular fertilization"
            ])
        
        return recommendations

    def get_disease_details(self, plant_type: str, disease: str) -> Dict:
        """
        Get detailed information about a specific plant disease.
        
        Args:
            plant_type: Type of plant (e.g., 'tomato', 'rose')
            disease: Name of the disease
            
        Returns:
            Dictionary containing detailed disease information
        """
        try:
            return self.disease_db.get(plant_type, {}).get(disease, {})
        except Exception as e:
            return {"error": f"Error retrieving disease details: {str(e)}"}

    def save_analysis(self, analysis: Dict, output_path: str) -> bool:
        """
        Save the analysis results to a JSON file.
        
        Args:
            analysis: Dictionary containing analysis results
            output_path: Path to save the results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving analysis: {str(e)}")
            return False

    def _load_disease_database(self) -> Dict:
        """Load the comprehensive plant disease database."""
        return {
            "apple": {
                "apple_scab": {
                    "symptoms": [
                        "Olive-green to black spots on leaves",
                        "Velvety texture on spots",
                        "Yellowing and premature leaf drop",
                        "Lesions on fruit and twigs",
                        "Corky, scabby spots on fruit"
                    ],
                    "causes": [
                        "Fungus Venturia inaequalis",
                        "Wet spring weather",
                        "Poor air circulation",
                        "Overhead watering"
                    ],
                    "treatment": [
                        "Apply fungicides in early spring",
                        "Remove and destroy infected leaves",
                        "Prune for better air circulation",
                        "Use resistant varieties"
                    ],
                    "prevention": [
                        "Plant resistant varieties",
                        "Space trees properly",
                        "Prune for good air flow",
                        "Clean up fallen leaves",
                        "Avoid overhead watering"
                    ]
                },
                "black_rot": {
                    "symptoms": [
                        "Purple spots on leaves",
                        "Fruit rot with concentric rings",
                        "Cankers on branches",
                        "Premature fruit drop",
                        "Leaf yellowing and wilting"
                    ],
                    "causes": [
                        "Fungus Botryosphaeria obtusa",
                        "Warm, wet weather",
                        "Poor sanitation",
                        "Wounded tissue"
                    ],
                    "treatment": [
                        "Remove infected fruit and branches",
                        "Apply fungicides during bloom",
                        "Prune out cankers",
                        "Improve air circulation"
                    ],
                    "prevention": [
                        "Plant resistant varieties",
                        "Remove mummified fruit",
                        "Prune properly",
                        "Avoid wounding trees",
                        "Clean up fallen debris"
                    ]
                }
            },
            "tomato": {
                "early_blight": {
                    "symptoms": [
                        "Small, dark brown to black spots on lower leaves",
                        "Concentric rings in the spots",
                        "Yellow halos around the spots",
                        "Leaves turning yellow and dropping",
                        "Lesions on stems and fruits"
                    ],
                    "causes": [
                        "Fungus Alternaria solani",
                        "Warm, humid weather",
                        "Poor air circulation",
                        "Overhead watering"
                    ],
                    "treatment": [
                        "Remove and destroy infected leaves",
                        "Apply copper-based fungicides",
                        "Improve air circulation",
                        "Water at the base of plants",
                        "Rotate crops annually"
                    ],
                    "prevention": [
                        "Use disease-resistant varieties",
                        "Space plants properly",
                        "Mulch around plants",
                        "Avoid overhead watering",
                        "Clean garden tools regularly"
                    ]
                },
                "late_blight": {
                    "symptoms": [
                        "Large, irregular brown spots on leaves",
                        "White fungal growth on undersides",
                        "Dark lesions on stems",
                        "Rapid plant collapse",
                        "Fruit rot with firm, brown spots"
                    ],
                    "causes": [
                        "Phytophthora infestans fungus",
                        "Cool, wet weather",
                        "High humidity",
                        "Poor drainage"
                    ],
                    "treatment": [
                        "Remove infected plants immediately",
                        "Apply fungicides preventatively",
                        "Improve soil drainage",
                        "Use drip irrigation"
                    ],
                    "prevention": [
                        "Plant resistant varieties",
                        "Space plants for good air flow",
                        "Water in the morning",
                        "Remove plant debris",
                        "Rotate crops"
                    ]
                }
            },
            "potato": {
                "early_blight": {
                    "symptoms": [
                        "Small, dark spots on leaves",
                        "Concentric rings in spots",
                        "Yellowing of leaves",
                        "Premature defoliation",
                        "Lesions on stems"
                    ],
                    "causes": [
                        "Fungus Alternaria solani",
                        "Warm, humid conditions",
                        "Poor air circulation",
                        "Overhead watering"
                    ],
                    "treatment": [
                        "Remove infected leaves",
                        "Apply fungicides",
                        "Improve air circulation",
                        "Water at soil level"
                    ],
                    "prevention": [
                        "Use certified seed potatoes",
                        "Rotate crops",
                        "Space plants properly",
                        "Remove plant debris",
                        "Avoid overhead watering"
                    ]
                },
                "late_blight": {
                    "symptoms": [
                        "Dark, water-soaked spots on leaves",
                        "White fungal growth on undersides",
                        "Rapid plant collapse",
                        "Brown lesions on stems",
                        "Rotting tubers"
                    ],
                    "causes": [
                        "Phytophthora infestans",
                        "Cool, wet weather",
                        "High humidity",
                        "Poor drainage"
                    ],
                    "treatment": [
                        "Remove infected plants",
                        "Apply fungicides",
                        "Improve drainage",
                        "Harvest early if necessary"
                    ],
                    "prevention": [
                        "Use resistant varieties",
                        "Plant certified seed",
                        "Rotate crops",
                        "Improve drainage",
                        "Monitor weather conditions"
                    ]
                }
            }
        } 