from PIL import Image
import numpy as np
import os
from typing import Dict, List, Tuple
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import requests
from io import BytesIO

class PlantDiseaseModel:
    def __init__(self):
        # Initialize the pre-trained model
        self.model = self._load_pretrained_model()
        self.disease_db = self._load_disease_database()
        self.class_names = self._load_class_names()
        
    def _load_pretrained_model(self) -> Model:
        """Load the pre-trained model."""
        try:
            # Load the saved model
            model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_model.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    "Model not found. Please run setup_model.py to create the model."
                )
            
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def _load_class_names(self) -> List[str]:
        """Load the class names for the PlantVillage dataset."""
        return [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Corn_(maize)___Cercospora_leaf_spot',
            'Corn_(maize)___Common_rust',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess the image for the model."""
        try:
            # Resize image to 224x224 (required by MobileNetV2)
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize
            image_array = np.array(image)
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        except Exception as e:
            print(f"Error in preprocess_image: {str(e)}")
            raise
        
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a plant image using the pre-trained model.
        
        Args:
            image_path: Path to the plant image
            
        Returns:
            Dictionary containing detailed analysis results
        """
        try:
            # Load and preprocess image
            print(f"Loading image from: {image_path}")
            image = Image.open(image_path)
            
            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                print("Converting image to RGB mode")
                image = image.convert('RGB')
            
            # Preprocess the image
            print("Preprocessing image...")
            processed_image = self.preprocess_image(image)
            
            # Get model predictions
            print("Getting model predictions...")
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get class name
            class_name = self.class_names[predicted_class]
            plant_type, disease = class_name.split('___')
            
            print(f"Detected: {plant_type} with {disease} (confidence: {confidence:.2f})")
            
            # Get detailed information
            disease_info = self.get_disease_details(plant_type.lower(), disease.lower())
            
            return {
                "plant_type": plant_type,
                "disease": disease,
                "confidence": confidence,
                "severity": self._determine_severity(confidence),
                "analysis": {
                    "visual_symptoms": disease_info.get("symptoms", []),
                    "stage": self._determine_stage(confidence),
                    "risk_factors": disease_info.get("causes", []),
                    "treatment_plan": {
                        "immediate_actions": disease_info.get("treatment", []),
                        "long_term_measures": disease_info.get("prevention", [])
                    },
                    "monitoring_schedule": self._get_monitoring_schedule(confidence),
                    "prevention_measures": disease_info.get("prevention", [])
                },
                "recommendations": self._generate_recommendations(confidence, disease_info)
            }
            
        except Exception as e:
            print(f"Error in analyze_image: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Failed to analyze image: {str(e)}",
                "status": "failed"
            }
            
    def _determine_severity(self, confidence: float) -> str:
        """Determine disease severity based on confidence score."""
        if confidence > 0.9:
            return "severe"
        elif confidence > 0.7:
            return "moderate"
        else:
            return "mild"
            
    def _determine_stage(self, confidence: float) -> str:
        """Determine disease stage based on confidence score."""
        if confidence > 0.9:
            return "advanced"
        elif confidence > 0.7:
            return "mid-stage"
        else:
            return "early"
            
    def _get_monitoring_schedule(self, confidence: float) -> Dict:
        """Generate monitoring schedule based on disease severity."""
        if confidence > 0.9:
            return {
                "daily": "Check for new symptoms and document changes",
                "weekly": "Apply treatment and assess effectiveness",
                "monthly": "Evaluate overall plant health and recovery"
            }
        elif confidence > 0.7:
            return {
                "daily": "Monitor for symptom progression",
                "weekly": "Apply preventive measures",
                "monthly": "Assess treatment effectiveness"
            }
        else:
            return {
                "daily": "Check for new symptoms",
                "weekly": "Apply preventive measures",
                "monthly": "Monitor overall plant health"
            }
            
    def _generate_recommendations(self, confidence: float, disease_info: Dict) -> List[str]:
        """Generate recommendations based on disease severity and information."""
        recommendations = []
        
        if confidence > 0.9:
            recommendations.extend([
                "Begin treatment immediately",
                "Isolate affected plants if possible",
                "Document symptom progression daily"
            ])
        elif confidence > 0.7:
            recommendations.extend([
                "Start treatment as soon as possible",
                "Monitor plant health closely",
                "Implement preventive measures"
            ])
        else:
            recommendations.extend([
                "Monitor for symptom progression",
                "Implement preventive measures",
                "Consider early treatment options"
            ])
            
        recommendations.extend(disease_info.get("treatment", []))
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