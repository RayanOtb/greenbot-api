import os
import requests
import tensorflow as tf
from tqdm import tqdm
import shutil
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def download_file(url, filename):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def setup_model():
    """Set up the pre-trained model for plant disease detection."""
    print("Setting up the plant disease detection model...")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        print("Loading base model...")
        # Load MobileNetV2 as base model
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(38, activation='softmax')(x)  # 38 classes in PlantVillage dataset
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the model
        model_path = os.path.join('models', 'plant_disease_model.h5')
        model.save(model_path, include_optimizer=True)
        
        print("Model setup completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        return True
    except Exception as e:
        print(f"Error setting up model: {str(e)}")
        return False

if __name__ == "__main__":
    setup_model() 