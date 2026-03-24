import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class GarbageClassifier:
    """
    Garbage Classification System
    Classifies garbage into: Plastic, Cardboard, Paper, Metal, Other
    """
    
    def __init__(self, model_path='models/garbage_classifier.h5'):
        """
        Initialize the garbage classifier
        Args:
            model_path: Path to the pre-trained model
        """
        self.classes = ['Plastic', 'Cardboard', 'Paper', 'Metal', 'Other']
        self.model_path = model_path
        self.model = self.load_model() if os.path.exists(model_path) else None
    
    def load_model(self):
        """Load pre-trained Keras model"""
        try:
            model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model prediction
        Args:
            image_path: Path to the image file
        Returns:
            Preprocessed image array
        """
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize to [0, 1]
        return img_array
    
    def predict(self, image_path):
        """
        Predict garbage type from image
        Args:
            image_path: Path to the image file
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'status': 'failed'
            }
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.classes[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get all predictions with probabilities
            all_predictions = {
                self.classes[i]: float(predictions[0][i]) 
                for i in range(len(self.classes))
            }
            
            return {
                'garbage_type': predicted_class,
                'confidence': round(confidence * 100, 2),
                'all_predictions': all_predictions,
                'status': 'success'
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

if __name__ == '__main__':
    classifier = GarbageClassifier()
    
    # Example usage
    test_image = 'samples/test_image.jpg'
    if os.path.exists(test_image):
        result = classifier.predict(test_image)
        print(f"Result: {result}")
    else:
        print(f"Test image not found: {test_image}")