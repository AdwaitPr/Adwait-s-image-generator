# This file will handle making predictions on new data
import numpy as np
from data_loader import load_data
from model import create_model
from train import train_model

def predict(features):
    _, _, _, _, scaler = load_data()
    model = train_model()
    
    # Scale the input features
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(scaled_features)
    return prediction[0]

if __name__ == "__main__":
    # Example usage
    new_flower = [5.1, 3.5, 1.4, 0.2]  # Example features
    print(f"Predicted class: {predict(new_flower)}")

