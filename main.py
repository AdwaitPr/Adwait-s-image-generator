# This file will serve as the entry point for your application
from train import train_model
from predict import predict

def main():
    print("Training the model...")
    model = train_model()
    
    print("\nMaking predictions...")
    new_flowers = [
        [5.1, 3.5, 1.4, 0.2],
        [6.3, 3.3, 6.0, 2.5],
        [5.9, 3.0, 4.2, 1.5]
    ]
    
    for flower in new_flowers:
        prediction = predict(flower)
        print(f"Features: {flower}, Predicted class: {prediction}")

if __name__ == "__main__":
    main()

