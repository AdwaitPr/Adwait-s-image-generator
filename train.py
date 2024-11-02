# This file will handle the training process
from data_loader import load_data
from model import create_model
from sklearn.metrics import classification_report

def train_model():
    X_train, X_test, y_train, y_test, _ = load_data()
    model = create_model()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    train_model()

