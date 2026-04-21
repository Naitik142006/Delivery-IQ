import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_and_save_model():
    print("Loading data...")
    df = pd.read_csv('food_delivery_cleaned.csv')
    
    # Select features and target
    X = df[['Distance_km', 'Order_Hour']]
    y = df['Delivery_Duration']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model trained successfully. R^2 score on test set: {score:.3f}")
    
    print("Saving model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
