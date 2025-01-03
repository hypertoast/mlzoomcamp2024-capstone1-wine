# train_utils.py
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from model import WineQualityModel

def load_data(filepath: str) -> pd.DataFrame:
    """Load the wine quality dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully, shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare data for training"""
    X = df.drop('quality', axis=1)
    y = df['quality']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save_model(df: pd.DataFrame, model_dir: str = 'model') -> WineQualityModel:
    """Train and save the model"""
    os.makedirs(model_dir, exist_ok=True)
    
    X_train, X_val, y_train, y_val = prepare_data(df)
    
    model = WineQualityModel()
    model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(model_dir, 'wine_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save feature importance
    importance_df = model.get_feature_importance()
    importance_path = os.path.join(model_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")
    
    return model

def test_predictions(model: WineQualityModel) -> None:
    """Test model predictions with sample data"""
    print("\nTesting predictions with sample data...")
    print(f"Class order: {model.model.classes_}")
    
    sample_wines = pd.DataFrame({
        'fixed acidity': [7.4, 8.1, 7.2],
        'volatile acidity': [0.7, 0.5, 0.3],
        'citric acid': [0.0, 0.3, 0.3],
        'residual sugar': [1.9, 2.4, 1.7],
        'chlorides': [0.076, 0.089, 0.082],
        'free sulfur dioxide': [11.0, 22.0, 15.0],
        'total sulfur dioxide': [34.0, 48.0, 42.0],
        'density': [0.9978, 0.9968, 0.9972],
        'pH': [3.51, 3.36, 3.42],
        'sulphates': [0.56, 0.72, 0.68],
        'alcohol': [9.4, 11.2, 12.1]
    })
    
    predictions, probabilities = model.predict(sample_wines)
    
    print("\nPrediction Results:")
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        print(f"\nWine {i+1}:")
        print(f"Predicted class: {pred}")
        print("Class probabilities:")
        # Print probabilities in the order of model.classes_
        for class_name, prob in zip(model.model.classes_, probs):
            print(f"  {class_name}: {prob:.3f}")