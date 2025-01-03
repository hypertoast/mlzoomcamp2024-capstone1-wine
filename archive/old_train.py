import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Tuple, Dict

class WineQualityModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        self.feature_names = None
        self.required_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features including engineering steps"""
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Verify all required features are present
        missing_features = [feat for feat in self.required_features 
                          if feat not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Create engineered features
        df['alcohol_to_density_ratio'] = df['alcohol'] / df['density']
        df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
        df['sulfur_ratio'] = (df['free sulfur dioxide'] / 
                            df['total sulfur dioxide'].replace(0, 1))  # Avoid division by zero
        
        # Create log transforms for skewed features
        skewed_features = ['residual sugar', 'chlorides', 'total sulfur dioxide', 
                          'free sulfur dioxide', 'sulphates']
        for feature in skewed_features:
            df[f'{feature}_log'] = np.log1p(df[feature])
        
        return df
    
    def create_quality_labels(self, quality: pd.Series) -> pd.Series:
        """Convert numeric quality scores to classes"""
        return pd.cut(quality, 
                     bins=[0, 5, 6, 10], 
                     labels=['low', 'medium', 'high'],
                     include_lowest=True)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model pipeline"""
        # Prepare features
        X_prepared = self.prepare_features(X)
        
        # Store feature names
        self.feature_names = X_prepared.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_prepared)
        
        # Create quality classes
        y_classes = self.create_quality_labels(y)
        
        # Fit model
        self.model.fit(X_scaled, y_classes)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict wine quality class and probabilities
        Returns: (predictions, probabilities)
        """
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet!")
        
        # Prepare features
        X_prepared = self.prepare_features(X)
        
        # Select features in the same order as training
        X_prepared = X_prepared[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_prepared)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance dataframe"""
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet!")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def train_and_save_model(df: pd.DataFrame, model_dir: str = 'model') -> WineQualityModel:
    """
    Train and save the model
    Args:
        df: Input DataFrame with wine quality data
        model_dir: Directory to save model artifacts
    Returns:
        Trained model
    """
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare data
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
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
    """
    Test model predictions with sample data
    Args:
        model: Trained WineQualityModel
    """
    print("\nTesting predictions with sample data...")
    
    # Sample test cases
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
    
    # Make predictions
    predictions, probabilities = model.predict(sample_wines)
    
    # Print results
    print("\nPrediction Results:")
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        print(f"\nWine {i+1}:")
        print(f"Predicted class: {pred}")
        print("Class probabilities:")
        for class_name, prob in zip(['low', 'medium', 'high'], probs):
            print(f"  {class_name}: {prob:.3f}")



def main():
    # Load data
    try:
        df = pd.read_csv('winequality.csv')
        print("Data loaded successfully, shape:", df.shape)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        # Train and save model
        print("\nTraining model...")
        model = train_and_save_model(df)
        
        # Print training summary
        importance_df = model.get_feature_importance()
        print("\nModel Training Summary:")
        print("Top 5 important features:")
        print(importance_df.head())
        
        # Test predictions
        test_predictions(model)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        return

def xmain():
    # Load data
    try:
        df = pd.read_csv('winequality.csv')
        print("Data loaded successfully, shape:", df.shape)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Train model
    try:
        print("\nTraining model...")
        model = WineQualityModel()
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        predictions, _ = model.predict(X_val)
        
        # Save model
        model_path = os.path.join('model', 'wine_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
        # Save feature importance
        importance_df = model.get_feature_importance()
        importance_path = os.path.join('model', 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to {importance_path}")
        
        # Print summary
        print("\nModel Training Summary:")
        print("Top 5 important features:")
        print(importance_df.head())
        
    except Exception as e:
        print(f"Error during training: {e}")
        return

if __name__ == "__main__":
    main()