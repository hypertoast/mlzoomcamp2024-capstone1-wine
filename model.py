# model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

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
        df = df.copy()
        
        missing_features = [feat for feat in self.required_features 
                        if feat not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        df['alcohol_to_density_ratio'] = df['alcohol'] / df['density']
        df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
        df['sulfur_ratio'] = (df['free sulfur dioxide'] / 
                            df['total sulfur dioxide'].replace(0, 1))
        
        # Fix: Update feature names to match DataFrame columns
        skewed_features = [
            'residual sugar',
            'chlorides',
            'total sulfur dioxide',  # Changed from total_sulfur_dioxide
            'free sulfur dioxide',   # Changed from free_sulfur_dioxide
            'sulphates'
        ]
        
        for feature in skewed_features:
            df[f'{feature}_log'] = np.log1p(df[feature])
        
        return df
    
    def create_quality_labels(self, quality: pd.Series) -> pd.Series:
        return pd.cut(quality, 
                     bins=[0, 5, 6, 10], 
                     labels=['low', 'medium', 'high'],
                     include_lowest=True)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_prepared = self.prepare_features(X)
        self.feature_names = X_prepared.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_prepared)
        y_classes = self.create_quality_labels(y)
        self.model.fit(X_scaled, y_classes)
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict wine quality class and probabilities
        Returns: (predictions, probabilities)
        """
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet!")
        
        X_prepared = self.prepare_features(X)
        X_prepared = X_prepared[self.feature_names]
        X_scaled = self.scaler.transform(X_prepared)
        
        probabilities = self.model.predict_proba(X_scaled)
        # Use argmax to get the index of highest probability and map to corresponding class
        predictions = np.array([self.model.classes_[np.argmax(p)] for p in probabilities])
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_names is None:
            raise ValueError("Model has not been fitted yet!")
            
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)