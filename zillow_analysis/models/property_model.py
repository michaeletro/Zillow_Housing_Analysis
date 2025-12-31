"""
Property Valuation Model

Machine learning models for predicting property values.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
from datetime import datetime


class PropertyValuationModel:
    """
    Machine learning model for property valuation.
    
    Supports multiple algorithms:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the valuation model.
        
        Args:
            model_type: Type of model - "random_forest", "gradient_boost", "xgboost", "lightgbm"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.trained = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model."""
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boost":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw property data.
        
        Args:
            df: DataFrame with property data
            
        Returns:
            DataFrame with prepared features
        """
        df = df.copy()
        
        # Calculate derived features
        if 'living_area' in df.columns and 'price' in df.columns:
            df['price_per_sqft'] = df['price'] / (df['living_area'] + 1)
        
        if 'year_built' in df.columns:
            df['age'] = datetime.now().year - df['year_built']
        
        if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
            df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
        
        if 'lot_size' in df.columns and 'living_area' in df.columns:
            df['lot_to_living_ratio'] = df['lot_size'] / (df['living_area'] + 1)
        
        # Encode categorical variables
        categorical_cols = ['property_type', 'city', 'state', 'status']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen labels
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'price') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select relevant features for modeling.
        
        Args:
            df: DataFrame with prepared features
            target_col: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Core numeric features
        numeric_features = [
            'bedrooms', 'bathrooms', 'living_area', 'lot_size',
            'year_built', 'age', 'tax_assessed_value',
            'latitude', 'longitude', 'days_on_zillow'
        ]
        
        # Derived features
        derived_features = [
            'price_per_sqft', 'bath_bed_ratio', 'lot_to_living_ratio'
        ]
        
        # Encoded categorical features
        encoded_features = [
            'property_type_encoded', 'city_encoded', 'state_encoded'
        ]
        
        # Combine all features
        all_features = numeric_features + derived_features + encoded_features
        
        # Select only available features
        available_features = [f for f in all_features if f in df.columns]
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Get features and target
        X = df[available_features].fillna(0)
        y = df[target_col]
        
        self.feature_names = available_features
        
        return X, y
    
    def train(self, 
              properties: List[Dict[str, Any]], 
              target_col: str = 'price',
              test_size: float = 0.2,
              validate: bool = True) -> Dict[str, float]:
        """
        Train the valuation model.
        
        Args:
            properties: List of property dictionaries
            target_col: Target variable to predict
            test_size: Proportion of data for testing
            validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with training metrics
        """
        # Convert to DataFrame
        df = pd.DataFrame(properties)
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Select features and target
        X, y = self.select_features(df, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
        }
        
        # Cross-validation
        if validate:
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='r2', n_jobs=-1
            )
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
        
        self.trained = True
        
        print(f"\nTraining complete!")
        print(f"Test R² Score: {metrics['test_r2']:.4f}")
        print(f"Test MAE: ${metrics['test_mae']:,.0f}")
        print(f"Test RMSE: ${metrics['test_rmse']:,.0f}")
        
        return metrics
    
    def predict(self, properties: List[Dict[str, Any]]) -> np.ndarray:
        """
        Predict property values.
        
        Args:
            properties: List of property dictionaries
            
        Returns:
            Array of predicted values
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame(properties)
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Select features
        X = df[self.feature_names].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance.head(top_n)
        else:
            return pd.DataFrame({'message': ['Feature importance not available for this model']})
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.trained = model_data['trained']
        
        print(f"Model loaded from {filepath}")
    
    def evaluate_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single property and provide detailed analysis.
        
        Args:
            property_data: Property dictionary
            
        Returns:
            Dictionary with evaluation results
        """
        # Predict value
        predicted_value = self.predict([property_data])[0]
        
        # Compare with asking price or zestimate
        actual_price = property_data.get('price') or property_data.get('zestimate')
        
        evaluation = {
            'predicted_value': predicted_value,
            'actual_price': actual_price,
            'difference': predicted_value - actual_price if actual_price else None,
            'percent_difference': ((predicted_value - actual_price) / actual_price * 100) 
                                  if actual_price else None,
            'evaluation': None
        }
        
        # Determine if property is undervalued, overvalued, or fairly priced
        if evaluation['percent_difference'] is not None:
            if evaluation['percent_difference'] > 10:
                evaluation['evaluation'] = "Overvalued - Model predicts higher value"
            elif evaluation['percent_difference'] < -10:
                evaluation['evaluation'] = "Undervalued - Model predicts lower value"
            else:
                evaluation['evaluation'] = "Fairly priced - Within 10% of model prediction"
        
        return evaluation
