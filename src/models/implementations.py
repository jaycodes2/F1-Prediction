"""
F1 Race Prediction Model Implementations.

This module contains specific model implementations for F1 race prediction,
including ensemble methods and specialized algorithms.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import logging

from .training import BaseModelTrainer, ModelTrainingError
from .interfaces import ModelTrainer


logger = logging.getLogger(__name__)


class F1RandomForestModel(BaseModelTrainer):
    """Random Forest model optimized for F1 race prediction."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state)
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Default hyperparameters optimized for F1 data
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': random_state
        }
        self.default_params.update(kwargs)
    
    def train(self, features: List[Dict], targets: List[float], 
              dates: Optional[List] = None, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train the Random Forest model."""
        logger.info("Training Random Forest model for F1 race prediction")
        
        # Prepare training data
        X, y, dates_array = self.prepare_training_data(features, targets, dates)
        self.feature_names = list(features[0].keys()) if features else None
        
        # Initialize model
        self.model = RandomForestRegressor(**self.default_params)
        
        # Hyperparameter tuning if requested
        if tune_hyperparameters:
            param_space = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            best_params = self.tuner.tune_hyperparameters(
                self.model, param_space, X, y
            )
            self.model.set_params(**best_params)
            logger.info(f"Best hyperparameters: {best_params}")
        
        # Train the model
        self.model.fit(X, y)
        
        # Evaluate model
        metrics = self.evaluate_model(self.model, X, y)
        
        # Cross-validation
        cv_results = self.cross_validate_model(self.model, X, y)
        metrics.update(cv_results)
        
        # Feature importance
        feature_importance = self.get_feature_importance(self.model, self.feature_names)
        
        logger.info(f"Model training completed. MAE: {metrics['mae']:.4f}")
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_type': 'RandomForest',
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def predict(self, features: List[Dict]) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ModelTrainingError("Model must be trained before making predictions")
        
        X, _, _ = self.prepare_training_data(features, [0] * len(features))
        return self.model.predict(X)
    
    def predict_proba(self, features: List[Dict]) -> Optional[np.ndarray]:
        """Random Forest doesn't provide probability estimates for regression."""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.model is None:
            return {'status': 'not_trained'}
        
        return {
            'model_type': 'RandomForest',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': self.model.n_features_in_,
            'feature_names': self.feature_names
        }


class F1GradientBoostingModel(BaseModelTrainer):
    """Gradient Boosting model for F1 race prediction."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        self.default_params = {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': random_state
        }
        self.default_params.update(kwargs)
    
    def train(self, features: List[Dict], targets: List[float], 
              dates: Optional[List] = None, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train the Gradient Boosting model."""
        logger.info("Training Gradient Boosting model for F1 race prediction")
        
        X, y, dates_array = self.prepare_training_data(features, targets, dates)
        self.feature_names = list(features[0].keys()) if features else None
        
        # Scale features for better performance
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingRegressor(**self.default_params)
        
        if tune_hyperparameters:
            param_space = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.7, 0.8, 0.9]
            }
            
            best_params = self.tuner.tune_hyperparameters(
                self.model, param_space, X_scaled, y
            )
            self.model.set_params(**best_params)
            logger.info(f"Best hyperparameters: {best_params}")
        
        self.model.fit(X_scaled, y)
        
        metrics = self.evaluate_model(self.model, X_scaled, y)
        cv_results = self.cross_validate_model(self.model, X_scaled, y)
        metrics.update(cv_results)
        
        feature_importance = self.get_feature_importance(self.model, self.feature_names)
        
        logger.info(f"Model training completed. MAE: {metrics['mae']:.4f}")
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_type': 'GradientBoosting',
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def predict(self, features: List[Dict]) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ModelTrainingError("Model must be trained before making predictions")
        
        X, _, _ = self.prepare_training_data(features, [0] * len(features))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, features: List[Dict]) -> Optional[np.ndarray]:
        """Gradient Boosting doesn't provide probability estimates for regression."""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.model is None:
            return {'status': 'not_trained'}
        
        return {
            'model_type': 'GradientBoosting',
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth,
            'n_features': self.model.n_features_in_,
            'feature_names': self.feature_names
        }


class F1EnsembleModel(BaseModelTrainer):
    """Ensemble model combining multiple algorithms for F1 race prediction."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state)
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self.feature_names = None
        
        # Initialize base models
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=random_state
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=random_state
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1, 
                random_state=random_state, n_jobs=-1, verbosity=0
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                random_state=random_state, n_jobs=-1, verbose=-1
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        # Models that need scaling
        self.models_need_scaling = {'ridge'}
    
    def train(self, features: List[Dict], targets: List[float], 
              dates: Optional[List] = None, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """Train the ensemble model."""
        logger.info("Training Ensemble model for F1 race prediction")
        
        X, y, dates_array = self.prepare_training_data(features, targets, dates)
        self.feature_names = list(features[0].keys()) if features else None
        
        # Split data for meta-learning
        X_train, X_val, y_train, y_val = self.split_temporal_data(X, y, test_size=0.2)
        
        model_predictions = {}
        model_scores = {}
        
        # Train each base model
        for name, model in self.base_models.items():
            logger.info(f"Training {name} model")
            
            # Scale features if needed
            if name in self.models_need_scaling:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                self.scalers[name] = scaler
                
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
            
            self.models[name] = model
            model_predictions[name] = val_pred
            
            # Calculate validation score (negative MAE for maximization)
            mae = np.mean(np.abs(y_val - val_pred))
            model_scores[name] = 1.0 / (1.0 + mae)  # Convert to weight
        
        # Calculate ensemble weights based on validation performance
        total_score = sum(model_scores.values())
        self.weights = {name: score / total_score for name, score in model_scores.items()}
        
        logger.info(f"Ensemble weights: {self.weights}")
        
        # Evaluate ensemble on validation set
        ensemble_pred = self._ensemble_predict(model_predictions)
        metrics = self.metrics_calculator.calculate_all_metrics(y_val, ensemble_pred)
        
        # Train final models on full dataset
        for name, model in self.models.items():
            if name in self.models_need_scaling:
                X_scaled = self.scalers[name].fit_transform(X)
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)
        
        logger.info(f"Ensemble training completed. Validation MAE: {metrics['mae']:.4f}")
        
        return {
            'metrics': metrics,
            'model_weights': self.weights,
            'model_type': 'Ensemble',
            'base_models': list(self.base_models.keys()),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def _ensemble_predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions from base models using learned weights."""
        ensemble_pred = np.zeros(len(next(iter(predictions.values()))))
        
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
    
    def predict(self, features: List[Dict]) -> np.ndarray:
        """Make predictions using the ensemble model."""
        if not self.models:
            raise ModelTrainingError("Model must be trained before making predictions")
        
        X, _, _ = self.prepare_training_data(features, [0] * len(features))
        
        predictions = {}
        for name, model in self.models.items():
            if name in self.models_need_scaling:
                X_scaled = self.scalers[name].transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        
        return self._ensemble_predict(predictions)
    
    def predict_proba(self, features: List[Dict]) -> Optional[np.ndarray]:
        """Ensemble doesn't provide probability estimates for regression."""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ensemble model."""
        if not self.models:
            return {'status': 'not_trained'}
        
        return {
            'model_type': 'Ensemble',
            'base_models': list(self.models.keys()),
            'weights': self.weights,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }


class F1NeuralNetworkModel(BaseModelTrainer):
    """Neural Network model for F1 race prediction."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        self.default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': random_state,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        self.default_params.update(kwargs)
    
    def train(self, features: List[Dict], targets: List[float], 
              dates: Optional[List] = None, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train the Neural Network model."""
        logger.info("Training Neural Network model for F1 race prediction")
        
        X, y, dates_array = self.prepare_training_data(features, targets, dates)
        self.feature_names = list(features[0].keys()) if features else None
        
        # Scale features (essential for neural networks)
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = MLPRegressor(**self.default_params)
        
        if tune_hyperparameters:
            param_space = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
            
            best_params = self.tuner.tune_hyperparameters(
                self.model, param_space, X_scaled, y
            )
            self.model.set_params(**best_params)
            logger.info(f"Best hyperparameters: {best_params}")
        
        self.model.fit(X_scaled, y)
        
        metrics = self.evaluate_model(self.model, X_scaled, y)
        cv_results = self.cross_validate_model(self.model, X_scaled, y)
        metrics.update(cv_results)
        
        logger.info(f"Model training completed. MAE: {metrics['mae']:.4f}")
        
        return {
            'metrics': metrics,
            'model_type': 'NeuralNetwork',
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'n_layers': len(self.model.hidden_layer_sizes),
            'n_iterations': self.model.n_iter_
        }
    
    def predict(self, features: List[Dict]) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ModelTrainingError("Model must be trained before making predictions")
        
        X, _, _ = self.prepare_training_data(features, [0] * len(features))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, features: List[Dict]) -> Optional[np.ndarray]:
        """Neural Network doesn't provide probability estimates for regression."""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.model is None:
            return {'status': 'not_trained'}
        
        return {
            'model_type': 'NeuralNetwork',
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'n_features': self.model.n_features_in_,
            'n_layers': len(self.model.hidden_layer_sizes),
            'n_iterations': self.model.n_iter_,
            'feature_names': self.feature_names
        }


class F1XGBoostModel(BaseModelTrainer):
    """XGBoost model optimized for F1 race prediction with ranking objective."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state)
        self.model = None
        self.feature_names = None
        
        # Default hyperparameters optimized for ranking
        self.default_params = {
            'objective': 'reg:squarederror',  # Can be changed to rank:pairwise for ranking
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': random_state,
            'n_jobs': -1
        }
        self.default_params.update(kwargs)
    
    def train(self, features: List[Dict], targets: List[float], 
              dates: Optional[List] = None, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train the XGBoost model."""
        logger.info("Training XGBoost model for F1 race prediction")
        
        X, y, dates_array = self.prepare_training_data(features, targets, dates)
        self.feature_names = list(features[0].keys()) if features else None
        
        self.model = xgb.XGBRegressor(**self.default_params)
        
        if tune_hyperparameters:
            param_space = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
            
            best_params = self.tuner.tune_hyperparameters(
                self.model, param_space, X, y
            )
            self.model.set_params(**best_params)
            logger.info(f"Best hyperparameters: {best_params}")
        
        self.model.fit(X, y)
        
        metrics = self.evaluate_model(self.model, X, y)
        cv_results = self.cross_validate_model(self.model, X, y)
        metrics.update(cv_results)
        
        feature_importance = self.get_feature_importance(self.model, self.feature_names)
        
        logger.info(f"Model training completed. MAE: {metrics['mae']:.4f}")
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_type': 'XGBoost',
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def predict(self, features: List[Dict]) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ModelTrainingError("Model must be trained before making predictions")
        
        X, _, _ = self.prepare_training_data(features, [0] * len(features))
        return self.model.predict(X)
    
    def predict_proba(self, features: List[Dict]) -> Optional[np.ndarray]:
        """XGBoost doesn't provide probability estimates for regression."""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.model is None:
            return {'status': 'not_trained'}
        
        return {
            'model_type': 'XGBoost',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'n_features': self.model.n_features_in_,
            'feature_names': self.feature_names
        }


class F1LightGBMModel(BaseModelTrainer):
    """LightGBM model for F1 race prediction with gradient boosting."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state)
        self.model = None
        self.feature_names = None
        
        self.default_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        self.default_params.update(kwargs)
    
    def train(self, features: List[Dict], targets: List[float], 
              dates: Optional[List] = None, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train the LightGBM model."""
        logger.info("Training LightGBM model for F1 race prediction")
        
        X, y, dates_array = self.prepare_training_data(features, targets, dates)
        self.feature_names = list(features[0].keys()) if features else None
        
        self.model = lgb.LGBMRegressor(**self.default_params)
        
        if tune_hyperparameters:
            param_space = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10, -1],
                'learning_rate': [0.05, 0.1, 0.15],
                'feature_fraction': [0.7, 0.8, 0.9],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
            
            best_params = self.tuner.tune_hyperparameters(
                self.model, param_space, X, y
            )
            self.model.set_params(**best_params)
            logger.info(f"Best hyperparameters: {best_params}")
        
        self.model.fit(X, y)
        
        metrics = self.evaluate_model(self.model, X, y)
        cv_results = self.cross_validate_model(self.model, X, y)
        metrics.update(cv_results)
        
        feature_importance = self.get_feature_importance(self.model, self.feature_names)
        
        logger.info(f"Model training completed. MAE: {metrics['mae']:.4f}")
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_type': 'LightGBM',
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def predict(self, features: List[Dict]) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ModelTrainingError("Model must be trained before making predictions")
        
        X, _, _ = self.prepare_training_data(features, [0] * len(features))
        return self.model.predict(X)
    
    def predict_proba(self, features: List[Dict]) -> Optional[np.ndarray]:
        """LightGBM doesn't provide probability estimates for regression."""
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.model is None:
            return {'status': 'not_trained'}
        
        return {
            'model_type': 'LightGBM',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'n_features': self.model.n_features_in_,
            'feature_names': self.feature_names
        }


class ModelFactory:
    """Factory class for creating F1 prediction models."""
    
    AVAILABLE_MODELS = {
        'random_forest': F1RandomForestModel,
        'gradient_boosting': F1GradientBoostingModel,
        'xgboost': F1XGBoostModel,
        'lightgbm': F1LightGBMModel,
        'ensemble': F1EnsembleModel,
        'neural_network': F1NeuralNetworkModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModelTrainer:
        """Create a model instance of the specified type."""
        if model_type not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(cls.AVAILABLE_MODELS.keys())}")
        
        model_class = cls.AVAILABLE_MODELS[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls.AVAILABLE_MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model type."""
        if model_type not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls.AVAILABLE_MODELS[model_type]
        
        return {
            'name': model_type,
            'class': model_class.__name__,
            'description': model_class.__doc__.strip() if model_class.__doc__ else "No description available"
        }