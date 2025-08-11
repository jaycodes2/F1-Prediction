"""
Model training infrastructure for F1 race prediction.
"""
import logging
import numpy as np
import pandas as pd
import pickle
import joblib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import spearmanr
import warnings

from ..models.interfaces import ModelTrainer
from ..config import config


logger = logging.getLogger(__name__)


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)


class TimeSeriesValidator:
    """
    Time-series cross-validation to prevent data leakage in temporal data.
    """
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None):
        """Generate time-series splits."""
        return self.tscv.split(X, y, groups)
    
    def validate_temporal_order(self, dates: List[datetime]) -> bool:
        """Validate that data is in temporal order."""
        if len(dates) < 2:
            return True
        
        for i in range(1, len(dates)):
            if dates[i] < dates[i-1]:
                return False
        return True


class HyperparameterTuner:
    """
    Hyperparameter tuning with support for grid search and random search.
    """
    
    def __init__(self, search_type: str = 'random', n_iter: int = 50, cv: int = 3, 
                 scoring: str = 'neg_mean_absolute_error', random_state: int = None):
        self.search_type = search_type
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state or config.model.random_state
    
    def tune_hyperparameters(self, model, param_space: Dict[str, Any], 
                           X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search or random search.
        
        Args:
            model: ML model to tune
            param_space: Parameter space to search
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best parameters found
        """
        try:
            if self.search_type == 'grid':
                search = GridSearchCV(
                    model, param_space, cv=self.cv, scoring=self.scoring,
                    n_jobs=-1, random_state=self.random_state
                )
            elif self.search_type == 'random':
                search = RandomizedSearchCV(
                    model, param_space, n_iter=self.n_iter, cv=self.cv,
                    scoring=self.scoring, n_jobs=-1, random_state=self.random_state
                )
            else:
                raise ValueError(f"Invalid search_type: {self.search_type}")
            
            # Suppress warnings during hyperparameter search
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train)
            
            logger.info(f"Best {self.search_type} search score: {search.best_score_:.4f}")
            logger.info(f"Best parameters: {search.best_params_}")
            
            return search.best_params_
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            raise ModelTrainingError(f"Hyperparameter tuning failed: {e}")


class ModelPersistence:
    """
    Model persistence and loading functionality.
    """
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir or config.data.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained model to save
            model_name: Name for the model file
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_{timestamp}.pkl"
            model_path = self.models_dir / model_filename
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata
            if metadata:
                metadata_path = self.models_dir / f"{model_name}_{timestamp}_metadata.json"
                import json
                with open(metadata_path, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    serializable_metadata = self._make_json_serializable(metadata)
                    json.dump(serializable_metadata, f, indent=2)
            
            logger.info(f"Model saved to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelTrainingError(f"Failed to save model: {e}")
    
    def load_model(self, model_path: str) -> Any:
        """Load a saved model."""
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelTrainingError(f"Failed to load model: {e}")
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """List all saved models with their metadata."""
        models = []
        
        for model_file in self.models_dir.glob("*.pkl"):
            model_info = {
                'name': model_file.stem,
                'path': str(model_file),
                'size_mb': model_file.stat().st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(model_file.stat().st_mtime)
            }
            
            # Look for corresponding metadata file
            metadata_file = model_file.with_suffix('').with_suffix('_metadata.json')
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        model_info['metadata'] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_file}: {e}")
            
            models.append(model_info)
        
        return sorted(models, key=lambda x: x['created'], reverse=True)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj


class MetricsCalculator:
    """
    Calculate evaluation metrics for F1 race prediction models.
    """
    
    def __init__(self):
        self.supported_metrics = [
            'mae', 'spearman_correlation', 'top_k_accuracy', 
            'position_accuracy', 'points_correlation'
        ]
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            k_values: List[int] = None) -> Dict[str, float]:
        """
        Calculate all supported metrics.
        
        Args:
            y_true: True finishing positions
            y_pred: Predicted finishing positions
            k_values: List of K values for top-K accuracy
            
        Returns:
            Dictionary of metric names and values
        """
        if k_values is None:
            k_values = [3, 5, 10]
        
        metrics = {}
        
        try:
            # Mean Absolute Error
            metrics['mae'] = self.mean_absolute_error(y_true, y_pred)
            
            # Spearman Rank Correlation
            metrics['spearman_correlation'] = self.spearman_correlation(y_true, y_pred)
            
            # Top-K Accuracy for different K values
            for k in k_values:
                metrics[f'top_{k}_accuracy'] = self.top_k_accuracy(y_true, y_pred, k)
            
            # Position-based accuracy (exact position matches)
            metrics['position_accuracy'] = self.position_accuracy(y_true, y_pred)
            
            # Additional metrics
            metrics['rmse'] = self.root_mean_squared_error(y_true, y_pred)
            metrics['median_absolute_error'] = self.median_absolute_error(y_true, y_pred)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise ModelTrainingError(f"Error calculating metrics: {e}")
        
        return metrics
    
    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))
    
    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def median_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Median Absolute Error."""
        return float(np.median(np.abs(y_true - y_pred)))
    
    def spearman_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Spearman rank correlation coefficient."""
        correlation, _ = spearmanr(y_true, y_pred)
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def top_k_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        Calculate Top-K accuracy (percentage of predictions within K positions).
        
        Args:
            y_true: True positions
            y_pred: Predicted positions
            k: Number of positions tolerance
            
        Returns:
            Top-K accuracy as a percentage
        """
        if len(y_true) == 0:
            return 0.0
        
        within_k = np.abs(y_true - y_pred) <= k
        return float(np.mean(within_k) * 100)
    
    def position_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate exact position accuracy."""
        if len(y_true) == 0:
            return 0.0
        
        exact_matches = y_true == np.round(y_pred)
        return float(np.mean(exact_matches) * 100)
    
    def calculate_position_distribution_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for position distribution analysis."""
        metrics = {}
        
        # Calculate error distribution
        errors = y_pred - y_true
        
        metrics['mean_error'] = float(np.mean(errors))
        metrics['std_error'] = float(np.std(errors))
        metrics['error_skewness'] = float(self._calculate_skewness(errors))
        
        # Position-specific metrics
        for position_range in [(1, 3), (4, 10), (11, 20)]:
            start, end = position_range
            mask = (y_true >= start) & (y_true <= end)
            
            if np.any(mask):
                range_mae = np.mean(np.abs(errors[mask]))
                metrics[f'mae_positions_{start}_{end}'] = float(range_mae)
        
        return metrics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness


class BaseModelTrainer(ModelTrainer):
    """
    Base model trainer with common functionality for all ML algorithms.
    """
    
    def __init__(self, random_state: int = None):
        self.random_state = random_state or config.model.random_state
        self.validator = TimeSeriesValidator()
        self.tuner = HyperparameterTuner(random_state=self.random_state)
        self.persistence = ModelPersistence()
        self.metrics_calculator = MetricsCalculator()
        
        # Training configuration
        self.config = {
            'test_size': config.model.test_size,
            'cv_folds': config.model.cv_folds,
            'enable_hyperparameter_tuning': True,
            'save_models': True,
            'calculate_feature_importance': True
        }
    
    def prepare_training_data(self, features: List[Dict[str, Any]], targets: List[int],
                            dates: List[datetime] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from feature dictionaries.
        
        Args:
            features: List of feature dictionaries
            targets: List of target values (finishing positions)
            dates: List of race dates for temporal ordering
            
        Returns:
            Tuple of (X, y, dates_array)
        """
        try:
            if len(features) != len(targets):
                raise ValueError("Features and targets must have the same length")
            
            # Convert feature dictionaries to DataFrame for easier handling
            df = pd.DataFrame(features)
            
            # Handle missing values
            df = df.fillna(0)  # Simple imputation - could be made more sophisticated
            
            # Convert to numpy arrays
            X = df.values.astype(np.float32)
            y = np.array(targets, dtype=np.float32)
            
            # Handle dates
            if dates:
                dates_array = np.array(dates)
                
                # Validate temporal order
                if not self.validator.validate_temporal_order(dates):
                    logger.warning("Data is not in temporal order - sorting by date")
                    # Sort by date
                    sort_indices = np.argsort(dates_array)
                    X = X[sort_indices]
                    y = y[sort_indices]
                    dates_array = dates_array[sort_indices]
            else:
                dates_array = None
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, dates_array
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise ModelTrainingError(f"Error preparing training data: {e}")
    
    def split_temporal_data(self, X: np.ndarray, y: np.ndarray, 
                          test_size: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally (no shuffling to maintain time order).
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        test_size = test_size or self.config['test_size']
        split_index = int(len(X) * (1 - test_size))
        
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        logger.info(f"Temporal split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, features: List[Dict[str, Any]], targets: List[int]) -> Any:
        """
        Train a model - to be implemented by subclasses.
        
        Args:
            features: List of feature dictionaries
            targets: List of target values
            
        Returns:
            Trained model
        """
        raise NotImplementedError("Subclasses must implement train_model")
    
    def evaluate_model(self, model: Any, test_features: List[Dict[str, Any]], 
                      test_targets: List[int]) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            test_features: Test feature dictionaries
            test_targets: Test target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Prepare test data
            X_test, y_test, _ = self.prepare_training_data(test_features, test_targets)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(y_test, y_pred)
            
            # Add position distribution metrics
            distribution_metrics = self.metrics_calculator.calculate_position_distribution_metrics(y_test, y_pred)
            metrics.update(distribution_metrics)
            
            logger.info(f"Model evaluation completed. MAE: {metrics['mae']:.3f}, "
                       f"Spearman: {metrics['spearman_correlation']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise ModelTrainingError(f"Error evaluating model: {e}")
    
    def tune_hyperparameters(self, model_type: str, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tune hyperparameters for a model type.
        
        Args:
            model_type: Type of model to tune
            param_space: Parameter space to search
            
        Returns:
            Best parameters found
        """
        # This method should be called with training data
        # Implementation depends on the specific model type
        raise NotImplementedError("Call tune_hyperparameters with training data")
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = None) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation metrics
        """
        try:
            cv_folds = cv_folds or self.config['cv_folds']
            validator = TimeSeriesValidator(n_splits=cv_folds)
            
            cv_scores = []
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(validator.split(X, y)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Clone and train model for this fold
                fold_model = self._clone_model(model)
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Predict and evaluate
                y_pred_fold = fold_model.predict(X_val_fold)
                fold_mae = mean_absolute_error(y_val_fold, y_pred_fold)
                cv_scores.append(fold_mae)
                
                # Calculate detailed metrics for this fold
                metrics = self.metrics_calculator.calculate_all_metrics(y_val_fold, y_pred_fold)
                fold_metrics.append(metrics)
                
                logger.debug(f"Fold {fold + 1}: MAE = {fold_mae:.3f}")
            
            # Aggregate metrics across folds
            cv_results = {
                'cv_mae_mean': np.mean(cv_scores),
                'cv_mae_std': np.std(cv_scores),
                'cv_folds': cv_folds
            }
            
            # Average other metrics across folds
            if fold_metrics:
                for metric_name in fold_metrics[0].keys():
                    metric_values = [fold[metric_name] for fold in fold_metrics]
                    cv_results[f'cv_{metric_name}_mean'] = np.mean(metric_values)
                    cv_results[f'cv_{metric_name}_std'] = np.std(metric_values)
            
            logger.info(f"Cross-validation completed. Mean MAE: {cv_results['cv_mae_mean']:.3f} "
                       f"(±{cv_results['cv_mae_std']:.3f})")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise ModelTrainingError(f"Error in cross-validation: {e}")
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model for cross-validation."""
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback: create new instance with same parameters
            return type(model)(**model.get_params())
    
    def get_feature_importance(self, model: Any, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature names and importance scores
        """
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                logger.warning("Model does not support feature importance extraction")
                return importance_dict
            
            if feature_names and len(feature_names) == len(importances):
                importance_dict = dict(zip(feature_names, importances))
            else:
                importance_dict = {f'feature_{i}': imp for i, imp in enumerate(importances)}
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def save_training_results(self, model: Any, metrics: Dict[str, float], 
                            model_name: str, additional_metadata: Dict[str, Any] = None) -> str:
        """
        Save trained model and results.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            model_name: Name for the saved model
            additional_metadata: Additional metadata to save
            
        Returns:
            Path to saved model
        """
        try:
            metadata = {
                'model_type': type(model).__name__,
                'training_date': datetime.now().isoformat(),
                'metrics': metrics,
                'random_state': self.random_state,
                'config': self.config
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            model_path = self.persistence.save_model(model, model_name, metadata)
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
            raise ModelTrainingError(f"Error saving training results: {e}")