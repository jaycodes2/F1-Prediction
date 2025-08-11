"""
Advanced ensemble methods for F1 race prediction.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
import logging

from .implementations import ModelFactory
from .training import BaseModelTrainer, ModelTrainingError, MetricsCalculator


logger = logging.getLogger(__name__)


class AdvancedEnsembleModel(BaseModelTrainer):
    """
    Advanced ensemble model with multiple combination strategies.
    """
    
    def __init__(self, random_state: int = 42, ensemble_method: str = 'weighted_average'):
        super().__init__(random_state=random_state)
        self.ensemble_method = ensemble_method
        self.base_models = {}
        self.meta_model = None
        self.model_weights = {}
        self.feature_names = None
        
        # Available ensemble methods
        self.available_methods = [
            'weighted_average', 'stacking', 'voting', 'dynamic_weighting'
        ]
        
        if ensemble_method not in self.available_methods:
            raise ValueError(f"Invalid ensemble method. Available: {self.available_methods}")
    
    def train(self, features: List[Dict], targets: List[float], 
              dates: Optional[List] = None, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """Train the advanced ensemble model."""
        logger.info(f"Training Advanced Ensemble with {self.ensemble_method} method")
        
        X, y, dates_array = self.prepare_training_data(features, targets, dates)
        self.feature_names = list(features[0].keys()) if features else None
        
        # Split data for meta-learning
        X_train, X_val, y_train, y_val = self.split_temporal_data(X, y, test_size=0.3)
        
        # Define base models to use
        base_model_configs = {
            'random_forest': {'random_state': self.random_state, 'n_estimators': 100},
            'xgboost': {'random_state': self.random_state, 'n_estimators': 100, 'verbosity': 0},
            'lightgbm': {'random_state': self.random_state, 'n_estimators': 100, 'verbose': -1},
            'gradient_boosting': {'random_state': self.random_state, 'n_estimators': 100}
        }
        
        # Train base models
        base_predictions_train = {}
        base_predictions_val = {}
        model_scores = {}
        
        for model_name, config in base_model_configs.items():
            logger.info(f"Training base model: {model_name}")
            
            try:
                model = ModelFactory.create_model(model_name, **config)
                
                # Convert back to feature dictionaries for the model's train method
                train_features = [dict(zip(self.feature_names, row)) for row in X_train]
                train_targets = y_train.tolist()
                
                # Train the model
                model.train(train_features, train_targets, tune_hyperparameters=False)
                
                # Get predictions for validation set
                val_features = [dict(zip(self.feature_names, row)) for row in X_val]
                val_predictions = model.predict(val_features)
                
                # Get predictions for training set (for stacking)
                train_predictions = model.predict(train_features)
                
                self.base_models[model_name] = model
                base_predictions_train[model_name] = train_predictions
                base_predictions_val[model_name] = val_predictions
                
                # Calculate model score
                mae = np.mean(np.abs(y_val - val_predictions))
                model_scores[model_name] = 1.0 / (1.0 + mae)  # Convert to weight
                
                logger.info(f"✓ {model_name} trained successfully (MAE: {mae:.3f})")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                continue
        
        if not self.base_models:
            raise ModelTrainingError("No base models were successfully trained")
        
        # Apply ensemble method
        if self.ensemble_method == 'weighted_average':
            self._train_weighted_average(model_scores)
        elif self.ensemble_method == 'stacking':
            self._train_stacking(base_predictions_train, y_train, base_predictions_val, y_val)
        elif self.ensemble_method == 'voting':
            self._train_voting()
        elif self.ensemble_method == 'dynamic_weighting':
            self._train_dynamic_weighting(base_predictions_val, y_val)
        
        # Evaluate ensemble
        ensemble_predictions = self._predict_ensemble(base_predictions_val)
        metrics = self.metrics_calculator.calculate_all_metrics(y_val, ensemble_predictions)
        
        logger.info(f"Ensemble training completed. MAE: {metrics['mae']:.3f}")
        
        return {
            'metrics': metrics,
            'ensemble_method': self.ensemble_method,
            'base_models': list(self.base_models.keys()),
            'model_weights': self.model_weights,
            'model_type': 'AdvancedEnsemble',
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def _train_weighted_average(self, model_scores: Dict[str, float]):
        """Train weighted average ensemble."""
        total_score = sum(model_scores.values())
        self.model_weights = {name: score / total_score for name, score in model_scores.items()}
        logger.info(f"Weighted average weights: {self.model_weights}")
    
    def _train_stacking(self, train_preds: Dict[str, np.ndarray], y_train: np.ndarray,
                       val_preds: Dict[str, np.ndarray], y_val: np.ndarray):
        """Train stacking ensemble with meta-learner."""
        # Create meta-features from base model predictions
        meta_features_train = np.column_stack(list(train_preds.values()))
        meta_features_val = np.column_stack(list(val_preds.values()))
        
        # Train meta-model
        self.meta_model = Ridge(alpha=1.0, random_state=self.random_state)
        self.meta_model.fit(meta_features_train, y_train)
        
        # Evaluate meta-model
        meta_predictions = self.meta_model.predict(meta_features_val)
        meta_mae = np.mean(np.abs(y_val - meta_predictions))
        
        logger.info(f"Stacking meta-model trained (MAE: {meta_mae:.3f})")
    
    def _train_voting(self):
        """Train voting ensemble (simple average)."""
        n_models = len(self.base_models)
        self.model_weights = {name: 1.0 / n_models for name in self.base_models.keys()}
        logger.info("Voting ensemble: equal weights for all models")
    
    def _train_dynamic_weighting(self, val_preds: Dict[str, np.ndarray], y_val: np.ndarray):
        """Train dynamic weighting based on prediction confidence."""
        # Calculate weights based on prediction variance and accuracy
        weights = {}
        
        for model_name, predictions in val_preds.items():
            # Calculate accuracy weight
            mae = np.mean(np.abs(y_val - predictions))
            accuracy_weight = 1.0 / (1.0 + mae)
            
            # Calculate confidence weight (inverse of prediction variance)
            pred_variance = np.var(predictions)
            confidence_weight = 1.0 / (1.0 + pred_variance)
            
            # Combine weights
            weights[model_name] = accuracy_weight * confidence_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.model_weights = {name: weight / total_weight for name, weight in weights.items()}
        
        logger.info(f"Dynamic weighting: {self.model_weights}")
    
    def _predict_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate ensemble predictions."""
        if self.ensemble_method == 'stacking' and self.meta_model is not None:
            meta_features = np.column_stack(list(base_predictions.values()))
            return self.meta_model.predict(meta_features)
        else:
            # Weighted average
            ensemble_pred = np.zeros(len(next(iter(base_predictions.values()))))
            for model_name, predictions in base_predictions.items():
                weight = self.model_weights.get(model_name, 0)
                ensemble_pred += weight * predictions
            return ensemble_pred
    
    def predict(self, features: List[Dict]) -> np.ndarray:
        """Make predictions using the ensemble."""
        if not self.base_models:
            raise ModelTrainingError("Ensemble must be trained before making predictions")
        
        # Get predictions from all base models
        base_predictions = {}
        for model_name, model in self.base_models.items():
            predictions = model.predict(features)
            base_predictions[model_name] = predictions
        
        return self._predict_ensemble(base_predictions)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        if not self.base_models:
            return {'status': 'not_trained'}
        
        return {
            'model_type': 'AdvancedEnsemble',
            'ensemble_method': self.ensemble_method,
            'base_models': list(self.base_models.keys()),
            'model_weights': self.model_weights,
            'n_base_models': len(self.base_models),
            'feature_names': self.feature_names
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison framework.
    """
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.evaluation_results = {}
    
    def evaluate_single_model(self, model, test_features: List[Dict], 
                            test_targets: List[float], model_name: str = None) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        model_name = model_name or type(model).__name__
        
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        predictions = model.predict(test_features)
        y_true = np.array(test_targets)
        
        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all_metrics(y_true, predictions)
        
        # Add position distribution metrics
        distribution_metrics = self.metrics_calculator.calculate_position_distribution_metrics(
            y_true, predictions
        )
        metrics.update(distribution_metrics)
        
        # Calculate additional evaluation metrics
        additional_metrics = self._calculate_additional_metrics(y_true, predictions)
        metrics.update(additional_metrics)
        
        # Get model info
        model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
        
        evaluation_result = {
            'model_name': model_name,
            'metrics': metrics,
            'model_info': model_info,
            'predictions': predictions.tolist(),
            'n_predictions': len(predictions)
        }
        
        self.evaluation_results[model_name] = evaluation_result
        
        logger.info(f"✓ {model_name} evaluation completed")
        return evaluation_result
    
    def compare_models(self, models: Dict[str, Any], test_features: List[Dict], 
                      test_targets: List[float]) -> Dict[str, Any]:
        """Compare multiple models side by side."""
        logger.info(f"Comparing {len(models)} models")
        
        comparison_results = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            try:
                result = self.evaluate_single_model(model, test_features, test_targets, model_name)
                comparison_results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'summary': summary,
            'best_models': self._identify_best_models(comparison_results)
        }
    
    def _calculate_additional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate additional evaluation metrics."""
        metrics = {}
        
        # Prediction range analysis
        metrics['pred_min'] = float(np.min(y_pred))
        metrics['pred_max'] = float(np.max(y_pred))
        metrics['pred_std'] = float(np.std(y_pred))
        
        # Error analysis
        errors = y_pred - y_true
        metrics['mean_error'] = float(np.mean(errors))
        metrics['error_std'] = float(np.std(errors))
        metrics['max_error'] = float(np.max(np.abs(errors)))
        
        # Percentile accuracies
        for percentile in [25, 50, 75, 90]:
            threshold = np.percentile(np.abs(errors), percentile)
            within_threshold = np.mean(np.abs(errors) <= threshold) * 100
            metrics[f'accuracy_p{percentile}'] = float(within_threshold)
        
        return metrics
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of model comparison."""
        summary = {
            'n_models': len(results),
            'metrics_comparison': {},
            'ranking': {}
        }
        
        # Extract key metrics for comparison
        key_metrics = ['mae', 'spearman_correlation', 'top_3_accuracy', 'position_accuracy']
        
        for metric in key_metrics:
            metric_values = {}
            for model_name, result in results.items():
                if 'metrics' in result and metric in result['metrics']:
                    metric_values[model_name] = result['metrics'][metric]
            
            if metric_values:
                summary['metrics_comparison'][metric] = {
                    'values': metric_values,
                    'best': min(metric_values.items(), key=lambda x: x[1]) if metric == 'mae' 
                           else max(metric_values.items(), key=lambda x: x[1]),
                    'worst': max(metric_values.items(), key=lambda x: x[1]) if metric == 'mae' 
                            else min(metric_values.items(), key=lambda x: x[1])
                }
        
        return summary
    
    def _identify_best_models(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Identify best performing models for different metrics."""
        best_models = {}
        
        metrics_to_check = {
            'lowest_mae': ('mae', min),
            'highest_spearman': ('spearman_correlation', max),
            'highest_top3_accuracy': ('top_3_accuracy', max),
            'highest_position_accuracy': ('position_accuracy', max)
        }
        
        for category, (metric, func) in metrics_to_check.items():
            metric_values = {}
            for model_name, result in results.items():
                if 'metrics' in result and metric in result['metrics']:
                    metric_values[model_name] = result['metrics'][metric]
            
            if metric_values:
                best_model = func(metric_values.items(), key=lambda x: x[1])
                best_models[category] = {
                    'model': best_model[0],
                    'value': best_model[1]
                }
        
        return best_models
    
    def generate_evaluation_report(self, output_file: str = None) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("F1 RACE PREDICTION MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Models evaluated: {len(self.evaluation_results)}")
        report_lines.append("")
        
        # Individual model results
        for model_name, result in self.evaluation_results.items():
            if 'error' in result:
                report_lines.append(f"{model_name}: FAILED - {result['error']}")
                continue
            
            metrics = result['metrics']
            report_lines.append(f"{model_name.upper()}")
            report_lines.append("-" * len(model_name))
            report_lines.append(f"MAE: {metrics.get('mae', 'N/A'):.3f}")
            report_lines.append(f"Spearman Correlation: {metrics.get('spearman_correlation', 'N/A'):.3f}")
            report_lines.append(f"Top-3 Accuracy: {metrics.get('top_3_accuracy', 'N/A'):.1f}%")
            report_lines.append(f"Position Accuracy: {metrics.get('position_accuracy', 'N/A'):.1f}%")
            report_lines.append(f"RMSE: {metrics.get('rmse', 'N/A'):.3f}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_file}")
        
        return report_text


class CrossValidationEvaluator:
    """
    Advanced cross-validation evaluation for time series data.
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_model_cv(self, model_factory_func, features: List[Dict], 
                         targets: List[float], **model_kwargs) -> Dict[str, Any]:
        """Evaluate a model using time-series cross-validation."""
        logger.info(f"Starting {self.n_splits}-fold cross-validation")
        
        # Prepare data
        trainer = BaseModelTrainer()
        X, y, _ = trainer.prepare_training_data(features, targets)
        
        # Time series splits
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Convert back to feature dictionaries
            train_features_fold = [dict(zip(features[0].keys(), row)) for row in X_train_fold]
            val_features_fold = [dict(zip(features[0].keys(), row)) for row in X_val_fold]
            
            try:
                # Create and train model
                model = model_factory_func(**model_kwargs)
                model.train(train_features_fold, y_train_fold.tolist(), tune_hyperparameters=False)
                
                # Make predictions
                predictions = model.predict(val_features_fold)
                
                # Calculate metrics
                fold_metrics = self.metrics_calculator.calculate_all_metrics(y_val_fold, predictions)
                fold_results.append(fold_metrics)
                
            except Exception as e:
                logger.error(f"Fold {fold + 1} failed: {e}")
                continue
        
        if not fold_results:
            raise ModelTrainingError("All cross-validation folds failed")
        
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results)
        
        logger.info(f"Cross-validation completed. Mean MAE: {cv_results['mae_mean']:.3f}")
        return cv_results
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate cross-validation results across folds."""
        if not fold_results:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        all_metrics = set()
        for fold_result in fold_results:
            all_metrics.update(fold_result.keys())
        
        # Calculate mean and std for each metric
        for metric in all_metrics:
            values = [fold_result.get(metric, 0) for fold_result in fold_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        
        aggregated['n_folds'] = len(fold_results)
        return aggregated