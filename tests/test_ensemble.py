"""
Tests for advanced ensemble methods and model evaluation.
"""
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch
from pathlib import Path

from src.models.ensemble import (
    AdvancedEnsembleModel, ModelEvaluator, CrossValidationEvaluator
)
from src.models.implementations import ModelFactory
from src.models.training import ModelTrainingError


class TestAdvancedEnsembleModel:
    """Tests for AdvancedEnsembleModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        features = []
        targets = []
        
        for i in range(60):  # Need sufficient data for ensemble
            features.append({
                'qualifying_position': np.random.randint(1, 21),
                'driver_points': np.random.randint(0, 100),
                'constructor_points': np.random.randint(0, 200),
                'track_temperature': np.random.uniform(20, 40)
            })
            targets.append(np.random.randint(1, 21))
        
        return features, targets
    
    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        ensemble = AdvancedEnsembleModel(random_state=42, ensemble_method='weighted_average')
        
        assert ensemble.random_state == 42
        assert ensemble.ensemble_method == 'weighted_average'
        assert len(ensemble.base_models) == 0
        assert ensemble.meta_model is None
    
    def test_invalid_ensemble_method(self):
        """Test initialization with invalid ensemble method."""
        with pytest.raises(ValueError):
            AdvancedEnsembleModel(ensemble_method='invalid_method')
    
    def test_weighted_average_ensemble(self, sample_data):
        """Test weighted average ensemble method."""
        features, targets = sample_data
        
        ensemble = AdvancedEnsembleModel(
            random_state=42, 
            ensemble_method='weighted_average'
        )
        
        results = ensemble.train(features, targets, tune_hyperparameters=False)
        
        assert ensemble.ensemble_method == 'weighted_average'
        assert len(ensemble.base_models) > 0
        assert len(ensemble.model_weights) > 0
        assert results['model_type'] == 'AdvancedEnsemble'
        assert 'metrics' in results
        
        # Check that weights sum to approximately 1
        total_weight = sum(ensemble.model_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_stacking_ensemble(self, sample_data):
        """Test stacking ensemble method."""
        features, targets = sample_data
        
        ensemble = AdvancedEnsembleModel(
            random_state=42, 
            ensemble_method='stacking'
        )
        
        results = ensemble.train(features, targets, tune_hyperparameters=False)
        
        assert ensemble.ensemble_method == 'stacking'
        assert ensemble.meta_model is not None
        assert len(ensemble.base_models) > 0
        assert results['ensemble_method'] == 'stacking'
    
    def test_voting_ensemble(self, sample_data):
        """Test voting ensemble method."""
        features, targets = sample_data
        
        ensemble = AdvancedEnsembleModel(
            random_state=42, 
            ensemble_method='voting'
        )
        
        results = ensemble.train(features, targets, tune_hyperparameters=False)
        
        assert ensemble.ensemble_method == 'voting'
        assert len(ensemble.model_weights) > 0
        
        # All weights should be equal for voting
        weights = list(ensemble.model_weights.values())
        assert all(abs(w - weights[0]) < 0.01 for w in weights)
    
    def test_dynamic_weighting_ensemble(self, sample_data):
        """Test dynamic weighting ensemble method."""
        features, targets = sample_data
        
        ensemble = AdvancedEnsembleModel(
            random_state=42, 
            ensemble_method='dynamic_weighting'
        )
        
        results = ensemble.train(features, targets, tune_hyperparameters=False)
        
        assert ensemble.ensemble_method == 'dynamic_weighting'
        assert len(ensemble.model_weights) > 0
        assert results['ensemble_method'] == 'dynamic_weighting'
    
    def test_ensemble_prediction(self, sample_data):
        """Test ensemble prediction."""
        features, targets = sample_data
        
        ensemble = AdvancedEnsembleModel(random_state=42)
        ensemble.train(features, targets, tune_hyperparameters=False)
        
        # Make predictions
        test_features = features[:10]
        predictions = ensemble.predict(test_features)
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
        assert all(1 <= p <= 21 for p in predictions)
    
    def test_predict_without_training(self, sample_data):
        """Test prediction without training raises error."""
        features, _ = sample_data
        ensemble = AdvancedEnsembleModel()
        
        with pytest.raises(ModelTrainingError):
            ensemble.predict(features[:5])
    
    def test_get_model_info(self, sample_data):
        """Test getting ensemble model info."""
        features, targets = sample_data
        
        # Before training
        ensemble = AdvancedEnsembleModel()
        info = ensemble.get_model_info()
        assert info['status'] == 'not_trained'
        
        # After training
        ensemble.train(features, targets, tune_hyperparameters=False)
        info = ensemble.get_model_info()
        
        assert info['model_type'] == 'AdvancedEnsemble'
        assert 'base_models' in info
        assert 'model_weights' in info
        assert 'n_base_models' in info
        assert info['n_base_models'] > 0


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create test evaluator."""
        return ModelEvaluator()
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        features = []
        targets = []
        
        for i in range(30):
            features.append({
                'feature_1': np.random.rand(),
                'feature_2': np.random.rand(),
                'feature_3': np.random.rand()
            })
            targets.append(np.random.randint(1, 21))
        
        model = ModelFactory.create_model('random_forest', random_state=42)
        model.train(features, targets, tune_hyperparameters=False)
        
        return model, features[:10], targets[:10]
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.metrics_calculator is not None
        assert len(evaluator.evaluation_results) == 0
    
    def test_evaluate_single_model(self, evaluator, trained_model):
        """Test single model evaluation."""
        model, test_features, test_targets = trained_model
        
        result = evaluator.evaluate_single_model(
            model, test_features, test_targets, 'test_model'
        )
        
        assert result['model_name'] == 'test_model'
        assert 'metrics' in result
        assert 'model_info' in result
        assert 'predictions' in result
        assert 'n_predictions' in result
        
        # Check that metrics are calculated
        metrics = result['metrics']
        assert 'mae' in metrics
        assert 'spearman_correlation' in metrics
        assert isinstance(metrics['mae'], float)
        assert metrics['mae'] >= 0
    
    def test_compare_models(self, evaluator):
        """Test model comparison."""
        # Create sample data
        features = []
        targets = []
        
        for i in range(40):
            features.append({
                'feature_1': np.random.rand(),
                'feature_2': np.random.rand()
            })
            targets.append(np.random.randint(1, 21))
        
        # Create and train multiple models
        models = {}
        for model_type in ['random_forest']:  # Use only one for speed
            model = ModelFactory.create_model(model_type, random_state=42)
            model.train(features[:30], targets[:30], tune_hyperparameters=False)
            models[model_type] = model
        
        # Compare models
        comparison = evaluator.compare_models(
            models, features[30:], targets[30:]
        )
        
        assert 'individual_results' in comparison
        assert 'summary' in comparison
        assert 'best_models' in comparison
        
        # Check individual results
        individual = comparison['individual_results']
        assert len(individual) == len(models)
        
        # Check summary
        summary = comparison['summary']
        assert 'n_models' in summary
        assert summary['n_models'] == len(models)
    
    def test_generate_evaluation_report(self, evaluator, trained_model):
        """Test evaluation report generation."""
        model, test_features, test_targets = trained_model
        
        # Evaluate a model first
        evaluator.evaluate_single_model(
            model, test_features, test_targets, 'test_model'
        )
        
        # Generate report
        report = evaluator.generate_evaluation_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'F1 RACE PREDICTION MODEL EVALUATION REPORT' in report
        assert 'test_model' in report.upper()
    
    def test_empty_evaluation_report(self, evaluator):
        """Test report generation with no results."""
        report = evaluator.generate_evaluation_report()
        assert report == "No evaluation results available."


class TestCrossValidationEvaluator:
    """Tests for CrossValidationEvaluator class."""
    
    @pytest.fixture
    def cv_evaluator(self):
        """Create test CV evaluator."""
        return CrossValidationEvaluator(n_splits=3)  # Small number for testing
    
    @pytest.fixture
    def cv_data(self):
        """Create data for cross-validation testing."""
        np.random.seed(42)
        features = []
        targets = []
        
        for i in range(50):  # Need enough data for CV splits
            features.append({
                'feature_1': np.random.rand(),
                'feature_2': np.random.rand(),
                'feature_3': np.random.rand()
            })
            targets.append(np.random.randint(1, 21))
        
        return features, targets
    
    def test_cv_evaluator_initialization(self, cv_evaluator):
        """Test CV evaluator initialization."""
        assert cv_evaluator.n_splits == 3
        assert cv_evaluator.metrics_calculator is not None
    
    def test_evaluate_model_cv(self, cv_evaluator, cv_data):
        """Test cross-validation evaluation."""
        features, targets = cv_data
        
        # Define model factory function
        def model_factory(**kwargs):
            return ModelFactory.create_model('random_forest', **kwargs)
        
        # Run cross-validation
        cv_results = cv_evaluator.evaluate_model_cv(
            model_factory, features, targets, random_state=42
        )
        
        assert isinstance(cv_results, dict)
        assert 'mae_mean' in cv_results
        assert 'mae_std' in cv_results
        assert 'n_folds' in cv_results
        assert cv_results['n_folds'] <= 3  # May be less if some folds fail
        
        # Check that we have mean and std for key metrics
        for metric in ['mae', 'spearman_correlation']:
            assert f'{metric}_mean' in cv_results
            assert f'{metric}_std' in cv_results
            assert isinstance(cv_results[f'{metric}_mean'], (int, float))
            assert isinstance(cv_results[f'{metric}_std'], (int, float))
    
    def test_cv_with_insufficient_data(self, cv_evaluator):
        """Test CV with insufficient data."""
        # Very small dataset
        features = [{'feature_1': 0.5}] * 5
        targets = [1, 2, 3, 4, 5]
        
        def model_factory(**kwargs):
            return ModelFactory.create_model('random_forest', **kwargs)
        
        # This might fail or have limited folds
        try:
            cv_results = cv_evaluator.evaluate_model_cv(
                model_factory, features, targets, random_state=42
            )
            # If it succeeds, check basic structure
            assert isinstance(cv_results, dict)
        except ModelTrainingError:
            # Expected for insufficient data
            pass


class TestIntegrationEnsembleEvaluation:
    """Integration tests for ensemble and evaluation systems."""
    
    @pytest.fixture
    def comprehensive_data(self):
        """Create comprehensive test data."""
        np.random.seed(42)
        features = []
        targets = []
        
        for i in range(80):
            qualifying_pos = np.random.randint(1, 21)
            
            features.append({
                'qualifying_position': qualifying_pos,
                'driver_points': np.random.randint(0, 400),
                'constructor_points': np.random.randint(0, 600),
                'track_temperature': np.random.uniform(15, 45),
                'weather_dry': np.random.choice([0, 1]),
                'driver_experience': np.random.randint(0, 300)
            })
            
            # Somewhat realistic finishing position
            base_finish = qualifying_pos + np.random.normal(0, 3)
            finish_pos = max(1, min(20, int(round(base_finish))))
            targets.append(finish_pos)
        
        return features, targets
    
    def test_ensemble_vs_individual_evaluation(self, comprehensive_data):
        """Test ensemble performance vs individual models."""
        features, targets = comprehensive_data
        
        # Split data
        train_features = features[:60]
        train_targets = targets[:60]
        test_features = features[60:]
        test_targets = targets[60:]
        
        # Train individual model
        individual_model = ModelFactory.create_model('random_forest', random_state=42)
        individual_model.train(train_features, train_targets, tune_hyperparameters=False)
        
        # Train ensemble model
        ensemble_model = AdvancedEnsembleModel(random_state=42, ensemble_method='weighted_average')
        ensemble_model.train(train_features, train_targets, tune_hyperparameters=False)
        
        # Evaluate both models
        evaluator = ModelEvaluator()
        
        individual_result = evaluator.evaluate_single_model(
            individual_model, test_features, test_targets, 'individual'
        )
        
        ensemble_result = evaluator.evaluate_single_model(
            ensemble_model, test_features, test_targets, 'ensemble'
        )
        
        # Both should produce valid results
        assert individual_result['metrics']['mae'] >= 0
        assert ensemble_result['metrics']['mae'] >= 0
        
        # Both should have reasonable predictions
        assert len(individual_result['predictions']) == len(test_features)
        assert len(ensemble_result['predictions']) == len(test_features)
    
    def test_full_evaluation_pipeline(self, comprehensive_data):
        """Test complete evaluation pipeline."""
        features, targets = comprehensive_data
        
        # Create multiple models
        models = {}
        
        # Individual models
        for model_type in ['random_forest']:  # Use one for speed
            model = ModelFactory.create_model(model_type, random_state=42)
            model.train(features[:50], targets[:50], tune_hyperparameters=False)
            models[model_type] = model
        
        # Ensemble model
        ensemble = AdvancedEnsembleModel(random_state=42)
        ensemble.train(features[:50], targets[:50], tune_hyperparameters=False)
        models['ensemble'] = ensemble
        
        # Comprehensive evaluation
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(
            models, features[50:], targets[50:]
        )
        
        # Generate report
        report = evaluator.generate_evaluation_report()
        
        # Verify results
        assert len(comparison['individual_results']) == len(models)
        assert 'best_models' in comparison
        assert len(report) > 0
        
        # Check that all models were evaluated successfully
        for model_name in models.keys():
            assert model_name in comparison['individual_results']
            result = comparison['individual_results'][model_name]
            if 'error' not in result:
                assert 'metrics' in result
                assert result['metrics']['mae'] >= 0
    
    def test_cross_validation_integration(self, comprehensive_data):
        """Test cross-validation with different ensemble methods."""
        features, targets = comprehensive_data
        
        cv_evaluator = CrossValidationEvaluator(n_splits=3)
        
        # Test individual model CV
        def rf_factory(**kwargs):
            return ModelFactory.create_model('random_forest', **kwargs)
        
        rf_cv_results = cv_evaluator.evaluate_model_cv(
            rf_factory, features, targets, random_state=42
        )
        
        # Test ensemble model CV
        def ensemble_factory(**kwargs):
            return AdvancedEnsembleModel(ensemble_method='voting', **kwargs)
        
        ensemble_cv_results = cv_evaluator.evaluate_model_cv(
            ensemble_factory, features, targets, random_state=42
        )
        
        # Both should produce valid CV results
        assert 'mae_mean' in rf_cv_results
        assert 'mae_mean' in ensemble_cv_results
        assert rf_cv_results['n_folds'] > 0
        assert ensemble_cv_results['n_folds'] > 0