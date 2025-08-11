"""
Tests for F1 model implementations.
"""
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch
from pathlib import Path

from src.models.implementations import (
    F1RandomForestModel, F1GradientBoostingModel, F1EnsembleModel,
    F1NeuralNetworkModel, F1XGBoostModel, F1LightGBMModel, ModelFactory
)
from src.models.training import ModelTrainingError


class TestF1RandomForestModel:
    """Tests for F1RandomForestModel class."""
    
    @pytest.fixture
    def model(self):
        """Create test model instance."""
        return F1RandomForestModel(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        features = []
        targets = []
        
        for i in range(50):
            features.append({
                'qualifying_position': np.random.randint(1, 21),
                'driver_points': np.random.randint(0, 100),
                'constructor_points': np.random.randint(0, 200),
                'track_temperature': np.random.uniform(20, 40),
                'weather_dry': np.random.choice([0, 1])
            })
            targets.append(np.random.randint(1, 21))  # Finishing positions
        
        return features, targets
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert model.model is None
        assert model.feature_names is None
        assert 'n_estimators' in model.default_params
        assert model.default_params['random_state'] == 42
    
    def test_train_model_basic(self, model, sample_data):
        """Test basic model training."""
        features, targets = sample_data
        
        # Train without hyperparameter tuning for speed
        results = model.train(features, targets, tune_hyperparameters=False)
        
        assert model.model is not None
        assert model.feature_names is not None
        assert len(model.feature_names) == 5
        
        assert 'metrics' in results
        assert 'feature_importance' in results
        assert 'model_type' in results
        assert results['model_type'] == 'RandomForest'
        
        # Check metrics
        metrics = results['metrics']
        assert 'mae' in metrics
        assert 'spearman_correlation' in metrics
        assert isinstance(metrics['mae'], float)
        assert metrics['mae'] >= 0
    
    def test_predict(self, model, sample_data):
        """Test model prediction."""
        features, targets = sample_data
        
        # Train model first
        model.train(features, targets, tune_hyperparameters=False)
        
        # Make predictions
        test_features = features[:10]
        predictions = model.predict(test_features)
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
        assert all(1 <= p <= 21 for p in predictions)  # Should be reasonable positions
    
    def test_predict_without_training(self, model, sample_data):
        """Test prediction without training raises error."""
        features, _ = sample_data
        
        with pytest.raises(ModelTrainingError):
            model.predict(features[:5])
    
    def test_get_model_info(self, model, sample_data):
        """Test model info retrieval."""
        # Before training
        info = model.get_model_info()
        assert info['status'] == 'not_trained'
        
        # After training
        features, targets = sample_data
        model.train(features, targets, tune_hyperparameters=False)
        
        info = model.get_model_info()
        assert info['model_type'] == 'RandomForest'
        assert 'n_estimators' in info
        assert 'n_features' in info
        assert info['feature_names'] == model.feature_names


class TestF1XGBoostModel:
    """Tests for F1XGBoostModel class."""
    
    @pytest.fixture
    def model(self):
        """Create test model instance."""
        return F1XGBoostModel(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        features = []
        targets = []
        
        for i in range(30):  # Smaller dataset for faster testing
            features.append({
                'qualifying_position': np.random.randint(1, 21),
                'driver_rating': np.random.uniform(0, 100),
                'car_performance': np.random.uniform(0, 10)
            })
            targets.append(np.random.randint(1, 21))
        
        return features, targets
    
    def test_model_initialization(self, model):
        """Test XGBoost model initialization."""
        assert model.random_state == 42
        assert model.model is None
        assert 'objective' in model.default_params
        assert model.default_params['random_state'] == 42
    
    def test_train_model(self, model, sample_data):
        """Test XGBoost model training."""
        features, targets = sample_data
        
        results = model.train(features, targets, tune_hyperparameters=False)
        
        assert model.model is not None
        assert results['model_type'] == 'XGBoost'
        assert 'metrics' in results
        assert 'feature_importance' in results
    
    def test_predict(self, model, sample_data):
        """Test XGBoost prediction."""
        features, targets = sample_data
        
        model.train(features, targets, tune_hyperparameters=False)
        predictions = model.predict(features[:5])
        
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestF1LightGBMModel:
    """Tests for F1LightGBMModel class."""
    
    @pytest.fixture
    def model(self):
        """Create test model instance."""
        return F1LightGBMModel(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        features = []
        targets = []
        
        for i in range(30):
            features.append({
                'grid_position': np.random.randint(1, 21),
                'driver_experience': np.random.randint(0, 300),
                'team_budget': np.random.uniform(100, 500)
            })
            targets.append(np.random.randint(1, 21))
        
        return features, targets
    
    def test_model_initialization(self, model):
        """Test LightGBM model initialization."""
        assert model.random_state == 42
        assert model.model is None
        assert 'objective' in model.default_params
        assert model.default_params['random_state'] == 42
    
    def test_train_model(self, model, sample_data):
        """Test LightGBM model training."""
        features, targets = sample_data
        
        results = model.train(features, targets, tune_hyperparameters=False)
        
        assert model.model is not None
        assert results['model_type'] == 'LightGBM'
        assert 'metrics' in results


class TestF1EnsembleModel:
    """Tests for F1EnsembleModel class."""
    
    @pytest.fixture
    def model(self):
        """Create test ensemble model instance."""
        return F1EnsembleModel(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        features = []
        targets = []
        
        for i in range(60):  # Need more data for ensemble
            features.append({
                'qualifying_position': np.random.randint(1, 21),
                'driver_points': np.random.randint(0, 100),
                'constructor_points': np.random.randint(0, 200),
                'track_temperature': np.random.uniform(20, 40)
            })
            targets.append(np.random.randint(1, 21))
        
        return features, targets
    
    def test_model_initialization(self, model):
        """Test ensemble model initialization."""
        assert model.random_state == 42
        assert len(model.base_models) > 0
        assert 'random_forest' in model.base_models
        assert 'xgboost' in model.base_models
        assert 'lightgbm' in model.base_models
    
    def test_train_ensemble(self, model, sample_data):
        """Test ensemble model training."""
        features, targets = sample_data
        
        results = model.train(features, targets, tune_hyperparameters=False)
        
        assert len(model.models) > 0
        assert len(model.weights) > 0
        assert results['model_type'] == 'Ensemble'
        assert 'model_weights' in results
        assert 'base_models' in results
        
        # Check that weights sum to approximately 1
        total_weight = sum(model.weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_ensemble_predict(self, model, sample_data):
        """Test ensemble prediction."""
        features, targets = sample_data
        
        model.train(features, targets, tune_hyperparameters=False)
        predictions = model.predict(features[:10])
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_get_model_info(self, model, sample_data):
        """Test ensemble model info."""
        features, targets = sample_data
        model.train(features, targets, tune_hyperparameters=False)
        
        info = model.get_model_info()
        assert info['model_type'] == 'Ensemble'
        assert 'base_models' in info
        assert 'weights' in info
        assert len(info['base_models']) > 0


class TestF1NeuralNetworkModel:
    """Tests for F1NeuralNetworkModel class."""
    
    @pytest.fixture
    def model(self):
        """Create test neural network model instance."""
        return F1NeuralNetworkModel(random_state=42, max_iter=50)  # Reduce iterations for testing
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        features = []
        targets = []
        
        for i in range(40):
            features.append({
                'qualifying_position': np.random.randint(1, 21),
                'driver_skill': np.random.uniform(0, 1),
                'car_speed': np.random.uniform(0, 1),
                'weather_factor': np.random.uniform(0, 1)
            })
            targets.append(np.random.randint(1, 21))
        
        return features, targets
    
    def test_model_initialization(self, model):
        """Test neural network model initialization."""
        assert model.random_state == 42
        assert model.model is None
        assert model.scaler is not None
        assert 'hidden_layer_sizes' in model.default_params
    
    def test_train_model(self, model, sample_data):
        """Test neural network model training."""
        features, targets = sample_data
        
        results = model.train(features, targets, tune_hyperparameters=False)
        
        assert model.model is not None
        assert results['model_type'] == 'NeuralNetwork'
        assert 'n_iterations' in results
    
    def test_predict(self, model, sample_data):
        """Test neural network prediction."""
        features, targets = sample_data
        
        model.train(features, targets, tune_hyperparameters=False)
        predictions = model.predict(features[:5])
        
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestModelFactory:
    """Tests for ModelFactory class."""
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = ModelFactory.list_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'random_forest' in models
        assert 'xgboost' in models
        assert 'lightgbm' in models
        assert 'ensemble' in models
        assert 'neural_network' in models
    
    def test_create_model_valid_types(self):
        """Test creating models of valid types."""
        for model_type in ['random_forest', 'xgboost', 'lightgbm', 'ensemble']:
            model = ModelFactory.create_model(model_type, random_state=42)
            assert model is not None
            assert hasattr(model, 'train')
            assert hasattr(model, 'predict')
    
    def test_create_model_invalid_type(self):
        """Test creating model with invalid type."""
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model_type')
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = ModelFactory.get_model_info('random_forest')
        
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'class' in info
        assert 'description' in info
        assert info['name'] == 'random_forest'
    
    def test_get_model_info_invalid_type(self):
        """Test getting info for invalid model type."""
        with pytest.raises(ValueError):
            ModelFactory.get_model_info('invalid_model_type')
    
    def test_create_model_with_custom_params(self):
        """Test creating model with custom parameters."""
        model = ModelFactory.create_model(
            'random_forest', 
            random_state=123, 
            n_estimators=50
        )
        
        assert model.random_state == 123
        assert model.default_params['n_estimators'] == 50


class TestModelIntegration:
    """Integration tests for model training and prediction pipeline."""
    
    @pytest.fixture
    def comprehensive_data(self):
        """Create comprehensive training data."""
        np.random.seed(42)  # For reproducible tests
        
        features = []
        targets = []
        
        for i in range(100):
            # Create more realistic F1 data
            qualifying_pos = np.random.randint(1, 21)
            
            features.append({
                'qualifying_position': qualifying_pos,
                'driver_championship_points': np.random.randint(0, 400),
                'constructor_championship_points': np.random.randint(0, 600),
                'driver_wins_season': np.random.randint(0, 10),
                'constructor_wins_season': np.random.randint(0, 15),
                'track_temperature': np.random.uniform(15, 45),
                'air_temperature': np.random.uniform(10, 40),
                'humidity': np.random.uniform(30, 90),
                'wind_speed': np.random.uniform(0, 20),
                'weather_dry': np.random.choice([0, 1], p=[0.2, 0.8]),
                'track_grip': np.random.uniform(0.7, 1.0),
                'fuel_load': np.random.uniform(50, 110),
                'tire_compound': np.random.randint(1, 4),
                'engine_power': np.random.uniform(800, 1000),
                'aerodynamic_efficiency': np.random.uniform(0.6, 1.0)
            })
            
            # Create somewhat realistic finishing position based on qualifying
            # Better qualifying positions tend to finish better, but with noise
            base_finish = qualifying_pos + np.random.normal(0, 3)
            finish_pos = max(1, min(20, int(round(base_finish))))
            targets.append(finish_pos)
        
        return features, targets
    
    def test_model_comparison(self, comprehensive_data):
        """Test and compare different model types."""
        features, targets = comprehensive_data
        
        models_to_test = ['random_forest', 'xgboost', 'lightgbm']
        results = {}
        
        for model_type in models_to_test:
            model = ModelFactory.create_model(model_type, random_state=42)
            
            # Train model
            training_results = model.train(features, targets, tune_hyperparameters=False)
            
            # Make predictions
            predictions = model.predict(features[:20])
            
            results[model_type] = {
                'mae': training_results['metrics']['mae'],
                'spearman_correlation': training_results['metrics']['spearman_correlation'],
                'predictions_range': (min(predictions), max(predictions))
            }
        
        # All models should produce reasonable results
        for model_type, result in results.items():
            assert result['mae'] >= 0
            assert -1 <= result['spearman_correlation'] <= 1
            assert 1 <= result['predictions_range'][0] <= 21
            assert 1 <= result['predictions_range'][1] <= 21
    
    def test_ensemble_vs_individual_models(self, comprehensive_data):
        """Test that ensemble model works with individual models."""
        features, targets = comprehensive_data
        
        # Train individual models
        rf_model = ModelFactory.create_model('random_forest', random_state=42)
        rf_results = rf_model.train(features, targets, tune_hyperparameters=False)
        
        # Train ensemble model
        ensemble_model = ModelFactory.create_model('ensemble', random_state=42)
        ensemble_results = ensemble_model.train(features, targets, tune_hyperparameters=False)
        
        # Both should produce valid results
        assert rf_results['metrics']['mae'] >= 0
        assert ensemble_results['metrics']['mae'] >= 0
        
        # Ensemble should use multiple base models
        assert len(ensemble_results['base_models']) > 1
        assert 'random_forest' in ensemble_results['base_models']
    
    def test_feature_importance_extraction(self, comprehensive_data):
        """Test feature importance extraction across different models."""
        features, targets = comprehensive_data
        
        models_with_importance = ['random_forest', 'xgboost', 'lightgbm']
        
        for model_type in models_with_importance:
            model = ModelFactory.create_model(model_type, random_state=42)
            results = model.train(features, targets, tune_hyperparameters=False)
            
            importance = results['feature_importance']
            assert isinstance(importance, dict)
            assert len(importance) > 0
            
            # Check that all feature names are present
            feature_names = set(features[0].keys())
            importance_names = set(importance.keys())
            assert feature_names == importance_names
            
            # Check that importance values are reasonable
            for feature, score in importance.items():
                assert isinstance(score, (int, float, np.number))
                assert score >= 0