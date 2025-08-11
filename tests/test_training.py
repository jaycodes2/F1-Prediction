"""
Tests for model training infrastructure.
"""
import pytest
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.training import (
    BaseModelTrainer, TimeSeriesValidator, HyperparameterTuner,
    ModelPersistence, MetricsCalculator, ModelTrainingError
)


class TestTimeSeriesValidator:
    """Tests for TimeSeriesValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a test validator."""
        return TimeSeriesValidator(n_splits=3)
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.n_splits == 3
        assert validator.tscv is not None
    
    def test_split_generation(self, validator):
        """Test time series split generation."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        splits = list(validator.split(X, y))
        assert len(splits) == 3
        
        # Check that each split has train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert max(train_idx) < min(test_idx)  # Time series property
    
    def test_validate_temporal_order_valid(self, validator):
        """Test temporal order validation with valid data."""
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 2, 1),
            datetime(2024, 3, 1)
        ]
        assert validator.validate_temporal_order(dates) is True
    
    def test_validate_temporal_order_invalid(self, validator):
        """Test temporal order validation with invalid data."""
        dates = [
            datetime(2024, 3, 1),
            datetime(2024, 1, 1),  # Out of order
            datetime(2024, 2, 1)
        ]
        assert validator.validate_temporal_order(dates) is False
    
    def test_validate_temporal_order_empty(self, validator):
        """Test temporal order validation with empty data."""
        assert validator.validate_temporal_order([]) is True
        assert validator.validate_temporal_order([datetime.now()]) is True


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner class."""
    
    @pytest.fixture
    def tuner(self):
        """Create a test tuner."""
        return HyperparameterTuner(search_type='random', n_iter=5, cv=2)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        return X, y
    
    def test_tuner_initialization(self, tuner):
        """Test tuner initialization."""
        assert tuner.search_type == 'random'
        assert tuner.n_iter == 5
        assert tuner.cv == 2
    
    def test_tune_hyperparameters_random_search(self, tuner, sample_data):
        """Test hyperparameter tuning with random search."""
        X, y = sample_data
        model = RandomForestRegressor(random_state=42)
        param_space = {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5, 7]
        }
        
        best_params = tuner.tune_hyperparameters(model, param_space, X, y)
        
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert best_params['n_estimators'] in param_space['n_estimators']
        assert best_params['max_depth'] in param_space['max_depth']
    
    def test_tune_hyperparameters_grid_search(self, sample_data):
        """Test hyperparameter tuning with grid search."""
        tuner = HyperparameterTuner(search_type='grid', cv=2)
        X, y = sample_data
        model = LinearRegression()
        param_space = {
            'fit_intercept': [True, False]
        }
        
        best_params = tuner.tune_hyperparameters(model, param_space, X, y)
        
        assert isinstance(best_params, dict)
        assert 'fit_intercept' in best_params
        assert best_params['fit_intercept'] in param_space['fit_intercept']
    
    def test_tune_hyperparameters_invalid_search_type(self, sample_data):
        """Test hyperparameter tuning with invalid search type."""
        tuner = HyperparameterTuner(search_type='invalid')
        X, y = sample_data
        model = RandomForestRegressor()
        param_space = {'n_estimators': [10, 20]}
        
        with pytest.raises(ModelTrainingError):
            tuner.tune_hyperparameters(model, param_space, X, y)


class TestModelPersistence:
    """Tests for ModelPersistence class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def persistence(self, temp_dir):
        """Create test persistence instance."""
        return ModelPersistence(models_dir=temp_dir)
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model."""
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model.fit(X, y)
        return model
    
    def test_persistence_initialization(self, persistence, temp_dir):
        """Test persistence initialization."""
        assert persistence.models_dir == Path(temp_dir)
        assert persistence.models_dir.exists()
    
    def test_save_model(self, persistence, sample_model):
        """Test model saving."""
        metadata = {
            'model_type': 'RandomForest',
            'accuracy': 0.85,
            'training_date': datetime.now()
        }
        
        model_path = persistence.save_model(sample_model, 'test_model', metadata)
        
        assert Path(model_path).exists()
        assert 'test_model' in model_path
        assert model_path.endswith('.pkl')
    
    def test_load_model(self, persistence, sample_model):
        """Test model loading."""
        # Save model first
        model_path = persistence.save_model(sample_model, 'test_model')
        
        # Load model
        loaded_model = persistence.load_model(model_path)
        
        assert loaded_model is not None
        assert type(loaded_model) == type(sample_model)
        assert loaded_model.n_estimators == sample_model.n_estimators
    
    def test_load_nonexistent_model(self, persistence):
        """Test loading non-existent model."""
        with pytest.raises(ModelTrainingError):
            persistence.load_model('nonexistent_model.pkl')
    
    def test_list_saved_models(self, persistence, sample_model):
        """Test listing saved models."""
        # Initially empty
        models = persistence.list_saved_models()
        assert len(models) == 0
        
        # Save a model
        persistence.save_model(sample_model, 'test_model')
        
        # List models
        models = persistence.list_saved_models()
        assert len(models) == 1
        assert 'test_model' in models[0]['name']
        assert 'path' in models[0]
        assert 'size_mb' in models[0]
        assert 'created' in models[0]
    
    def test_make_json_serializable(self, persistence):
        """Test JSON serialization helper."""
        data = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3]),
            'datetime': datetime(2024, 1, 1),
            'nested': {
                'list': [np.int32(1), np.float32(2.5)]
            }
        }
        
        serializable = persistence._make_json_serializable(data)
        
        assert isinstance(serializable['numpy_int'], int)
        assert isinstance(serializable['numpy_float'], float)
        assert isinstance(serializable['numpy_array'], list)
        assert isinstance(serializable['datetime'], str)
        assert isinstance(serializable['nested']['list'][0], int)
        assert isinstance(serializable['nested']['list'][1], float)


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create test metrics calculator."""
        return MetricsCalculator()
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = np.array([1.1, 2.2, 2.8, 4.5, 4.9, 6.2, 7.1, 8.3, 9.2, 9.8])
        return y_true, y_pred
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initialization."""
        assert len(calculator.supported_metrics) > 0
        assert 'mae' in calculator.supported_metrics
        assert 'spearman_correlation' in calculator.supported_metrics
    
    def test_mean_absolute_error(self, calculator, sample_predictions):
        """Test MAE calculation."""
        y_true, y_pred = sample_predictions
        mae = calculator.mean_absolute_error(y_true, y_pred)
        
        assert isinstance(mae, float)
        assert mae >= 0
        # Should be around 0.3 for our sample data
        assert 0.1 <= mae <= 0.5
    
    def test_spearman_correlation(self, calculator, sample_predictions):
        """Test Spearman correlation calculation."""
        y_true, y_pred = sample_predictions
        correlation = calculator.spearman_correlation(y_true, y_pred)
        
        assert isinstance(correlation, float)
        assert -1 <= correlation <= 1
        # Should be high positive correlation for our sample data
        assert correlation > 0.9
    
    def test_top_k_accuracy(self, calculator, sample_predictions):
        """Test Top-K accuracy calculation."""
        y_true, y_pred = sample_predictions
        
        # Test different K values
        top_1_acc = calculator.top_k_accuracy(y_true, y_pred, k=1)
        top_3_acc = calculator.top_k_accuracy(y_true, y_pred, k=3)
        
        assert isinstance(top_1_acc, float)
        assert isinstance(top_3_acc, float)
        assert 0 <= top_1_acc <= 100
        assert 0 <= top_3_acc <= 100
        assert top_3_acc >= top_1_acc  # Top-3 should be >= Top-1
    
    def test_position_accuracy(self, calculator):
        """Test exact position accuracy calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 4, 4, 5])  # 4 out of 5 correct
        
        accuracy = calculator.position_accuracy(y_true, y_pred)
        
        assert isinstance(accuracy, float)
        assert accuracy == 80.0  # 4/5 * 100
    
    def test_calculate_all_metrics(self, calculator, sample_predictions):
        """Test calculation of all metrics."""
        y_true, y_pred = sample_predictions
        
        metrics = calculator.calculate_all_metrics(y_true, y_pred, k_values=[3, 5])
        
        assert isinstance(metrics, dict)
        assert 'mae' in metrics
        assert 'spearman_correlation' in metrics
        assert 'top_3_accuracy' in metrics
        assert 'top_5_accuracy' in metrics
        assert 'position_accuracy' in metrics
        assert 'rmse' in metrics
        assert 'median_absolute_error' in metrics
    
    def test_calculate_position_distribution_metrics(self, calculator, sample_predictions):
        """Test position distribution metrics."""
        y_true, y_pred = sample_predictions
        
        metrics = calculator.calculate_position_distribution_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'mean_error' in metrics
        assert 'std_error' in metrics
        assert 'error_skewness' in metrics
        
        # Should have position-specific MAE metrics
        position_metrics = [k for k in metrics.keys() if 'mae_positions' in k]
        assert len(position_metrics) > 0
    
    def test_metrics_with_empty_data(self, calculator):
        """Test metrics calculation with empty data."""
        y_true = np.array([])
        y_pred = np.array([])
        
        # Should handle empty data gracefully
        top_k_acc = calculator.top_k_accuracy(y_true, y_pred, k=3)
        position_acc = calculator.position_accuracy(y_true, y_pred)
        
        assert top_k_acc == 0.0
        assert position_acc == 0.0


class TestBaseModelTrainer:
    """Tests for BaseModelTrainer class."""
    
    @pytest.fixture
    def trainer(self):
        """Create test trainer."""
        return BaseModelTrainer(random_state=42)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        features = []
        for i in range(20):
            features.append({
                'feature_1': np.random.rand(),
                'feature_2': np.random.rand(),
                'feature_3': np.random.rand()
            })
        return features
    
    @pytest.fixture
    def sample_targets(self):
        """Create sample target data."""
        return list(range(1, 21))  # Positions 1-20
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.random_state == 42
        assert trainer.validator is not None
        assert trainer.tuner is not None
        assert trainer.persistence is not None
        assert trainer.metrics_calculator is not None
        assert isinstance(trainer.config, dict)
    
    def test_prepare_training_data(self, trainer, sample_features, sample_targets):
        """Test training data preparation."""
        X, y, dates_array = trainer.prepare_training_data(sample_features, sample_targets)
        
        assert X.shape[0] == len(sample_features)
        assert X.shape[1] == 3  # 3 features
        assert len(y) == len(sample_targets)
        assert dates_array is None  # No dates provided
        assert X.dtype == np.float32
        assert y.dtype == np.float32
    
    def test_prepare_training_data_with_dates(self, trainer, sample_features, sample_targets):
        """Test training data preparation with dates."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)]
        
        X, y, dates_array = trainer.prepare_training_data(sample_features, sample_targets, dates)
        
        assert dates_array is not None
        assert len(dates_array) == len(dates)
    
    def test_prepare_training_data_mismatched_lengths(self, trainer, sample_features):
        """Test training data preparation with mismatched lengths."""
        targets = [1, 2, 3]  # Different length than features
        
        with pytest.raises(ModelTrainingError):
            trainer.prepare_training_data(sample_features, targets)
    
    def test_split_temporal_data(self, trainer):
        """Test temporal data splitting."""
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        X_train, X_test, y_train, y_test = trainer.split_temporal_data(X, y, test_size=0.2)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_evaluate_model(self, trainer, sample_features, sample_targets):
        """Test model evaluation."""
        # Create a simple mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array(sample_targets, dtype=float)
        
        metrics = trainer.evaluate_model(mock_model, sample_features, sample_targets)
        
        assert isinstance(metrics, dict)
        assert 'mae' in metrics
        assert 'spearman_correlation' in metrics
        assert metrics['mae'] == 0.0  # Perfect predictions
        assert metrics['spearman_correlation'] == 1.0  # Perfect correlation
    
    def test_cross_validate_model(self, trainer):
        """Test cross-validation."""
        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        
        # Create a simple model
        model = LinearRegression()
        
        cv_results = trainer.cross_validate_model(model, X, y, cv_folds=3)
        
        assert isinstance(cv_results, dict)
        assert 'cv_mae_mean' in cv_results
        assert 'cv_mae_std' in cv_results
        assert 'cv_folds' in cv_results
        assert cv_results['cv_folds'] == 3
    
    def test_get_feature_importance_tree_model(self, trainer):
        """Test feature importance extraction from tree-based model."""
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        model.fit(X, y)
        
        feature_names = ['feature_1', 'feature_2', 'feature_3']
        importance = trainer.get_feature_importance(model, feature_names)
        
        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert all(name in importance for name in feature_names)
        assert all(isinstance(score, (int, float)) for score in importance.values())
    
    def test_get_feature_importance_linear_model(self, trainer):
        """Test feature importance extraction from linear model."""
        model = LinearRegression()
        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        model.fit(X, y)
        
        importance = trainer.get_feature_importance(model)
        
        assert isinstance(importance, dict)
        assert len(importance) == 3
    
    def test_get_feature_importance_unsupported_model(self, trainer):
        """Test feature importance extraction from unsupported model."""
        model = Mock()  # Mock model without feature_importances_ or coef_
        
        importance = trainer.get_feature_importance(model)
        
        assert isinstance(importance, dict)
        assert len(importance) == 0
    
    @patch('src.models.training.BaseModelTrainer.persistence')
    def test_save_training_results(self, mock_persistence, trainer):
        """Test saving training results."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'TestModel'
        
        metrics = {'mae': 0.5, 'spearman_correlation': 0.8}
        
        mock_persistence.save_model.return_value = '/path/to/model.pkl'
        
        model_path = trainer.save_training_results(mock_model, metrics, 'test_model')
        
        assert model_path == '/path/to/model.pkl'
        mock_persistence.save_model.assert_called_once()
        
        # Check that metadata was passed correctly
        call_args = mock_persistence.save_model.call_args
        metadata = call_args[0][2]  # Third argument is metadata
        
        assert 'model_type' in metadata
        assert 'training_date' in metadata
        assert 'metrics' in metadata
        assert metadata['metrics'] == metrics