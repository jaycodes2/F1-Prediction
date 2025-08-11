"""
Tests for the prediction engine and related services.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.services.prediction_engine import (
    PredictionEngine, ConfidenceCalculator, PredictionRequest, PredictionResult
)
from src.models.data_models import PositionPrediction


class TestConfidenceCalculator:
    """Tests for ConfidenceCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create test confidence calculator."""
        return ConfidenceCalculator()
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.confidence_factors is not None
        assert len(calculator.confidence_factors) == 4
        assert 'model_agreement' in calculator.confidence_factors
        assert 'prediction_variance' in calculator.confidence_factors
    
    def test_single_model_confidence(self, calculator):
        """Test confidence calculation with single model."""
        predictions = [np.array([1, 2, 3, 4, 5])]
        
        confidence = calculator.calculate_prediction_confidence(predictions)
        
        assert isinstance(confidence, dict)
        assert 'overall_confidence' in confidence
        assert 0 <= confidence['overall_confidence'] <= 1
        assert 'model_agreement' in confidence
        assert 'prediction_variance' in confidence
    
    def test_multiple_model_confidence(self, calculator):
        """Test confidence calculation with multiple models."""
        predictions = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1.1, 2.2, 2.9, 4.1, 5.2]),
            np.array([0.9, 1.8, 3.1, 3.9, 4.8])
        ]
        
        confidence = calculator.calculate_prediction_confidence(predictions)
        
        assert isinstance(confidence, dict)
        assert 'overall_confidence' in confidence
        assert 'model_agreement' in confidence
        assert confidence['model_agreement'] > 0.5  # Should have good agreement
    
    def test_confidence_with_accuracies(self, calculator):
        """Test confidence calculation with model accuracies."""
        predictions = [np.array([1, 2, 3, 4, 5])]
        accuracies = [0.85, 0.90, 0.78]
        
        confidence = calculator.calculate_prediction_confidence(
            predictions, model_accuracies=accuracies
        )
        
        assert 'historical_accuracy' in confidence
        assert confidence['historical_accuracy'] == np.mean(accuracies)
    
    def test_position_confidence(self, calculator):
        """Test position-specific confidence calculation."""
        # Very consistent predictions
        consistent_preds = [5.0, 5.1, 4.9, 5.0, 5.2]
        consistent_conf = calculator.calculate_position_confidence(consistent_preds)
        
        # Very inconsistent predictions
        inconsistent_preds = [1.0, 10.0, 3.0, 15.0, 7.0]
        inconsistent_conf = calculator.calculate_position_confidence(inconsistent_preds)
        
        assert consistent_conf > inconsistent_conf
        assert 0 <= consistent_conf <= 1
        assert 0 <= inconsistent_conf <= 1
    
    def test_empty_predictions_confidence(self, calculator):
        """Test confidence calculation with empty predictions."""
        confidence = calculator.calculate_prediction_confidence([])
        assert confidence['overall_confidence'] == 0.0
        
        position_conf = calculator.calculate_position_confidence([])
        assert position_conf == 0.0


class TestPredictionRequest:
    """Tests for PredictionRequest class."""
    
    @pytest.fixture
    def sample_request(self):
        """Create sample prediction request."""
        return PredictionRequest(
            race_name="Monaco Grand Prix",
            circuit="Monaco",
            date=datetime(2024, 5, 26),
            drivers=[
                {
                    'driver_id': 'HAM',
                    'name': 'Lewis Hamilton',
                    'grid_position': 3,
                    'championship_points': 150
                },
                {
                    'driver_id': 'VER',
                    'name': 'Max Verstappen',
                    'grid_position': 1,
                    'championship_points': 200
                }
            ],
            weather={
                'conditions': 'dry',
                'track_temp': 35.0,
                'air_temp': 28.0,
                'humidity': 65.0
            }
        )
    
    def test_request_creation(self, sample_request):
        """Test prediction request creation."""
        assert sample_request.race_name == "Monaco Grand Prix"
        assert sample_request.circuit == "Monaco"
        assert len(sample_request.drivers) == 2
        assert sample_request.weather['conditions'] == 'dry'
    
    def test_request_to_dict(self, sample_request):
        """Test request serialization to dictionary."""
        request_dict = sample_request.to_dict()
        
        assert isinstance(request_dict, dict)
        assert request_dict['race_name'] == "Monaco Grand Prix"
        assert request_dict['circuit'] == "Monaco"
        assert isinstance(request_dict['date'], str)  # Should be ISO format
        assert len(request_dict['drivers']) == 2


class TestPredictionEngine:
    """Tests for PredictionEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create test prediction engine."""
        return PredictionEngine(model_type='random_forest')
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        features = []
        targets = []
        
        for i in range(50):
            features.append({
                'qualifying_position': np.random.randint(1, 21),
                'driver_championship_points': np.random.randint(0, 400),
                'constructor_championship_points': np.random.randint(0, 600),
                'track_temperature': np.random.uniform(15, 45),
                'weather_dry': np.random.choice([0, 1])
            })
            targets.append(np.random.randint(1, 21))
        
        return features, targets
    
    @pytest.fixture
    def sample_request(self):
        """Create sample prediction request."""
        return PredictionRequest(
            race_name="Test Grand Prix",
            circuit="Test Circuit",
            date=datetime.now(),
            drivers=[
                {
                    'driver_id': f'D{i}',
                    'name': f'Driver {i}',
                    'grid_position': i + 1,
                    'championship_points': np.random.randint(0, 200),
                    'constructor_points': np.random.randint(0, 300),
                    'experience_races': np.random.randint(0, 200)
                }
                for i in range(10)
            ],
            weather={
                'conditions': 'dry',
                'track_temp': 30.0,
                'air_temp': 25.0,
                'humidity': 60.0,
                'wind_speed': 5.0,
                'grip_level': 0.9
            }
        )
    
    def test_engine_initialization(self, engine):
        """Test prediction engine initialization."""
        assert engine.model_type == 'random_forest'
        assert engine.model is None
        assert engine.feature_extractor is not None
        assert engine.confidence_calculator is not None
        assert len(engine.prediction_cache) == 0
    
    def test_model_initialization_success(self, engine, sample_training_data):
        """Test successful model initialization."""
        features, targets = sample_training_data
        
        success = engine.initialize_model(features, targets)
        
        assert success is True
        assert engine.model is not None
        assert len(engine.model_accuracies) > 0
    
    def test_model_initialization_without_data(self, engine):
        """Test model initialization without training data."""
        success = engine.initialize_model()
        
        assert success is True
        assert engine.model is not None
    
    def test_ensemble_model_initialization(self):
        """Test ensemble model initialization."""
        engine = PredictionEngine(model_type='ensemble')
        success = engine.initialize_model()
        
        assert success is True
        assert engine.model is not None
    
    def test_predict_race_without_initialization(self, engine, sample_request):
        """Test prediction without model initialization raises error."""
        with pytest.raises(ValueError):
            engine.predict_race(sample_request)
    
    def test_predict_race_success(self, engine, sample_training_data, sample_request):
        """Test successful race prediction."""
        features, targets = sample_training_data
        engine.initialize_model(features, targets)
        
        result = engine.predict_race(sample_request)
        
        assert isinstance(result, PredictionResult)
        assert result.race_name == sample_request.race_name
        assert len(result.predictions) == len(sample_request.drivers)
        assert 0 <= result.confidence_score <= 1
        assert result.generated_at is not None
        
        # Check predictions are sorted by position
        positions = [pred.predicted_position for pred in result.predictions]
        assert positions == sorted(positions)
        
        # Check all positions are valid
        for pred in result.predictions:
            assert 1 <= pred.predicted_position <= 20
            assert 0 <= pred.confidence <= 1
            assert isinstance(pred.probability_distribution, dict)
    
    def test_predict_batch(self, engine, sample_training_data):
        """Test batch prediction."""
        features, targets = sample_training_data
        engine.initialize_model(features, targets)
        
        # Create multiple requests
        requests = []
        for i in range(3):
            request = PredictionRequest(
                race_name=f"Race {i+1}",
                circuit=f"Circuit {i+1}",
                date=datetime.now() + timedelta(days=i),
                drivers=[
                    {
                        'driver_id': f'D{j}',
                        'name': f'Driver {j}',
                        'grid_position': j + 1
                    }
                    for j in range(5)
                ],
                weather={'conditions': 'dry', 'track_temp': 25.0}
            )
            requests.append(request)
        
        results = engine.predict_batch(requests)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.race_name == f"Race {i+1}"
            assert len(result.predictions) == 5
    
    def test_prediction_caching(self, engine, sample_training_data, sample_request):
        """Test prediction caching functionality."""
        features, targets = sample_training_data
        engine.initialize_model(features, targets)
        
        # First prediction
        result1 = engine.predict_race(sample_request)
        assert len(engine.prediction_cache) == 1
        
        # Second prediction (should use cache)
        result2 = engine.predict_race(sample_request)
        assert len(engine.prediction_cache) == 1
        
        # Results should be identical (from cache)
        assert result1.generated_at == result2.generated_at
    
    def test_cache_key_generation(self, engine, sample_request):
        """Test cache key generation."""
        key1 = engine._generate_cache_key(sample_request)
        key2 = engine._generate_cache_key(sample_request)
        
        # Same request should generate same key
        assert key1 == key2
        
        # Different request should generate different key
        sample_request.race_name = "Different Race"
        key3 = engine._generate_cache_key(sample_request)
        assert key1 != key3
    
    def test_feature_extraction(self, engine, sample_request):
        """Test race feature extraction."""
        features = engine._extract_race_features(sample_request)
        
        assert len(features) == len(sample_request.drivers)
        
        for feature_dict in features:
            assert isinstance(feature_dict, dict)
            assert 'qualifying_position' in feature_dict
            assert 'track_temperature' in feature_dict
            assert 'weather_dry' in feature_dict
            
            # Check feature values are reasonable
            assert 1 <= feature_dict['qualifying_position'] <= 20
            assert feature_dict['weather_dry'] in [0, 1]
    
    def test_position_probabilities(self, engine):
        """Test position probability calculation."""
        predicted_pos = 5.0
        probabilities = engine._calculate_position_probabilities(predicted_pos)
        
        assert isinstance(probabilities, dict)
        assert len(probabilities) == 20  # F1 has max 20 positions
        
        # Check probabilities sum to 1
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.01
        
        # Check highest probability is around predicted position
        max_prob_pos = max(probabilities.items(), key=lambda x: x[1])[0]
        assert abs(max_prob_pos - predicted_pos) <= 2
    
    def test_prediction_insights(self, engine, sample_training_data, sample_request):
        """Test prediction insights generation."""
        features, targets = sample_training_data
        engine.initialize_model(features, targets)
        
        result = engine.predict_race(sample_request)
        insights = engine.get_prediction_insights(result)
        
        assert isinstance(insights, dict)
        assert 'race_name' in insights
        assert 'total_drivers' in insights
        assert 'confidence_level' in insights
        assert 'most_likely_winner' in insights
        assert 'confidence_distribution' in insights
        
        # Check confidence level categorization
        assert insights['confidence_level'] in ['high', 'medium', 'low']
        
        # Check most likely winner
        winner = insights['most_likely_winner']
        assert 'driver' in winner
        assert 'confidence' in winner
    
    def test_model_accuracy_update(self, engine):
        """Test model accuracy tracking."""
        actual_results = [1, 2, 3, 4, 5]
        predicted_results = [1, 3, 2, 4, 6]
        
        engine.update_model_accuracy(actual_results, predicted_results)
        
        assert engine.model_type in engine.model_accuracies
        assert 0 <= engine.model_accuracies[engine.model_type] <= 1
    
    def test_model_status(self, engine, sample_training_data):
        """Test model status reporting."""
        # Before initialization
        status = engine.get_model_status()
        assert status['model_initialized'] is False
        assert status['cache_size'] == 0
        
        # After initialization
        features, targets = sample_training_data
        engine.initialize_model(features, targets)
        
        status = engine.get_model_status()
        assert status['model_initialized'] is True
        assert status['model_type'] == 'random_forest'
    
    def test_cache_cleaning(self, engine):
        """Test cache cleaning functionality."""
        # Fill cache with old entries
        old_time = datetime.now() - timedelta(hours=2)
        
        for i in range(5):
            engine.prediction_cache[f'key_{i}'] = {
                'result': Mock(),
                'timestamp': old_time
            }
        
        # Add recent entry
        engine.prediction_cache['recent_key'] = {
            'result': Mock(),
            'timestamp': datetime.now()
        }
        
        # Clean cache
        engine._clean_cache()
        
        # Only recent entry should remain
        assert len(engine.prediction_cache) == 1
        assert 'recent_key' in engine.prediction_cache


class TestPredictionEngineIntegration:
    """Integration tests for prediction engine with real models."""
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        # Create engine
        engine = PredictionEngine(model_type='random_forest')
        
        # Create training data
        training_features = []
        training_targets = []
        
        for i in range(60):
            training_features.append({
                'qualifying_position': np.random.randint(1, 21),
                'driver_championship_points': np.random.randint(0, 400),
                'constructor_championship_points': np.random.randint(0, 600),
                'track_temperature': np.random.uniform(15, 45),
                'weather_dry': np.random.choice([0, 1]),
                'driver_experience': np.random.randint(0, 300),
                'car_performance_rating': np.random.uniform(0.5, 1.0)
            })
            training_targets.append(np.random.randint(1, 21))
        
        # Initialize and train
        success = engine.initialize_model(training_features, training_targets)
        assert success is True
        
        # Create prediction request
        request = PredictionRequest(
            race_name="Integration Test GP",
            circuit="Test Circuit",
            date=datetime.now(),
            drivers=[
                {
                    'driver_id': f'DRIVER_{i}',
                    'name': f'Test Driver {i}',
                    'grid_position': i + 1,
                    'championship_points': np.random.randint(0, 200),
                    'constructor_points': np.random.randint(0, 300),
                    'wins_season': np.random.randint(0, 5),
                    'constructor_wins': np.random.randint(0, 8),
                    'experience_races': np.random.randint(0, 200),
                    'car_rating': np.random.uniform(0.5, 1.0),
                    'fuel_load': np.random.uniform(50, 110),
                    'tire_compound': np.random.randint(1, 4),
                    'engine_power': np.random.uniform(800, 1000),
                    'aero_efficiency': np.random.uniform(0.6, 1.0)
                }
                for i in range(15)
            ],
            weather={
                'conditions': 'dry',
                'track_temp': 32.0,
                'air_temp': 26.0,
                'humidity': 55.0,
                'wind_speed': 8.0,
                'grip_level': 0.85
            }
        )
        
        # Make prediction
        result = engine.predict_race(request)
        
        # Verify result
        assert isinstance(result, PredictionResult)
        assert result.race_name == "Integration Test GP"
        assert len(result.predictions) == 15
        assert 0 <= result.confidence_score <= 1
        
        # Verify predictions are valid
        for pred in result.predictions:
            assert 1 <= pred.predicted_position <= 20
            assert 0 <= pred.confidence <= 1
            assert len(pred.probability_distribution) == 20
        
        # Generate insights
        insights = engine.get_prediction_insights(result)
        assert 'most_likely_winner' in insights
        assert 'confidence_level' in insights
        
        # Test batch prediction
        batch_requests = [request] * 3
        batch_results = engine.predict_batch(batch_requests)
        assert len(batch_results) == 3
        
        # Update accuracy with mock results
        actual_positions = list(range(1, 16))
        predicted_positions = [pred.predicted_position for pred in result.predictions]
        engine.update_model_accuracy(actual_positions, predicted_positions)
        
        # Check model status
        status = engine.get_model_status()
        assert status['model_initialized'] is True
        assert len(status['model_accuracies']) > 0