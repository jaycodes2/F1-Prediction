"""
Tests for advanced feature engineering.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from src.features.advanced_features import (
    AdvancedFeatureEngineer, TrackSpecificFeatures, WeatherFeatures,
    AdvancedFeatureEngineeringError
)
from src.models.data_models import (
    RaceData, RaceResult, QualifyingResult, 
    WeatherData, LapTime, RaceStatus
)


class TestTrackSpecificFeatures:
    """Tests for TrackSpecificFeatures class."""
    
    @pytest.fixture
    def track_features(self):
        """Create track features instance."""
        return TrackSpecificFeatures()
    
    def test_get_track_features_known_circuit(self, track_features):
        """Test getting features for a known circuit."""
        monaco_features = track_features.get_track_features('monaco')
        
        assert 'overtaking_difficulty' in monaco_features
        assert 'street_circuit' in monaco_features
        assert 'safety_car_probability' in monaco_features
        
        # Monaco should have high overtaking difficulty
        assert monaco_features['overtaking_difficulty'] > 0.8
        assert monaco_features['street_circuit'] == 1.0
    
    def test_get_track_features_unknown_circuit(self, track_features):
        """Test getting features for an unknown circuit."""
        unknown_features = track_features.get_track_features('unknown_circuit')
        
        # Should return default characteristics
        assert unknown_features == track_features.default_characteristics
    
    def test_calculate_track_suitability(self, track_features):
        """Test track suitability calculation."""
        driver_characteristics = {
            'overtaking_skill': 0.8,
            'wet_weather_skill': 0.9,
            'consistency': 0.7
        }
        
        # Monaco should be less suitable for overtaking-focused drivers
        monaco_suitability = track_features.calculate_track_suitability('monaco', driver_characteristics)
        
        # Monza should be more suitable for overtaking-focused drivers
        monza_suitability = track_features.calculate_track_suitability('monza', driver_characteristics)
        
        assert 0 <= monaco_suitability <= 1
        assert 0 <= monza_suitability <= 1
        # Monza should be more suitable for this driver profile
        assert monza_suitability > monaco_suitability
    
    def test_calculate_track_suitability_empty_characteristics(self, track_features):
        """Test track suitability with empty driver characteristics."""
        suitability = track_features.calculate_track_suitability('monaco', {})
        assert suitability == 0.5  # Default neutral suitability


class TestWeatherFeatures:
    """Tests for WeatherFeatures class."""
    
    @pytest.fixture
    def weather_features(self):
        """Create weather features instance."""
        return WeatherFeatures()
    
    @pytest.fixture
    def sample_weather(self):
        """Create sample weather data."""
        return WeatherData(
            temperature=25.0,
            humidity=60.0,
            pressure=1013.25,
            wind_speed=10.0,
            wind_direction=180,
            rainfall=False,
            track_temp=30.0
        )
    
    def test_extract_weather_features_basic(self, weather_features, sample_weather):
        """Test basic weather feature extraction."""
        features = weather_features.extract_weather_features(sample_weather)
        
        # Check basic features
        assert features['temperature'] == 25.0
        assert features['humidity'] == 60.0
        assert features['rainfall'] == 0.0
        assert features['weather_dry'] == 1.0
        assert features['weather_wet'] == 0.0
        
        # Check derived features
        assert 'temp_track_diff' in features
        assert 'temp_optimal' in features
        assert 'weather_extreme' in features
        
        assert features['temp_track_diff'] == 5.0  # 30 - 25
    
    def test_extract_weather_features_with_rain(self, weather_features):
        """Test weather feature extraction with rain."""
        rainy_weather = WeatherData(
            temperature=15.0,
            humidity=90.0,
            pressure=1005.0,
            wind_speed=20.0,
            wind_direction=270,
            rainfall=True,
            track_temp=18.0
        )
        
        features = weather_features.extract_weather_features(rainy_weather)
        
        assert features['rainfall'] == 1.0
        assert features['weather_dry'] == 0.0
        assert features['weather_wet'] == 1.0
        assert features['humidity_high'] == 1.0
        assert features['wind_strong'] == 1.0
        assert features['weather_extreme'] > 0.5  # Should be considered extreme
    
    def test_extract_weather_features_with_historical(self, weather_features, sample_weather):
        """Test weather feature extraction with historical comparison."""
        historical_weather = [
            WeatherData(20.0, 50.0, 1010.0, 5.0, 90, False, 25.0),
            WeatherData(22.0, 55.0, 1015.0, 8.0, 180, False, 27.0),
            WeatherData(24.0, 65.0, 1012.0, 12.0, 270, False, 29.0)
        ]
        
        features = weather_features.extract_weather_features(sample_weather, historical_weather)
        
        # Should include comparison features
        assert 'temp_vs_avg' in features
        assert 'humidity_vs_avg' in features
        assert 'wind_vs_avg' in features
        
        # Current temp (25) vs historical avg (22) should be positive
        assert features['temp_vs_avg'] > 0
    
    def test_calculate_temperature_optimality(self, weather_features):
        """Test temperature optimality calculation."""
        # Optimal temperature
        assert weather_features._calculate_temperature_optimality(25.0) == 1.0
        
        # Sub-optimal temperatures
        assert weather_features._calculate_temperature_optimality(10.0) < 1.0
        assert weather_features._calculate_temperature_optimality(40.0) < 1.0
        
        # Extreme temperatures
        assert weather_features._calculate_temperature_optimality(-10.0) == 0.0
        assert weather_features._calculate_temperature_optimality(60.0) == 0.0
    
    def test_calculate_weather_extremity(self, weather_features):
        """Test weather extremity calculation."""
        # Normal weather
        normal_weather = WeatherData(25.0, 50.0, 1013.0, 5.0, 180, False, 30.0)
        normal_extremity = weather_features._calculate_weather_extremity(normal_weather)
        
        # Extreme weather
        extreme_weather = WeatherData(5.0, 95.0, 950.0, 40.0, 180, True, 10.0)
        extreme_extremity = weather_features._calculate_weather_extremity(extreme_weather)
        
        assert 0 <= normal_extremity <= 1
        assert 0 <= extreme_extremity <= 1
        assert extreme_extremity > normal_extremity


class TestAdvancedFeatureEngineer:
    """Tests for AdvancedFeatureEngineer class."""
    
    @pytest.fixture
    def engineer(self):
        """Create advanced feature engineer instance."""
        return AdvancedFeatureEngineer()
    
    @pytest.fixture
    def sample_race_series(self):
        """Create a series of races for testing."""
        races = []
        base_date = datetime(2024, 3, 1)
        
        for i in range(5):
            weather = WeatherData(25.0 + i, 60.0, 1013.25, 5.0, 180, False, 30.0 + i)
            
            results = [
                RaceResult(
                    driver_id='hamilton',
                    constructor_id='mercedes',
                    grid_position=i + 1,
                    final_position=i + 1,
                    points=max(25 - i * 3, 0),
                    fastest_lap=None,
                    status=RaceStatus.FINISHED,
                    laps_completed=57
                ),
                RaceResult(
                    driver_id='verstappen',
                    constructor_id='red_bull',
                    grid_position=i + 2,
                    final_position=i + 2,
                    points=max(18 - i * 2, 0),
                    fastest_lap=None,
                    status=RaceStatus.FINISHED,
                    laps_completed=57
                )
            ]
            
            qualifying = [
                QualifyingResult(
                    driver_id='hamilton',
                    constructor_id='mercedes',
                    position=i + 1,
                    q1_time=LapTime(91000, 1),
                    q2_time=LapTime(90500, 1),
                    q3_time=LapTime(90000, 1)
                ),
                QualifyingResult(
                    driver_id='verstappen',
                    constructor_id='red_bull',
                    position=i + 2,
                    q1_time=LapTime(91100, 1),
                    q2_time=LapTime(90600, 1),
                    q3_time=LapTime(90100, 1)
                )
            ]
            
            race = RaceData(
                season=2024,
                round=i + 1,
                circuit_id='silverstone' if i == 0 else f'circuit_{i}',
                race_name=f'Race {i + 1}',
                date=base_date + timedelta(weeks=i * 2),
                results=results,
                qualifying=qualifying,
                weather=weather
            )
            races.append(race)
        
        return races
    
    def test_engineer_initialization(self, engineer):
        """Test engineer initialization."""
        assert engineer.base_extractor is not None
        assert engineer.rolling_calculator is not None
        assert engineer.track_features is not None
        assert engineer.weather_features is not None
        assert isinstance(engineer.config, dict)
    
    def test_engineer_race_features_success(self, engineer, sample_race_series):
        """Test successful race feature engineering."""
        target_race = sample_race_series[-1]  # Last race
        historical_races = sample_race_series[:-1]  # All but last
        
        features = engineer.engineer_race_features(target_race, historical_races)
        
        assert isinstance(features, dict)
        assert 'hamilton' in features
        assert 'verstappen' in features
        
        # Check Hamilton's features
        hamilton_features = features['hamilton']
        assert isinstance(hamilton_features, dict)
        assert len(hamilton_features) > 10  # Should have many features
        
        # Check for different feature categories
        feature_names = list(hamilton_features.keys())
        
        # Should have rolling statistics features
        rolling_features = [f for f in feature_names if 'rolling' in f]
        assert len(rolling_features) > 0
        
        # Should have track features
        track_features = [f for f in feature_names if 'track' in f]
        assert len(track_features) > 0
        
        # Should have weather features
        weather_features = [f for f in feature_names if 'weather' in f]
        assert len(weather_features) > 0
    
    def test_engineer_race_features_no_historical_data(self, engineer, sample_race_series):
        """Test feature engineering with no historical data."""
        target_race = sample_race_series[0]
        
        with pytest.raises(AdvancedFeatureEngineeringError):
            engineer.engineer_race_features(target_race, [])
    
    def test_engineer_race_features_specific_drivers(self, engineer, sample_race_series):
        """Test feature engineering for specific drivers only."""
        target_race = sample_race_series[-1]
        historical_races = sample_race_series[:-1]
        
        features = engineer.engineer_race_features(
            target_race, historical_races, include_target_drivers=['hamilton']
        )
        
        assert 'hamilton' in features
        assert 'verstappen' not in features
        assert len(features) == 1
    
    def test_engineer_comparative_features(self, engineer, sample_race_series):
        """Test comparative feature engineering."""
        target_race = sample_race_series[-1]
        historical_races = sample_race_series[:-1]
        
        comparative_features = engineer.engineer_comparative_features(target_race, historical_races)
        
        assert isinstance(comparative_features, dict)
        
        # Should have field-level features
        assert 'field_size' in comparative_features or len(comparative_features) == 0  # Might be empty if insufficient data
    
    def test_get_feature_importance_weights(self, engineer):
        """Test feature importance weights."""
        weights = engineer.get_feature_importance_weights()
        
        assert isinstance(weights, dict)
        assert len(weights) > 0
        
        # Check that weights sum to approximately 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Check for expected categories
        assert 'recent_form' in weights
        assert 'rolling_stats' in weights
        assert 'track_specific' in weights
    
    def test_validate_engineered_features(self, engineer):
        """Test feature validation."""
        # Create sample features
        sample_features = {
            'hamilton': {
                'recent_form': 0.8,
                'rolling_3_avg_position': 2.5,
                'track_overtaking_difficulty': 0.6,
                'weather_temperature': 25.0
            },
            'verstappen': {
                'recent_form': 0.9,
                'rolling_3_avg_position': 1.8,
                'track_overtaking_difficulty': 0.6,
                # Missing weather_temperature
            }
        }
        
        validation_report = engineer.validate_engineered_features(sample_features)
        
        assert 'total_drivers' in validation_report
        assert 'total_features' in validation_report
        assert 'quality_score' in validation_report
        assert 'missing_features' in validation_report
        
        assert validation_report['total_drivers'] == 2
        assert validation_report['total_features'] == 4
        assert 0 <= validation_report['quality_score'] <= 1
        
        # weather_temperature should be in missing features (only 50% coverage)
        assert 'weather_temperature' in validation_report['missing_features']
    
    def test_validate_engineered_features_empty(self, engineer):
        """Test feature validation with empty features."""
        validation_report = engineer.validate_engineered_features({})
        
        assert validation_report['total_drivers'] == 0
        assert validation_report['quality_score'] == 0.0
    
    def test_engineer_driver_features_integration(self, engineer, sample_race_series):
        """Test that driver feature engineering integrates all components."""
        target_race = sample_race_series[-1]
        historical_races = sample_race_series[:-1]
        
        # Test the internal method directly
        hamilton_features = engineer._engineer_driver_features('hamilton', target_race, historical_races)
        
        assert isinstance(hamilton_features, dict)
        assert len(hamilton_features) > 20  # Should have many features from different sources
        
        # Check for features from different components
        feature_names = list(hamilton_features.keys())
        
        # Base features
        base_features = ['recent_form', 'constructor_performance', 'track_experience']
        assert any(f in feature_names for f in base_features)
        
        # Rolling features
        rolling_features = [f for f in feature_names if 'rolling' in f]
        assert len(rolling_features) > 0
        
        # Track features
        track_features = [f for f in feature_names if 'track' in f]
        assert len(track_features) > 0
        
        # Weather features
        weather_features = [f for f in feature_names if 'weather' in f]
        assert len(weather_features) > 0
        
        # Grid/qualifying features
        grid_features = [f for f in feature_names if 'grid' in f]
        assert len(grid_features) > 0