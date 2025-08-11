"""
Tests for base feature extractor.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from src.features.base_extractor import BaseFeatureExtractor, FeatureExtractionError
from src.models.data_models import (
    RaceData, RaceResult, QualifyingResult, 
    WeatherData, LapTime, RaceStatus, DriverFeatures
)


class TestBaseFeatureExtractor:
    """Tests for the BaseFeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a test feature extractor."""
        return BaseFeatureExtractor()
    
    @pytest.fixture
    def sample_race_data(self):
        """Create sample race data for testing."""
        weather = WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        
        results = []
        qualifying = []
        
        # Create sample drivers with varying performance
        drivers_data = [
            ('hamilton', 'mercedes', 1, 1, 25.0),
            ('verstappen', 'red_bull', 2, 2, 18.0),
            ('leclerc', 'ferrari', 3, 3, 15.0),
            ('russell', 'mercedes', 4, 4, 12.0),
            ('sainz', 'ferrari', 5, 5, 10.0)
        ]
        
        for i, (driver_id, constructor_id, grid, position, points) in enumerate(drivers_data):
            race_result = RaceResult(
                driver_id=driver_id,
                constructor_id=constructor_id,
                grid_position=grid,
                final_position=position,
                points=points,
                fastest_lap=LapTime(90000 + i * 100, 45) if i == 0 else None,
                status=RaceStatus.FINISHED,
                laps_completed=57
            )
            results.append(race_result)
            
            qual_result = QualifyingResult(
                driver_id=driver_id,
                constructor_id=constructor_id,
                position=grid,
                q1_time=LapTime(91000 + i * 100, 1),
                q2_time=LapTime(90500 + i * 100, 1),
                q3_time=LapTime(90000 + i * 100, 1)
            )
            qualifying.append(qual_result)
        
        return RaceData(
            season=2024,
            round=1,
            circuit_id='bahrain',
            race_name='Bahrain Grand Prix',
            date=datetime(2024, 3, 2, 15, 0, 0),
            results=results,
            qualifying=qualifying,
            weather=weather
        )
    
    @pytest.fixture
    def historical_race_data(self, sample_race_data):
        """Create historical race data for testing."""
        races = []
        
        # Create 5 races with varying results for Hamilton
        for i in range(5):
            race = RaceData(
                season=2024,
                round=i + 1,
                circuit_id=f'circuit_{i}',
                race_name=f'Race {i + 1}',
                date=datetime(2024, 2, 1) + timedelta(weeks=i),  # Earlier dates
                results=[
                    RaceResult(
                        driver_id='hamilton',
                        constructor_id='mercedes',
                        grid_position=i + 1,
                        final_position=min(i + 1, 5),  # Varying performance
                        points=max(25 - i * 3, 0),
                        fastest_lap=None,
                        status=RaceStatus.FINISHED,
                        laps_completed=57
                    ),
                    RaceResult(
                        driver_id='verstappen',
                        constructor_id='red_bull',
                        grid_position=i + 2,
                        final_position=min(i + 2, 6),
                        points=max(18 - i * 2, 0),
                        fastest_lap=None,
                        status=RaceStatus.FINISHED,
                        laps_completed=57
                    ),
                    RaceResult(
                        driver_id='leclerc',
                        constructor_id='ferrari',
                        grid_position=i + 3,
                        final_position=min(i + 3, 7),
                        points=max(15 - i * 2, 0),
                        fastest_lap=None,
                        status=RaceStatus.FINISHED,
                        laps_completed=57
                    ),
                    RaceResult(
                        driver_id='russell',
                        constructor_id='mercedes',
                        grid_position=i + 4,
                        final_position=min(i + 4, 8),
                        points=max(12 - i * 2, 0),
                        fastest_lap=None,
                        status=RaceStatus.FINISHED,
                        laps_completed=57
                    ),
                    RaceResult(
                        driver_id='sainz',
                        constructor_id='ferrari',
                        grid_position=i + 5,
                        final_position=min(i + 5, 9),
                        points=max(10 - i * 2, 0),
                        fastest_lap=None,
                        status=RaceStatus.FINISHED,
                        laps_completed=57
                    )
                ],
                qualifying=[
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
                    ),
                    QualifyingResult(
                        driver_id='leclerc',
                        constructor_id='ferrari',
                        position=i + 3,
                        q1_time=LapTime(91200, 1),
                        q2_time=LapTime(90700, 1),
                        q3_time=LapTime(90200, 1)
                    ),
                    QualifyingResult(
                        driver_id='russell',
                        constructor_id='mercedes',
                        position=i + 4,
                        q1_time=LapTime(91300, 1),
                        q2_time=LapTime(90800, 1),
                        q3_time=LapTime(90300, 1)
                    ),
                    QualifyingResult(
                        driver_id='sainz',
                        constructor_id='ferrari',
                        position=i + 5,
                        q1_time=LapTime(91400, 1),
                        q2_time=LapTime(90900, 1),
                        q3_time=LapTime(90400, 1)
                    )
                ],
                weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
            )
            races.append(race)
        
        return races
    
    def test_extractor_initialization(self, extractor):
        """Test feature extractor initialization."""
        assert extractor.min_races_for_features == 3
        assert len(extractor.driver_features) > 0
        assert len(extractor.constructor_features) > 0
        assert len(extractor.qualifying_features) > 0
    
    def test_extract_driver_features_success(self, extractor, historical_race_data):
        """Test successful driver feature extraction."""
        features = extractor.extract_driver_features(historical_race_data, 'hamilton')
        
        assert isinstance(features, DriverFeatures)
        assert features.driver_id == 'hamilton'
        assert 0 <= features.recent_form <= 1
        assert 0 <= features.constructor_performance <= 1
        assert features.track_experience == 5  # 5 races
        assert -1 <= features.qualifying_delta <= 1
        assert features.championship_position > 0
        assert features.points_total >= 0
    
    def test_extract_driver_features_no_data(self, extractor):
        """Test driver feature extraction with no data."""
        with pytest.raises(FeatureExtractionError):
            extractor.extract_driver_features([], 'hamilton')
    
    def test_extract_driver_features_driver_not_found(self, extractor, historical_race_data):
        """Test driver feature extraction when driver not found."""
        with pytest.raises(FeatureExtractionError):
            extractor.extract_driver_features(historical_race_data, 'nonexistent_driver')
    
    def test_extract_base_driver_features(self, extractor, historical_race_data):
        """Test base driver feature extraction."""
        # Find Hamilton's results
        driver_results = []
        driver_qualifying = []
        
        for race in historical_race_data:
            for result in race.results:
                if result.driver_id == 'hamilton':
                    driver_results.append((race, result))
                    break
            
            for qual in race.qualifying:
                if qual.driver_id == 'hamilton':
                    driver_qualifying.append((race, qual))
                    break
        
        features = extractor._extract_base_driver_features(driver_results, driver_qualifying)
        
        assert 'avg_finish_position' in features
        assert 'avg_grid_position' in features
        assert 'points_total' in features
        assert 'finish_rate' in features
        assert 'podium_rate' in features
        assert 'dnf_rate' in features
        assert features['finish_rate'] == 1.0  # All races finished
        assert features['points_total'] > 0
    
    def test_calculate_recent_form(self, extractor, historical_race_data):
        """Test recent form calculation."""
        # Find Hamilton's results
        driver_results = []
        for race in historical_race_data:
            for result in race.results:
                if result.driver_id == 'hamilton':
                    driver_results.append((race, result))
                    break
        
        form = extractor._calculate_recent_form(driver_results, window=3)
        
        assert 0 <= form <= 1
        assert isinstance(form, float)
    
    def test_calculate_recent_form_empty(self, extractor):
        """Test recent form calculation with empty data."""
        form = extractor._calculate_recent_form([])
        assert form == 0.5  # Neutral form
    
    def test_calculate_constructor_performance(self, extractor, historical_race_data):
        """Test constructor performance calculation."""
        # Find Hamilton's results
        driver_results = []
        for race in historical_race_data:
            for result in race.results:
                if result.driver_id == 'hamilton':
                    driver_results.append((race, result))
                    break
        
        performance = extractor._calculate_constructor_performance(driver_results)
        
        assert 0 <= performance <= 1
        assert isinstance(performance, float)
    
    def test_calculate_qualifying_delta(self, extractor, historical_race_data):
        """Test qualifying delta calculation."""
        # Find Hamilton's qualifying
        driver_qualifying = []
        for race in historical_race_data:
            for qual in race.qualifying:
                if qual.driver_id == 'hamilton':
                    driver_qualifying.append((race, qual))
                    break
        
        delta = extractor._calculate_qualifying_delta(driver_qualifying)
        
        assert -1 <= delta <= 1
        assert isinstance(delta, float)
    
    def test_estimate_championship_position(self, extractor):
        """Test championship position estimation."""
        # Test various point totals
        assert extractor._estimate_championship_position(350) == 1
        assert extractor._estimate_championship_position(250) == 2
        assert extractor._estimate_championship_position(150) == 4
        assert extractor._estimate_championship_position(50) == 8
        assert extractor._estimate_championship_position(5) <= 20
    
    def test_calculate_rolling_stats(self, extractor, historical_race_data):
        """Test rolling statistics calculation."""
        rolling_stats = extractor.calculate_rolling_stats(historical_race_data, window=3)
        
        assert 'driver_form' in rolling_stats
        assert 'constructor_performance' in rolling_stats
        assert 'hamilton' in rolling_stats['driver_form']
        assert 'verstappen' in rolling_stats['driver_form']
        assert 'mercedes' in rolling_stats['constructor_performance']
        assert 'red_bull' in rolling_stats['constructor_performance']
        
        # Check structure of rolling stats
        hamilton_form = rolling_stats['driver_form']['hamilton']
        assert len(hamilton_form) == 5  # 5 races
        assert all('form' in entry for entry in hamilton_form)
        assert all('race_date' in entry for entry in hamilton_form)
    
    def test_calculate_rolling_stats_empty(self, extractor):
        """Test rolling statistics with empty data."""
        rolling_stats = extractor.calculate_rolling_stats([], window=3)
        assert rolling_stats == {}
    
    def test_encode_categorical_features(self, extractor):
        """Test categorical feature encoding."""
        features = {
            'constructor_id': 'mercedes',
            'circuit_id': 'bahrain',
            'weather_condition': 'dry',
            'numeric_feature': 0.5
        }
        
        encoded = extractor.encode_categorical_features(features)
        
        # Check that categorical features are one-hot encoded
        assert 'constructor_id_mercedes' in encoded
        assert encoded['constructor_id_mercedes'] == 1.0
        assert 'constructor_id_ferrari' in encoded
        assert encoded['constructor_id_ferrari'] == 0.0
        
        assert 'circuit_id_bahrain' in encoded
        assert encoded['circuit_id_bahrain'] == 1.0
        
        assert 'weather_condition_dry' in encoded
        assert encoded['weather_condition_dry'] == 1.0
        assert 'weather_condition_wet' in encoded
        assert encoded['weather_condition_wet'] == 0.0
        
        # Check that numeric features are preserved
        assert encoded['numeric_feature'] == 0.5
        
        # Check that original categorical features are removed
        assert 'constructor_id' not in encoded
        assert 'circuit_id' not in encoded
        assert 'weather_condition' not in encoded
    
    def test_extract_race_features(self, extractor, sample_race_data, historical_race_data):
        """Test complete race feature extraction."""
        race_features = extractor.extract_race_features(sample_race_data, historical_race_data)
        
        assert 'race_metadata' in race_features
        assert 'weather_features' in race_features
        assert 'grid_features' in race_features
        assert 'driver_features' in race_features
        assert 'constructor_features' in race_features
        
        # Check race metadata
        metadata = race_features['race_metadata']
        assert metadata['season'] == 2024
        assert metadata['round'] == 1
        assert metadata['circuit_id'] == 'bahrain'
        
        # Check weather features
        weather = race_features['weather_features']
        assert 'temperature' in weather
        assert 'humidity' in weather
        assert 'rainfall' in weather
        assert weather['weather_condition'] == 'dry'
        
        # Check driver features
        assert 'hamilton' in race_features['driver_features']
        hamilton_features = race_features['driver_features']['hamilton']
        assert 'grid_position' in hamilton_features
        assert 'constructor_id' in hamilton_features
    
    def test_extract_race_metadata_features(self, extractor, sample_race_data):
        """Test race metadata feature extraction."""
        metadata = extractor._extract_race_metadata_features(sample_race_data)
        
        assert metadata['season'] == 2024
        assert metadata['round'] == 1
        assert metadata['circuit_id'] == 'bahrain'
        assert metadata['month'] == 3
        assert 'day_of_year' in metadata
        assert 'is_weekend' in metadata
    
    def test_extract_weather_features(self, extractor, sample_race_data):
        """Test weather feature extraction."""
        weather_features = extractor._extract_weather_features(sample_race_data)
        
        assert weather_features['temperature'] == 25.0
        assert weather_features['humidity'] == 60.0
        assert weather_features['rainfall'] == 0.0
        assert weather_features['weather_condition'] == 'dry'
        assert 'temp_track_diff' in weather_features
    
    def test_extract_grid_features(self, extractor, sample_race_data):
        """Test grid feature extraction."""
        grid_features = extractor._extract_grid_features(sample_race_data)
        
        assert 'grid_spread' in grid_features
        assert 'avg_grid_position' in grid_features
        assert 'grid_competitiveness' in grid_features
        assert grid_features['grid_spread'] == 4  # Positions 1-5
        assert grid_features['avg_grid_position'] == 3.0  # Average of 1,2,3,4,5
    
    def test_get_encoding_lists(self, extractor):
        """Test encoding list getters."""
        constructors = extractor._get_constructor_encoding()
        circuits = extractor._get_circuit_encoding()
        weather = extractor._get_weather_encoding()
        
        assert isinstance(constructors, list)
        assert isinstance(circuits, list)
        assert isinstance(weather, list)
        
        assert 'mercedes' in constructors
        assert 'ferrari' in constructors
        assert 'bahrain' in circuits
        assert 'monaco' in circuits
        assert 'dry' in weather
        assert 'wet' in weather