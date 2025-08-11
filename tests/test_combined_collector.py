"""
Tests for combined data collector.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.data.combined_collector import CombinedDataCollector
from src.data.ergast_client import ErgastAPIError
from src.data.fastf1_client import FastF1Error
from src.models.data_models import (
    RaceData, RaceResult, QualifyingResult, 
    WeatherData, LapTime, RaceStatus
)


class TestCombinedDataCollector:
    """Tests for the CombinedDataCollector class."""
    
    @pytest.fixture
    def collector(self):
        """Create a test collector."""
        with patch('src.data.combined_collector.ErgastAPIClient'):
            with patch('src.data.combined_collector.FastF1Client'):
                return CombinedDataCollector()
    
    @pytest.fixture
    def sample_race_data(self):
        """Create sample race data."""
        weather = WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        
        race_result = RaceResult(
            driver_id='hamilton',
            constructor_id='mercedes',
            grid_position=1,
            final_position=1,
            points=25.0,
            fastest_lap=None,
            status=RaceStatus.FINISHED,
            laps_completed=57
        )
        
        qual_result = QualifyingResult(
            driver_id='hamilton',
            constructor_id='mercedes',
            position=1,
            q1_time=None,
            q2_time=None,
            q3_time=None
        )
        
        return RaceData(
            season=2024,
            round=1,
            circuit_id='bahrain',
            race_name='Bahrain Grand Prix',
            date=datetime(2024, 3, 2),
            results=[race_result],
            qualifying=[qual_result],
            weather=weather
        )
    
    def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert hasattr(collector, 'ergast_client')
        assert hasattr(collector, 'fastf1_client')
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_collect_race_data_success(self, collector, sample_race_data):
        """Test successful race data collection with both sources."""
        # Setup mocks
        collector.ergast_client = Mock()
        collector.fastf1_client = Mock()
        
        collector.ergast_client.collect_race_data.return_value = sample_race_data
        
        # Mock FastF1 weather data
        enhanced_weather = WeatherData(26.0, 58.0, 1013.0, 6.0, 185, False, 32.0)
        collector.fastf1_client.get_weather_data.return_value = enhanced_weather
        
        # Mock FastF1 qualifying times
        collector.fastf1_client.get_qualifying_times.return_value = {
            'hamilton': {
                'Q1': LapTime(90123, 1),
                'Q2': LapTime(89456, 1),
                'Q3': LapTime(88789, 1)
            }
        }
        
        # Mock fastest lap
        collector.fastf1_client.get_fastest_lap.return_value = LapTime(88500, 45)
        
        result = collector.collect_race_data(2024, 1)
        
        # Verify Ergast was called
        collector.ergast_client.collect_race_data.assert_called_once_with(2024, 1)
        
        # Verify FastF1 enhancements were applied
        assert result.weather.temperature == 26.0  # Enhanced weather
        assert result.qualifying[0].q1_time.time_ms == 90123  # Enhanced qualifying
        assert result.results[0].fastest_lap.time_ms == 88500  # Enhanced fastest lap
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_collect_race_data_ergast_only(self, collector, sample_race_data):
        """Test race data collection when FastF1 fails."""
        # Setup mocks
        collector.ergast_client = Mock()
        collector.fastf1_client = Mock()
        
        collector.ergast_client.collect_race_data.return_value = sample_race_data
        
        # FastF1 calls fail
        collector.fastf1_client.get_weather_data.side_effect = FastF1Error("No data")
        collector.fastf1_client.get_qualifying_times.side_effect = FastF1Error("No data")
        collector.fastf1_client.get_fastest_lap.side_effect = FastF1Error("No data")
        
        result = collector.collect_race_data(2024, 1)
        
        # Should still return data from Ergast
        assert result.season == 2024
        assert result.round == 1
        assert result.weather.temperature == 25.0  # Original weather data
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_collect_race_data_ergast_failure(self, collector):
        """Test race data collection when Ergast fails."""
        # Setup mocks
        collector.ergast_client = Mock()
        collector.fastf1_client = Mock()
        
        collector.ergast_client.collect_race_data.side_effect = ErgastAPIError("API error")
        
        with pytest.raises(ErgastAPIError):
            collector.collect_race_data(2024, 1)
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_validate_data_valid(self, collector, sample_race_data):
        """Test data validation with valid data."""
        # Setup mocks
        collector.ergast_client = Mock()
        collector.ergast_client.validate_data.return_value = True
        
        is_valid = collector.validate_data(sample_race_data)
        
        assert is_valid is True
        collector.ergast_client.validate_data.assert_called_once_with(sample_race_data)
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_validate_data_ergast_invalid(self, collector, sample_race_data):
        """Test data validation when Ergast validation fails."""
        # Setup mocks
        collector.ergast_client = Mock()
        collector.ergast_client.validate_data.return_value = False
        
        is_valid = collector.validate_data(sample_race_data)
        
        assert is_valid is False
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_validate_data_invalid_weather(self, collector, sample_race_data):
        """Test data validation with invalid weather data."""
        # Setup mocks
        collector.ergast_client = Mock()
        collector.ergast_client.validate_data.return_value = True
        
        # Set invalid temperature
        sample_race_data.weather.temperature = -100.0
        
        is_valid = collector.validate_data(sample_race_data)
        
        assert is_valid is False
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_get_data_source_info(self, collector):
        """Test getting data source availability info."""
        # Setup mocks
        collector.ergast_client = Mock()
        collector.fastf1_client = Mock()
        
        collector.ergast_client.get_race_info.return_value = {'raceName': 'Test Race'}
        collector.fastf1_client.get_available_sessions.return_value = ['R', 'Q', 'FP1']
        
        info = collector.get_data_source_info(2024, 1)
        
        assert info['ergast_available'] is True
        assert info['fastf1_available'] is True
        assert info['fastf1_sessions'] == ['R', 'Q', 'FP1']
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_get_data_source_info_partial(self, collector):
        """Test getting data source info when only one source is available."""
        # Setup mocks
        collector.ergast_client = Mock()
        collector.fastf1_client = Mock()
        
        collector.ergast_client.get_race_info.side_effect = ErgastAPIError("No data")
        collector.fastf1_client.get_available_sessions.return_value = ['R', 'Q']
        
        info = collector.get_data_source_info(2024, 1)
        
        assert info['ergast_available'] is False
        assert info['fastf1_available'] is True
        assert info['fastf1_sessions'] == ['R', 'Q']
    
    @patch.object(CombinedDataCollector, '__init__', lambda x: None)
    def test_collect_historical_data(self, collector, sample_race_data):
        """Test historical data collection."""
        # Setup mocks
        collector.ergast_client = Mock()
        
        # Mock season schedule
        collector.ergast_client._make_request.return_value = {
            'MRData': {
                'RaceTable': {
                    'Races': [
                        {'round': '1'},
                        {'round': '2'}
                    ]
                }
            }
        }
        
        # Mock collect_race_data to return sample data
        collector.collect_race_data = Mock(return_value=sample_race_data)
        
        result = collector.collect_historical_data(2024, 2024)
        
        assert len(result) == 2  # Two races collected
        assert collector.collect_race_data.call_count == 2