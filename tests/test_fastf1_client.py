"""
Tests for FastF1 client.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data.fastf1_client import FastF1Client, FastF1Error
from src.models.data_models import WeatherData, LapTime


class TestFastF1Client:
    """Tests for the FastF1Client class."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        with patch('src.data.fastf1_client.fastf1.Cache.enable_cache'):
            with patch('src.data.fastf1_client.fastf1.set_log_level'):
                return FastF1Client()
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock FastF1 session."""
        session = Mock()
        session.name = "Race"
        session.date = datetime(2024, 3, 2, 15, 0, 0)
        session.drivers = ['HAM', 'VER', 'LEC', 'RUS', 'SAI', 'NOR', 'PIA', 'ALO', 
                          'STR', 'TSU', 'GAS', 'OCO', 'ALB', 'SAR', 'MAG', 'HUL', 
                          'BOT', 'ZHO', 'RIC', 'DEV']  # 20 drivers
        
        # Mock weather data
        weather_df = pd.DataFrame({
            'AirTemp': [25.0, 26.0, 25.5],
            'Humidity': [60.0, 58.0, 59.0],
            'Pressure': [1013.25, 1013.0, 1013.1],
            'WindSpeed': [5.0, 6.0, 5.5],
            'WindDirection': [180, 185, 182],
            'Rainfall': [False, False, False],
            'TrackTemp': [30.0, 31.0, 30.5]
        })
        session.weather_data = weather_df
        
        # Mock event data
        session.event = {
            'EventName': 'Bahrain Grand Prix',
            'EventKey': 'bahrain',
            'Country': 'Bahrain',
            'Location': 'Sakhir'
        }
        
        # Mock lap data
        lap_data = pd.DataFrame({
            'LapNumber': [1, 2, 3, 1, 2, 3],
            'Driver': ['HAM', 'HAM', 'HAM', 'VER', 'VER', 'VER'],
            'LapTime': [
                pd.Timedelta(seconds=90.123),
                pd.Timedelta(seconds=89.456),
                pd.Timedelta(seconds=88.789),
                pd.Timedelta(seconds=90.555),
                pd.Timedelta(seconds=89.888),
                pd.Timedelta(seconds=89.111)
            ],
            'IsPersonalBest': [False, False, True, False, False, True],
            'Sector1Time': [pd.Timedelta(seconds=30.1), pd.Timedelta(seconds=29.8), pd.Timedelta(seconds=29.5),
                           pd.Timedelta(seconds=30.2), pd.Timedelta(seconds=29.9), pd.Timedelta(seconds=29.6)],
            'Sector2Time': [pd.Timedelta(seconds=30.0), pd.Timedelta(seconds=29.7), pd.Timedelta(seconds=29.4),
                           pd.Timedelta(seconds=30.1), pd.Timedelta(seconds=29.8), pd.Timedelta(seconds=29.5)],
            'Sector3Time': [pd.Timedelta(seconds=30.0), pd.Timedelta(seconds=29.9), pd.Timedelta(seconds=29.8),
                           pd.Timedelta(seconds=30.2), pd.Timedelta(seconds=30.1), pd.Timedelta(seconds=30.0)]
        })
        session.laps = lap_data
        
        # Mock results data for qualifying
        results_df = pd.DataFrame({
            'Abbreviation': ['HAM', 'VER', 'LEC'],
            'Q1': [pd.Timedelta(seconds=90.123), pd.Timedelta(seconds=90.555), pd.Timedelta(seconds=91.0)],
            'Q2': [pd.Timedelta(seconds=89.456), pd.Timedelta(seconds=89.888), pd.NaT],
            'Q3': [pd.Timedelta(seconds=88.789), pd.Timedelta(seconds=89.111), pd.NaT]
        })
        session.results = results_df
        
        return session
    
    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.cache_dir == "data/fastf1_cache"
    
    @patch('src.data.fastf1_client.os.makedirs')
    @patch('src.data.fastf1_client.os.path.exists')
    def test_setup_cache_creates_directory(self, mock_exists, mock_makedirs, client):
        """Test cache directory creation."""
        mock_exists.return_value = False
        
        client._setup_cache()
        
        mock_makedirs.assert_called_once_with(client.cache_dir, exist_ok=True)
    
    @patch('src.data.fastf1_client.fastf1.get_session')
    def test_get_session_success(self, mock_get_session, client, mock_session):
        """Test successful session retrieval."""
        mock_get_session.return_value = mock_session
        
        session = client.get_session(2024, 1, 'R')
        
        assert session == mock_session
        mock_get_session.assert_called_once_with(2024, 1, 'R')
        mock_session.load.assert_called_once()
    
    @patch('src.data.fastf1_client.fastf1.get_session')
    def test_get_session_failure(self, mock_get_session, client):
        """Test session retrieval failure."""
        mock_get_session.side_effect = Exception("Session not found")
        
        with pytest.raises(FastF1Error) as exc_info:
            client.get_session(2024, 1, 'R')
        
        assert "Failed to load R session" in str(exc_info.value)
        assert exc_info.value.session_info == {"year": 2024, "race": 1, "session_type": "R"}
    
    @patch.object(FastF1Client, 'get_session')
    def test_get_weather_data_success(self, mock_get_session, client, mock_session):
        """Test successful weather data retrieval."""
        mock_get_session.return_value = mock_session
        
        weather = client.get_weather_data(2024, 1, 'R')
        
        assert isinstance(weather, WeatherData)
        assert weather.temperature == 25.5  # Latest reading
        assert weather.humidity == 59.0
        assert weather.pressure == 1013.1
        assert weather.wind_speed == 5.5
        assert weather.wind_direction == 182
        assert weather.rainfall is False
        assert weather.track_temp == 30.5
    
    @patch.object(FastF1Client, 'get_session')
    def test_get_weather_data_empty(self, mock_get_session, client, mock_session):
        """Test weather data retrieval with empty data."""
        mock_session.weather_data = pd.DataFrame()  # Empty DataFrame
        mock_get_session.return_value = mock_session
        
        weather = client.get_weather_data(2024, 1, 'R')
        
        # Should return default values
        assert isinstance(weather, WeatherData)
        assert weather.temperature == 25.0
        assert weather.humidity == 60.0
        assert weather.rainfall is False
    
    @patch.object(FastF1Client, 'get_session')
    def test_get_weather_data_error(self, mock_get_session, client):
        """Test weather data retrieval with error."""
        mock_get_session.side_effect = Exception("Session error")
        
        weather = client.get_weather_data(2024, 1, 'R')
        
        # Should return default values on error
        assert isinstance(weather, WeatherData)
        assert weather.temperature == 25.0
    
    @patch.object(FastF1Client, 'get_session')
    def test_get_qualifying_times(self, mock_get_session, client, mock_session):
        """Test qualifying times retrieval."""
        mock_get_session.return_value = mock_session
        
        # Mock the pick_driver method
        def mock_pick_driver(driver):
            return mock_session.laps[mock_session.laps['Driver'] == driver]
        
        mock_session.laps.pick_driver = mock_pick_driver
        
        qualifying_times = client.get_qualifying_times(2024, 1)
        
        assert 'ham' in qualifying_times
        assert 'ver' in qualifying_times
        
        # Check HAM's times
        ham_times = qualifying_times['ham']
        assert ham_times['Q1'] is not None
        assert ham_times['Q2'] is not None
        assert ham_times['Q3'] is not None
        assert ham_times['Q3'].time_ms == 88789  # 88.789 seconds
    
    @patch.object(FastF1Client, 'get_session')
    def test_get_lap_times(self, mock_get_session, client, mock_session):
        """Test lap times retrieval for a driver."""
        mock_get_session.return_value = mock_session
        
        # Mock the pick_driver method
        def mock_pick_driver(driver):
            return mock_session.laps[mock_session.laps['Driver'] == driver]
        
        mock_session.laps.pick_driver = mock_pick_driver
        
        lap_times = client.get_lap_times(2024, 1, 'HAM')
        
        assert len(lap_times) == 3  # HAM has 3 laps
        assert all(isinstance(lap, LapTime) for lap in lap_times)
        assert lap_times[0].lap_number == 1
        assert lap_times[0].time_ms == 90123  # 90.123 seconds
    
    @patch.object(FastF1Client, 'get_lap_times')
    def test_get_fastest_lap(self, mock_get_lap_times, client):
        """Test fastest lap retrieval."""
        mock_lap_times = [
            LapTime(time_ms=90123, lap_number=1),
            LapTime(time_ms=89456, lap_number=2),
            LapTime(time_ms=88789, lap_number=3)  # Fastest
        ]
        mock_get_lap_times.return_value = mock_lap_times
        
        fastest = client.get_fastest_lap(2024, 1, 'HAM')
        
        assert fastest is not None
        assert fastest.time_ms == 88789
        assert fastest.lap_number == 3
    
    @patch.object(FastF1Client, 'get_lap_times')
    def test_get_fastest_lap_no_data(self, mock_get_lap_times, client):
        """Test fastest lap retrieval with no data."""
        mock_get_lap_times.return_value = []
        
        fastest = client.get_fastest_lap(2024, 1, 'HAM')
        
        assert fastest is None
    
    @patch.object(FastF1Client, 'get_session')
    def test_get_telemetry_summary(self, mock_get_session, client, mock_session):
        """Test telemetry summary retrieval."""
        mock_get_session.return_value = mock_session
        
        # Mock the pick_driver method
        def mock_pick_driver(driver):
            return mock_session.laps[mock_session.laps['Driver'] == driver]
        
        mock_session.laps.pick_driver = mock_pick_driver
        
        summary = client.get_telemetry_summary(2024, 1, 'HAM')
        
        assert summary['total_laps'] == 3
        assert summary['valid_laps'] == 3
        assert summary['fastest_lap_time'] == 88.789
        assert 'average_lap_time' in summary
        assert 'avg_sector1' in summary
        assert 'avg_sector2' in summary
        assert 'avg_sector3' in summary
    
    @patch.object(FastF1Client, 'get_session')
    def test_get_session_info(self, mock_get_session, client, mock_session):
        """Test session info retrieval."""
        mock_get_session.return_value = mock_session
        
        info = client.get_session_info(2024, 1, 'R')
        
        assert info['session_name'] == 'Race'
        assert info['session_type'] == 'R'
        assert info['circuit_name'] == 'Bahrain Grand Prix'
        assert info['circuit_key'] == 'bahrain'
        assert info['country'] == 'Bahrain'
        assert info['location'] == 'Sakhir'
        assert len(info['drivers']) == 20
    
    @patch.object(FastF1Client, 'get_session')
    def test_validate_session_data_valid(self, mock_get_session, client, mock_session):
        """Test session data validation with valid data."""
        mock_get_session.return_value = mock_session
        
        is_valid = client.validate_session_data(2024, 1, 'R')
        
        assert is_valid is True
    
    @patch.object(FastF1Client, 'get_session')
    def test_validate_session_data_no_laps(self, mock_get_session, client, mock_session):
        """Test session data validation with no lap data."""
        mock_session.laps = pd.DataFrame()  # Empty DataFrame
        mock_get_session.return_value = mock_session
        
        is_valid = client.validate_session_data(2024, 1, 'R')
        
        assert is_valid is False
    
    @patch.object(FastF1Client, 'get_session')
    def test_validate_session_data_no_drivers(self, mock_get_session, client, mock_session):
        """Test session data validation with no drivers."""
        mock_session.drivers = []
        mock_get_session.return_value = mock_session
        
        is_valid = client.validate_session_data(2024, 1, 'R')
        
        assert is_valid is False
    
    @patch('src.data.fastf1_client.fastf1.get_session')
    def test_get_available_sessions(self, mock_get_session, client):
        """Test getting available sessions."""
        # Mock successful sessions
        mock_session = Mock()
        mock_session.load = Mock()
        
        # Only R and Q sessions available
        def side_effect(year, race, session_type):
            if session_type in ['R', 'Q']:
                return mock_session
            else:
                raise Exception("Session not available")
        
        mock_get_session.side_effect = side_effect
        
        available = client.get_available_sessions(2024, 1)
        
        assert 'R' in available
        assert 'Q' in available
        assert 'FP1' not in available
        assert len(available) == 2