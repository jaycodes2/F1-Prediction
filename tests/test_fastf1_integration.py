"""
Integration tests for FastF1 client.
These tests make real API calls and should be run sparingly.
"""
import pytest
from src.data.fastf1_client import FastF1Client


@pytest.mark.integration
class TestFastF1Integration:
    """Integration tests that make real FastF1 calls."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return FastF1Client()
    
    def test_get_weather_data_real(self, client):
        """Test getting real weather data."""
        # Test with 2023 Bahrain GP (should be stable data)
        weather = client.get_weather_data(2023, 1, 'R')
        
        assert weather.temperature > 0
        assert weather.humidity >= 0
        assert weather.pressure > 0
        assert weather.wind_speed >= 0
        assert weather.wind_direction >= 0
        assert weather.track_temp > 0
    
    def test_get_session_info_real(self, client):
        """Test getting real session info."""
        info = client.get_session_info(2023, 1, 'R')
        
        assert info['session_name'] == 'Race'
        assert info['session_type'] == 'R'
        assert 'circuit_name' in info
        assert len(info['drivers']) > 15  # Should have ~20 drivers
    
    def test_validate_session_data_real(self, client):
        """Test validating real session data."""
        is_valid = client.validate_session_data(2023, 1, 'R')
        
        # Should be valid for a completed race
        assert is_valid is True
    
    def test_get_available_sessions_real(self, client):
        """Test getting available sessions for a real race."""
        available = client.get_available_sessions(2023, 1)
        
        # Should have at least Race and Qualifying
        assert 'R' in available
        assert 'Q' in available
        assert len(available) >= 2