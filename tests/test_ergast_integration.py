"""
Integration tests for Ergast API client.
These tests make real API calls and should be run sparingly.
"""
import pytest
from src.data.ergast_client import ErgastAPIClient


@pytest.mark.integration
class TestErgastAPIIntegration:
    """Integration tests that make real API calls."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return ErgastAPIClient()
    
    def test_collect_race_data_real_api(self, client):
        """Test collecting real race data from 2023 season."""
        # Test with 2023 Bahrain GP (should be stable data)
        race_data = client.collect_race_data(2023, 1)
        
        assert race_data.season == 2023
        assert race_data.round == 1
        assert race_data.circuit_id == "bahrain"
        assert len(race_data.results) > 15  # Should have ~20 drivers
        assert len(race_data.qualifying) > 15
        
        # Validate data quality
        assert client.validate_data(race_data) is True
        
        # Check that we have reasonable data
        winner = min(race_data.results, key=lambda x: x.final_position)
        assert winner.final_position == 1
        assert winner.points > 0
    
    def test_get_driver_standings_real_api(self, client):
        """Test getting real driver standings."""
        standings = client.get_driver_standings(2023, 1)
        
        assert len(standings) > 15
        assert all('Driver' in standing for standing in standings)
        assert all('points' in standing for standing in standings)
    
    def test_rate_limiting_works(self, client):
        """Test that rate limiting doesn't cause errors with multiple requests."""
        # Make several requests in quick succession
        for round_num in range(1, 4):  # First 3 races of 2023
            race_info = client.get_race_info(2023, round_num)
            assert race_info['season'] == '2023'
            assert race_info['round'] == str(round_num)