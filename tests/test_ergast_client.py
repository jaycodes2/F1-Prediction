"""
Tests for Ergast API client.
"""
import pytest
import time
import requests.exceptions
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.data.ergast_client import (
    ErgastAPIClient, ErgastAPIError, RateLimiter
)
from src.models.data_models import RaceStatus, LapTime


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests_per_second=2)
        assert limiter.max_requests_per_second == 2
        assert limiter.min_interval == 0.5
    
    def test_rate_limiter_waits_when_needed(self):
        """Test that rate limiter waits when requests are too frequent."""
        limiter = RateLimiter(max_requests_per_second=10)  # 0.1s interval
        
        # First request should not wait
        start_time = time.time()
        limiter.wait_if_needed()
        first_duration = time.time() - start_time
        assert first_duration < 0.01  # Should be very fast
        
        # Second request immediately after should wait
        start_time = time.time()
        limiter.wait_if_needed()
        second_duration = time.time() - start_time
        assert second_duration >= 0.09  # Should wait ~0.1s


class TestErgastAPIClient:
    """Tests for the ErgastAPIClient class."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return ErgastAPIClient()
    
    @pytest.fixture
    def mock_race_response(self):
        """Mock response for race data."""
        return {
            'MRData': {
                'RaceTable': {
                    'Races': [{
                        'season': '2024',
                        'round': '1',
                        'raceName': 'Bahrain Grand Prix',
                        'date': '2024-03-02',
                        'time': '15:00:00Z',
                        'Circuit': {
                            'circuitId': 'bahrain'
                        },
                        'Results': [{
                            'Driver': {'driverId': 'hamilton'},
                            'Constructor': {'constructorId': 'mercedes'},
                            'grid': '1',
                            'position': '1',
                            'points': '25',
                            'status': 'Finished',
                            'laps': '57',
                            'FastestLap': {
                                'Time': {'time': '1:31.447'}
                            }
                        }]
                    }]
                }
            }
        }
    
    @pytest.fixture
    def mock_qualifying_response(self):
        """Mock response for qualifying data."""
        return {
            'MRData': {
                'RaceTable': {
                    'Races': [{
                        'QualifyingResults': [{
                            'Driver': {'driverId': 'hamilton'},
                            'Constructor': {'constructorId': 'mercedes'},
                            'position': '1',
                            'Q1': '1:32.123',
                            'Q2': '1:31.456',
                            'Q3': '1:30.789'
                        }]
                    }]
                }
            }
        }
    
    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.base_url == "http://ergast.com/api/f1"
        assert client.timeout == 30
        assert client.max_retries == 3
        assert isinstance(client.rate_limiter, RateLimiter)
    
    @patch('src.data.ergast_client.requests.Session.get')
    def test_make_request_success(self, mock_get, client):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_get.return_value = mock_response
        
        result = client._make_request('test/endpoint')
        
        assert result == {'test': 'data'}
        mock_get.assert_called_once()
    
    @patch('src.data.ergast_client.requests.Session.get')
    def test_make_request_retry_on_failure(self, mock_get, client):
        """Test request retry logic on failure."""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = Exception("Server error")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {'test': 'data'}
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]
        
        with patch('time.sleep'):  # Speed up test
            result = client._make_request('test/endpoint')
        
        assert result == {'test': 'data'}
        assert mock_get.call_count == 2
    
    @patch('src.data.ergast_client.requests.Session.get')
    def test_make_request_max_retries_exceeded(self, mock_get, client):
        """Test that exception is raised when max retries exceeded."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")
        mock_get.return_value = mock_response
        
        with patch('time.sleep'):  # Speed up test
            with pytest.raises(ErgastAPIError):
                client._make_request('test/endpoint')
        
        assert mock_get.call_count == client.max_retries + 1
    
    def test_parse_lap_time_valid(self, client):
        """Test parsing valid lap time."""
        lap_time = client._parse_lap_time("1:31.447")
        
        assert lap_time is not None
        assert lap_time.time_ms == 91447  # 1*60*1000 + 31*1000 + 447
        assert lap_time.lap_number == 1
    
    def test_parse_lap_time_invalid(self, client):
        """Test parsing invalid lap time."""
        assert client._parse_lap_time("") is None
        assert client._parse_lap_time("invalid") is None
        assert client._parse_lap_time(None) is None
    
    def test_parse_race_status(self, client):
        """Test parsing race status."""
        assert client._parse_race_status("Finished") == RaceStatus.FINISHED
        assert client._parse_race_status("1") == RaceStatus.FINISHED
        assert client._parse_race_status("Disqualified") == RaceStatus.DSQ
        assert client._parse_race_status("Did not start") == RaceStatus.DNS
        assert client._parse_race_status("Accident") == RaceStatus.DNF
    
    @patch.object(ErgastAPIClient, '_make_request')
    def test_get_race_results(self, mock_request, client, mock_race_response):
        """Test getting race results."""
        mock_request.return_value = mock_race_response
        
        results = client.get_race_results(2024, 1)
        
        assert len(results) == 1
        assert results[0]['Driver']['driverId'] == 'hamilton'
        mock_request.assert_called_once_with('2024/1/results')
    
    @patch.object(ErgastAPIClient, '_make_request')
    def test_get_race_results_no_data(self, mock_request, client):
        """Test getting race results when no data available."""
        mock_request.return_value = {
            'MRData': {'RaceTable': {'Races': []}}
        }
        
        results = client.get_race_results(2024, 1)
        assert results == []
    
    @patch.object(ErgastAPIClient, '_make_request')
    def test_get_qualifying_results(self, mock_request, client, mock_qualifying_response):
        """Test getting qualifying results."""
        mock_request.return_value = mock_qualifying_response
        
        results = client.get_qualifying_results(2024, 1)
        
        assert len(results) == 1
        assert results[0]['Driver']['driverId'] == 'hamilton'
        mock_request.assert_called_once_with('2024/1/qualifying')
    
    @patch.object(ErgastAPIClient, 'get_race_info')
    @patch.object(ErgastAPIClient, 'get_race_results')
    @patch.object(ErgastAPIClient, 'get_qualifying_results')
    def test_collect_race_data(self, mock_qual, mock_results, mock_info, client):
        """Test collecting complete race data."""
        # Setup mocks
        mock_info.return_value = {
            'raceName': 'Bahrain Grand Prix',
            'date': '2024-03-02',
            'time': '15:00:00Z',
            'Circuit': {'circuitId': 'bahrain'}
        }
        
        mock_results.return_value = [{
            'Driver': {'driverId': 'hamilton'},
            'Constructor': {'constructorId': 'mercedes'},
            'grid': '1',
            'position': '1',
            'points': '25',
            'status': 'Finished',
            'laps': '57',
            'FastestLap': {'Time': {'time': '1:31.447'}}
        }]
        
        mock_qual.return_value = [{
            'Driver': {'driverId': 'hamilton'},
            'Constructor': {'constructorId': 'mercedes'},
            'position': '1',
            'Q1': '1:32.123',
            'Q2': '1:31.456',
            'Q3': '1:30.789'
        }]
        
        race_data = client.collect_race_data(2024, 1)
        
        assert race_data.season == 2024
        assert race_data.round == 1
        assert race_data.circuit_id == 'bahrain'
        assert race_data.race_name == 'Bahrain Grand Prix'
        assert len(race_data.results) == 1
        assert len(race_data.qualifying) == 1
        assert race_data.results[0].driver_id == 'hamilton'
        assert race_data.results[0].final_position == 1
    
    def test_validate_data_valid(self, client):
        """Test data validation with valid data."""
        from src.models.data_models import RaceData, RaceResult, WeatherData
        
        race_data = RaceData(
            season=2024,
            round=1,
            circuit_id='bahrain',
            race_name='Test Race',
            date=datetime.now(),
            results=[
                RaceResult('hamilton', 'mercedes', 1, 1, 25.0, None, RaceStatus.FINISHED, 57),
                RaceResult('verstappen', 'red_bull', 2, 2, 18.0, None, RaceStatus.FINISHED, 57)
            ],
            qualifying=[],
            weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        )
        
        assert client.validate_data(race_data) is True
    
    def test_validate_data_no_results(self, client):
        """Test data validation with no results."""
        from src.models.data_models import RaceData, WeatherData
        
        race_data = RaceData(
            season=2024,
            round=1,
            circuit_id='bahrain',
            race_name='Test Race',
            date=datetime.now(),
            results=[],
            qualifying=[],
            weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        )
        
        assert client.validate_data(race_data) is False
    
    def test_validate_data_duplicate_drivers(self, client):
        """Test data validation with duplicate drivers."""
        from src.models.data_models import RaceData, RaceResult, WeatherData
        
        race_data = RaceData(
            season=2024,
            round=1,
            circuit_id='bahrain',
            race_name='Test Race',
            date=datetime.now(),
            results=[
                RaceResult('hamilton', 'mercedes', 1, 1, 25.0, None, RaceStatus.FINISHED, 57),
                RaceResult('hamilton', 'mercedes', 2, 2, 18.0, None, RaceStatus.FINISHED, 57)
            ],
            qualifying=[],
            weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        )
        
        assert client.validate_data(race_data) is False