"""
Ergast API client for collecting F1 race data.
"""
import time
import requests
import requests.exceptions
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

from ..models.data_models import (
    RaceData, RaceResult, QualifyingResult, 
    WeatherData, LapTime, RaceStatus
)
from ..models.interfaces import DataCollector
from ..config import config


logger = logging.getLogger(__name__)


class ErgastAPIError(Exception):
    """Custom exception for Ergast API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, max_requests_per_second: int = 4):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_interval:
            sleep_time = self.min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


class ErgastAPIClient(DataCollector):
    """Client for interacting with the Ergast F1 API."""
    
    def __init__(self):
        self.base_url = config.api.ergast_base_url
        self.timeout = config.api.request_timeout
        self.max_retries = config.api.max_retries
        self.backoff_factor = config.api.retry_backoff_factor
        self.rate_limiter = RateLimiter(config.api.ergast_rate_limit)
        
        # Setup session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'F1-Race-Prediction-System/1.0'
        })
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request with retry logic."""
        if params is None:
            params = {}
        
        # Always request JSON format
        params['format'] = 'json'
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                logger.debug(f"Making request to {url} (attempt {attempt + 1})")
                response = self.session.get(url, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries:
                        wait_time = self.backoff_factor * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ErgastAPIError(
                            "Rate limit exceeded after all retries", 
                            response.status_code
                        )
                else:
                    # For other status codes, retry if we have attempts left
                    if attempt < self.max_retries:
                        wait_time = self.backoff_factor * (2 ** attempt)
                        logger.warning(f"HTTP {response.status_code}, retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        try:
                            response.raise_for_status()
                        except requests.exceptions.HTTPError as e:
                            raise ErgastAPIError(f"HTTP error after {self.max_retries} retries: {e}", response.status_code)
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.warning(f"Request failed: {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    raise ErgastAPIError(f"Request failed after {self.max_retries} retries: {e}")
        
        raise ErgastAPIError("Unexpected error in request handling")
    
    def get_race_results(self, season: int, round: int) -> List[Dict[str, Any]]:
        """Get race results for a specific race."""
        endpoint = f"{season}/{round}/results"
        
        try:
            data = self._make_request(endpoint)
            race_table = data['MRData']['RaceTable']
            
            if not race_table['Races']:
                logger.warning(f"No race data found for season {season}, round {round}")
                return []
            
            return race_table['Races'][0]['Results']
            
        except KeyError as e:
            raise ErgastAPIError(f"Unexpected API response structure: {e}")
    
    def get_qualifying_results(self, season: int, round: int) -> List[Dict[str, Any]]:
        """Get qualifying results for a specific race."""
        endpoint = f"{season}/{round}/qualifying"
        
        try:
            data = self._make_request(endpoint)
            race_table = data['MRData']['RaceTable']
            
            if not race_table['Races']:
                logger.warning(f"No qualifying data found for season {season}, round {round}")
                return []
            
            return race_table['Races'][0]['QualifyingResults']
            
        except KeyError as e:
            raise ErgastAPIError(f"Unexpected API response structure: {e}")
    
    def get_race_info(self, season: int, round: int) -> Dict[str, Any]:
        """Get basic race information."""
        endpoint = f"{season}/{round}"
        
        try:
            data = self._make_request(endpoint)
            race_table = data['MRData']['RaceTable']
            
            if not race_table['Races']:
                raise ErgastAPIError(f"No race found for season {season}, round {round}")
            
            return race_table['Races'][0]
            
        except KeyError as e:
            raise ErgastAPIError(f"Unexpected API response structure: {e}")
    
    def get_driver_standings(self, season: int, round: int = None) -> List[Dict[str, Any]]:
        """Get driver standings for a season or after a specific round."""
        if round:
            endpoint = f"{season}/{round}/driverStandings"
        else:
            endpoint = f"{season}/driverStandings"
        
        try:
            data = self._make_request(endpoint)
            standings_table = data['MRData']['StandingsTable']
            
            if not standings_table['StandingsLists']:
                return []
            
            return standings_table['StandingsLists'][0]['DriverStandings']
            
        except KeyError as e:
            raise ErgastAPIError(f"Unexpected API response structure: {e}")
    
    def get_constructor_standings(self, season: int, round: int = None) -> List[Dict[str, Any]]:
        """Get constructor standings for a season or after a specific round."""
        if round:
            endpoint = f"{season}/{round}/constructorStandings"
        else:
            endpoint = f"{season}/constructorStandings"
        
        try:
            data = self._make_request(endpoint)
            standings_table = data['MRData']['StandingsTable']
            
            if not standings_table['StandingsLists']:
                return []
            
            return standings_table['StandingsLists'][0]['ConstructorStandings']
            
        except KeyError as e:
            raise ErgastAPIError(f"Unexpected API response structure: {e}")
    
    def _parse_lap_time(self, time_str: str) -> Optional[LapTime]:
        """Parse lap time string into LapTime object."""
        if not time_str:
            return None
        
        try:
            # Parse time format like "1:23.456"
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds_parts = parts[1].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                
                total_ms = (minutes * 60 + seconds) * 1000 + milliseconds
                return LapTime(time_ms=total_ms, lap_number=1)  # Lap number not available in this context
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse lap time '{time_str}': {e}")
        
        return None
    
    def _parse_race_status(self, status_str: str) -> RaceStatus:
        """Parse race status string into RaceStatus enum."""
        status_lower = status_str.lower()
        
        if 'finished' in status_lower or status_str.isdigit():
            return RaceStatus.FINISHED
        elif 'disqualified' in status_lower:
            return RaceStatus.DSQ
        elif 'did not start' in status_lower:
            return RaceStatus.DNS
        else:
            return RaceStatus.DNF
    
    def collect_race_data(self, season: int, round: int) -> RaceData:
        """Collect complete race data for a specific race."""
        try:
            # Get basic race info
            race_info = self.get_race_info(season, round)
            
            # Get race results
            race_results_raw = self.get_race_results(season, round)
            
            # Get qualifying results
            qualifying_results_raw = self.get_qualifying_results(season, round)
            
            # Parse race results
            race_results = []
            for result in race_results_raw:
                fastest_lap = None
                if 'FastestLap' in result and 'Time' in result['FastestLap']:
                    fastest_lap = self._parse_lap_time(result['FastestLap']['Time']['time'])
                
                race_result = RaceResult(
                    driver_id=result['Driver']['driverId'],
                    constructor_id=result['Constructor']['constructorId'],
                    grid_position=int(result['grid']),
                    final_position=int(result['position']) if result['position'].isdigit() else 999,
                    points=float(result['points']),
                    fastest_lap=fastest_lap,
                    status=self._parse_race_status(result['status']),
                    laps_completed=int(result['laps'])
                )
                race_results.append(race_result)
            
            # Parse qualifying results
            qualifying_results = []
            for qual in qualifying_results_raw:
                q1_time = self._parse_lap_time(qual.get('Q1', ''))
                q2_time = self._parse_lap_time(qual.get('Q2', ''))
                q3_time = self._parse_lap_time(qual.get('Q3', ''))
                
                qualifying_result = QualifyingResult(
                    driver_id=qual['Driver']['driverId'],
                    constructor_id=qual['Constructor']['constructorId'],
                    position=int(qual['position']),
                    q1_time=q1_time,
                    q2_time=q2_time,
                    q3_time=q3_time
                )
                qualifying_results.append(qualifying_result)
            
            # Create placeholder weather data (Ergast doesn't provide weather)
            weather = WeatherData(
                temperature=25.0,  # Default values
                humidity=60.0,
                pressure=1013.25,
                wind_speed=5.0,
                wind_direction=180,
                rainfall=False,
                track_temp=30.0
            )
            
            # Parse race date
            race_date = datetime.strptime(
                f"{race_info['date']} {race_info.get('time', '00:00:00Z')}", 
                "%Y-%m-%d %H:%M:%SZ"
            )
            
            return RaceData(
                season=season,
                round=round,
                circuit_id=race_info['Circuit']['circuitId'],
                race_name=race_info['raceName'],
                date=race_date,
                results=race_results,
                qualifying=qualifying_results,
                weather=weather
            )
            
        except Exception as e:
            logger.error(f"Failed to collect race data for {season}/{round}: {e}")
            raise ErgastAPIError(f"Failed to collect race data: {e}")
    
    def collect_historical_data(self, start_year: int, end_year: int) -> List[RaceData]:
        """Collect historical data for a range of years."""
        all_race_data = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Collecting data for season {year}")
            
            try:
                # Get season schedule first
                season_data = self._make_request(f"{year}")
                races = season_data['MRData']['RaceTable']['Races']
                
                for race in races:
                    round_num = int(race['round'])
                    try:
                        race_data = self.collect_race_data(year, round_num)
                        all_race_data.append(race_data)
                        logger.debug(f"Collected data for {year} round {round_num}")
                    except ErgastAPIError as e:
                        logger.warning(f"Failed to collect data for {year}/{round_num}: {e}")
                        continue
                        
            except ErgastAPIError as e:
                logger.error(f"Failed to get season schedule for {year}: {e}")
                continue
        
        logger.info(f"Collected {len(all_race_data)} races from {start_year}-{end_year}")
        return all_race_data
    
    def validate_data(self, data: RaceData) -> bool:
        """Validate collected race data for quality and completeness."""
        try:
            # Check basic data integrity
            if not data.results:
                logger.warning(f"No race results for {data.season}/{data.round}")
                return False
            
            if not data.qualifying:
                logger.warning(f"No qualifying results for {data.season}/{data.round}")
                # Don't fail validation, as some old races might not have qualifying data
            
            # Check for reasonable number of drivers
            num_drivers = len(data.results)
            if num_drivers < config.data.min_drivers_per_race or num_drivers > config.data.max_drivers_per_race:
                logger.warning(f"Unusual number of drivers ({num_drivers}) for {data.season}/{data.round}")
            
            # Check for data consistency
            driver_ids = {result.driver_id for result in data.results}
            if len(driver_ids) != len(data.results):
                logger.error(f"Duplicate drivers found in race results for {data.season}/{data.round}")
                return False
            
            # Check position consistency
            positions = [result.final_position for result in data.results if result.final_position < 999]
            if len(set(positions)) != len(positions):
                logger.error(f"Duplicate finishing positions found for {data.season}/{data.round}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating race data: {e}")
            return False
    
    def __del__(self):
        """Clean up session on destruction."""
        if hasattr(self, 'session'):
            self.session.close()