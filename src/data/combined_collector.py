"""
Combined data collector that integrates Ergast and FastF1 data sources.
"""
import logging
from typing import List, Optional
from datetime import datetime

from .ergast_client import ErgastAPIClient, ErgastAPIError
from .fastf1_client import FastF1Client, FastF1Error
from ..models.data_models import RaceData, WeatherData
from ..models.interfaces import DataCollector


logger = logging.getLogger(__name__)


class CombinedDataCollector(DataCollector):
    """
    Data collector that combines Ergast API and FastF1 data sources.
    Uses Ergast for race results and standings, FastF1 for weather and telemetry.
    """
    
    def __init__(self):
        self.ergast_client = ErgastAPIClient()
        self.fastf1_client = FastF1Client()
    
    def collect_race_data(self, season: int, round: int) -> RaceData:
        """
        Collect complete race data combining both data sources.
        
        Args:
            season: F1 season year
            round: Race round number
            
        Returns:
            RaceData object with combined information
        """
        logger.info(f"Collecting combined race data for {season} round {round}")
        
        try:
            # Get base race data from Ergast (results, qualifying, basic info)
            race_data = self.ergast_client.collect_race_data(season, round)
            logger.debug("Successfully collected Ergast data")
            
            # Enhance with FastF1 weather data
            try:
                weather_data = self.fastf1_client.get_weather_data(season, round, 'R')
                race_data.weather = weather_data
                logger.debug("Successfully enhanced with FastF1 weather data")
            except FastF1Error as e:
                logger.warning(f"Could not get FastF1 weather data: {e}")
                # Keep the default weather data from Ergast client
            
            # Enhance qualifying times with FastF1 data if available
            try:
                fastf1_qualifying = self.fastf1_client.get_qualifying_times(season, round)
                
                # Update qualifying results with more accurate times from FastF1
                for qual_result in race_data.qualifying:
                    driver_id = qual_result.driver_id
                    if driver_id in fastf1_qualifying:
                        fastf1_times = fastf1_qualifying[driver_id]
                        
                        # Update times if FastF1 has better data
                        if fastf1_times['Q1'] and not qual_result.q1_time:
                            qual_result.q1_time = fastf1_times['Q1']
                        if fastf1_times['Q2'] and not qual_result.q2_time:
                            qual_result.q2_time = fastf1_times['Q2']
                        if fastf1_times['Q3'] and not qual_result.q3_time:
                            qual_result.q3_time = fastf1_times['Q3']
                
                logger.debug("Successfully enhanced qualifying times with FastF1 data")
                
            except FastF1Error as e:
                logger.warning(f"Could not enhance qualifying times with FastF1: {e}")
            
            # Enhance race results with fastest lap data from FastF1
            try:
                for race_result in race_data.results:
                    if not race_result.fastest_lap:
                        # Try to get fastest lap from FastF1
                        driver_abbr = race_result.driver_id.upper()[:3]  # Convert to abbreviation
                        fastest_lap = self.fastf1_client.get_fastest_lap(season, round, driver_abbr)
                        if fastest_lap:
                            race_result.fastest_lap = fastest_lap
                
                logger.debug("Successfully enhanced fastest lap times with FastF1 data")
                
            except FastF1Error as e:
                logger.warning(f"Could not enhance fastest lap times with FastF1: {e}")
            
            return race_data
            
        except ErgastAPIError as e:
            logger.error(f"Failed to collect base race data from Ergast: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error collecting combined race data: {e}")
            raise
    
    def collect_historical_data(self, start_year: int, end_year: int) -> List[RaceData]:
        """
        Collect historical data for a range of years using combined sources.
        
        Args:
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            
        Returns:
            List of RaceData objects
        """
        logger.info(f"Collecting combined historical data from {start_year} to {end_year}")
        
        all_race_data = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing season {year}")
            
            try:
                # Get season schedule from Ergast
                season_data = self.ergast_client._make_request(f"{year}")
                races = season_data['MRData']['RaceTable']['Races']
                
                for race in races:
                    round_num = int(race['round'])
                    
                    try:
                        race_data = self.collect_race_data(year, round_num)
                        all_race_data.append(race_data)
                        logger.debug(f"Successfully collected data for {year} round {round_num}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to collect data for {year}/{round_num}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to get season schedule for {year}: {e}")
                continue
        
        logger.info(f"Collected {len(all_race_data)} races from combined sources")
        return all_race_data
    
    def validate_data(self, data: RaceData) -> bool:
        """
        Validate collected race data using both clients' validation logic.
        
        Args:
            data: RaceData to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Use Ergast client's validation as the primary validator
        ergast_valid = self.ergast_client.validate_data(data)
        
        if not ergast_valid:
            return False
        
        # Additional validation for FastF1 enhanced data
        try:
            # Check weather data quality
            weather = data.weather
            if weather.temperature < -50 or weather.temperature > 60:
                logger.warning(f"Suspicious air temperature: {weather.temperature}°C")
                return False
            
            if weather.track_temp < -50 or weather.track_temp > 80:
                logger.warning(f"Suspicious track temperature: {weather.track_temp}°C")
                return False
            
            if weather.humidity < 0 or weather.humidity > 100:
                logger.warning(f"Invalid humidity: {weather.humidity}%")
                return False
            
            # Check for enhanced qualifying times
            enhanced_count = 0
            for qual_result in data.qualifying:
                if qual_result.q1_time or qual_result.q2_time or qual_result.q3_time:
                    enhanced_count += 1
            
            if enhanced_count == 0:
                logger.warning("No qualifying times found in data")
                # Don't fail validation, but log the issue
            
            return True
            
        except Exception as e:
            logger.error(f"Error in combined data validation: {e}")
            return False
    
    def get_data_source_info(self, season: int, round: int) -> dict:
        """
        Get information about data source availability for a specific race.
        
        Args:
            season: F1 season year
            round: Race round number
            
        Returns:
            Dictionary with data source availability information
        """
        info = {
            'ergast_available': False,
            'fastf1_available': False,
            'fastf1_sessions': []
        }
        
        # Check Ergast availability
        try:
            race_info = self.ergast_client.get_race_info(season, round)
            info['ergast_available'] = True
        except ErgastAPIError:
            pass
        
        # Check FastF1 availability
        try:
            available_sessions = self.fastf1_client.get_available_sessions(season, round)
            info['fastf1_available'] = len(available_sessions) > 0
            info['fastf1_sessions'] = available_sessions
        except FastF1Error:
            pass
        
        return info