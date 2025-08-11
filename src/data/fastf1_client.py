"""
FastF1 client for collecting F1 telemetry and weather data.
"""
import os
import logging
import fastf1
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models.data_models import WeatherData, LapTime
from ..config import config


logger = logging.getLogger(__name__)


class FastF1Error(Exception):
    """Custom exception for FastF1 errors."""
    def __init__(self, message: str, session_info: Optional[Dict[str, Any]] = None):
        self.message = message
        self.session_info = session_info
        super().__init__(self.message)


class FastF1Client:
    """Client for interacting with FastF1 data."""
    
    def __init__(self):
        self.cache_dir = config.api.fastf1_cache_dir
        self._setup_cache()
        
        # Enable FastF1 cache
        fastf1.Cache.enable_cache(self.cache_dir)
        
        # Suppress FastF1 warnings for cleaner output
        fastf1.set_log_level('WARNING')
    
    def _setup_cache(self):
        """Setup cache directory for FastF1."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Created FastF1 cache directory: {self.cache_dir}")
    
    def get_session(self, year: int, race: int, session_type: str) -> fastf1.core.Session:
        """Get a FastF1 session object."""
        try:
            # Load the session
            session = fastf1.get_session(year, race, session_type)
            session.load()
            return session
            
        except Exception as e:
            raise FastF1Error(
                f"Failed to load {session_type} session for {year} round {race}: {e}",
                {"year": year, "race": race, "session_type": session_type}
            )
    
    def get_weather_data(self, year: int, race: int, session_type: str = 'R') -> WeatherData:
        """Get weather data for a race session."""
        try:
            session = self.get_session(year, race, session_type)
            
            # Get weather data from the session
            weather_data = session.weather_data
            
            if weather_data.empty:
                logger.warning(f"No weather data available for {year} round {race}")
                # Return default weather data
                return WeatherData(
                    temperature=25.0,
                    humidity=60.0,
                    pressure=1013.25,
                    wind_speed=5.0,
                    wind_direction=180,
                    rainfall=False,
                    track_temp=30.0
                )
            
            # Get the most recent weather reading
            latest_weather = weather_data.iloc[-1]
            
            return WeatherData(
                temperature=float(latest_weather.get('AirTemp', 25.0)),
                humidity=float(latest_weather.get('Humidity', 60.0)),
                pressure=float(latest_weather.get('Pressure', 1013.25)),
                wind_speed=float(latest_weather.get('WindSpeed', 5.0)),
                wind_direction=int(latest_weather.get('WindDirection', 180)),
                rainfall=bool(latest_weather.get('Rainfall', False)),
                track_temp=float(latest_weather.get('TrackTemp', 30.0))
            )
            
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            # Return default weather data on error
            return WeatherData(
                temperature=25.0,
                humidity=60.0,
                pressure=1013.25,
                wind_speed=5.0,
                wind_direction=180,
                rainfall=False,
                track_temp=30.0
            )
    
    def get_qualifying_times(self, year: int, race: int) -> Dict[str, Dict[str, Optional[LapTime]]]:
        """Get detailed qualifying times for all drivers."""
        try:
            session = self.get_session(year, race, 'Q')
            
            qualifying_times = {}
            
            # Get results for all drivers
            for driver_abbr in session.drivers:
                try:
                    driver_data = session.laps.pick_driver(driver_abbr)
                    
                    if driver_data.empty:
                        continue
                    
                    # Get best times from each qualifying session
                    q1_time = None
                    q2_time = None
                    q3_time = None
                    
                    # Q1 times (all drivers)
                    q1_laps = driver_data[driver_data['IsPersonalBest'] == True]
                    if not q1_laps.empty:
                        best_q1 = q1_laps.iloc[0]
                        if pd.notna(best_q1['LapTime']):
                            q1_time = LapTime(
                                time_ms=int(best_q1['LapTime'].total_seconds() * 1000),
                                lap_number=int(best_q1['LapNumber'])
                            )
                    
                    # For Q2 and Q3, we need to check session results
                    results = session.results
                    driver_result = results[results['Abbreviation'] == driver_abbr]
                    
                    if not driver_result.empty:
                        driver_info = driver_result.iloc[0]
                        
                        # Q2 time
                        if pd.notna(driver_info.get('Q2')):
                            q2_time = LapTime(
                                time_ms=int(driver_info['Q2'].total_seconds() * 1000),
                                lap_number=1  # Qualifying lap numbers not easily available
                            )
                        
                        # Q3 time
                        if pd.notna(driver_info.get('Q3')):
                            q3_time = LapTime(
                                time_ms=int(driver_info['Q3'].total_seconds() * 1000),
                                lap_number=1
                            )
                    
                    # Convert driver abbreviation to full driver ID (lowercase)
                    driver_id = driver_abbr.lower()
                    
                    qualifying_times[driver_id] = {
                        'Q1': q1_time,
                        'Q2': q2_time,
                        'Q3': q3_time
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing qualifying data for {driver_abbr}: {e}")
                    continue
            
            return qualifying_times
            
        except FastF1Error:
            raise
        except Exception as e:
            raise FastF1Error(f"Error getting qualifying times: {e}")
    
    def get_lap_times(self, year: int, race: int, driver_abbr: str) -> List[LapTime]:
        """Get all lap times for a specific driver in a race."""
        try:
            session = self.get_session(year, race, 'R')
            
            driver_laps = session.laps.pick_driver(driver_abbr)
            
            if driver_laps.empty:
                logger.warning(f"No lap data found for driver {driver_abbr}")
                return []
            
            lap_times = []
            for _, lap in driver_laps.iterrows():
                if pd.notna(lap['LapTime']):
                    lap_time = LapTime(
                        time_ms=int(lap['LapTime'].total_seconds() * 1000),
                        lap_number=int(lap['LapNumber'])
                    )
                    lap_times.append(lap_time)
            
            return lap_times
            
        except FastF1Error:
            raise
        except Exception as e:
            raise FastF1Error(f"Error getting lap times for {driver_abbr}: {e}")
    
    def get_fastest_lap(self, year: int, race: int, driver_abbr: str) -> Optional[LapTime]:
        """Get the fastest lap for a specific driver in a race."""
        try:
            lap_times = self.get_lap_times(year, race, driver_abbr)
            
            if not lap_times:
                return None
            
            # Find the fastest lap
            fastest = min(lap_times, key=lambda x: x.time_ms)
            return fastest
            
        except FastF1Error:
            raise
        except Exception as e:
            logger.error(f"Error getting fastest lap for {driver_abbr}: {e}")
            return None
    
    def get_telemetry_summary(self, year: int, race: int, driver_abbr: str) -> Dict[str, Any]:
        """Get telemetry summary for a driver."""
        try:
            session = self.get_session(year, race, 'R')
            
            driver_laps = session.laps.pick_driver(driver_abbr)
            
            if driver_laps.empty:
                return {}
            
            # Calculate summary statistics
            valid_laps = driver_laps[pd.notna(driver_laps['LapTime'])]
            
            if valid_laps.empty:
                return {}
            
            summary = {
                'total_laps': len(driver_laps),
                'valid_laps': len(valid_laps),
                'fastest_lap_time': valid_laps['LapTime'].min().total_seconds(),
                'average_lap_time': valid_laps['LapTime'].mean().total_seconds(),
                'median_lap_time': valid_laps['LapTime'].median().total_seconds(),
                'lap_time_std': valid_laps['LapTime'].std().total_seconds(),
            }
            
            # Add sector times if available
            if 'Sector1Time' in valid_laps.columns:
                summary['avg_sector1'] = valid_laps['Sector1Time'].mean().total_seconds()
            if 'Sector2Time' in valid_laps.columns:
                summary['avg_sector2'] = valid_laps['Sector2Time'].mean().total_seconds()
            if 'Sector3Time' in valid_laps.columns:
                summary['avg_sector3'] = valid_laps['Sector3Time'].mean().total_seconds()
            
            return summary
            
        except FastF1Error:
            raise
        except Exception as e:
            logger.error(f"Error getting telemetry summary for {driver_abbr}: {e}")
            return {}
    
    def get_session_info(self, year: int, race: int, session_type: str = 'R') -> Dict[str, Any]:
        """Get general session information."""
        try:
            session = self.get_session(year, race, session_type)
            
            info = {
                'session_name': session.name,
                'session_type': session_type,
                'date': session.date,
                'circuit_name': session.event.get('EventName', 'Unknown'),
                'circuit_key': session.event.get('EventKey', 'unknown'),
                'country': session.event.get('Country', 'Unknown'),
                'location': session.event.get('Location', 'Unknown'),
                'total_laps': len(session.laps) if hasattr(session, 'laps') else 0,
                'drivers': list(session.drivers) if hasattr(session, 'drivers') else []
            }
            
            return info
            
        except FastF1Error:
            raise
        except Exception as e:
            raise FastF1Error(f"Error getting session info: {e}")
    
    def validate_session_data(self, year: int, race: int, session_type: str = 'R') -> bool:
        """Validate that session data is available and complete."""
        try:
            session = self.get_session(year, race, session_type)
            
            # Check if session has basic data
            if not hasattr(session, 'laps') or session.laps.empty:
                logger.warning(f"No lap data for {year} round {race} {session_type}")
                return False
            
            # Check if we have drivers
            if not hasattr(session, 'drivers') or len(session.drivers) == 0:
                logger.warning(f"No driver data for {year} round {race} {session_type}")
                return False
            
            # Check for reasonable number of drivers
            num_drivers = len(session.drivers)
            if num_drivers < config.data.min_drivers_per_race:
                logger.warning(f"Too few drivers ({num_drivers}) for {year} round {race}")
                return False
            
            return True
            
        except FastF1Error:
            logger.error(f"FastF1 error validating session: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating session data: {e}")
            return False
    
    def get_available_sessions(self, year: int, race: int) -> List[str]:
        """Get list of available session types for a race."""
        available_sessions = []
        
        for session_type in ['FP1', 'FP2', 'FP3', 'Q', 'R']:
            try:
                session = fastf1.get_session(year, race, session_type)
                # Try to load minimal data to check availability
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                available_sessions.append(session_type)
            except Exception as e:
                logger.debug(f"Session {session_type} not available for {year}/{race}: {e}")
                continue
        
        return available_sessions