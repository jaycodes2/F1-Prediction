"""
Comprehensive data validation system for F1 race data.
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter

from ..models.data_models import RaceData, RaceResult, QualifyingResult, WeatherData
from ..config import config


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None, severity: str = "error"):
        self.message = message
        self.field = field
        self.severity = severity  # "error", "warning", "info"
        super().__init__(self.message)


class ValidationResult:
    """Result of data validation."""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.info: List[ValidationError] = []
        self.stats: Dict[str, Any] = {}
    
    def add_error(self, message: str, field: str = None):
        """Add a validation error."""
        error = ValidationError(message, field, "error")
        self.errors.append(error)
        self.is_valid = False
        logger.error(f"Validation error: {message}")
    
    def add_warning(self, message: str, field: str = None):
        """Add a validation warning."""
        warning = ValidationError(message, field, "warning")
        self.warnings.append(warning)
        logger.warning(f"Validation warning: {message}")
    
    def add_info(self, message: str, field: str = None):
        """Add validation info."""
        info = ValidationError(message, field, "info")
        self.info.append(info)
        logger.info(f"Validation info: {message}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'info_count': len(self.info),
            'stats': self.stats
        }


class DataValidator:
    """Comprehensive data validator for F1 race data."""
    
    def __init__(self):
        self.min_drivers = config.data.min_drivers_per_race
        self.max_drivers = config.data.max_drivers_per_race
        self.max_missing_percentage = config.data.max_missing_percentage
    
    def validate_race_data(self, data: RaceData) -> ValidationResult:
        """Perform comprehensive validation of race data."""
        result = ValidationResult()
        
        try:
            # Basic structure validation
            self._validate_basic_structure(data, result)
            
            # Race metadata validation
            self._validate_race_metadata(data, result)
            
            # Results validation
            self._validate_race_results(data, result)
            
            # Qualifying validation
            self._validate_qualifying_results(data, result)
            
            # Weather validation
            self._validate_weather_data(data, result)
            
            # Cross-validation between different data sections
            self._validate_data_consistency(data, result)
            
            # Statistical validation
            self._validate_statistical_patterns(data, result)
            
            # Calculate validation statistics
            self._calculate_validation_stats(data, result)
            
        except Exception as e:
            result.add_error(f"Unexpected validation error: {e}")
        
        return result
    
    def _validate_basic_structure(self, data: RaceData, result: ValidationResult):
        """Validate basic data structure."""
        if not isinstance(data, RaceData):
            result.add_error("Data is not a RaceData instance", "structure")
            return
        
        # Check required fields
        required_fields = ['season', 'round', 'circuit_id', 'race_name', 'date', 'results', 'qualifying', 'weather']
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field) is None:
                result.add_error(f"Missing required field: {field}", field)
    
    def _validate_race_metadata(self, data: RaceData, result: ValidationResult):
        """Validate race metadata."""
        # Season validation
        current_year = datetime.now().year
        if data.season < 1950 or data.season > current_year + 1:
            result.add_error(f"Invalid season: {data.season}", "season")
        
        # Round validation
        if data.round < 1 or data.round > 30:  # F1 seasons typically have 15-25 races
            result.add_warning(f"Unusual round number: {data.round}", "round")
        
        # Circuit ID validation
        if not data.circuit_id or len(data.circuit_id) < 3:
            result.add_error("Invalid circuit_id", "circuit_id")
        
        # Race name validation
        if not data.race_name or len(data.race_name) < 5:
            result.add_error("Invalid race_name", "race_name")
        
        # Date validation
        if not isinstance(data.date, datetime):
            result.add_error("Invalid date format", "date")
        else:
            # Check if date is reasonable
            min_date = datetime(1950, 1, 1)
            max_date = datetime.now() + timedelta(days=365)
            if data.date < min_date or data.date > max_date:
                result.add_warning(f"Unusual race date: {data.date}", "date")
    
    def _validate_race_results(self, data: RaceData, result: ValidationResult):
        """Validate race results."""
        if not data.results:
            result.add_error("No race results found", "results")
            return
        
        # Check number of drivers
        num_drivers = len(data.results)
        if num_drivers < self.min_drivers:
            result.add_error(f"Too few drivers: {num_drivers} (minimum: {self.min_drivers})", "results")
        elif num_drivers > self.max_drivers:
            result.add_warning(f"Too many drivers: {num_drivers} (maximum: {self.max_drivers})", "results")
        
        # Check for duplicate drivers
        driver_ids = [r.driver_id for r in data.results]
        duplicates = [driver for driver, count in Counter(driver_ids).items() if count > 1]
        if duplicates:
            result.add_error(f"Duplicate drivers found: {duplicates}", "results")
        
        # Check for duplicate positions
        positions = [r.final_position for r in data.results if r.final_position < 999]
        duplicate_positions = [pos for pos, count in Counter(positions).items() if count > 1]
        if duplicate_positions:
            result.add_error(f"Duplicate finishing positions: {duplicate_positions}", "results")
        
        # Validate individual results
        for i, race_result in enumerate(data.results):
            self._validate_single_race_result(race_result, i, result)
    
    def _validate_single_race_result(self, race_result: RaceResult, index: int, result: ValidationResult):
        """Validate a single race result."""
        prefix = f"results[{index}]"
        
        # Driver ID validation
        if not race_result.driver_id or len(race_result.driver_id) < 3:
            result.add_error(f"Invalid driver_id: {race_result.driver_id}", f"{prefix}.driver_id")
        
        # Constructor ID validation
        if not race_result.constructor_id or len(race_result.constructor_id) < 3:
            result.add_error(f"Invalid constructor_id: {race_result.constructor_id}", f"{prefix}.constructor_id")
        
        # Grid position validation
        if race_result.grid_position < 1 or race_result.grid_position > 30:
            result.add_warning(f"Unusual grid position: {race_result.grid_position}", f"{prefix}.grid_position")
        
        # Final position validation
        if race_result.final_position < 1 or race_result.final_position > 30:
            if race_result.final_position != 999:  # 999 is used for DNF
                result.add_warning(f"Unusual final position: {race_result.final_position}", f"{prefix}.final_position")
        
        # Points validation
        if race_result.points < 0 or race_result.points > 30:  # Max points in F1 is 26 (25 + fastest lap)
            result.add_warning(f"Unusual points: {race_result.points}", f"{prefix}.points")
        
        # Laps completed validation
        if race_result.laps_completed < 0 or race_result.laps_completed > 100:
            result.add_warning(f"Unusual laps completed: {race_result.laps_completed}", f"{prefix}.laps_completed")
    
    def _validate_qualifying_results(self, data: RaceData, result: ValidationResult):
        """Validate qualifying results."""
        if not data.qualifying:
            result.add_warning("No qualifying results found", "qualifying")
            return
        
        # Check for duplicate drivers in qualifying
        qual_driver_ids = [q.driver_id for q in data.qualifying]
        duplicates = [driver for driver, count in Counter(qual_driver_ids).items() if count > 1]
        if duplicates:
            result.add_error(f"Duplicate drivers in qualifying: {duplicates}", "qualifying")
        
        # Check for duplicate positions
        qual_positions = [q.position for q in data.qualifying]
        duplicate_positions = [pos for pos, count in Counter(qual_positions).items() if count > 1]
        if duplicate_positions:
            result.add_error(f"Duplicate qualifying positions: {duplicate_positions}", "qualifying")
        
        # Validate individual qualifying results
        for i, qual_result in enumerate(data.qualifying):
            self._validate_single_qualifying_result(qual_result, i, result)
    
    def _validate_single_qualifying_result(self, qual_result: QualifyingResult, index: int, result: ValidationResult):
        """Validate a single qualifying result."""
        prefix = f"qualifying[{index}]"
        
        # Position validation
        if qual_result.position < 1 or qual_result.position > 30:
            result.add_warning(f"Unusual qualifying position: {qual_result.position}", f"{prefix}.position")
        
        # Time validation
        for session, time_obj in [("Q1", qual_result.q1_time), ("Q2", qual_result.q2_time), ("Q3", qual_result.q3_time)]:
            if time_obj:
                # Check for reasonable lap times (between 60 and 150 seconds)
                time_seconds = time_obj.time_ms / 1000
                if time_seconds < 60 or time_seconds > 150:
                    result.add_warning(f"Unusual {session} time: {time_seconds:.3f}s", f"{prefix}.{session.lower()}_time")
    
    def _validate_weather_data(self, data: RaceData, result: ValidationResult):
        """Validate weather data."""
        weather = data.weather
        
        if not isinstance(weather, WeatherData):
            result.add_error("Invalid weather data structure", "weather")
            return
        
        # Temperature validation
        if weather.temperature < -20 or weather.temperature > 60:
            result.add_warning(f"Unusual air temperature: {weather.temperature}°C", "weather.temperature")
        
        if weather.track_temp < -20 or weather.track_temp > 80:
            result.add_warning(f"Unusual track temperature: {weather.track_temp}°C", "weather.track_temp")
        
        # Humidity validation
        if weather.humidity < 0 or weather.humidity > 100:
            result.add_error(f"Invalid humidity: {weather.humidity}%", "weather.humidity")
        
        # Pressure validation
        if weather.pressure < 900 or weather.pressure > 1100:
            result.add_warning(f"Unusual pressure: {weather.pressure} mbar", "weather.pressure")
        
        # Wind validation
        if weather.wind_speed < 0 or weather.wind_speed > 100:
            result.add_warning(f"Unusual wind speed: {weather.wind_speed} km/h", "weather.wind_speed")
        
        if weather.wind_direction < 0 or weather.wind_direction > 360:
            result.add_error(f"Invalid wind direction: {weather.wind_direction}°", "weather.wind_direction")
    
    def _validate_data_consistency(self, data: RaceData, result: ValidationResult):
        """Validate consistency between different data sections."""
        # Check if drivers in results match drivers in qualifying
        result_drivers = set(r.driver_id for r in data.results)
        qual_drivers = set(q.driver_id for q in data.qualifying)
        
        missing_in_qualifying = result_drivers - qual_drivers
        missing_in_results = qual_drivers - result_drivers
        
        if missing_in_qualifying:
            result.add_warning(f"Drivers in results but not in qualifying: {missing_in_qualifying}", "consistency")
        
        if missing_in_results:
            result.add_warning(f"Drivers in qualifying but not in results: {missing_in_results}", "consistency")
        
        # Check constructor consistency
        driver_constructors = {}
        for race_result in data.results:
            driver_constructors[race_result.driver_id] = race_result.constructor_id
        
        for qual_result in data.qualifying:
            if qual_result.driver_id in driver_constructors:
                if qual_result.constructor_id != driver_constructors[qual_result.driver_id]:
                    result.add_error(
                        f"Constructor mismatch for {qual_result.driver_id}: "
                        f"qualifying={qual_result.constructor_id}, race={driver_constructors[qual_result.driver_id]}",
                        "consistency"
                    )
    
    def _validate_statistical_patterns(self, data: RaceData, result: ValidationResult):
        """Validate statistical patterns in the data."""
        if not data.results:
            return
        
        # Check points distribution
        total_points = sum(r.points for r in data.results)
        if total_points == 0:
            result.add_warning("No points awarded in race", "statistics")
        elif total_points > 200:  # Typical race awards ~150 points total
            result.add_warning(f"Unusually high total points: {total_points}", "statistics")
        
        # Check finishing rate
        finished_drivers = sum(1 for r in data.results if r.final_position < 999)
        finish_rate = finished_drivers / len(data.results)
        if finish_rate < 0.5:
            result.add_warning(f"Low finish rate: {finish_rate:.1%}", "statistics")
        
        # Check lap completion patterns
        max_laps = max(r.laps_completed for r in data.results)
        if max_laps < 30:
            result.add_warning(f"Unusually short race: {max_laps} laps", "statistics")
        elif max_laps > 100:
            result.add_warning(f"Unusually long race: {max_laps} laps", "statistics")
    
    def _calculate_validation_stats(self, data: RaceData, result: ValidationResult):
        """Calculate validation statistics."""
        stats = {
            'num_drivers': len(data.results),
            'num_qualifying': len(data.qualifying),
            'finish_rate': 0,
            'points_awarded': 0,
            'max_laps': 0,
            'weather_enhanced': False
        }
        
        if data.results:
            finished = sum(1 for r in data.results if r.final_position < 999)
            stats['finish_rate'] = finished / len(data.results)
            stats['points_awarded'] = sum(r.points for r in data.results)
            stats['max_laps'] = max(r.laps_completed for r in data.results)
        
        # Check if weather data appears to be enhanced (not default values)
        default_weather = WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        stats['weather_enhanced'] = (
            data.weather.temperature != default_weather.temperature or
            data.weather.humidity != default_weather.humidity or
            data.weather.pressure != default_weather.pressure
        )
        
        result.stats = stats
    
    def validate_multiple_races(self, races: List[RaceData]) -> Dict[str, Any]:
        """Validate multiple races and provide aggregate statistics."""
        if not races:
            return {'error': 'No races provided for validation'}
        
        aggregate_result = {
            'total_races': len(races),
            'valid_races': 0,
            'races_with_warnings': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'common_issues': Counter(),
            'season_stats': {},
            'validation_details': []
        }
        
        for race in races:
            validation_result = self.validate_race_data(race)
            
            # Update aggregate stats
            if validation_result.is_valid:
                aggregate_result['valid_races'] += 1
            
            if validation_result.warnings:
                aggregate_result['races_with_warnings'] += 1
            
            aggregate_result['total_errors'] += len(validation_result.errors)
            aggregate_result['total_warnings'] += len(validation_result.warnings)
            
            # Track common issues
            for error in validation_result.errors:
                aggregate_result['common_issues'][error.message] += 1
            for warning in validation_result.warnings:
                aggregate_result['common_issues'][warning.message] += 1
            
            # Season statistics
            season = race.season
            if season not in aggregate_result['season_stats']:
                aggregate_result['season_stats'][season] = {
                    'races': 0,
                    'valid_races': 0,
                    'total_drivers': 0
                }
            
            aggregate_result['season_stats'][season]['races'] += 1
            if validation_result.is_valid:
                aggregate_result['season_stats'][season]['valid_races'] += 1
            aggregate_result['season_stats'][season]['total_drivers'] += len(race.results)
            
            # Store individual validation details
            aggregate_result['validation_details'].append({
                'season': race.season,
                'round': race.round,
                'circuit_id': race.circuit_id,
                'is_valid': validation_result.is_valid,
                'error_count': len(validation_result.errors),
                'warning_count': len(validation_result.warnings),
                'stats': validation_result.stats
            })
        
        return aggregate_result