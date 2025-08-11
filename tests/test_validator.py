"""
Tests for data validator.
"""
import pytest
from datetime import datetime, timedelta

from src.data.validator import DataValidator, ValidationResult, ValidationError
from src.models.data_models import (
    RaceData, RaceResult, QualifyingResult, 
    WeatherData, LapTime, RaceStatus
)


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_validation_result_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.info) == 0
        assert result.stats == {}
    
    def test_add_error(self):
        """Test adding validation errors."""
        result = ValidationResult()
        result.add_error("Test error", "test_field")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].message == "Test error"
        assert result.errors[0].field == "test_field"
        assert result.errors[0].severity == "error"
    
    def test_add_warning(self):
        """Test adding validation warnings."""
        result = ValidationResult()
        result.add_warning("Test warning", "test_field")
        
        assert result.is_valid is True  # Warnings don't invalidate
        assert len(result.warnings) == 1
        assert result.warnings[0].message == "Test warning"
        assert result.warnings[0].severity == "warning"
    
    def test_get_summary(self):
        """Test getting validation summary."""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_warning("Warning 1")
        result.add_info("Info 1")
        result.stats = {'test': 'value'}
        
        summary = result.get_summary()
        assert summary['is_valid'] is False
        assert summary['error_count'] == 1
        assert summary['warning_count'] == 1
        assert summary['info_count'] == 1
        assert summary['stats'] == {'test': 'value'}


class TestDataValidator:
    """Tests for DataValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a test validator."""
        return DataValidator()
    
    @pytest.fixture
    def valid_race_data(self):
        """Create valid race data for testing."""
        weather = WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        
        results = []
        qualifying = []
        
        # Create 20 drivers
        for i in range(20):
            driver_id = f"driver_{i:02d}"
            constructor_id = f"team_{i//2:02d}"  # 2 drivers per team
            
            race_result = RaceResult(
                driver_id=driver_id,
                constructor_id=constructor_id,
                grid_position=i + 1,
                final_position=i + 1,
                points=max(0, 25 - i) if i < 10 else 0,
                fastest_lap=LapTime(90000 + i * 100, 45) if i == 0 else None,
                status=RaceStatus.FINISHED,
                laps_completed=57
            )
            results.append(race_result)
            
            qual_result = QualifyingResult(
                driver_id=driver_id,
                constructor_id=constructor_id,
                position=i + 1,
                q1_time=LapTime(91000 + i * 100, 1),
                q2_time=LapTime(90500 + i * 100, 1) if i < 15 else None,
                q3_time=LapTime(90000 + i * 100, 1) if i < 10 else None
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
    
    def test_validate_valid_race_data(self, validator, valid_race_data):
        """Test validation of valid race data."""
        result = validator.validate_race_data(valid_race_data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.stats['num_drivers'] == 20
        assert result.stats['num_qualifying'] == 20
        assert result.stats['finish_rate'] == 1.0
        assert result.stats['points_awarded'] > 0
    
    def test_validate_basic_structure_errors(self, validator):
        """Test validation of basic structure errors."""
        # Test with None
        result = validator.validate_race_data(None)
        assert result.is_valid is False
        assert any("not a RaceData instance" in error.message for error in result.errors)
    
    def test_validate_race_metadata(self, validator, valid_race_data):
        """Test race metadata validation."""
        # Test invalid season
        valid_race_data.season = 1900
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Invalid season" in error.message for error in result.errors)
        
        # Reset and test invalid round
        valid_race_data.season = 2024
        valid_race_data.round = 50
        result = validator.validate_race_data(valid_race_data)
        assert any("Unusual round number" in warning.message for warning in result.warnings)
        
        # Reset and test invalid circuit_id
        valid_race_data.round = 1
        valid_race_data.circuit_id = "x"
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Invalid circuit_id" in error.message for error in result.errors)
    
    def test_validate_race_results_errors(self, validator, valid_race_data):
        """Test race results validation errors."""
        # Test too few drivers
        valid_race_data.results = valid_race_data.results[:10]  # Only 10 drivers
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Too few drivers" in error.message for error in result.errors)
        
        # Test duplicate drivers
        valid_race_data.results = valid_race_data.results[:18]  # Reset to valid count
        valid_race_data.results[1].driver_id = valid_race_data.results[0].driver_id  # Duplicate
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Duplicate drivers found" in error.message for error in result.errors)
        
        # Test duplicate positions
        valid_race_data.results[1].driver_id = "unique_driver"  # Fix duplicate driver
        valid_race_data.results[1].final_position = valid_race_data.results[0].final_position  # Duplicate position
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Duplicate finishing positions" in error.message for error in result.errors)
    
    def test_validate_single_race_result(self, validator, valid_race_data):
        """Test individual race result validation."""
        # Test invalid driver_id
        valid_race_data.results[0].driver_id = "x"
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Invalid driver_id" in error.message for error in result.errors)
        
        # Test unusual points
        valid_race_data.results[0].driver_id = "valid_driver"
        valid_race_data.results[0].points = 50  # Too many points
        result = validator.validate_race_data(valid_race_data)
        assert any("Unusual points" in warning.message for warning in result.warnings)
    
    def test_validate_qualifying_results(self, validator, valid_race_data):
        """Test qualifying results validation."""
        # Test duplicate drivers in qualifying
        valid_race_data.qualifying[1].driver_id = valid_race_data.qualifying[0].driver_id
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Duplicate drivers in qualifying" in error.message for error in result.errors)
        
        # Test unusual qualifying time
        valid_race_data.qualifying[1].driver_id = "unique_qual_driver"
        valid_race_data.qualifying[0].q1_time = LapTime(30000, 1)  # 30 seconds - too fast
        result = validator.validate_race_data(valid_race_data)
        assert any("Unusual Q1 time" in warning.message for warning in result.warnings)
    
    def test_validate_weather_data(self, validator, valid_race_data):
        """Test weather data validation."""
        # Test invalid temperature
        valid_race_data.weather.temperature = -50
        result = validator.validate_race_data(valid_race_data)
        assert any("Unusual air temperature" in warning.message for warning in result.warnings)
        
        # Test invalid humidity
        valid_race_data.weather.temperature = 25  # Reset
        valid_race_data.weather.humidity = 150  # Invalid
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Invalid humidity" in error.message for error in result.errors)
        
        # Test invalid wind direction
        valid_race_data.weather.humidity = 60  # Reset
        valid_race_data.weather.wind_direction = 400  # Invalid
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Invalid wind direction" in error.message for error in result.errors)
    
    def test_validate_data_consistency(self, validator, valid_race_data):
        """Test data consistency validation."""
        # Remove a driver from qualifying but keep in results
        valid_race_data.qualifying = valid_race_data.qualifying[1:]  # Remove first driver
        result = validator.validate_race_data(valid_race_data)
        assert any("Drivers in results but not in qualifying" in warning.message for warning in result.warnings)
        
        # Test constructor mismatch
        valid_race_data.qualifying = valid_race_data.qualifying + [QualifyingResult(
            driver_id=valid_race_data.results[0].driver_id,
            constructor_id="different_constructor",
            position=21,
            q1_time=None, q2_time=None, q3_time=None
        )]
        result = validator.validate_race_data(valid_race_data)
        assert result.is_valid is False
        assert any("Constructor mismatch" in error.message for error in result.errors)
    
    def test_validate_statistical_patterns(self, validator, valid_race_data):
        """Test statistical pattern validation."""
        # Test no points awarded
        for result in valid_race_data.results:
            result.points = 0
        result = validator.validate_race_data(valid_race_data)
        assert any("No points awarded" in warning.message for warning in result.warnings)
        
        # Test low finish rate
        for i, race_result in enumerate(valid_race_data.results):
            if i > 5:  # Only first 6 finish
                race_result.final_position = 999
                race_result.status = RaceStatus.DNF
        result = validator.validate_race_data(valid_race_data)
        assert any("Low finish rate" in warning.message for warning in result.warnings)
    
    def test_validate_multiple_races(self, validator, valid_race_data):
        """Test validation of multiple races."""
        # Create multiple races
        races = [valid_race_data]
        
        # Add an invalid race
        invalid_race = RaceData(
            season=2024,
            round=2,
            circuit_id='test',
            race_name='Test Race',
            date=datetime(2024, 3, 9),
            results=[],  # No results - invalid
            qualifying=[],
            weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        )
        races.append(invalid_race)
        
        aggregate_result = validator.validate_multiple_races(races)
        
        assert aggregate_result['total_races'] == 2
        assert aggregate_result['valid_races'] == 1
        assert aggregate_result['total_errors'] > 0
        assert 2024 in aggregate_result['season_stats']
        assert len(aggregate_result['validation_details']) == 2
    
    def test_validate_empty_race_list(self, validator):
        """Test validation of empty race list."""
        result = validator.validate_multiple_races([])
        assert 'error' in result
        assert result['error'] == 'No races provided for validation'