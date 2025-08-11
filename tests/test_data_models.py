"""
Tests for core data models.
"""
import pytest
from datetime import datetime
from src.models.data_models import (
    RaceData, RaceResult, QualifyingResult, 
    WeatherData, LapTime, RaceStatus
)


def test_lap_time_creation():
    """Test LapTime dataclass creation."""
    lap_time = LapTime(time_ms=90000, lap_number=1)
    assert lap_time.time_ms == 90000
    assert lap_time.lap_number == 1


def test_weather_data_creation():
    """Test WeatherData dataclass creation."""
    weather = WeatherData(
        temperature=25.0,
        humidity=60.0,
        pressure=1013.25,
        wind_speed=10.0,
        wind_direction=180,
        rainfall=False,
        track_temp=35.0
    )
    assert weather.temperature == 25.0
    assert weather.rainfall is False


def test_race_result_creation():
    """Test RaceResult dataclass creation."""
    fastest_lap = LapTime(time_ms=85000, lap_number=45)
    result = RaceResult(
        driver_id="hamilton",
        constructor_id="mercedes",
        grid_position=1,
        final_position=1,
        points=25.0,
        fastest_lap=fastest_lap,
        status=RaceStatus.FINISHED,
        laps_completed=58
    )
    assert result.driver_id == "hamilton"
    assert result.points == 25.0
    assert result.status == RaceStatus.FINISHED


def test_race_data_creation():
    """Test RaceData dataclass creation."""
    weather = WeatherData(25.0, 60.0, 1013.25, 10.0, 180, False, 35.0)
    result = RaceResult(
        "hamilton", "mercedes", 1, 1, 25.0, 
        None, RaceStatus.FINISHED, 58
    )
    qualifying = QualifyingResult(
        "hamilton", "mercedes", 1, 
        LapTime(90000, 1), LapTime(89000, 1), LapTime(88000, 1)
    )
    
    race_data = RaceData(
        season=2024,
        round=1,
        circuit_id="bahrain",
        race_name="Bahrain Grand Prix",
        date=datetime(2024, 3, 2),
        results=[result],
        qualifying=[qualifying],
        weather=weather
    )
    
    assert race_data.season == 2024
    assert race_data.circuit_id == "bahrain"
    assert len(race_data.results) == 1
    assert len(race_data.qualifying) == 1