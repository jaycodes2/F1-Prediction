"""
Core data models for the F1 Race Prediction System.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum


class RaceStatus(Enum):
    FINISHED = "finished"
    DNF = "dnf"
    DSQ = "disqualified"
    DNS = "dns"


@dataclass
class LapTime:
    """Represents a lap time in milliseconds."""
    time_ms: int
    lap_number: int


@dataclass
class WeatherData:
    """Weather conditions during a race session."""
    temperature: float  # Celsius
    humidity: float  # Percentage
    pressure: float  # mbar
    wind_speed: float  # km/h
    wind_direction: int  # degrees
    rainfall: bool
    track_temp: float  # Celsius


@dataclass
class QualifyingResult:
    """Qualifying session result for a driver."""
    driver_id: str
    constructor_id: str
    position: int
    q1_time: Optional[LapTime]
    q2_time: Optional[LapTime]
    q3_time: Optional[LapTime]


@dataclass
class RaceResult:
    """Race result for a single driver."""
    driver_id: str
    constructor_id: str
    grid_position: int
    final_position: int
    points: float
    fastest_lap: Optional[LapTime]
    status: RaceStatus
    laps_completed: int


@dataclass
class RaceData:
    """Complete race data including results and conditions."""
    season: int
    round: int
    circuit_id: str
    race_name: str
    date: datetime
    results: List[RaceResult]
    qualifying: List[QualifyingResult]
    weather: WeatherData


@dataclass
class DriverFeatures:
    """Engineered features for a driver."""
    driver_id: str
    recent_form: float  # Rolling average of recent finishes
    constructor_performance: float
    track_experience: int
    weather_performance: float
    qualifying_delta: float  # Difference from expected qualifying position
    championship_position: int
    points_total: int


@dataclass
class PositionPrediction:
    """Prediction for a single driver's finishing position."""
    driver_id: str
    predicted_position: int
    probability_distribution: List[float]  # Probability for each position
    expected_points: float
    confidence_score: float


@dataclass
class RacePrediction:
    """Complete race prediction with insights."""
    predicted_positions: List[PositionPrediction]
    confidence_scores: List[float]
    overtaking_probabilities: Dict[str, float]
    insights: List[str]
    prediction_timestamp: datetime


@dataclass
class RaceParameters:
    """Input parameters for race prediction."""
    circuit_id: str
    season: int
    drivers: List[str]  # Driver IDs
    constructors: List[str]  # Constructor IDs
    grid_positions: Dict[str, int]  # Driver ID -> Grid position
    weather: WeatherData
    qualifying_times: Optional[Dict[str, float]] = None