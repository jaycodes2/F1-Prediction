"""
Core interfaces for the F1 Race Prediction System.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .data_models import (
    RaceData, RaceParameters, RacePrediction, 
    DriverFeatures, PositionPrediction
)


class DataCollector(ABC):
    """Interface for data collection services."""
    
    @abstractmethod
    def collect_race_data(self, season: int, round: int) -> RaceData:
        """Collect data for a specific race."""
        pass
    
    @abstractmethod
    def collect_historical_data(self, start_year: int, end_year: int) -> List[RaceData]:
        """Collect historical data for a range of years."""
        pass
    
    @abstractmethod
    def validate_data(self, data: RaceData) -> bool:
        """Validate collected data for quality and completeness."""
        pass


class FeatureEngineer(ABC):
    """Interface for feature engineering services."""
    
    @abstractmethod
    def extract_driver_features(self, race_data: List[RaceData], driver_id: str) -> DriverFeatures:
        """Extract features for a specific driver."""
        pass
    
    @abstractmethod
    def calculate_rolling_stats(self, races: List[RaceData], window: int) -> Dict[str, Any]:
        """Calculate rolling statistics over a window of races."""
        pass
    
    @abstractmethod
    def encode_categorical_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Encode categorical variables for ML models."""
        pass


class ModelTrainer(ABC):
    """Interface for model training services."""
    
    @abstractmethod
    def train_model(self, features: List[Dict[str, Any]], targets: List[int]) -> Any:
        """Train a prediction model."""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, test_features: List[Dict[str, Any]], 
                      test_targets: List[int]) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def tune_hyperparameters(self, model_type: str, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Tune model hyperparameters."""
        pass


class PredictionEngine(ABC):
    """Interface for prediction services."""
    
    @abstractmethod
    def predict_race(self, race_params: RaceParameters) -> RacePrediction:
        """Generate predictions for a single race."""
        pass
    
    @abstractmethod
    def predict_batch(self, batch_params: List[RaceParameters]) -> List[RacePrediction]:
        """Generate predictions for multiple races."""
        pass
    
    @abstractmethod
    def get_prediction_confidence(self, prediction: RacePrediction) -> float:
        """Calculate overall confidence score for a prediction."""
        pass


class DataStorage(ABC):
    """Interface for data storage services."""
    
    @abstractmethod
    def save_race_data(self, data: RaceData) -> bool:
        """Save race data to storage."""
        pass
    
    @abstractmethod
    def load_race_data(self, season: int, round: int) -> Optional[RaceData]:
        """Load race data from storage."""
        pass
    
    @abstractmethod
    def save_features(self, features: Dict[str, Any], identifier: str) -> bool:
        """Save engineered features to storage."""
        pass
    
    @abstractmethod
    def load_features(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load engineered features from storage."""
        pass


class WebInterface(ABC):
    """Interface for web interface components."""
    
    @abstractmethod
    def render_input_form(self) -> Dict[str, Any]:
        """Render and collect user inputs."""
        pass
    
    @abstractmethod
    def display_predictions(self, predictions: RacePrediction) -> None:
        """Display prediction results."""
        pass
    
    @abstractmethod
    def generate_visualizations(self, data: Dict[str, Any]) -> Any:
        """Generate charts and visualizations."""
        pass