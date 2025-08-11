"""
Configuration management for the F1 Race Prediction System.
"""
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class APIConfig:
    """Configuration for external APIs."""
    ergast_base_url: str = "http://ergast.com/api/f1"
    ergast_rate_limit: int = 4  # requests per second
    fastf1_cache_dir: str = "data/fastf1_cache"
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 0.3


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Model-specific parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    
    lgb_n_estimators: int = 100
    lgb_learning_rate: float = 0.1
    lgb_num_leaves: int = 31


@dataclass
class DataConfig:
    """Configuration for data processing."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "models"
    
    # Feature engineering
    rolling_window_size: int = 5
    min_races_for_features: int = 3
    
    # Data validation
    max_missing_percentage: float = 0.1
    min_drivers_per_race: int = 18
    max_drivers_per_race: int = 22


@dataclass
class WebConfig:
    """Configuration for web interface."""
    app_title: str = "F1 Race Prediction System"
    page_icon: str = "🏁"
    layout: str = "wide"
    
    # Visualization settings
    chart_height: int = 400
    chart_width: int = 600
    color_palette: list = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                "#FF1801", "#AAAAAA", "#0600EF", "#FF8700",
                "#005AFF", "#00D2BE", "#FF073A", "#FFFFFF"
            ]


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.api = APIConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.web = WebConfig()
        
        # Override with environment variables if available
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        # API configuration
        if os.getenv("ERGAST_BASE_URL"):
            self.api.ergast_base_url = os.getenv("ERGAST_BASE_URL")
        
        if os.getenv("FASTF1_CACHE_DIR"):
            self.api.fastf1_cache_dir = os.getenv("FASTF1_CACHE_DIR")
        
        # Model configuration
        if os.getenv("RANDOM_STATE"):
            self.model.random_state = int(os.getenv("RANDOM_STATE"))
        
        # Data configuration
        if os.getenv("RAW_DATA_DIR"):
            self.data.raw_data_dir = os.getenv("RAW_DATA_DIR")
        
        if os.getenv("PROCESSED_DATA_DIR"):
            self.data.processed_data_dir = os.getenv("PROCESSED_DATA_DIR")
        
        if os.getenv("MODELS_DIR"):
            self.data.models_dir = os.getenv("MODELS_DIR")


# Global configuration instance
config = Config()