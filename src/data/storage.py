"""
Data storage system for F1 race data.
"""
import os
import json
import pickle
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from ..models.data_models import RaceData
from ..models.interfaces import DataStorage
from ..config import config


logger = logging.getLogger(__name__)


class DataStorageError(Exception):
    """Custom exception for data storage errors."""
    def __init__(self, message: str, operation: str = None):
        self.message = message
        self.operation = operation
        super().__init__(self.message)


class FileDataStorage(DataStorage):
    """
    File-based data storage implementation.
    Stores race data as JSON files with metadata and pickle files for complex objects.
    """
    
    def __init__(self):
        self.raw_data_dir = Path(config.data.raw_data_dir)
        self.processed_data_dir = Path(config.data.processed_data_dir)
        self.features_dir = self.processed_data_dir / "features"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.raw_data_dir, self.processed_data_dir, self.features_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def _get_race_filename(self, season: int, round: int, format: str = "json") -> Path:
        """Generate filename for race data."""
        filename = f"race_{season}_{round:02d}.{format}"
        return self.raw_data_dir / str(season) / filename
    
    def _get_metadata_filename(self, season: int, round: int) -> Path:
        """Generate filename for race metadata."""
        filename = f"race_{season}_{round:02d}_metadata.json"
        return self.raw_data_dir / str(season) / filename
    
    def _serialize_race_data(self, data: RaceData) -> Dict[str, Any]:
        """Convert RaceData to serializable dictionary."""
        from ..models.data_models import RaceStatus
        from enum import Enum
        
        def convert_to_serializable(obj):
            if obj is None:
                return None
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif hasattr(obj, '__dict__'):
                # Handle dataclass objects
                result = {}
                for key, value in obj.__dict__.items():
                    result[key] = convert_to_serializable(value)
                return result
            else:
                return str(obj)  # Fallback to string representation
        
        return convert_to_serializable(data)
    
    def _deserialize_race_data(self, data_dict: Dict[str, Any]) -> RaceData:
        """Convert dictionary back to RaceData object."""
        from ..models.data_models import (
            RaceData, RaceResult, QualifyingResult, 
            WeatherData, LapTime, RaceStatus
        )
        
        # Deserialize weather data
        weather_dict = data_dict['weather']
        weather = WeatherData(**weather_dict)
        
        # Deserialize race results
        results = []
        for result_dict in data_dict['results']:
            # Handle fastest lap
            fastest_lap = None
            if result_dict['fastest_lap']:
                fastest_lap = LapTime(**result_dict['fastest_lap'])
            
            # Handle race status
            status = RaceStatus(result_dict['status'])
            
            result = RaceResult(
                driver_id=result_dict['driver_id'],
                constructor_id=result_dict['constructor_id'],
                grid_position=result_dict['grid_position'],
                final_position=result_dict['final_position'],
                points=result_dict['points'],
                fastest_lap=fastest_lap,
                status=status,
                laps_completed=result_dict['laps_completed']
            )
            results.append(result)
        
        # Deserialize qualifying results
        qualifying = []
        for qual_dict in data_dict['qualifying']:
            # Handle lap times
            q1_time = LapTime(**qual_dict['q1_time']) if qual_dict['q1_time'] else None
            q2_time = LapTime(**qual_dict['q2_time']) if qual_dict['q2_time'] else None
            q3_time = LapTime(**qual_dict['q3_time']) if qual_dict['q3_time'] else None
            
            qual = QualifyingResult(
                driver_id=qual_dict['driver_id'],
                constructor_id=qual_dict['constructor_id'],
                position=qual_dict['position'],
                q1_time=q1_time,
                q2_time=q2_time,
                q3_time=q3_time
            )
            qualifying.append(qual)
        
        # Parse date
        date = datetime.fromisoformat(data_dict['date'])
        
        return RaceData(
            season=data_dict['season'],
            round=data_dict['round'],
            circuit_id=data_dict['circuit_id'],
            race_name=data_dict['race_name'],
            date=date,
            results=results,
            qualifying=qualifying,
            weather=weather
        )
    
    def save_race_data(self, data: RaceData) -> bool:
        """Save race data to storage."""
        try:
            # Ensure season directory exists
            season_dir = self.raw_data_dir / str(data.season)
            season_dir.mkdir(parents=True, exist_ok=True)
            
            # Serialize and save race data
            race_file = self._get_race_filename(data.season, data.round)
            serialized_data = self._serialize_race_data(data)
            
            with open(race_file, 'w', encoding='utf-8') as f:
                json.dump(serialized_data, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            metadata = {
                'season': data.season,
                'round': data.round,
                'circuit_id': data.circuit_id,
                'race_name': data.race_name,
                'date': data.date.isoformat(),
                'num_drivers': len(data.results),
                'num_qualifying': len(data.qualifying),
                'saved_at': datetime.now().isoformat(),
                'file_size': race_file.stat().st_size
            }
            
            metadata_file = self._get_metadata_filename(data.season, data.round)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved race data for {data.season} round {data.round}")
            return True
            
        except Exception as e:
            season_info = getattr(data, 'season', 'unknown') if data else 'unknown'
            round_info = getattr(data, 'round', 'unknown') if data else 'unknown'
            logger.error(f"Failed to save race data for {season_info}/{round_info}: {e}")
            raise DataStorageError(f"Failed to save race data: {e}", "save_race_data")
    
    def load_race_data(self, season: int, round: int) -> Optional[RaceData]:
        """Load race data from storage."""
        try:
            race_file = self._get_race_filename(season, round)
            
            if not race_file.exists():
                logger.debug(f"Race data file not found: {race_file}")
                return None
            
            with open(race_file, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            race_data = self._deserialize_race_data(data_dict)
            logger.debug(f"Loaded race data for {season} round {round}")
            return race_data
            
        except Exception as e:
            logger.error(f"Failed to load race data for {season}/{round}: {e}")
            raise DataStorageError(f"Failed to load race data: {e}", "load_race_data")
    
    def save_features(self, features: Dict[str, Any], identifier: str) -> bool:
        """Save engineered features to storage."""
        try:
            features_file = self.features_dir / f"{identifier}.pkl"
            
            with open(features_file, 'wb') as f:
                pickle.dump(features, f)
            
            # Save metadata
            metadata = {
                'identifier': identifier,
                'feature_count': len(features),
                'saved_at': datetime.now().isoformat(),
                'file_size': features_file.stat().st_size
            }
            
            metadata_file = self.features_dir / f"{identifier}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved features for {identifier}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save features for {identifier}: {e}")
            raise DataStorageError(f"Failed to save features: {e}", "save_features")
    
    def load_features(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load engineered features from storage."""
        try:
            features_file = self.features_dir / f"{identifier}.pkl"
            
            if not features_file.exists():
                logger.debug(f"Features file not found: {features_file}")
                return None
            
            with open(features_file, 'rb') as f:
                features = pickle.load(f)
            
            logger.debug(f"Loaded features for {identifier}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to load features for {identifier}: {e}")
            raise DataStorageError(f"Failed to load features: {e}", "load_features")
    
    def list_available_races(self, season: int = None) -> List[Dict[str, Any]]:
        """List all available race data."""
        available_races = []
        
        try:
            if season:
                season_dirs = [self.raw_data_dir / str(season)]
            else:
                season_dirs = [d for d in self.raw_data_dir.iterdir() if d.is_dir()]
            
            for season_dir in season_dirs:
                if not season_dir.exists():
                    continue
                
                for metadata_file in season_dir.glob("*_metadata.json"):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        available_races.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
                        continue
            
            # Sort by season and round
            available_races.sort(key=lambda x: (x['season'], x['round']))
            return available_races
            
        except Exception as e:
            logger.error(f"Failed to list available races: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {
                'total_races': 0,
                'total_features': 0,
                'storage_size_mb': 0,
                'seasons': [],
                'last_updated': None
            }
            
            # Count races and calculate size
            total_size = 0
            seasons = set()
            
            for race_file in self.raw_data_dir.rglob("race_*.json"):
                # Skip metadata files for race counting
                if "_metadata" in race_file.name:
                    total_size += race_file.stat().st_size  # But include in size calculation
                    continue
                    
                stats['total_races'] += 1
                total_size += race_file.stat().st_size
                
                # Extract season from path
                season = int(race_file.parent.name)
                seasons.add(season)
            
            # Count features
            for features_file in self.features_dir.glob("*.pkl"):
                stats['total_features'] += 1
                total_size += features_file.stat().st_size
            
            # Also count feature metadata files in size
            for metadata_file in self.features_dir.glob("*_metadata.json"):
                total_size += metadata_file.stat().st_size
            
            stats['storage_size_mb'] = round(total_size / (1024 * 1024), 2)
            stats['seasons'] = sorted(list(seasons))
            
            # Find most recent file
            all_files = list(self.raw_data_dir.rglob("*.json")) + list(self.features_dir.glob("*.pkl"))
            if all_files:
                most_recent = max(all_files, key=lambda f: f.stat().st_mtime)
                stats['last_updated'] = datetime.fromtimestamp(most_recent.stat().st_mtime).isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def cleanup_old_data(self, keep_seasons: int = 5) -> bool:
        """Clean up old data, keeping only the most recent seasons."""
        try:
            available_races = self.list_available_races()
            if not available_races:
                return True
            
            # Get unique seasons and sort
            seasons = sorted(set(race['season'] for race in available_races), reverse=True)
            
            if len(seasons) <= keep_seasons:
                logger.info("No old data to clean up")
                return True
            
            # Remove old seasons
            seasons_to_remove = seasons[keep_seasons:]
            removed_count = 0
            
            for season in seasons_to_remove:
                season_dir = self.raw_data_dir / str(season)
                if season_dir.exists():
                    import shutil
                    shutil.rmtree(season_dir)
                    removed_count += 1
                    logger.info(f"Removed data for season {season}")
            
            logger.info(f"Cleaned up {removed_count} old seasons")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False