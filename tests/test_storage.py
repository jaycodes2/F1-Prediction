"""
Tests for data storage system.
"""
import pytest
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

from src.data.storage import FileDataStorage, DataStorageError
from src.models.data_models import (
    RaceData, RaceResult, QualifyingResult, 
    WeatherData, LapTime, RaceStatus
)


class TestFileDataStorage:
    """Tests for the FileDataStorage class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage instance."""
        temp_dir = tempfile.mkdtemp()
        
        with patch('src.data.storage.config') as mock_config:
            mock_config.data.raw_data_dir = str(Path(temp_dir) / "raw")
            mock_config.data.processed_data_dir = str(Path(temp_dir) / "processed")
            
            storage = FileDataStorage()
            
            yield storage
            
            # Cleanup
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_race_data(self):
        """Create sample race data for testing."""
        weather = WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        
        race_result = RaceResult(
            driver_id='hamilton',
            constructor_id='mercedes',
            grid_position=1,
            final_position=1,
            points=25.0,
            fastest_lap=LapTime(88500, 45),
            status=RaceStatus.FINISHED,
            laps_completed=57
        )
        
        qual_result = QualifyingResult(
            driver_id='hamilton',
            constructor_id='mercedes',
            position=1,
            q1_time=LapTime(90123, 1),
            q2_time=LapTime(89456, 1),
            q3_time=LapTime(88789, 1)
        )
        
        return RaceData(
            season=2024,
            round=1,
            circuit_id='bahrain',
            race_name='Bahrain Grand Prix',
            date=datetime(2024, 3, 2, 15, 0, 0),
            results=[race_result],
            qualifying=[qual_result],
            weather=weather
        )
    
    def test_storage_initialization(self, temp_storage):
        """Test storage initialization creates directories."""
        assert temp_storage.raw_data_dir.exists()
        assert temp_storage.processed_data_dir.exists()
        assert temp_storage.features_dir.exists()
    
    def test_save_and_load_race_data(self, temp_storage, sample_race_data):
        """Test saving and loading race data."""
        # Save race data
        success = temp_storage.save_race_data(sample_race_data)
        assert success is True
        
        # Check files were created
        race_file = temp_storage._get_race_filename(2024, 1)
        metadata_file = temp_storage._get_metadata_filename(2024, 1)
        assert race_file.exists()
        assert metadata_file.exists()
        
        # Load race data
        loaded_data = temp_storage.load_race_data(2024, 1)
        assert loaded_data is not None
        
        # Verify data integrity
        assert loaded_data.season == sample_race_data.season
        assert loaded_data.round == sample_race_data.round
        assert loaded_data.circuit_id == sample_race_data.circuit_id
        assert loaded_data.race_name == sample_race_data.race_name
        assert loaded_data.date == sample_race_data.date
        
        # Check results
        assert len(loaded_data.results) == 1
        result = loaded_data.results[0]
        assert result.driver_id == 'hamilton'
        assert result.constructor_id == 'mercedes'
        assert result.final_position == 1
        assert result.points == 25.0
        assert result.fastest_lap.time_ms == 88500
        assert result.status == RaceStatus.FINISHED
        
        # Check qualifying
        assert len(loaded_data.qualifying) == 1
        qual = loaded_data.qualifying[0]
        assert qual.driver_id == 'hamilton'
        assert qual.position == 1
        assert qual.q1_time.time_ms == 90123
        assert qual.q2_time.time_ms == 89456
        assert qual.q3_time.time_ms == 88789
        
        # Check weather
        assert loaded_data.weather.temperature == 25.0
        assert loaded_data.weather.humidity == 60.0
    
    def test_load_nonexistent_race_data(self, temp_storage):
        """Test loading non-existent race data."""
        loaded_data = temp_storage.load_race_data(2024, 99)
        assert loaded_data is None
    
    def test_save_and_load_features(self, temp_storage):
        """Test saving and loading features."""
        features = {
            'driver_form': [0.8, 0.7, 0.9],
            'constructor_performance': 0.85,
            'track_specific': {'sector1': 0.9, 'sector2': 0.8}
        }
        
        # Save features
        success = temp_storage.save_features(features, 'test_features')
        assert success is True
        
        # Check files were created
        features_file = temp_storage.features_dir / 'test_features.pkl'
        metadata_file = temp_storage.features_dir / 'test_features_metadata.json'
        assert features_file.exists()
        assert metadata_file.exists()
        
        # Load features
        loaded_features = temp_storage.load_features('test_features')
        assert loaded_features is not None
        assert loaded_features == features
    
    def test_load_nonexistent_features(self, temp_storage):
        """Test loading non-existent features."""
        loaded_features = temp_storage.load_features('nonexistent')
        assert loaded_features is None
    
    def test_list_available_races(self, temp_storage, sample_race_data):
        """Test listing available races."""
        # Initially empty
        races = temp_storage.list_available_races()
        assert len(races) == 0
        
        # Save some race data
        temp_storage.save_race_data(sample_race_data)
        
        # Create another race
        sample_race_data.round = 2
        sample_race_data.race_name = 'Saudi Arabian Grand Prix'
        temp_storage.save_race_data(sample_race_data)
        
        # List races
        races = temp_storage.list_available_races()
        assert len(races) == 2
        
        # Check sorting (by season, then round)
        assert races[0]['round'] == 1
        assert races[1]['round'] == 2
        
        # Test filtering by season
        races_2024 = temp_storage.list_available_races(season=2024)
        assert len(races_2024) == 2
        
        races_2023 = temp_storage.list_available_races(season=2023)
        assert len(races_2023) == 0
    
    def test_get_storage_stats(self, temp_storage, sample_race_data):
        """Test getting storage statistics."""
        # Initially empty
        stats = temp_storage.get_storage_stats()
        assert stats['total_races'] == 0
        assert stats['total_features'] == 0
        assert stats['storage_size_mb'] == 0
        
        # Add some data
        temp_storage.save_race_data(sample_race_data)
        temp_storage.save_features({'test': 'data'}, 'test_features')
        
        # Get updated stats
        stats = temp_storage.get_storage_stats()
        assert stats['total_races'] == 1
        assert stats['total_features'] == 1
        assert stats['storage_size_mb'] >= 0  # Files might be very small
        assert 2024 in stats['seasons']
        assert stats['last_updated'] is not None
    
    def test_serialize_deserialize_race_data(self, temp_storage, sample_race_data):
        """Test race data serialization and deserialization."""
        # Serialize
        serialized = temp_storage._serialize_race_data(sample_race_data)
        assert isinstance(serialized, dict)
        assert serialized['season'] == 2024
        assert serialized['round'] == 1
        
        # Deserialize
        deserialized = temp_storage._deserialize_race_data(serialized)
        assert isinstance(deserialized, RaceData)
        assert deserialized.season == sample_race_data.season
        assert deserialized.round == sample_race_data.round
        assert deserialized.circuit_id == sample_race_data.circuit_id
    
    def test_cleanup_old_data(self, temp_storage):
        """Test cleaning up old data."""
        # Create data for multiple seasons
        for season in [2020, 2021, 2022, 2023, 2024]:
            race_data = RaceData(
                season=season,
                round=1,
                circuit_id='test',
                race_name='Test Race',
                date=datetime(season, 3, 1),
                results=[],
                qualifying=[],
                weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
            )
            temp_storage.save_race_data(race_data)
        
        # Verify all seasons exist
        stats = temp_storage.get_storage_stats()
        assert len(stats['seasons']) == 5
        
        # Cleanup, keeping only 3 most recent seasons
        success = temp_storage.cleanup_old_data(keep_seasons=3)
        assert success is True
        
        # Verify cleanup
        stats = temp_storage.get_storage_stats()
        assert len(stats['seasons']) == 3
        assert 2022 in stats['seasons']
        assert 2023 in stats['seasons']
        assert 2024 in stats['seasons']
        assert 2020 not in stats['seasons']
        assert 2021 not in stats['seasons']
    
    def test_error_handling(self, temp_storage):
        """Test error handling in storage operations."""
        # Test saving invalid data
        with pytest.raises(DataStorageError):
            temp_storage.save_race_data(None)
        
        # Test loading from corrupted file
        race_file = temp_storage._get_race_filename(2024, 1)
        race_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create corrupted JSON file
        with open(race_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(DataStorageError):
            temp_storage.load_race_data(2024, 1)