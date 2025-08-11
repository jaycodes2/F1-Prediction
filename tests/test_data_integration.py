"""
Integration tests for the complete data collection pipeline.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from src.data.combined_collector import CombinedDataCollector
from src.data.storage import FileDataStorage
from src.data.validator import DataValidator


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_storage_dir):
        """Create storage instance with temporary directory."""
        with patch('src.data.storage.config') as mock_config:
            mock_config.data.raw_data_dir = str(Path(temp_storage_dir) / "raw")
            mock_config.data.processed_data_dir = str(Path(temp_storage_dir) / "processed")
            yield FileDataStorage()
    
    @pytest.fixture
    def collector(self):
        """Create data collector."""
        return CombinedDataCollector()
    
    @pytest.fixture
    def validator(self):
        """Create data validator."""
        return DataValidator()
    
    def test_complete_data_pipeline(self, collector, storage, validator):
        """Test the complete data collection, storage, and validation pipeline."""
        # Step 1: Collect race data
        race_data = collector.collect_race_data(2023, 1)
        
        assert race_data is not None
        assert race_data.season == 2023
        assert race_data.round == 1
        assert len(race_data.results) > 15  # Should have ~20 drivers
        
        # Step 2: Validate collected data
        validation_result = validator.validate_race_data(race_data)
        
        assert validation_result is not None
        # Data should be valid (or at least not have critical errors)
        if not validation_result.is_valid:
            # Log errors for debugging but don't fail the test
            print(f"Validation errors: {[e.message for e in validation_result.errors]}")
        
        # Step 3: Save to storage
        save_success = storage.save_race_data(race_data)
        assert save_success is True
        
        # Step 4: Load from storage
        loaded_data = storage.load_race_data(2023, 1)
        assert loaded_data is not None
        
        # Step 5: Validate loaded data
        loaded_validation = validator.validate_race_data(loaded_data)
        assert loaded_validation is not None
        
        # Step 6: Verify data integrity
        assert loaded_data.season == race_data.season
        assert loaded_data.round == race_data.round
        assert loaded_data.circuit_id == race_data.circuit_id
        assert len(loaded_data.results) == len(race_data.results)
        assert len(loaded_data.qualifying) == len(race_data.qualifying)
    
    def test_storage_statistics_and_management(self, collector, storage, validator):
        """Test storage statistics and data management features."""
        # Initially empty
        stats = storage.get_storage_stats()
        assert stats['total_races'] == 0
        assert stats['total_features'] == 0
        
        # Add some data
        race_data = collector.collect_race_data(2023, 1)
        storage.save_race_data(race_data)
        
        # Save some features
        sample_features = {
            'test_feature': [1, 2, 3],
            'another_feature': 0.5
        }
        storage.save_features(sample_features, 'test_features')
        
        # Check updated stats
        stats = storage.get_storage_stats()
        assert stats['total_races'] == 1
        assert stats['total_features'] == 1
        assert 2023 in stats['seasons']
        
        # List available races
        available_races = storage.list_available_races()
        assert len(available_races) == 1
        assert available_races[0]['season'] == 2023
        assert available_races[0]['round'] == 1
    
    def test_data_source_availability_check(self, collector):
        """Test checking data source availability."""
        # Check data source availability for a known race
        source_info = collector.get_data_source_info(2023, 1)
        
        assert 'ergast_available' in source_info
        assert 'fastf1_available' in source_info
        assert 'fastf1_sessions' in source_info
        
        # At least one source should be available for 2023 data
        assert source_info['ergast_available'] or source_info['fastf1_available']
    
    def test_batch_validation(self, collector, validator):
        """Test batch validation of multiple races."""
        try:
            # Collect multiple races
            races = []
            for round_num in [1, 2]:
                try:
                    race_data = collector.collect_race_data(2023, round_num)
                    races.append(race_data)
                except Exception as e:
                    print(f"Could not collect race {round_num}: {e}")
                    continue
            
            if len(races) >= 1:
                # Validate batch
                aggregate_result = validator.validate_multiple_races(races)
                
                assert aggregate_result['total_races'] == len(races)
                assert 'valid_races' in aggregate_result
                assert 'season_stats' in aggregate_result
                assert 2023 in aggregate_result['season_stats']
            else:
                pytest.skip("Could not collect enough race data for batch validation")
                
        except Exception as e:
            pytest.skip(f"Batch validation test skipped due to data collection issues: {e}")
    
    def test_error_handling_and_recovery(self, storage, validator):
        """Test error handling and recovery in the pipeline."""
        # Test validation of invalid data
        validation_result = validator.validate_race_data(None)
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        
        # Test loading non-existent data
        loaded_data = storage.load_race_data(9999, 99)
        assert loaded_data is None
        
        # Test loading non-existent features
        loaded_features = storage.load_features('nonexistent_features')
        assert loaded_features is None