"""
Debug feature generation and model training.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_realistic_data_features():
    """Test realistic data feature generation."""
    print("🔍 Testing Realistic Data Feature Generation")
    print("=" * 50)
    
    try:
        from web.utils.realistic_f1_data import RealisticF1Data
        
        f1_data = RealisticF1Data()
        training_data = f1_data.generate_realistic_training_data(3)
        
        print(f"Generated {len(training_data['features'])} training samples")
        
        # Check first sample
        sample_features = training_data['features'][0]
        print(f"\nSample features ({len(sample_features)} total):")
        
        for i, (key, value) in enumerate(sorted(sample_features.items())):
            print(f"  {i+1:2d}. {key:<30} = {value}")
        
        return training_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_data_helper_features():
    """Test data helper feature generation."""
    print("\n🔍 Testing Data Helper Feature Generation")
    print("=" * 50)
    
    try:
        from web.utils.data_helpers import DataHelper
        
        data_helper = DataHelper()
        training_data = data_helper.generate_training_data(3)
        
        print(f"Generated {len(training_data['features'])} training samples")
        
        # Check first sample
        sample_features = training_data['features'][0]
        print(f"\nSample features ({len(sample_features)} total):")
        
        for i, (key, value) in enumerate(sorted(sample_features.items())):
            print(f"  {i+1:2d}. {key:<30} = {value}")
        
        return training_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_prediction_engine_features():
    """Test prediction engine feature extraction."""
    print("\n🔍 Testing Prediction Engine Feature Extraction")
    print("=" * 50)
    
    try:
        from src.services.prediction_engine import PredictionEngine, PredictionRequest
        from datetime import datetime
        
        engine = PredictionEngine(model_type='random_forest')
        
        # Create test request
        request = PredictionRequest(
            race_name="Test Race",
            circuit="Silverstone",
            date=datetime.now(),
            drivers=[
                {'driver_id': 'VER', 'name': 'Max Verstappen', 'grid_position': 1},
                {'driver_id': 'NOR', 'name': 'Lando Norris', 'grid_position': 2}
            ],
            weather={
                'conditions': 'dry',
                'track_temp': 30.0,
                'air_temp': 25.0,
                'humidity': 60.0
            }
        )
        
        # Extract features
        features = engine._extract_race_features(request)
        
        print(f"Extracted features for {len(features)} drivers")
        
        # Check first driver features
        sample_features = features[0]
        print(f"\nSample features ({len(sample_features)} total):")
        
        for i, (key, value) in enumerate(sorted(sample_features.items())):
            print(f"  {i+1:2d}. {key:<30} = {value}")
        
        return features
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all feature tests."""
    print("🏎️  Feature Generation Debug Test")
    print("=" * 60)
    
    # Test realistic data
    realistic_data = test_realistic_data_features()
    
    # Test data helper
    helper_data = test_data_helper_features()
    
    # Test prediction engine
    engine_features = test_prediction_engine_features()
    
    # Compare feature counts
    print("\n📊 Feature Count Comparison:")
    print("=" * 30)
    
    if realistic_data:
        print(f"Realistic F1 Data: {len(realistic_data['features'][0])} features")
    
    if helper_data:
        print(f"Data Helper: {len(helper_data['features'][0])} features")
    
    if engine_features:
        print(f"Prediction Engine: {len(engine_features[0])} features")
    
    # Check if they match
    if realistic_data and helper_data and engine_features:
        realistic_keys = set(realistic_data['features'][0].keys())
        helper_keys = set(helper_data['features'][0].keys())
        engine_keys = set(engine_features[0].keys())
        
        print(f"\nFeature Set Comparison:")
        if realistic_keys == helper_keys == engine_keys:
            print("✅ All feature sets match!")
        else:
            print("❌ Feature sets don't match:")
            print(f"  Realistic: {len(realistic_keys)} features")
            print(f"  Helper: {len(helper_keys)} features")
            print(f"  Engine: {len(engine_keys)} features")
            
            # Show differences
            if realistic_keys != helper_keys:
                print(f"  Realistic vs Helper diff: {realistic_keys.symmetric_difference(helper_keys)}")
            
            if helper_keys != engine_keys:
                print(f"  Helper vs Engine diff: {helper_keys.symmetric_difference(engine_keys)}")

if __name__ == "__main__":
    main()