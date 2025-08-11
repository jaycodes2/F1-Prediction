"""
Test feature consistency between training and prediction data.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web.utils.data_helpers import DataHelper
from src.services.prediction_engine import PredictionEngine, PredictionRequest
from datetime import datetime

def test_feature_consistency():
    """Test that training and prediction features match."""
    print("🔍 Testing Feature Consistency")
    print("=" * 50)
    
    # Generate training data
    data_helper = DataHelper()
    training_data = data_helper.generate_training_data(5)
    
    print("Training Data Features:")
    training_features = training_data['features'][0]
    print(f"Feature count: {len(training_features)}")
    print("Feature names:")
    for i, feature_name in enumerate(sorted(training_features.keys())):
        print(f"  {i+1:2d}. {feature_name}")
    
    # Create prediction engine and initialize
    engine = PredictionEngine(model_type='random_forest')
    success = engine.initialize_model(training_data['features'], training_data['targets'])
    
    if not success:
        print("❌ Failed to initialize model")
        return
    
    print(f"\n✅ Model initialized successfully")
    
    # Create prediction request
    request = PredictionRequest(
        race_name="Test Race",
        circuit="Test Circuit",
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
    
    # Extract prediction features
    prediction_features = engine._extract_race_features(request)
    
    print(f"\nPrediction Data Features:")
    pred_features = prediction_features[0]
    print(f"Feature count: {len(pred_features)}")
    print("Feature names:")
    for i, feature_name in enumerate(sorted(pred_features.keys())):
        print(f"  {i+1:2d}. {feature_name}")
    
    # Compare feature sets
    training_keys = set(training_features.keys())
    prediction_keys = set(pred_features.keys())
    
    print(f"\nFeature Comparison:")
    print(f"Training features: {len(training_keys)}")
    print(f"Prediction features: {len(prediction_keys)}")
    
    if training_keys == prediction_keys:
        print("✅ Feature sets match perfectly!")
    else:
        missing_in_prediction = training_keys - prediction_keys
        extra_in_prediction = prediction_keys - training_keys
        
        if missing_in_prediction:
            print(f"❌ Missing in prediction: {missing_in_prediction}")
        
        if extra_in_prediction:
            print(f"❌ Extra in prediction: {extra_in_prediction}")
    
    # Try to make a prediction
    print(f"\nTesting Prediction:")
    try:
        result = engine.predict_race(request)
        print(f"✅ Prediction successful! Confidence: {result.confidence_score:.3f}")
        
        for i, pred in enumerate(result.predictions):
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            print(f"  P{i+1}: {driver_name}")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_consistency()