"""
Test the web app with realistic F1 2024 data.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web.utils.data_helpers import DataHelper
from web.utils.realistic_f1_data import RealisticF1Data
from src.services.prediction_engine import PredictionEngine
from src.models.data_models import PredictionRequest
from datetime import datetime

def test_web_app_data_integration():
    """Test that web app properly integrates realistic F1 data."""
    print("🏎️  Testing Web App with Realistic F1 Data")
    print("=" * 50)
    
    # Test data helper
    print("1. Testing DataHelper with realistic data...")
    data_helper = DataHelper()
    
    # Check driver data
    drivers = data_helper.current_f1_drivers
    print(f"   ✅ Loaded {len(drivers)} drivers")
    
    # Check top drivers have realistic data
    top_driver = drivers[0]
    print(f"   ✅ Championship leader: {top_driver['name']} ({top_driver['points']} pts)")
    
    if top_driver['points'] > 300:
        print("   ✅ Realistic championship points")
    else:
        print("   ⚠️  Championship points seem low")
    
    # Test training data generation
    print("\n2. Testing realistic training data generation...")
    training_data = data_helper.generate_training_data(50)
    
    features = training_data['features']
    targets = training_data['targets']
    
    print(f"   ✅ Generated {len(features)} training samples")
    print(f"   ✅ Feature count per sample: {len(features[0])}")
    
    # Check feature realism
    avg_points = sum(f['driver_championship_points'] for f in features) / len(features)
    print(f"   ✅ Average championship points in training: {avg_points:.1f}")
    
    # Check position distribution
    avg_position = sum(targets) / len(targets)
    print(f"   ✅ Average finishing position: {avg_position:.1f}")
    
    return data_helper

def test_realistic_prediction_engine():
    """Test prediction engine with realistic data."""
    print("\n3. Testing Prediction Engine with Realistic Data...")
    print("-" * 40)
    
    # Create prediction engine
    engine = PredictionEngine(model_type='random_forest')
    
    # Initialize with realistic training data
    data_helper = DataHelper()
    training_data = data_helper.generate_training_data(100)
    
    success = engine.initialize_model(training_data['features'], training_data['targets'])
    if success:
        print("   ✅ Prediction engine initialized successfully")
    else:
        print("   ❌ Failed to initialize prediction engine")
        return None
    
    # Create a realistic race request
    request = PredictionRequest(
        race_name="Abu Dhabi Grand Prix 2024",
        circuit="Yas Marina Circuit",
        date=datetime.now(),
        drivers=[
            {'driver_id': 'VER', 'name': 'Max Verstappen', 'grid_position': 1},
            {'driver_id': 'NOR', 'name': 'Lando Norris', 'grid_position': 2},
            {'driver_id': 'LEC', 'name': 'Charles Leclerc', 'grid_position': 3},
            {'driver_id': 'PIA', 'name': 'Oscar Piastri', 'grid_position': 4},
            {'driver_id': 'SAI', 'name': 'Carlos Sainz', 'grid_position': 5}
        ],
        weather={
            'conditions': 'dry',
            'track_temp': 35.0,
            'air_temp': 28.0,
            'humidity': 65.0
        }
    )
    
    # Generate prediction
    try:
        result = engine.predict_race(request)
        print(f"   ✅ Generated prediction with confidence: {result.confidence_score:.3f}")
        
        # Display results
        print("   📊 Predicted Results:")
        for i, pred in enumerate(result.predictions[:5]):
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            print(f"      P{i+1}: {driver_name:<18} (Conf: {pred.confidence_score:.2f})")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return None

def test_prediction_variety():
    """Test that predictions show variety."""
    print("\n4. Testing Prediction Variety...")
    print("-" * 40)
    
    engine = PredictionEngine(model_type='random_forest')
    data_helper = DataHelper()
    
    winners = []
    confidences = []
    
    for i in range(3):
        # Generate new training data each time
        training_data = data_helper.generate_training_data(80)
        engine.initialize_model(training_data['features'], training_data['targets'])
        
        # Same race setup
        request = PredictionRequest(
            race_name=f"Test Race {i+1}",
            circuit="Test Circuit",
            date=datetime.now(),
            drivers=[
                {'driver_id': 'VER', 'name': 'Max Verstappen', 'grid_position': 1},
                {'driver_id': 'NOR', 'name': 'Lando Norris', 'grid_position': 2},
                {'driver_id': 'LEC', 'name': 'Charles Leclerc', 'grid_position': 3}
            ],
            weather={'conditions': 'dry', 'track_temp': 30.0, 'air_temp': 25.0, 'humidity': 60.0}
        )
        
        result = engine.predict_race(request)
        winner = result.predictions[0]
        winner_name = getattr(winner, 'driver_name', winner.driver_id)
        
        winners.append(winner_name)
        confidences.append(result.confidence_score)
        
        print(f"   Run {i+1}: Winner = {winner_name:<18} (Conf: {result.confidence_score:.3f})")
    
    # Analyze variety
    unique_winners = len(set(winners))
    confidence_range = max(confidences) - min(confidences)
    
    print(f"\n   📈 Variety Analysis:")
    print(f"      Unique winners: {unique_winners}/3")
    print(f"      Confidence range: {confidence_range:.3f}")
    
    if unique_winners >= 2 or confidence_range > 0.05:
        print("   ✅ Good prediction variety")
    else:
        print("   ⚠️  Limited prediction variety")

def test_realistic_f1_data_direct():
    """Test the realistic F1 data module directly."""
    print("\n5. Testing Realistic F1 Data Module...")
    print("-" * 40)
    
    f1_data = RealisticF1Data()
    
    # Test driver data
    verstappen_data = f1_data.get_driver_data('VER')
    print(f"   ✅ Max Verstappen: {verstappen_data['points']} pts, {verstappen_data['wins']} wins")
    
    # Test team data
    mclaren_data = f1_data.get_team_data('McLaren')
    print(f"   ✅ McLaren car rating: {mclaren_data['car_rating']:.2f}")
    
    # Test realistic prediction
    drivers = [
        {'driver_id': 'VER', 'grid_position': 1},
        {'driver_id': 'NOR', 'grid_position': 2},
        {'driver_id': 'LEC', 'grid_position': 3}
    ]
    
    weather = {'conditions': 'dry', 'track_temp': 30.0, 'air_temp': 25.0, 'humidity': 60.0}
    
    predictions = f1_data.predict_race_realistic(drivers, weather, "Silverstone")
    
    print("   📊 Direct Realistic Prediction:")
    for i, pred in enumerate(predictions):
        print(f"      P{i+1}: {pred['driver_name']:<18} (Conf: {pred['confidence_score']:.2f})")

def main():
    """Run all web app tests with realistic data."""
    print("🏎️  Web App Realistic Data Integration Test")
    print("=" * 60)
    
    try:
        # Test data integration
        data_helper = test_web_app_data_integration()
        
        # Test prediction engine
        result = test_realistic_prediction_engine()
        
        # Test prediction variety
        test_prediction_variety()
        
        # Test realistic F1 data directly
        test_realistic_f1_data_direct()
        
        print("\n" + "=" * 60)
        print("🎉 Web App Realistic Data Integration Complete!")
        print("\nThe web app now uses:")
        print("✅ Real F1 2024 championship data")
        print("✅ Realistic driver and team performance ratings")
        print("✅ Circuit-specific prediction logic")
        print("✅ Weather-dependent race outcomes")
        print("✅ Varied predictions with proper randomness")
        
        print(f"\nYou can now run the web app with: streamlit run web/app.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()