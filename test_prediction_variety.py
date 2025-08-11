"""
Simple test to verify prediction variety.
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.prediction_engine import PredictionEngine, PredictionRequest
from web.utils.data_helpers import DataHelper

def test_predictions():
    """Test that predictions vary between runs."""
    print("🏎️  Testing F1 Prediction Variety")
    print("=" * 40)
    
    # Create data helper
    data_helper = DataHelper()
    
    # Create a consistent race request
    request = PredictionRequest(
        race_name="Test Grand Prix",
        circuit="Test Circuit",
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
            'track_temp': 30.0,
            'air_temp': 25.0,
            'humidity': 60.0
        }
    )
    
    predictions_list = []
    
    # Generate multiple predictions
    for i in range(3):
        print(f"\nGeneration {i+1}:")
        
        try:
            # Create new engine each time
            engine = PredictionEngine(model_type='random_forest')
            
            # Generate new training data each time
            training_data = data_helper.generate_training_data(100)
            engine.initialize_model(training_data['features'], training_data['targets'])
            
            # Make prediction
            print(f"  Making prediction...")
            result = engine.predict_race(request)
            print(f"  Got result with {len(result.predictions)} predictions")
            
            # Extract winner and top 3
            winner = result.predictions[0]
            winner_name = getattr(winner, 'driver_name', winner.driver_id)
            
            top_3 = []
            for pred in result.predictions[:3]:
                driver_name = getattr(pred, 'driver_name', pred.driver_id)
                top_3.append(f"{driver_name} (P{pred.predicted_position})")
            
            print(f"  Winner: {winner_name}")
            print(f"  Confidence: {result.confidence_score:.3f}")
            print(f"  Top 3: {', '.join(top_3)}")
            
            predictions_list.append({
                'winner': winner_name,
                'confidence': result.confidence_score,
                'top_3': top_3
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Check for variety
    print("\n" + "=" * 40)
    print("VARIETY ANALYSIS:")
    
    if len(predictions_list) > 0:
        winners = [p['winner'] for p in predictions_list]
        confidences = [p['confidence'] for p in predictions_list]
        
        unique_winners = len(set(winners))
        confidence_range = max(confidences) - min(confidences) if len(confidences) > 1 else 0
        
        print(f"Unique winners: {unique_winners}/{len(predictions_list)}")
        print(f"Confidence range: {confidence_range:.3f}")
        
        if unique_winners > 1 or confidence_range > 0.05:
            print("✅ Predictions show good variety!")
        else:
            print("⚠️  Predictions may be too similar")
    else:
        print("❌ No successful predictions generated")

if __name__ == "__main__":
    test_predictions()