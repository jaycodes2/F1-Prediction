"""
Demo script for F1 model implementations.
"""
import numpy as np
from src.models.implementations import ModelFactory

def create_sample_data(n_samples=100):
    """Create sample F1 training data."""
    features = []
    targets = []
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(n_samples):
        # Create realistic F1 features
        qualifying_pos = np.random.randint(1, 21)
        
        features.append({
            'qualifying_position': qualifying_pos,
            'driver_championship_points': np.random.randint(0, 400),
            'constructor_championship_points': np.random.randint(0, 600),
            'track_temperature': np.random.uniform(15, 45),
            'weather_dry': np.random.choice([0, 1], p=[0.2, 0.8]),
            'driver_experience': np.random.randint(0, 300),
            'car_performance_rating': np.random.uniform(0.5, 1.0)
        })
        
        # Create somewhat realistic finishing position
        # Better qualifying positions tend to finish better, but with noise
        base_finish = qualifying_pos + np.random.normal(0, 3)
        finish_pos = max(1, min(20, int(round(base_finish))))
        targets.append(finish_pos)
    
    return features, targets

def demo_model_training():
    """Demonstrate model training with different algorithms."""
    print("=== F1 Model Implementations Demo ===\n")
    
    # Create sample data
    print("Creating sample F1 training data...")
    features, targets = create_sample_data(80)
    test_features, test_targets = create_sample_data(20)
    
    print(f"Training data: {len(features)} samples")
    print(f"Test data: {len(test_features)} samples")
    print(f"Features per sample: {len(features[0])}")
    print()
    
    # Test different model types
    models_to_test = ['random_forest', 'xgboost', 'lightgbm']
    results = {}
    
    for model_type in models_to_test:
        print(f"--- Testing {model_type.upper()} Model ---")
        
        try:
            # Create model
            model = ModelFactory.create_model(model_type, random_state=42)
            print(f"✓ Model created successfully")
            
            # Train model (without hyperparameter tuning for speed)
            training_results = model.train(features, targets, tune_hyperparameters=False)
            print(f"✓ Model trained successfully")
            
            # Get model info
            model_info = model.get_model_info()
            print(f"✓ Model type: {model_info['model_type']}")
            
            # Make predictions
            predictions = model.predict(test_features)
            print(f"✓ Generated {len(predictions)} predictions")
            
            # Display metrics
            metrics = training_results['metrics']
            print(f"✓ Training MAE: {metrics['mae']:.3f}")
            print(f"✓ Spearman Correlation: {metrics['spearman_correlation']:.3f}")
            print(f"✓ Top-3 Accuracy: {metrics['top_3_accuracy']:.1f}%")
            
            # Feature importance
            if 'feature_importance' in training_results:
                importance = training_results['feature_importance']
                top_features = list(importance.items())[:3]
                print(f"✓ Top 3 important features:")
                for feature, score in top_features:
                    print(f"   - {feature}: {score:.3f}")
            
            results[model_type] = {
                'mae': metrics['mae'],
                'spearman': metrics['spearman_correlation'],
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"✗ Error with {model_type}: {e}")
            results[model_type] = {'error': str(e)}
        
        print()
    
    # Test ensemble model
    print("--- Testing ENSEMBLE Model ---")
    try:
        ensemble_model = ModelFactory.create_model('ensemble', random_state=42)
        ensemble_results = ensemble_model.train(features, targets, tune_hyperparameters=False)
        
        print(f"✓ Ensemble model trained successfully")
        print(f"✓ Base models: {', '.join(ensemble_results['base_models'])}")
        print(f"✓ Model weights: {ensemble_results['model_weights']}")
        
        ensemble_predictions = ensemble_model.predict(test_features)
        ensemble_metrics = ensemble_results['metrics']
        
        print(f"✓ Ensemble MAE: {ensemble_metrics['mae']:.3f}")
        print(f"✓ Ensemble Spearman: {ensemble_metrics['spearman_correlation']:.3f}")
        
        results['ensemble'] = {
            'mae': ensemble_metrics['mae'],
            'spearman': ensemble_metrics['spearman_correlation'],
            'predictions': ensemble_predictions
        }
        
    except Exception as e:
        print(f"✗ Error with ensemble: {e}")
        results['ensemble'] = {'error': str(e)}
    
    print()
    
    # Compare results
    print("=== Model Comparison ===")
    print(f"{'Model':<15} {'MAE':<8} {'Spearman':<10} {'Status'}")
    print("-" * 45)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name:<15} {'N/A':<8} {'N/A':<10} Error")
        else:
            mae = result['mae']
            spearman = result['spearman']
            print(f"{model_name:<15} {mae:<8.3f} {spearman:<10.3f} Success")
    
    print()
    
    # Test ModelFactory functionality
    print("=== ModelFactory Features ===")
    available_models = ModelFactory.list_available_models()
    print(f"Available models: {', '.join(available_models)}")
    
    for model_type in available_models[:3]:  # Test first 3
        try:
            info = ModelFactory.get_model_info(model_type)
            print(f"✓ {model_type}: {info['description'][:50]}...")
        except Exception as e:
            print(f"✗ {model_type}: Error getting info - {e}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo_model_training()