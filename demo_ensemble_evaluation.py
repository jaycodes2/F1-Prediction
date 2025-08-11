"""
Demo script for advanced ensemble methods and model evaluation.
"""
import numpy as np
from src.models.ensemble import AdvancedEnsembleModel, ModelEvaluator, CrossValidationEvaluator
from src.models.implementations import ModelFactory

def create_realistic_f1_data(n_samples=120):
    """Create realistic F1 training data with correlations."""
    np.random.seed(42)
    features = []
    targets = []
    
    for i in range(n_samples):
        # Create correlated features that make sense for F1
        qualifying_pos = np.random.randint(1, 21)
        
        # Driver skill affects both qualifying and race performance
        driver_skill = np.random.uniform(0.3, 1.0)
        
        # Car performance affects both qualifying and race
        car_performance = np.random.uniform(0.4, 1.0)
        
        # Weather affects race more than qualifying
        weather_impact = np.random.uniform(0.8, 1.2)
        
        features.append({
            'qualifying_position': qualifying_pos,
            'driver_championship_points': int(driver_skill * 400),
            'constructor_championship_points': int(car_performance * 600),
            'driver_wins_season': int(driver_skill * 10),
            'constructor_wins_season': int(car_performance * 15),
            'track_temperature': np.random.uniform(15, 45),
            'air_temperature': np.random.uniform(10, 40),
            'humidity': np.random.uniform(30, 90),
            'weather_dry': np.random.choice([0, 1], p=[0.2, 0.8]),
            'track_grip': np.random.uniform(0.7, 1.0) * weather_impact,
            'fuel_load': np.random.uniform(50, 110),
            'tire_compound': np.random.randint(1, 4),
            'driver_experience': int(driver_skill * 300),
            'car_reliability': car_performance
        })
        
        # Create realistic finishing position based on multiple factors
        base_finish = (
            qualifying_pos * 0.6 +  # Qualifying has strong influence
            (1 - driver_skill) * 10 +  # Better drivers finish better
            (1 - car_performance) * 8 +  # Better cars finish better
            np.random.normal(0, 2) * weather_impact  # Weather adds noise
        )
        
        finish_pos = max(1, min(20, int(round(base_finish))))
        targets.append(finish_pos)
    
    return features, targets

def demo_ensemble_methods():
    """Demonstrate different ensemble methods."""
    print("=== Advanced Ensemble Methods Demo ===\n")
    
    # Create realistic data
    print("Creating realistic F1 training data...")
    features, targets = create_realistic_f1_data(100)
    
    # Split data
    train_features = features[:70]
    train_targets = targets[:70]
    test_features = features[70:]
    test_targets = targets[70:]
    
    print(f"Training data: {len(train_features)} samples")
    print(f"Test data: {len(test_features)} samples")
    print(f"Features per sample: {len(train_features[0])}")
    print()
    
    # Test different ensemble methods
    ensemble_methods = ['weighted_average', 'stacking', 'voting', 'dynamic_weighting']
    ensemble_results = {}
    
    for method in ensemble_methods:
        print(f"--- Testing {method.upper().replace('_', ' ')} Ensemble ---")
        
        try:
            # Create and train ensemble
            ensemble = AdvancedEnsembleModel(
                random_state=42, 
                ensemble_method=method
            )
            
            training_results = ensemble.train(
                train_features, train_targets, 
                tune_hyperparameters=False
            )
            
            print(f"✓ Ensemble trained successfully")
            print(f"✓ Base models: {', '.join(training_results['base_models'])}")
            
            if method == 'stacking':
                print(f"✓ Meta-model: {type(ensemble.meta_model).__name__}")
            else:
                print(f"✓ Model weights: {ensemble.model_weights}")
            
            # Make predictions
            predictions = ensemble.predict(test_features)
            print(f"✓ Generated {len(predictions)} predictions")
            
            # Calculate test metrics
            from src.models.training import MetricsCalculator
            calc = MetricsCalculator()
            test_metrics = calc.calculate_all_metrics(
                np.array(test_targets), predictions
            )
            
            print(f"✓ Test MAE: {test_metrics['mae']:.3f}")
            print(f"✓ Test Spearman: {test_metrics['spearman_correlation']:.3f}")
            print(f"✓ Test Top-3 Accuracy: {test_metrics['top_3_accuracy']:.1f}%")
            
            ensemble_results[method] = {
                'model': ensemble,
                'metrics': test_metrics,
                'training_results': training_results
            }
            
        except Exception as e:
            print(f"✗ Error with {method}: {e}")
            ensemble_results[method] = {'error': str(e)}
        
        print()
    
    return ensemble_results, test_features, test_targets

def demo_model_evaluation(ensemble_results, test_features, test_targets):
    """Demonstrate comprehensive model evaluation."""
    print("=== Comprehensive Model Evaluation ===\n")
    
    # Create individual models for comparison
    print("Training individual models for comparison...")
    individual_models = {}
    
    for model_type in ['random_forest', 'xgboost', 'lightgbm']:
        try:
            model = ModelFactory.create_model(model_type, random_state=42)
            # Use same training data as ensembles
            train_features = test_features  # For demo, using test as train
            train_targets = test_targets
            
            model.train(train_features, train_targets, tune_hyperparameters=False)
            individual_models[model_type] = model
            print(f"✓ {model_type} trained")
        except Exception as e:
            print(f"✗ {model_type} failed: {e}")
    
    print()
    
    # Combine all models for evaluation
    all_models = individual_models.copy()
    
    # Add successful ensemble models
    for method, result in ensemble_results.items():
        if 'model' in result:
            all_models[f'ensemble_{method}'] = result['model']
    
    print(f"Evaluating {len(all_models)} models...")
    
    # Comprehensive evaluation
    evaluator = ModelEvaluator()
    comparison = evaluator.compare_models(all_models, test_features, test_targets)
    
    print("\n--- Individual Model Results ---")
    for model_name, result in comparison['individual_results'].items():
        if 'error' in result:
            print(f"{model_name}: FAILED - {result['error']}")
        else:
            metrics = result['metrics']
            print(f"{model_name}:")
            print(f"  MAE: {metrics['mae']:.3f}")
            print(f"  Spearman: {metrics['spearman_correlation']:.3f}")
            print(f"  Top-3 Accuracy: {metrics['top_3_accuracy']:.1f}%")
            print(f"  Position Accuracy: {metrics['position_accuracy']:.1f}%")
    
    print("\n--- Best Models by Category ---")
    best_models = comparison['best_models']
    for category, info in best_models.items():
        print(f"{category.replace('_', ' ').title()}: {info['model']} ({info['value']:.3f})")
    
    # Generate detailed report
    print("\n--- Generating Detailed Report ---")
    report = evaluator.generate_evaluation_report()
    
    # Save report to file
    with open('model_evaluation_report.txt', 'w') as f:
        f.write(report)
    print("✓ Detailed report saved to 'model_evaluation_report.txt'")
    
    return comparison

def demo_cross_validation():
    """Demonstrate cross-validation evaluation."""
    print("\n=== Cross-Validation Evaluation ===\n")
    
    # Create data for CV
    features, targets = create_realistic_f1_data(80)
    
    cv_evaluator = CrossValidationEvaluator(n_splits=4)
    
    # Test individual model CV
    print("--- Random Forest Cross-Validation ---")
    def rf_factory(**kwargs):
        return ModelFactory.create_model('random_forest', **kwargs)
    
    try:
        rf_cv = cv_evaluator.evaluate_model_cv(
            rf_factory, features, targets, random_state=42
        )
        
        print(f"✓ Cross-validation completed ({rf_cv['n_folds']} folds)")
        print(f"✓ Mean MAE: {rf_cv['mae_mean']:.3f} (±{rf_cv['mae_std']:.3f})")
        print(f"✓ Mean Spearman: {rf_cv['spearman_correlation_mean']:.3f} (±{rf_cv['spearman_correlation_std']:.3f})")
        print(f"✓ Mean Top-3 Accuracy: {rf_cv['top_3_accuracy_mean']:.1f}% (±{rf_cv['top_3_accuracy_std']:.1f}%)")
        
    except Exception as e:
        print(f"✗ Random Forest CV failed: {e}")
    
    print()
    
    # Test ensemble CV
    print("--- Ensemble Cross-Validation ---")
    def ensemble_factory(**kwargs):
        return AdvancedEnsembleModel(ensemble_method='voting', **kwargs)
    
    try:
        ensemble_cv = cv_evaluator.evaluate_model_cv(
            ensemble_factory, features, targets, random_state=42
        )
        
        print(f"✓ Ensemble CV completed ({ensemble_cv['n_folds']} folds)")
        print(f"✓ Mean MAE: {ensemble_cv['mae_mean']:.3f} (±{ensemble_cv['mae_std']:.3f})")
        print(f"✓ Mean Spearman: {ensemble_cv['spearman_correlation_mean']:.3f} (±{ensemble_cv['spearman_correlation_std']:.3f})")
        
    except Exception as e:
        print(f"✗ Ensemble CV failed: {e}")

def demo_feature_importance_analysis(ensemble_results):
    """Demonstrate feature importance analysis across models."""
    print("\n=== Feature Importance Analysis ===\n")
    
    # Collect feature importance from all models
    all_importance = {}
    
    # From ensemble models
    for method, result in ensemble_results.items():
        if 'training_results' in result and 'feature_importance' in result['training_results']:
            # For ensembles, we can get importance from base models
            model = result['model']
            if hasattr(model, 'base_models'):
                for base_name, base_model in model.base_models.items():
                    importance = base_model.get_feature_importance(model.feature_names)
                    if importance:
                        all_importance[f'{method}_{base_name}'] = importance
    
    if not all_importance:
        print("No feature importance data available")
        return
    
    # Aggregate feature importance across models
    feature_scores = {}
    for model_name, importance in all_importance.items():
        for feature, score in importance.items():
            if feature not in feature_scores:
                feature_scores[feature] = []
            feature_scores[feature].append(score)
    
    # Calculate average importance
    avg_importance = {}
    for feature, scores in feature_scores.items():
        avg_importance[feature] = np.mean(scores)
    
    # Sort by importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("Average Feature Importance (across all models):")
    print("-" * 50)
    for feature, importance in sorted_features[:10]:  # Top 10
        print(f"{feature:<30} {importance:.4f}")
    
    print(f"\nTotal features analyzed: {len(sorted_features)}")
    print(f"Models contributing: {len(all_importance)}")

def main():
    """Run the complete ensemble and evaluation demo."""
    print("🏎️  F1 RACE PREDICTION - ADVANCED ENSEMBLE & EVALUATION DEMO 🏎️\n")
    
    try:
        # Demo ensemble methods
        ensemble_results, test_features, test_targets = demo_ensemble_methods()
        
        # Demo model evaluation
        comparison = demo_model_evaluation(ensemble_results, test_features, test_targets)
        
        # Demo cross-validation
        demo_cross_validation()
        
        # Demo feature importance
        demo_feature_importance_analysis(ensemble_results)
        
        print("\n" + "="*60)
        print("🎯 DEMO SUMMARY")
        print("="*60)
        
        successful_ensembles = sum(1 for r in ensemble_results.values() if 'model' in r)
        total_models_evaluated = len(comparison['individual_results']) if comparison else 0
        
        print(f"✅ Ensemble methods tested: {len(ensemble_results)}")
        print(f"✅ Successful ensembles: {successful_ensembles}")
        print(f"✅ Total models evaluated: {total_models_evaluated}")
        print(f"✅ Evaluation report generated: model_evaluation_report.txt")
        
        if comparison and 'best_models' in comparison:
            best_overall = comparison['best_models'].get('lowest_mae', {})
            if best_overall:
                print(f"🏆 Best performing model: {best_overall['model']} (MAE: {best_overall['value']:.3f})")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()