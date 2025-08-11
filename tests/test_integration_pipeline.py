"""
Integration tests for the complete training and evaluation pipeline.
"""
import pytest
import numpy as np
from src.models.implementations import ModelFactory
from src.models.ensemble import AdvancedEnsembleModel, ModelEvaluator
from src.models.training import BaseModelTrainer, MetricsCalculator


class TestCompleteTrainingPipeline:
    """Integration tests for complete training and evaluation pipeline."""
    
    @pytest.fixture
    def pipeline_data(self):
        """Create comprehensive data for pipeline testing."""
        np.random.seed(42)
        features = []
        targets = []
        
        for i in range(100):
            qualifying_pos = np.random.randint(1, 21)
            
            features.append({
                'qualifying_position': qualifying_pos,
                'driver_championship_points': np.random.randint(0, 400),
                'constructor_championship_points': np.random.randint(0, 600),
                'track_temperature': np.random.uniform(15, 45),
                'weather_dry': np.random.choice([0, 1]),
                'driver_experience': np.random.randint(0, 300),
                'car_performance': np.random.uniform(0.5, 1.0)
            })
            
            # Create realistic finishing position
            base_finish = qualifying_pos + np.random.normal(0, 3)
            finish_pos = max(1, min(20, int(round(base_finish))))
            targets.append(finish_pos)
        
        return features, targets
    
    def test_end_to_end_individual_model_pipeline(self, pipeline_data):
        """Test complete pipeline with individual model."""
        features, targets = pipeline_data
        
        # Split data
        train_features = features[:70]
        train_targets = targets[:70]
        test_features = features[70:]
        test_targets = targets[70:]
        
        # 1. Create model
        model = ModelFactory.create_model('random_forest', random_state=42)
        assert model is not None
        
        # 2. Train model
        training_results = model.train(
            train_features, train_targets, 
            tune_hyperparameters=False
        )
        
        assert 'metrics' in training_results
        assert 'feature_importance' in training_results
        assert training_results['model_type'] == 'RandomForest'
        
        # 3. Make predictions
        predictions = model.predict(test_features)
        assert len(predictions) == len(test_features)
        assert all(1 <= p <= 21 for p in predictions)
        
        # 4. Evaluate model
        evaluator = ModelEvaluator()
        evaluation = evaluator.evaluate_single_model(
            model, test_features, test_targets, 'test_rf'
        )
        
        assert evaluation['model_name'] == 'test_rf'
        assert 'metrics' in evaluation
        assert evaluation['metrics']['mae'] >= 0
        
        # 5. Get model info
        model_info = model.get_model_info()
        assert model_info['model_type'] == 'RandomForest'
        assert 'n_features' in model_info
    
    def test_end_to_end_ensemble_pipeline(self, pipeline_data):
        """Test complete pipeline with ensemble model."""
        features, targets = pipeline_data
        
        # Split data
        train_features = features[:70]
        train_targets = targets[:70]
        test_features = features[70:]
        test_targets = targets[70:]
        
        # 1. Create ensemble
        ensemble = AdvancedEnsembleModel(
            random_state=42, 
            ensemble_method='weighted_average'
        )
        
        # 2. Train ensemble
        training_results = ensemble.train(
            train_features, train_targets, 
            tune_hyperparameters=False
        )
        
        assert training_results['model_type'] == 'AdvancedEnsemble'
        assert 'base_models' in training_results
        assert len(training_results['base_models']) > 0
        
        # 3. Make predictions
        predictions = ensemble.predict(test_features)
        assert len(predictions) == len(test_features)
        
        # 4. Evaluate ensemble
        evaluator = ModelEvaluator()
        evaluation = evaluator.evaluate_single_model(
            ensemble, test_features, test_targets, 'test_ensemble'
        )
        
        assert evaluation['model_name'] == 'test_ensemble'
        assert 'metrics' in evaluation
        
        # 5. Get ensemble info
        ensemble_info = ensemble.get_model_info()
        assert ensemble_info['model_type'] == 'AdvancedEnsemble'
        assert 'base_models' in ensemble_info
    
    def test_model_comparison_pipeline(self, pipeline_data):
        """Test complete model comparison pipeline."""
        features, targets = pipeline_data
        
        # Split data
        train_features = features[:60]
        train_targets = targets[:60]
        test_features = features[60:]
        test_targets = targets[60:]
        
        # Create and train multiple models
        models = {}
        
        # Individual models
        for model_type in ['random_forest', 'xgboost']:
            model = ModelFactory.create_model(model_type, random_state=42)
            model.train(train_features, train_targets, tune_hyperparameters=False)
            models[model_type] = model
        
        # Ensemble model
        ensemble = AdvancedEnsembleModel(random_state=42, ensemble_method='voting')
        ensemble.train(train_features, train_targets, tune_hyperparameters=False)
        models['ensemble'] = ensemble
        
        # Compare all models
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(models, test_features, test_targets)
        
        # Verify comparison results
        assert 'individual_results' in comparison
        assert 'summary' in comparison
        assert 'best_models' in comparison
        
        # Check individual results
        individual = comparison['individual_results']
        assert len(individual) == len(models)
        
        for model_name in models.keys():
            assert model_name in individual
            result = individual[model_name]
            if 'error' not in result:
                assert 'metrics' in result
                assert result['metrics']['mae'] >= 0
        
        # Check best models identification
        best_models = comparison['best_models']
        assert 'lowest_mae' in best_models
        assert 'highest_spearman' in best_models
    
    def test_metrics_calculation_consistency(self, pipeline_data):
        """Test that metrics are calculated consistently across the pipeline."""
        features, targets = pipeline_data
        
        # Train a model
        model = ModelFactory.create_model('random_forest', random_state=42)
        model.train(features[:50], targets[:50], tune_hyperparameters=False)
        
        # Get test data
        test_features = features[50:70]
        test_targets = targets[50:70]
        
        # Make predictions
        predictions = model.predict(test_features)
        
        # Calculate metrics using different methods
        
        # Method 1: Direct MetricsCalculator
        calc = MetricsCalculator()
        direct_metrics = calc.calculate_all_metrics(
            np.array(test_targets), predictions
        )
        
        # Method 2: Through model evaluation
        evaluator = ModelEvaluator()
        eval_result = evaluator.evaluate_single_model(
            model, test_features, test_targets
        )
        eval_metrics = eval_result['metrics']
        
        # Method 3: Through BaseModelTrainer
        trainer = BaseModelTrainer()
        trainer_metrics = trainer.evaluate_model(model, test_features, test_targets)
        
        # Compare key metrics (allowing for small floating point differences)
        key_metrics = ['mae', 'spearman_correlation']
        
        for metric in key_metrics:
            if metric in direct_metrics and metric in eval_metrics:
                assert abs(direct_metrics[metric] - eval_metrics[metric]) < 0.001
            
            if metric in direct_metrics and metric in trainer_metrics:
                assert abs(direct_metrics[metric] - trainer_metrics[metric]) < 0.001
    
    def test_model_persistence_integration(self, pipeline_data):
        """Test model persistence integration in the pipeline."""
        features, targets = pipeline_data
        
        # Train a model
        model = ModelFactory.create_model('random_forest', random_state=42)
        training_results = model.train(
            features[:50], targets[:50], 
            tune_hyperparameters=False
        )
        
        # Save model results
        model_path = model.save_training_results(
            model.model, 
            training_results['metrics'], 
            'test_integration_model'
        )
        
        assert model_path is not None
        assert isinstance(model_path, str)
        assert 'test_integration_model' in model_path
        
        # Verify model can still make predictions after saving
        test_features = features[50:60]
        predictions = model.predict(test_features)
        assert len(predictions) == len(test_features)
    
    def test_error_handling_in_pipeline(self, pipeline_data):
        """Test error handling throughout the pipeline."""
        features, targets = pipeline_data
        
        # Test with invalid model type
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model_type')
        
        # Test prediction without training
        model = ModelFactory.create_model('random_forest')
        with pytest.raises(Exception):  # Should raise ModelTrainingError or similar
            model.predict(features[:5])
        
        # Test with mismatched data lengths
        model = ModelFactory.create_model('random_forest')
        with pytest.raises(Exception):
            model.train(features[:10], targets[:5])  # Mismatched lengths
    
    def test_feature_importance_pipeline(self, pipeline_data):
        """Test feature importance extraction throughout the pipeline."""
        features, targets = pipeline_data
        
        # Test with tree-based models that support feature importance
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            model = ModelFactory.create_model(model_type, random_state=42)
            training_results = model.train(
                features[:50], targets[:50], 
                tune_hyperparameters=False
            )
            
            # Check that feature importance is extracted
            assert 'feature_importance' in training_results
            importance = training_results['feature_importance']
            
            assert isinstance(importance, dict)
            assert len(importance) > 0
            
            # Check that all features are represented
            feature_names = set(features[0].keys())
            importance_names = set(importance.keys())
            assert feature_names == importance_names
            
            # Check that importance values are reasonable
            for feature, score in importance.items():
                assert isinstance(score, (int, float, np.number))
                assert score >= 0
    
    def test_cross_validation_integration(self, pipeline_data):
        """Test cross-validation integration in the pipeline."""
        features, targets = pipeline_data
        
        # Create a model
        model = ModelFactory.create_model('random_forest', random_state=42)
        
        # Prepare data for CV
        trainer = BaseModelTrainer()
        X, y, _ = trainer.prepare_training_data(features, targets)
        
        # Run cross-validation
        cv_results = trainer.cross_validate_model(
            model.model, X, y, cv_folds=3
        )
        
        assert isinstance(cv_results, dict)
        assert 'cv_mae_mean' in cv_results
        assert 'cv_mae_std' in cv_results
        assert 'cv_folds' in cv_results
        assert cv_results['cv_folds'] == 3
        
        # Check that CV metrics are reasonable
        assert cv_results['cv_mae_mean'] >= 0
        assert cv_results['cv_mae_std'] >= 0