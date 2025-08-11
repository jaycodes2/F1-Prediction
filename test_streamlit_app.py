"""
Test script to verify Streamlit application components work correctly.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.services.prediction_engine import PredictionEngine, PredictionRequest
        print("✅ Prediction engine imports successful")
    except ImportError as e:
        print(f"❌ Prediction engine import failed: {e}")
        return False
    
    try:
        from src.services.results_formatter import ResultsFormatter
        print("✅ Results formatter imports successful")
    except ImportError as e:
        print(f"❌ Results formatter import failed: {e}")
        return False
    
    try:
        from web.components.input_forms import RaceInputForm
        from web.components.visualizations import PredictionVisualizer
        from web.components.insights_display import InsightsDisplay
        print("✅ Web components imports successful")
    except ImportError as e:
        print(f"❌ Web components import failed: {e}")
        return False
    
    try:
        from web.utils.session_state import SessionStateManager
        from web.utils.data_helpers import DataHelper
        print("✅ Web utilities imports successful")
    except ImportError as e:
        print(f"❌ Web utilities import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without Streamlit."""
    print("\nTesting basic functionality...")
    
    try:
        from src.services.prediction_engine import PredictionEngine
        from web.utils.data_helpers import DataHelper
        
        # Test data helper
        data_helper = DataHelper()
        training_data = data_helper.generate_training_data(10)
        
        assert 'features' in training_data
        assert 'targets' in training_data
        assert len(training_data['features']) == 10
        assert len(training_data['targets']) == 10
        
        print("✅ Data helper functionality works")
        
        # Test prediction engine initialization
        engine = PredictionEngine(model_type='random_forest')
        success = engine.initialize_model(training_data['features'], training_data['targets'])
        
        assert success is True
        print("✅ Prediction engine initialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_streamlit_compatibility():
    """Test Streamlit compatibility."""
    print("\nTesting Streamlit compatibility...")
    
    try:
        import streamlit as st
        print("✅ Streamlit is available")
        
        # Test that we can import plotly
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly is available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Streamlit compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏎️  F1 Race Prediction Streamlit App - Component Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test Streamlit compatibility
    if not test_streamlit_compatibility():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! The Streamlit app should work correctly.")
        print("\nTo run the app:")
        print("  streamlit run web/app.py")
        print("  or")
        print("  python demo_streamlit_app.py")
    else:
        print("❌ Some tests failed. Please fix the issues before running the app.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install streamlit plotly")
        print("2. Check that all source files are present")
        print("3. Verify Python path configuration")

if __name__ == "__main__":
    main()