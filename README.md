# F1 Race Prediction System

A machine learning application that predicts Formula 1 race finishing positions based on historical data, qualifying results, driver performance, and track characteristics.

## Project Structure

```
f1-race-prediction/
├── src/
│   ├── data/           # Data collection and storage
│   ├── models/         # ML models and data structures
│   ├── services/       # Business logic and prediction services
│   └── web/           # Web interface components
├── data/
│   ├── raw/           # Raw data from APIs
│   └── processed/     # Processed and engineered features
├── models/            # Trained model artifacts
├── tests/             # Test modules
└── requirements.txt   # Python dependencies
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure as needed
5. Run the application:
   ```bash
   streamlit run src/web/app.py
   ```

## Features

- Historical F1 data collection from Ergast API and FastF1
- Advanced feature engineering with rolling statistics
- Ensemble machine learning models (Random Forest, XGBoost, LightGBM)
- Interactive web interface for race predictions
- Batch prediction capabilities
- Comprehensive model evaluation and validation

## Development Status

This project is currently under development. See the implementation plan in `.kiro/specs/f1-race-prediction/tasks.md` for current progress.