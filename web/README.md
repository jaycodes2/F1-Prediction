# F1 Race Prediction Web Application

A comprehensive Streamlit-based web interface for Formula 1 race prediction using advanced machine learning models.

## 🏎️ Features

### Core Functionality
- **Race Prediction**: Generate predictions for F1 races with customizable parameters
- **Multiple ML Models**: Choose from Ensemble, Random Forest, XGBoost, and LightGBM models
- **Interactive Visualizations**: Rich charts and graphs powered by Plotly
- **Comprehensive Insights**: AI-generated race insights and analysis
- **Batch Processing**: Upload CSV files for multiple race predictions
- **Export Capabilities**: Download results in CSV or JSON format

### User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Intuitive Navigation**: Easy-to-use sidebar navigation
- **Real-time Updates**: Live prediction generation and visualization
- **Session Management**: Track prediction history and preferences
- **Custom Styling**: F1-themed design with team colors

### Analysis Features
- **Position Predictions**: Detailed finishing position forecasts
- **Confidence Analysis**: Prediction reliability assessment
- **Podium Probabilities**: Top-3 finishing chances for each driver
- **Overtaking Scenarios**: Track-specific overtaking analysis
- **Weather Impact**: Weather condition effects on race outcomes
- **Statistical Summaries**: Comprehensive prediction statistics

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit plotly pandas numpy
```

### Running the Application
```bash
# From project root directory
streamlit run web/app.py

# Or use the demo script
python demo_streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📱 Application Structure

### Pages

#### 🏁 Race Prediction
- Configure race parameters (circuit, drivers, weather)
- Select prediction model and confidence threshold
- Generate real-time predictions with visualizations
- View detailed position predictions and insights

#### 📊 Results Analysis
- **Visualizations Tab**: Interactive charts and graphs
- **Insights Tab**: AI-generated race insights and analysis
- **Detailed Data Tab**: Complete prediction data with export options
- **Comparisons Tab**: Compare multiple predictions

#### 📋 Batch Processing
- Upload CSV files with multiple race configurations
- Process batch predictions efficiently
- Download batch results
- Sample CSV template provided

#### ℹ️ About
- Application information and features
- Model performance metrics
- Technical details and documentation

### Components

#### Input Forms (`web/components/input_forms.py`)
- **RaceInputForm**: Race configuration interface
- Driver selection and grid position setup
- Weather conditions configuration
- Preset race configurations (Monaco, Silverstone, Monza)

#### Visualizations (`web/components/visualizations.py`)
- **PredictionVisualizer**: Chart generation for predictions
- Position predictions bar charts
- Confidence distribution histograms
- Podium probability analysis
- Overtaking scenario visualizations
- Weather impact analysis

#### Insights Display (`web/components/insights_display.py`)
- **InsightsDisplay**: Formatted insight presentation
- Key race insights with confidence indicators
- Podium analysis with battle detection
- Statistical summaries and confidence analysis
- Driver spotlight features

### Utilities

#### Session State (`web/utils/session_state.py`)
- **SessionStateManager**: Streamlit session management
- Prediction history tracking
- User preferences storage
- Session statistics calculation

#### Data Helpers (`web/utils/data_helpers.py`)
- **DataHelper**: Data processing utilities
- Training data generation
- CSV import/export functionality
- Prediction statistics calculation

## 🎯 Usage Examples

### Basic Race Prediction
1. Navigate to "Race Prediction" page
2. Configure race details (name, circuit, date)
3. Set up drivers and grid positions
4. Configure weather conditions
5. Click "Generate Prediction"
6. View results and insights

### Batch Processing
1. Go to "Batch Processing" page
2. Download sample CSV template
3. Fill in race configurations
4. Upload completed CSV file
5. Process batch predictions
6. Download results

### Advanced Analysis
1. Generate a race prediction
2. Navigate to "Results Analysis" page
3. Explore visualizations in different tabs
4. Review detailed insights and statistics
5. Export data for further analysis

## 🔧 Configuration

### Model Settings
- **Prediction Model**: Choose ML algorithm
- **Confidence Threshold**: Set minimum confidence level
- **Advanced Options**: Additional model parameters

### User Preferences
- **Theme**: Application appearance
- **Default Drivers**: Number of drivers to include
- **Show Advanced Options**: Display expert settings

## 📊 Data Formats

### CSV Upload Format
```csv
race_name,circuit,date,drivers,weather
Monaco Grand Prix 2024,Circuit de Monaco,2024-05-26,"[{""driver_id"":""VER"",""name"":""Max Verstappen"",""grid_position"":1}]","{""conditions"":""dry"",""track_temp"":42.0}"
```

### JSON Export Format
```json
{
  "race_info": {
    "race_name": "Monaco Grand Prix 2024",
    "overall_confidence": 0.85
  },
  "position_predictions": [...],
  "key_insights": [...],
  "statistical_summary": {...}
}
```

## 🎨 Customization

### Styling
The application uses custom CSS for F1-themed styling:
- Team colors for visualizations
- Responsive design elements
- Confidence level color coding
- Interactive hover effects

### Adding New Features
1. Create component in appropriate directory
2. Import in main application
3. Add to navigation if needed
4. Update documentation

## 🐛 Troubleshooting

### Common Issues

**Application won't start:**
- Check Streamlit installation: `pip install streamlit`
- Verify all dependencies are installed
- Run from project root directory

**Import errors:**
- Ensure all required packages are installed
- Check Python path configuration
- Verify project structure

**Prediction errors:**
- Check input data format
- Verify model initialization
- Review error messages in console

**Performance issues:**
- Reduce number of drivers for faster predictions
- Use simpler models for quick testing
- Clear session data regularly

### Getting Help
- Check console output for error messages
- Review application logs
- Verify input data formats
- Test with sample data first

## 🚀 Deployment

### Local Development
```bash
streamlit run web/app.py --server.port 8501
```

### Production Deployment
- **Streamlit Cloud**: Connect GitHub repository
- **Heroku**: Use Procfile with streamlit command
- **Docker**: Create container with Streamlit app
- **AWS/GCP**: Deploy on cloud platforms

### Environment Variables
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 📈 Performance

### Optimization Tips
- Use caching for expensive operations
- Limit visualization complexity
- Implement lazy loading for large datasets
- Optimize model inference speed

### Monitoring
- Track prediction generation times
- Monitor memory usage
- Log user interactions
- Analyze error rates

## 🔒 Security

### Best Practices
- Validate all user inputs
- Sanitize uploaded files
- Implement rate limiting
- Use secure session management

### Data Privacy
- No personal data collection
- Session data stored locally
- Optional analytics tracking
- Clear data retention policies

## 📝 License

This project is part of the F1 Race Prediction System. See main project LICENSE for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request
5. Follow code style guidelines

## 📞 Support

For issues and questions:
- Check troubleshooting section
- Review documentation
- Submit GitHub issues
- Contact development team