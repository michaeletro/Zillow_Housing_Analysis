# Zillow Housing Analysis

A comprehensive Python package for analyzing Zillow property data, building property valuation models, and predicting real estate values using machine learning.

## 🏠 Overview

This project provides tools to:
- **Fetch property data** from Zillow API (supports RapidAPI integration)
- **Extract comprehensive property information** including prices, features, and market trends
- **Clean and preprocess** real estate data for analysis
- **Engineer features** for better model performance
- **Train machine learning models** to predict property valuations
- **Evaluate properties** to identify undervalued/overvalued opportunities
- **Visualize insights** with comprehensive charts and dashboards

## 📋 Features

### Data Collection
- Zillow API integration with fallback methods
- Property search by location (city, state, zip code)
- Market trend analysis
- Comparable properties (comps) analysis
- Property detail extraction

### Data Processing
- Automated data cleaning and validation
- Feature engineering and derived metrics
- Missing value handling
- Outlier detection and treatment
- Location data normalization

### Machine Learning Models
- **Multiple algorithms supported:**
  - Random Forest
  - Gradient Boosting
  - XGBoost (default)
  - LightGBM
- Feature importance analysis
- Cross-validation
- Model persistence (save/load)

### Visualization
- Price distribution analysis
- Correlation matrices
- Feature importance charts
- Prediction analysis plots
- Geographic distribution maps
- Market trend dashboards

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/michaeletro/Zillow_Housing_Analysis.git
cd Zillow_Housing_Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
from zillow_analysis import ZillowAPIClient, PropertyValuationModel
from zillow_analysis.utils import clean_property_data, calculate_derived_features

# Initialize API client
client = ZillowAPIClient()

# Search for properties
properties = client.search_properties(
    location="New York, NY",
    status="forSale",
    max_results=100
)

# Get market trends
market_data = client.get_market_trends("New York, NY")

# Clean and prepare data
df = clean_property_data(properties)
df = calculate_derived_features(df)

# Train valuation model
model = PropertyValuationModel(model_type="xgboost")
metrics = model.train(df.to_dict('records'))

# Evaluate a property
evaluation = model.evaluate_property(properties[0])
print(f"Predicted Value: ${evaluation['predicted_value']:,.0f}")
print(f"Evaluation: {evaluation['evaluation']}")
```

### Run Example Script

```bash
python example_usage.py
```

### Interactive Analysis

Launch the Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebooks/zillow_analysis_demo.ipynb
```

## 📁 Project Structure

```
Zillow_Housing_Analysis/
├── zillow_analysis/          # Main package
│   ├── api/                  # API client modules
│   │   └── zillow_client.py  # Zillow API integration
│   ├── models/               # Data models and ML models
│   │   ├── property_data.py  # Property data structures
│   │   └── property_model.py # Valuation models
│   └── utils/                # Utility functions
│       ├── data_preprocessing.py  # Data cleaning
│       └── visualization.py       # Plotting functions
├── notebooks/                # Jupyter notebooks
│   └── zillow_analysis_demo.ipynb
├── tests/                    # Unit tests
├── example_usage.py          # Example script
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## 🔧 Configuration

### API Keys

The package supports multiple data sources:

1. **RapidAPI (Recommended):**
   - Sign up at [RapidAPI](https://rapidapi.com/)
   - Subscribe to Zillow API
   - Set `RAPIDAPI_KEY` environment variable

2. **Sample Data:**
   - Works without API keys for testing
   - Generates realistic sample data

### Environment Variables

Create a `.env` file with:

```bash
RAPIDAPI_KEY=your_rapidapi_key_here
DEFAULT_MODEL_TYPE=xgboost
TEST_SIZE=0.2
RANDOM_STATE=42
```

## 📊 Model Performance

The property valuation models achieve:
- **R² Score:** 0.85-0.95 (depending on data quality)
- **Mean Absolute Error (MAE):** Typically 5-10% of property value
- **Features:** 15+ engineered features including:
  - Property characteristics (beds, baths, sqft)
  - Location features (city, state, coordinates)
  - Derived metrics (price per sqft, age, ratios)
  - Market indicators

## 🎯 Use Cases

1. **Property Investment Analysis:**
   - Identify undervalued properties
   - Compare asking prices to model predictions
   - Analyze market trends

2. **Portfolio Valuation:**
   - Batch property valuation
   - Market comparison analysis
   - Investment opportunity scoring

3. **Market Research:**
   - Neighborhood price analysis
   - Feature importance insights
   - Trend forecasting

4. **Academic Research:**
   - Real estate market analysis
   - Machine learning experiments
   - Data science education

## 📈 Example Analysis

```python
# Find undervalued properties
predictions = model.predict(properties)
df['predicted_price'] = predictions
df['percent_diff'] = ((df['predicted_price'] - df['price']) / df['price']) * 100

undervalued = df[df['percent_diff'] < -10].sort_values('percent_diff')
print(f"Found {len(undervalued)} undervalued properties")

# Visualize results
from zillow_analysis.utils import plot_prediction_analysis
plot_prediction_analysis(df['price'], df['predicted_price'])
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Property valuations are estimates based on available data and should not be the sole basis for investment decisions. Always consult with real estate professionals and conduct proper due diligence.

## 🔗 Resources

- [Zillow API Documentation](https://www.zillow.com/howto/api/APIOverview.htm)
- [RapidAPI Zillow](https://rapidapi.com/apimaker/api/zillow-com1)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note:** This project requires API keys for full functionality. Sample data is provided for testing without API access.
