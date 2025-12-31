# Zillow Housing Analysis - Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [API Client Usage](#api-client-usage)
3. [Data Processing](#data-processing)
4. [Model Training](#model-training)
5. [Making Predictions](#making-predictions)
6. [Visualization](#visualization)
7. [Advanced Usage](#advanced-usage)

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/michaeletro/Zillow_Housing_Analysis.git
cd Zillow_Housing_Analysis

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from zillow_analysis import ZillowAPIClient, PropertyValuationModel

# Initialize client and search for properties
client = ZillowAPIClient()
properties = client.search_properties("New York, NY", max_results=50)

# Train a model
model = PropertyValuationModel()
model.train(properties)

# Evaluate a property
evaluation = model.evaluate_property(properties[0])
print(f"Predicted value: ${evaluation['predicted_value']:,.0f}")
```

## API Client Usage

### Initialize the Client

```python
from zillow_analysis import ZillowAPIClient

# Option 1: Use environment variables
client = ZillowAPIClient()

# Option 2: Provide API keys directly
client = ZillowAPIClient(rapidapi_key="your_key_here")
```

### Search for Properties

```python
# Search by city and state
properties = client.search_properties(
    location="San Francisco, CA",
    status="forSale",
    max_results=100
)

# Search by zip code
properties = client.search_properties(
    location="94102",
    status="forRent",
    max_results=50
)
```

### Get Property Details

```python
# Get detailed information for a specific property
details = client.get_property_details(zpid="12345678")
print(details['price_history'])
print(details['tax_history'])
```

### Get Market Trends

```python
# Get market trends for a location
trends = client.get_market_trends("Seattle, WA")
print(f"Median price: ${trends['median_home_price']:,}")
print(f"Price change (1yr): {trends['price_change_1yr']}%")
```

### Get Comparable Properties

```python
# Get comparable properties
comps = client.get_comparable_properties(zpid="12345678", count=10)
for comp in comps:
    print(f"{comp['address']}: ${comp['price']:,}")
```

### Export Data

```python
# Export to CSV
client.export_properties_to_csv(properties, "properties.csv")

# Export to JSON
client.export_properties_to_json(properties, "properties.json")
```

## Data Processing

### Clean Property Data

```python
from zillow_analysis.utils import clean_property_data
import pandas as pd

# Convert to DataFrame and clean
df = clean_property_data(properties)

# This removes duplicates, handles missing values, and filters outliers
print(f"Cleaned {len(df)} properties")
```

### Calculate Derived Features

```python
from zillow_analysis.utils import calculate_derived_features

# Add calculated features
df = calculate_derived_features(df)

# New features include:
# - price_per_sqft
# - bath_bed_ratio
# - lot_living_ratio
# - age
# - age_category
# - size_category
```

### Handle Missing Values

```python
from zillow_analysis.utils import handle_missing_values

# Strategy options: 'median', 'mean', 'mode', 'drop'
df = handle_missing_values(df, strategy='median')
```

### Detect Outliers

```python
from zillow_analysis.utils import detect_outliers

# Detect outliers in specific columns
df = detect_outliers(df, columns=['price', 'living_area'], method='iqr')

# Check outlier flags
outliers = df[df['price_outlier'] == True]
print(f"Found {len(outliers)} price outliers")
```

### Aggregate by Location

```python
from zillow_analysis.utils import aggregate_by_location

# Aggregate statistics by city
city_stats = aggregate_by_location(df, group_by='city')
print(city_stats[['city', 'price_mean', 'price_median']])
```

## Model Training

### Initialize a Model

```python
from zillow_analysis import PropertyValuationModel

# Choose model type: 'random_forest', 'gradient_boost', 'xgboost', 'lightgbm'
model = PropertyValuationModel(model_type='xgboost')
```

### Train the Model

```python
# Train on property data
properties_list = df.to_dict('records')
metrics = model.train(
    properties=properties_list,
    target_col='price',
    test_size=0.2,
    validate=True
)

# View performance metrics
print(f"R² Score: {metrics['test_r2']:.4f}")
print(f"MAE: ${metrics['test_mae']:,.0f}")
print(f"RMSE: ${metrics['test_rmse']:,.0f}")
```

### Feature Importance

```python
# Get feature importance
importance = model.get_feature_importance(top_n=15)
print(importance)

# Visualize
from zillow_analysis.utils import plot_feature_importance
plot_feature_importance(importance)
```

### Save and Load Models

```python
# Save trained model
model.save_model('models/my_valuation_model.pkl')

# Load saved model
new_model = PropertyValuationModel()
new_model.load_model('models/my_valuation_model.pkl')
```

## Making Predictions

### Predict Property Values

```python
# Predict for multiple properties
predictions = model.predict(properties_list)

# Add to DataFrame
df['predicted_price'] = predictions
df['difference'] = df['predicted_price'] - df['price']
```

### Evaluate Single Property

```python
# Detailed evaluation of a property
property_data = properties[0]
evaluation = model.evaluate_property(property_data)

print(f"Asking Price: ${evaluation['actual_price']:,.0f}")
print(f"Predicted Value: ${evaluation['predicted_value']:,.0f}")
print(f"Difference: ${evaluation['difference']:,.0f}")
print(f"Assessment: {evaluation['evaluation']}")
```

### Find Investment Opportunities

```python
# Find undervalued properties
df['percent_diff'] = ((df['predicted_price'] - df['price']) / df['price']) * 100
undervalued = df[df['percent_diff'] < -10].sort_values('percent_diff')

print(f"Found {len(undervalued)} undervalued properties")
print(undervalued[['address', 'price', 'predicted_price', 'percent_diff']].head())
```

## Visualization

### Price Distribution

```python
from zillow_analysis.utils import plot_price_distribution

plot_price_distribution(df, save_path='price_dist.png')
```

### Correlation Matrix

```python
from zillow_analysis.utils import plot_correlation_matrix

# Select specific columns
columns = ['price', 'bedrooms', 'bathrooms', 'living_area', 'lot_size']
plot_correlation_matrix(df, columns=columns, save_path='correlation.png')
```

### Price vs Features

```python
from zillow_analysis.utils import plot_price_vs_features

features = ['living_area', 'year_built', 'bedrooms']
plot_price_vs_features(df, features=features, save_path='price_vs_features.png')
```

### Geographic Distribution

```python
from zillow_analysis.utils import plot_geographic_distribution

plot_geographic_distribution(df, save_path='geo_map.png')
```

### Prediction Analysis

```python
from zillow_analysis.utils import plot_prediction_analysis

plot_prediction_analysis(df['price'], df['predicted_price'], save_path='predictions.png')
```

### Market Trends

```python
from zillow_analysis.utils import plot_market_trends

market_data = client.get_market_trends("Boston, MA")
plot_market_trends(market_data, save_path='market_trends.png')
```

### Summary Dashboard

```python
from zillow_analysis.utils import create_summary_dashboard

create_summary_dashboard(df, save_path='dashboard.png')
```

## Advanced Usage

### Compare Multiple Models

```python
from zillow_analysis import PropertyValuationModel

models = ['random_forest', 'gradient_boost', 'xgboost', 'lightgbm']
results = {}

for model_type in models:
    model = PropertyValuationModel(model_type=model_type)
    metrics = model.train(properties_list, validate=False)
    results[model_type] = metrics['test_r2']

# Find best model
best_model = max(results, key=results.get)
print(f"Best model: {best_model} (R² = {results[best_model]:.4f})")
```

### Batch Property Evaluation

```python
# Evaluate multiple properties efficiently
evaluations = []
for prop in properties_list:
    eval_result = model.evaluate_property(prop)
    evaluations.append({
        'zpid': prop['zpid'],
        'address': prop['address'],
        'asking_price': eval_result['actual_price'],
        'predicted_price': eval_result['predicted_value'],
        'evaluation': eval_result['evaluation']
    })

eval_df = pd.DataFrame(evaluations)
```

### Custom Feature Engineering

```python
# Add your own custom features
df['price_premium'] = (df['price'] - df['tax_assessed_value']) / df['tax_assessed_value']
df['luxury_score'] = (df['bathrooms'] / df['bedrooms']) * (df['living_area'] / 1000)
df['value_score'] = df['price'] / (df['living_area'] * df['age'] + 1)

# Use in model training
properties_with_custom = df.to_dict('records')
model.train(properties_with_custom)
```

### Time-Based Analysis

```python
# Filter by days on market
recent = df[df['days_on_zillow'] < 30]
stale = df[df['days_on_zillow'] > 180]

print(f"Recent listings: {len(recent)}")
print(f"Stale listings: {len(stale)}")
print(f"Average price difference: ${(recent['price'].mean() - stale['price'].mean()):,.0f}")
```

### Location-Based Analysis

```python
# Compare neighborhoods
from zillow_analysis.utils import aggregate_by_location

# Get stats by city
city_analysis = aggregate_by_location(df, group_by='city')

# Sort by median price
city_analysis = city_analysis.sort_values('price_median', ascending=False)
print("Top 5 most expensive cities:")
print(city_analysis[['city', 'price_median', 'price_per_sqft_mean']].head())
```

### Portfolio Analysis

```python
# Analyze a portfolio of properties
portfolio_zpids = ['zpid1', 'zpid2', 'zpid3']
portfolio = df[df['zpid'].isin(portfolio_zpids)]

portfolio_value = portfolio['predicted_price'].sum()
portfolio_cost = portfolio['price'].sum()
potential_gain = portfolio_value - portfolio_cost

print(f"Portfolio Summary:")
print(f"  Properties: {len(portfolio)}")
print(f"  Total Cost: ${portfolio_cost:,.0f}")
print(f"  Predicted Value: ${portfolio_value:,.0f}")
print(f"  Potential Gain: ${potential_gain:,.0f} ({(potential_gain/portfolio_cost)*100:.2f}%)")
```

## Tips and Best Practices

1. **API Rate Limiting**: Be mindful of API rate limits. Use the built-in delay mechanism.

2. **Data Quality**: Always clean and validate data before training models.

3. **Model Selection**: Try multiple models and use cross-validation to select the best one.

4. **Feature Engineering**: The quality of features often matters more than the model choice.

5. **Regular Updates**: Property data changes frequently. Retrain models regularly with fresh data.

6. **Validation**: Always validate predictions with market knowledge and comparable sales.

7. **Outliers**: Handle extreme values carefully - they may be legitimate luxury properties or data errors.

8. **Location Specificity**: Train separate models for different markets when possible.

## Troubleshooting

### No API Key

If you don't have API keys, the system will use sample data generation for testing.

### Memory Issues

For large datasets, process in batches:

```python
# Process in chunks
chunk_size = 1000
for i in range(0, len(properties), chunk_size):
    chunk = properties[i:i+chunk_size]
    # Process chunk
```

### Poor Model Performance

- Increase training data size
- Add more relevant features
- Try different model types
- Tune hyperparameters
- Check for data quality issues

## Support

For issues or questions:
- Open an issue on GitHub
- Check the example scripts
- Review the Jupyter notebook tutorial

## Next Steps

1. Run `example_usage.py` to see the full workflow
2. Explore `notebooks/zillow_analysis_demo.ipynb` for interactive analysis
3. Customize models and features for your specific use case
4. Build your own analysis tools on top of this foundation
