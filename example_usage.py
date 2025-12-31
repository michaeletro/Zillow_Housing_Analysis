"""
Example script demonstrating how to use the Zillow Housing Analysis package.

This script shows:
1. Fetching property data from Zillow API
2. Cleaning and preprocessing the data
3. Training a valuation model
4. Making predictions and evaluating properties
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zillow_analysis import ZillowAPIClient, PropertyValuationModel
from zillow_analysis.utils import (
    clean_property_data, 
    calculate_derived_features,
    plot_price_distribution,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_prediction_analysis
)
import pandas as pd
import numpy as np


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("Zillow Housing Analysis - Example Usage")
    print("=" * 80)
    print()
    
    # Step 1: Initialize the API client
    print("Step 1: Initializing Zillow API Client...")
    client = ZillowAPIClient()
    print("✓ Client initialized")
    print()
    
    # Step 2: Search for properties
    print("Step 2: Searching for properties...")
    location = "New York, NY"
    properties = client.search_properties(
        location=location,
        status="forSale",
        max_results=100
    )
    print(f"✓ Found {len(properties)} properties in {location}")
    print()
    
    # Step 3: Get market trends
    print("Step 3: Fetching market trends...")
    market_data = client.get_market_trends(location)
    print(f"✓ Market data retrieved for {location}")
    print(f"   Median Home Price: ${market_data['median_home_price']:,}")
    print(f"   Price Change (1yr): {market_data['price_change_1yr']:.2f}%")
    print(f"   Days on Market: {market_data['days_on_market']}")
    print()
    
    # Step 4: Clean and preprocess data
    print("Step 4: Cleaning and preprocessing data...")
    df = clean_property_data(properties)
    df = calculate_derived_features(df)
    print(f"✓ Data cleaned: {len(df)} properties ready for analysis")
    print()
    
    # Step 5: Data exploration
    print("Step 5: Basic data statistics...")
    print(df[['price', 'bedrooms', 'bathrooms', 'living_area']].describe())
    print()
    
    # Step 6: Train valuation model
    print("Step 6: Training property valuation model...")
    model = PropertyValuationModel(model_type="xgboost")
    
    # Prepare data for training
    properties_list = df.to_dict('records')
    
    # Train the model
    metrics = model.train(properties_list, validate=True)
    print(f"✓ Model training complete!")
    print(f"   Test R² Score: {metrics['test_r2']:.4f}")
    print(f"   Test MAE: ${metrics['test_mae']:,.0f}")
    print(f"   Test RMSE: ${metrics['test_rmse']:,.0f}")
    if 'cv_r2_mean' in metrics:
        print(f"   Cross-validation R² (mean): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print()
    
    # Step 7: Feature importance
    print("Step 7: Analyzing feature importance...")
    importance = model.get_feature_importance(top_n=10)
    print("Top 10 Most Important Features:")
    print(importance.to_string(index=False))
    print()
    
    # Step 8: Evaluate a sample property
    print("Step 8: Evaluating a sample property...")
    sample_property = properties[0]
    evaluation = model.evaluate_property(sample_property)
    
    print(f"Property Address: {sample_property.get('address', 'N/A')}")
    print(f"Asking Price: ${evaluation['actual_price']:,.0f}")
    print(f"Model Predicted Value: ${evaluation['predicted_value']:,.0f}")
    if evaluation['difference']:
        print(f"Difference: ${evaluation['difference']:,.0f} ({evaluation['percent_difference']:.2f}%)")
    print(f"Evaluation: {evaluation['evaluation']}")
    print()
    
    # Step 9: Save model
    print("Step 9: Saving trained model...")
    model_path = "zillow_analysis/data/property_valuation_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"✓ Model saved to {model_path}")
    print()
    
    # Step 10: Export data
    print("Step 10: Exporting property data...")
    client.export_properties_to_csv(properties, "zillow_analysis/data/properties.csv")
    client.export_properties_to_json(properties, "zillow_analysis/data/properties.json")
    print("✓ Data exported to CSV and JSON")
    print()
    
    # Step 11: Visualizations (optional - commented out for non-interactive environments)
    print("Step 11: Creating visualizations...")
    print("Note: Visualizations are available but skipped in this example.")
    print("      Uncomment the following lines to generate plots:")
    print("      - plot_price_distribution(df)")
    print("      - plot_correlation_matrix(df)")
    print("      - plot_feature_importance(importance)")
    print()
    
    # Uncomment these lines to generate visualizations:
    # plot_price_distribution(df, save_path='zillow_analysis/data/price_distribution.png')
    # plot_correlation_matrix(df, save_path='zillow_analysis/data/correlation_matrix.png')
    # plot_feature_importance(importance, save_path='zillow_analysis/data/feature_importance.png')
    
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the exported data files in zillow_analysis/data/")
    print("2. Check the trained model for making predictions")
    print("3. Use the Jupyter notebook for interactive analysis")
    print("4. Customize the model parameters for better results")
    print()


if __name__ == "__main__":
    main()
