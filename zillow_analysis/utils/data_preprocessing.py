"""
Data Preprocessing Utilities

Functions for cleaning and preparing property data for analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


def clean_property_data(properties: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Clean and standardize property data.
    
    Args:
        properties: List of property dictionaries
        
    Returns:
        Cleaned DataFrame
    """
    df = pd.DataFrame(properties)
    
    # Remove duplicates based on zpid
    if 'zpid' in df.columns:
        df = df.drop_duplicates(subset=['zpid'], keep='first')
    
    # Handle missing values for numeric columns
    numeric_cols = ['price', 'bedrooms', 'bathrooms', 'living_area', 
                   'lot_size', 'year_built', 'tax_assessed_value']
    
    for col in numeric_cols:
        if col in df.columns:
            # Fill missing with median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # Remove outliers using IQR method for price
    if 'price' in df.columns:
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    
    # Convert year_built to age
    if 'year_built' in df.columns:
        df['age'] = datetime.now().year - df['year_built']
        df['age'] = df['age'].clip(lower=0)  # Ensure non-negative
    
    # Ensure positive values for key metrics
    positive_cols = ['living_area', 'lot_size', 'bedrooms', 'bathrooms']
    for col in positive_cols:
        if col in df.columns:
            df = df[df[col] > 0]
    
    return df


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features from raw property data.
    
    Args:
        df: DataFrame with property data
        
    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()
    
    # Price per square foot
    if 'price' in df.columns and 'living_area' in df.columns:
        df['price_per_sqft'] = df['price'] / df['living_area']
    
    # Bathroom to bedroom ratio
    if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
        df['bath_bed_ratio'] = df['bathrooms'] / df['bedrooms']
    
    # Lot to living area ratio
    if 'lot_size' in df.columns and 'living_area' in df.columns:
        df['lot_living_ratio'] = df['lot_size'] / df['living_area']
    
    # Price to tax assessment ratio
    if 'price' in df.columns and 'tax_assessed_value' in df.columns:
        df['price_tax_ratio'] = df['price'] / (df['tax_assessed_value'] + 1)
    
    # Total rooms
    if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Property age category
    if 'age' in df.columns:
        df['age_category'] = pd.cut(
            df['age'], 
            bins=[-1, 5, 20, 50, 100, 200],
            labels=['New', 'Modern', 'Mature', 'Old', 'Historic']
        )
    
    # Size category
    if 'living_area' in df.columns:
        df['size_category'] = pd.cut(
            df['living_area'],
            bins=[0, 1000, 2000, 3000, 10000],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
    
    return df


def normalize_location_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and standardize location data.
    
    Args:
        df: DataFrame with property data
        
    Returns:
        DataFrame with normalized location data
    """
    df = df.copy()
    
    # Standardize state abbreviations
    if 'state' in df.columns:
        df['state'] = df['state'].str.strip().str.upper()
    
    # Standardize city names
    if 'city' in df.columns:
        df['city'] = df['city'].str.strip().str.title()
    
    # Clean zip codes
    if 'zipcode' in df.columns:
        df['zipcode'] = df['zipcode'].astype(str).str.strip()
        # Extract first 5 digits
        df['zipcode'] = df['zipcode'].str[:5]
    
    return df


def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers in specified columns.
    
    Args:
        df: DataFrame with property data
        columns: List of columns to check for outliers
        method: Method to use - 'iqr' or 'zscore'
        
    Returns:
        DataFrame with outlier flags
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            df[f'{col}_outlier'] = z_scores > 3
    
    return df


def aggregate_by_location(df: pd.DataFrame, group_by: str = 'city') -> pd.DataFrame:
    """
    Aggregate property data by location.
    
    Args:
        df: DataFrame with property data
        group_by: Column to group by ('city', 'state', 'zipcode')
        
    Returns:
        Aggregated DataFrame
    """
    if group_by not in df.columns:
        raise ValueError(f"Column {group_by} not found in DataFrame")
    
    agg_dict = {}
    
    if 'price' in df.columns:
        agg_dict['price'] = ['mean', 'median', 'std', 'min', 'max', 'count']
    
    if 'price_per_sqft' in df.columns:
        agg_dict['price_per_sqft'] = ['mean', 'median']
    
    if 'living_area' in df.columns:
        agg_dict['living_area'] = ['mean', 'median']
    
    if 'bedrooms' in df.columns:
        agg_dict['bedrooms'] = ['mean']
    
    if 'days_on_zillow' in df.columns:
        agg_dict['days_on_zillow'] = ['mean', 'median']
    
    aggregated = df.groupby(group_by).agg(agg_dict)
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    
    return aggregated.reset_index()


def split_train_test(df: pd.DataFrame, 
                     test_size: float = 0.2, 
                     random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.
    
    Args:
        df: DataFrame with property data
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'median',
                          threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with property data
        strategy: Strategy for filling missing values ('median', 'mean', 'mode', 'drop')
        threshold: If strategy is 'drop', drop columns with more than this proportion of missing
        
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    if strategy == 'drop':
        # Drop columns with too many missing values
        missing_prop = df.isnull().sum() / len(df)
        cols_to_keep = missing_prop[missing_prop < threshold].index
        df = df[cols_to_keep]
        
        # Drop rows with any remaining missing values
        df = df.dropna()
    
    elif strategy in ['median', 'mean']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                fill_value = df[col].median() if strategy == 'median' else df[col].mean()
                df[col] = df[col].fillna(fill_value)
    
    elif strategy == 'mode':
        for col in df.columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_value)
    
    return df
