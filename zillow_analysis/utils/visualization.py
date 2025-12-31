"""
Data Visualization Utilities

Functions for visualizing property data and analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_price_distribution(df: pd.DataFrame, 
                            price_col: str = 'price',
                            save_path: Optional[str] = None):
    """
    Plot price distribution histogram and box plot.
    
    Args:
        df: DataFrame with property data
        price_col: Name of price column
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df[price_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Price Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df[price_col], vert=True)
    axes[1].set_ylabel('Price ($)')
    axes[1].set_title('Price Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame with property data
        columns: List of columns to include (None for all numeric)
        save_path: Optional path to save the figure
    """
    if columns:
        corr_df = df[columns]
    else:
        corr_df = df.select_dtypes(include=[np.number])
    
    correlation = corr_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_price_vs_features(df: pd.DataFrame,
                           features: List[str],
                           price_col: str = 'price',
                           save_path: Optional[str] = None):
    """
    Plot price vs various features.
    
    Args:
        df: DataFrame with property data
        features: List of feature columns to plot
        price_col: Name of price column
        save_path: Optional path to save the figure
    """
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        if feature in df.columns:
            axes[idx].scatter(df[feature], df[price_col], alpha=0.5)
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel(price_col)
            axes[idx].set_title(f'{price_col} vs {feature}')
            axes[idx].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df[feature].dropna(), 
                          df.loc[df[feature].notna(), price_col], 1)
            p = np.poly1d(z)
            axes[idx].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_price_by_category(df: pd.DataFrame,
                           category_col: str,
                           price_col: str = 'price',
                           save_path: Optional[str] = None):
    """
    Plot price distribution by category.
    
    Args:
        df: DataFrame with property data
        category_col: Name of categorical column
        price_col: Name of price column
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Box plot
    df.boxplot(column=price_col, by=category_col, figsize=(12, 6))
    plt.xlabel(category_col)
    plt.ylabel(price_col)
    plt.title(f'{price_col} Distribution by {category_col}')
    plt.suptitle('')  # Remove default title
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_geographic_distribution(df: pd.DataFrame,
                                 lat_col: str = 'latitude',
                                 lon_col: str = 'longitude',
                                 price_col: str = 'price',
                                 save_path: Optional[str] = None):
    """
    Plot geographic distribution of properties.
    
    Args:
        df: DataFrame with property data
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        price_col: Name of price column for color coding
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(df[lon_col], 
                         df[lat_col], 
                         c=df[price_col],
                         cmap='viridis',
                         alpha=0.6,
                         s=50)
    
    plt.colorbar(scatter, label=price_col)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of Properties')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 15,
                           save_path: Optional[str] = None):
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_analysis(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            save_path: Optional[str] = None):
    """
    Plot prediction vs actual values analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Scatter plot: Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2)
    axes[0].set_xlabel('Actual Price')
    axes[0].set_ylabel('Predicted Price')
    axes[0].set_title('Predicted vs Actual')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Price')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residuals Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_market_trends(market_data: Dict[str, Any],
                       save_path: Optional[str] = None):
    """
    Plot market trends visualization.
    
    Args:
        market_data: Dictionary with market data
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Price metrics
    axes[0, 0].bar(['Median Price', 'Price/SqFt'], 
                   [market_data['median_home_price'], 
                    market_data['median_price_per_sqft']*10])  # Scale for visibility
    axes[0, 0].set_ylabel('Value ($)')
    axes[0, 0].set_title('Price Metrics')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Price changes
    axes[0, 1].bar(['1 Year', '5 Year'], 
                   [market_data['price_change_1yr'], 
                    market_data['price_change_5yr']])
    axes[0, 1].set_ylabel('Change (%)')
    axes[0, 1].set_title('Price Changes')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Market metrics
    axes[1, 0].bar(['Inventory', 'Days on Market'], 
                   [market_data['inventory'], 
                    market_data['days_on_market']*10])  # Scale for visibility
    axes[1, 0].set_ylabel('Count / Days')
    axes[1, 0].set_title('Market Metrics')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Demand and forecast
    axes[1, 1].bar(['Demand Score', 'Forecast (1yr)'], 
                   [market_data['demand_score'], 
                    market_data['forecast_1yr']*10])  # Scale for visibility
    axes[1, 1].set_ylabel('Score / % Change')
    axes[1, 1].set_title('Demand & Forecast')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Market Trends for {market_data['location']}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_summary_dashboard(df: pd.DataFrame,
                            price_col: str = 'price',
                            save_path: Optional[str] = None):
    """
    Create a comprehensive summary dashboard.
    
    Args:
        df: DataFrame with property data
        price_col: Name of price column
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Price distribution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(df[price_col], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Price ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Price Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Summary statistics
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""
    Summary Statistics
    
    Count: {len(df):,}
    Mean: ${df[price_col].mean():,.0f}
    Median: ${df[price_col].median():,.0f}
    Std: ${df[price_col].std():,.0f}
    Min: ${df[price_col].min():,.0f}
    Max: ${df[price_col].max():,.0f}
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=10, 
            verticalalignment='center', family='monospace')
    
    # Scatter plots for key features
    features = ['living_area', 'bedrooms', 'bathrooms']
    for idx, feature in enumerate(features):
        if feature in df.columns:
            ax = fig.add_subplot(gs[1 + idx//3, idx%3])
            ax.scatter(df[feature], df[price_col], alpha=0.5)
            ax.set_xlabel(feature)
            ax.set_ylabel(price_col)
            ax.set_title(f'{price_col} vs {feature}')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Property Analysis Dashboard', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
