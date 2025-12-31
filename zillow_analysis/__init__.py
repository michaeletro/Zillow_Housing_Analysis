"""
Zillow Housing Analysis Package
A comprehensive tool for analyzing Zillow property data and predicting valuations.
"""

__version__ = "1.0.0"
__author__ = "Zillow Analysis Team"

from .api.zillow_client import ZillowAPIClient
from .models.property_model import PropertyValuationModel
from .models.property_data import PropertyData

__all__ = [
    "ZillowAPIClient",
    "PropertyValuationModel",
    "PropertyData",
]
