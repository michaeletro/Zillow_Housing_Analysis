"""
Property Data Models

Defines data structures for property information.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class PropertyData(BaseModel):
    """Property data model with validation."""
    
    zpid: str = Field(..., description="Zillow Property ID")
    address: str
    city: str
    state: str
    zipcode: str
    
    # Pricing
    price: Optional[float] = None
    zestimate: Optional[float] = None
    rent_zestimate: Optional[float] = None
    tax_assessed_value: Optional[float] = None
    
    # Property characteristics
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    living_area: Optional[float] = None
    lot_size: Optional[float] = None
    year_built: Optional[int] = None
    property_type: Optional[str] = None
    
    # Status
    status: Optional[str] = None
    days_on_zillow: Optional[int] = None
    
    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Metadata
    description: Optional[str] = None
    views: Optional[int] = None
    saves: Optional[int] = None
    last_sold_date: Optional[str] = None
    last_sold_price: Optional[float] = None
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "zpid": "12345678",
                "address": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zipcode": "10001",
                "price": 500000,
                "bedrooms": 3,
                "bathrooms": 2,
                "living_area": 1500,
                "property_type": "Single Family"
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "PropertyData":
        """Create PropertyData from API response."""
        return cls(**data)


class PropertyFeatures(BaseModel):
    """Extended property features for analysis."""
    
    # Calculated features
    price_per_sqft: Optional[float] = None
    age: Optional[int] = None
    bathrooms_per_bedroom: Optional[float] = None
    
    # Market features
    neighborhood_median_price: Optional[float] = None
    price_vs_neighborhood: Optional[float] = None  # Percentage difference
    
    # Time features
    days_on_market: Optional[int] = None
    season_listed: Optional[str] = None
    
    # Proximity features (in miles)
    distance_to_school: Optional[float] = None
    distance_to_transit: Optional[float] = None
    distance_to_downtown: Optional[float] = None
    
    # Quality indicators
    school_rating: Optional[float] = None
    walkability_score: Optional[int] = None
    crime_index: Optional[float] = None


class MarketData(BaseModel):
    """Market trend data for a location."""
    
    location: str
    median_home_price: float
    median_price_per_sqft: float
    price_change_1yr: float
    price_change_5yr: float
    inventory: int
    days_on_market: int
    forecast_1yr: float
    demand_score: float
    last_updated: str
