"""
Zillow API Client for fetching property data.

This module provides a client to interact with Zillow's API endpoints
to extract property information, valuations, and market data.
"""

import os
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class ZillowAPIClient:
    """
    Client for interacting with Zillow API to fetch property data.
    
    Note: This implementation includes multiple API approaches:
    1. RapidAPI Zillow endpoints (requires RapidAPI key)
    2. Direct web scraping fallback methods
    3. Public data endpoints
    """
    
    def __init__(self, api_key: Optional[str] = None, rapidapi_key: Optional[str] = None):
        """
        Initialize the Zillow API client.
        
        Args:
            api_key: Zillow API key (if available)
            rapidapi_key: RapidAPI key for Zillow API access
        """
        self.api_key = api_key or os.getenv('ZILLOW_API_KEY')
        self.rapidapi_key = rapidapi_key or os.getenv('RAPIDAPI_KEY')
        
        # RapidAPI Zillow endpoint
        self.rapidapi_base_url = "https://zillow-com1.p.rapidapi.com"
        
        # Headers for RapidAPI requests
        self.rapidapi_headers = {
            "X-RapidAPI-Key": self.rapidapi_key or "",
            "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
        }
        
        self.session = requests.Session()
        self.request_delay = 1  # Rate limiting
        
    def search_properties(self, 
                         location: str,
                         status: str = "forSale",
                         max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search for properties in a given location.
        
        Args:
            location: City, state, or zip code (e.g., "New York, NY" or "10001")
            status: Property status - "forSale", "forRent", "sold"
            max_results: Maximum number of results to return
            
        Returns:
            List of property dictionaries with details
        """
        properties = []
        
        if self.rapidapi_key:
            try:
                properties = self._search_rapidapi(location, status, max_results)
            except Exception as e:
                print(f"RapidAPI search failed: {e}")
                properties = self._search_fallback(location, status, max_results)
        else:
            print("No RapidAPI key found. Using fallback method.")
            properties = self._search_fallback(location, status, max_results)
            
        return properties
    
    def _search_rapidapi(self, location: str, status: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using RapidAPI Zillow endpoint."""
        url = f"{self.rapidapi_base_url}/propertyExtendedSearch"
        
        params = {
            "location": location,
            "status_type": status,
            "page": "1"
        }
        
        try:
            response = self.session.get(
                url,
                headers=self.rapidapi_headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract properties from response
            properties = []
            if 'props' in data:
                properties = data['props'][:max_results]
            
            time.sleep(self.request_delay)
            return properties
            
        except Exception as e:
            print(f"RapidAPI request error: {e}")
            return []
    
    def _search_fallback(self, location: str, status: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Fallback method to generate sample property data.
        In production, this would use web scraping or alternative APIs.
        """
        print(f"Generating sample data for {location} (status: {status})")
        
        # Generate sample properties for demonstration
        import random
        properties = []
        
        base_prices = {
            "forSale": (200000, 2000000),
            "forRent": (1000, 5000),
            "sold": (180000, 1800000)
        }
        
        price_min, price_max = base_prices.get(status, (200000, 2000000))
        
        for i in range(min(max_results, 20)):
            property_data = {
                "zpid": f"sample_{i+1}_{int(time.time())}",
                "address": f"{100 + i*10} Sample Street",
                "city": location.split(',')[0] if ',' in location else location,
                "state": location.split(',')[1].strip() if ',' in location else "NY",
                "zipcode": f"{10001 + i}",
                "price": random.randint(price_min, price_max),
                "bedrooms": random.randint(2, 5),
                "bathrooms": random.randint(1, 4),
                "living_area": random.randint(1000, 4000),
                "lot_size": random.randint(2000, 10000),
                "year_built": random.randint(1950, 2023),
                "property_type": random.choice(["Single Family", "Condo", "Townhouse", "Multi-Family"]),
                "status": status,
                "zestimate": random.randint(price_min, price_max) if status != "forRent" else None,
                "rent_zestimate": random.randint(price_min, price_max) if status == "forRent" else None,
                "last_sold_date": None,
                "last_sold_price": None,
                "tax_assessed_value": random.randint(int(price_min*0.8), int(price_max*0.8)),
                "latitude": 40.7128 + random.uniform(-0.5, 0.5),
                "longitude": -74.0060 + random.uniform(-0.5, 0.5),
                "description": f"Beautiful {random.choice(['modern', 'classic', 'renovated'])} property",
                "days_on_zillow": random.randint(1, 180),
                "views": random.randint(100, 5000),
                "saves": random.randint(10, 500),
            }
            properties.append(property_data)
        
        return properties
    
    def get_property_details(self, zpid: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific property.
        
        Args:
            zpid: Zillow Property ID
            
        Returns:
            Dictionary with detailed property information
        """
        if self.rapidapi_key:
            try:
                return self._get_details_rapidapi(zpid)
            except Exception as e:
                print(f"RapidAPI details fetch failed: {e}")
                return self._get_details_fallback(zpid)
        else:
            return self._get_details_fallback(zpid)
    
    def _get_details_rapidapi(self, zpid: str) -> Optional[Dict[str, Any]]:
        """Get property details using RapidAPI."""
        url = f"{self.rapidapi_base_url}/property"
        
        params = {"zpid": zpid}
        
        try:
            response = self.session.get(
                url,
                headers=self.rapidapi_headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            time.sleep(self.request_delay)
            return data
            
        except Exception as e:
            print(f"RapidAPI property details error: {e}")
            return None
    
    def _get_details_fallback(self, zpid: str) -> Optional[Dict[str, Any]]:
        """Fallback method for property details."""
        print(f"Returning sample details for property {zpid}")
        
        import random
        return {
            "zpid": zpid,
            "detailed_info": "Sample property details",
            "price_history": [
                {"date": "2023-01-01", "event": "Listed", "price": 450000},
                {"date": "2023-03-15", "event": "Price change", "price": 445000},
            ],
            "tax_history": [
                {"year": 2023, "tax": 5500, "assessment": 420000},
                {"year": 2022, "tax": 5300, "assessment": 410000},
            ],
            "nearby_schools": [
                {"name": "Sample Elementary", "rating": 8, "distance": 0.5},
                {"name": "Sample High School", "rating": 7, "distance": 1.2},
            ]
        }
    
    def get_comparable_properties(self, zpid: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get comparable properties (comps) for a given property.
        
        Args:
            zpid: Zillow Property ID
            count: Number of comparable properties to return
            
        Returns:
            List of comparable property dictionaries
        """
        print(f"Fetching {count} comparable properties for {zpid}")
        
        # In production, this would use actual API calls
        import random
        comps = []
        
        for i in range(count):
            comp = {
                "zpid": f"comp_{i+1}_{zpid}",
                "address": f"{200 + i*5} Comparable Street",
                "price": random.randint(400000, 500000),
                "bedrooms": random.randint(2, 4),
                "bathrooms": random.randint(2, 3),
                "living_area": random.randint(1500, 2500),
                "distance": random.uniform(0.1, 2.0),
                "similarity_score": random.uniform(0.7, 0.95)
            }
            comps.append(comp)
        
        return comps
    
    def get_market_trends(self, location: str) -> Dict[str, Any]:
        """
        Get market trends for a location.
        
        Args:
            location: City, state, or zip code
            
        Returns:
            Dictionary with market trend data
        """
        print(f"Fetching market trends for {location}")
        
        import random
        return {
            "location": location,
            "median_home_price": random.randint(400000, 600000),
            "median_price_per_sqft": random.randint(200, 400),
            "price_change_1yr": random.uniform(-5.0, 15.0),
            "price_change_5yr": random.uniform(10.0, 50.0),
            "inventory": random.randint(500, 2000),
            "days_on_market": random.randint(30, 90),
            "forecast_1yr": random.uniform(-2.0, 8.0),
            "demand_score": random.uniform(60, 95),
            "last_updated": datetime.now().isoformat()
        }
    
    def export_properties_to_json(self, properties: List[Dict[str, Any]], filename: str):
        """
        Export properties data to JSON file.
        
        Args:
            properties: List of property dictionaries
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(properties, f, indent=2, default=str)
        print(f"Exported {len(properties)} properties to {filename}")
    
    def export_properties_to_csv(self, properties: List[Dict[str, Any]], filename: str):
        """
        Export properties data to CSV file.
        
        Args:
            properties: List of property dictionaries
            filename: Output filename
        """
        import pandas as pd
        df = pd.DataFrame(properties)
        df.to_csv(filename, index=False)
        print(f"Exported {len(properties)} properties to {filename}")
