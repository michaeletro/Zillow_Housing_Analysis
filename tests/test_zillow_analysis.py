"""
Unit tests for Zillow Housing Analysis package.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zillow_analysis import ZillowAPIClient, PropertyValuationModel, PropertyData
from zillow_analysis.utils import clean_property_data, calculate_derived_features
import pandas as pd


class TestZillowAPIClient(unittest.TestCase):
    """Test cases for ZillowAPIClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = ZillowAPIClient()
    
    def test_client_initialization(self):
        """Test client initialization."""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.request_delay, 1)
    
    def test_search_properties(self):
        """Test property search."""
        properties = self.client.search_properties("New York, NY", max_results=10)
        self.assertIsInstance(properties, list)
        self.assertGreater(len(properties), 0)
        self.assertIn('zpid', properties[0])
        self.assertIn('price', properties[0])
    
    def test_get_market_trends(self):
        """Test market trends retrieval."""
        trends = self.client.get_market_trends("New York, NY")
        self.assertIsInstance(trends, dict)
        self.assertIn('median_home_price', trends)
        self.assertIn('location', trends)


class TestPropertyData(unittest.TestCase):
    """Test cases for PropertyData model."""
    
    def test_property_data_creation(self):
        """Test PropertyData model creation."""
        data = {
            'zpid': '12345',
            'address': '123 Main St',
            'city': 'New York',
            'state': 'NY',
            'zipcode': '10001',
            'price': 500000,
            'bedrooms': 3,
            'bathrooms': 2
        }
        
        property_obj = PropertyData(**data)
        self.assertEqual(property_obj.zpid, '12345')
        self.assertEqual(property_obj.price, 500000)
        self.assertEqual(property_obj.bedrooms, 3)
    
    def test_property_to_dict(self):
        """Test PropertyData to_dict method."""
        data = {
            'zpid': '12345',
            'address': '123 Main St',
            'city': 'New York',
            'state': 'NY',
            'zipcode': '10001'
        }
        
        property_obj = PropertyData(**data)
        result = property_obj.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['zpid'], '12345')


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing utilities."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_properties = [
            {
                'zpid': '1',
                'address': '123 Main St',
                'city': 'New York',
                'state': 'NY',
                'zipcode': '10001',
                'price': 500000,
                'bedrooms': 3,
                'bathrooms': 2,
                'living_area': 1500,
                'lot_size': 3000,
                'year_built': 2000
            },
            {
                'zpid': '2',
                'address': '456 Oak Ave',
                'city': 'Brooklyn',
                'state': 'NY',
                'zipcode': '11201',
                'price': 600000,
                'bedrooms': 4,
                'bathrooms': 3,
                'living_area': 2000,
                'lot_size': 4000,
                'year_built': 2010
            }
        ]
    
    def test_clean_property_data(self):
        """Test data cleaning."""
        df = clean_property_data(self.sample_properties)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('age', df.columns)
    
    def test_calculate_derived_features(self):
        """Test feature engineering."""
        df = pd.DataFrame(self.sample_properties)
        df_with_features = calculate_derived_features(df)
        
        self.assertIn('price_per_sqft', df_with_features.columns)
        self.assertIn('bath_bed_ratio', df_with_features.columns)


class TestPropertyValuationModel(unittest.TestCase):
    """Test cases for PropertyValuationModel."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_properties = [
            {
                'zpid': f'{i}',
                'address': f'{i} Main St',
                'city': 'New York',
                'state': 'NY',
                'zipcode': '10001',
                'price': 500000 + i * 10000,
                'bedrooms': 2 + (i % 3),
                'bathrooms': 1 + (i % 2),
                'living_area': 1000 + i * 100,
                'lot_size': 2000 + i * 200,
                'year_built': 1990 + (i % 30),
                'property_type': 'Single Family',
                'status': 'forSale',
                'tax_assessed_value': 480000 + i * 10000,
                'latitude': 40.7128,
                'longitude': -74.0060,
                'days_on_zillow': 30
            }
            for i in range(50)
        ]
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = PropertyValuationModel(model_type="xgboost")
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, "xgboost")
        self.assertFalse(model.trained)
    
    def test_model_training(self):
        """Test model training."""
        model = PropertyValuationModel(model_type="random_forest")
        metrics = model.train(self.sample_properties, validate=False)
        
        self.assertTrue(model.trained)
        self.assertIn('test_r2', metrics)
        self.assertIn('test_mae', metrics)
        self.assertGreater(metrics['test_r2'], 0)
    
    def test_model_prediction(self):
        """Test model predictions."""
        model = PropertyValuationModel(model_type="random_forest")
        model.train(self.sample_properties, validate=False)
        
        predictions = model.predict([self.sample_properties[0]])
        self.assertEqual(len(predictions), 1)
        self.assertGreater(predictions[0], 0)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = PropertyValuationModel(model_type="random_forest")
        model.train(self.sample_properties, validate=False)
        
        importance = model.get_feature_importance(top_n=5)
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertLessEqual(len(importance), 5)


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == '__main__':
    run_tests()
