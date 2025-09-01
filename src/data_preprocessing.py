"""
Data preprocessing module for supermarket analytics assignment.
Handles cleaning, normalization, and transformation of supermarket transaction data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupermarketDataPreprocessor:
    """
    Data preprocessor for supermarket transaction datasets.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.items_df = None
        self.sales_df = None
        self.promotion_df = None
        self.supermarkets_df = None
        self.merged_df = None
        
    def load_data(self):
        """Load all CSV files from the data directory."""
        try:
            self.items_df = pd.read_csv(self.data_dir / "items.csv")
            self.sales_df = pd.read_csv(self.data_dir / "Sales.csv")
            self.promotion_df = pd.read_csv(self.data_dir / "Promotion.csv")
            self.supermarkets_df = pd.read_csv(self.data_dir / "Supermarkets.csv")
            
            logger.info("All datasets loaded successfully")
            logger.info(f"Items shape: {self.items_df.shape}")
            logger.info(f"Sales shape: {self.sales_df.shape}")
            logger.info(f"Promotion shape: {self.promotion_df.shape}")
            logger.info(f"Supermarkets shape: {self.supermarkets_df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self):
        """Clean and normalize all datasets."""
        logger.info("Starting data cleaning process...")
        
        # Clean items data
        self._clean_items_data()
        
        # Clean sales data
        self._clean_sales_data()
        
        # Clean promotion data
        self._clean_promotion_data()
        
        # Clean supermarkets data
        self._clean_supermarkets_data()
        
        logger.info("Data cleaning completed")
    
    def _clean_items_data(self):
        """Clean items dataset."""
        # Handle potential column name issues
        if 'descrption' in self.items_df.columns:
            self.items_df.rename(columns={'descrption': 'description'}, inplace=True)
        
        # Remove duplicates
        self.items_df.drop_duplicates(subset=['code'], inplace=True)
        
        # Handle missing values
        self.items_df.fillna('Unknown', inplace=True)
        
        logger.info(f"Items data cleaned. Shape: {self.items_df.shape}")
    
    def _clean_sales_data(self):
        """Clean sales dataset."""
        # Convert time to proper format
        self.sales_df['time'] = pd.to_numeric(self.sales_df['time'], errors='coerce')
        
        # Handle missing values in numerical columns
        numerical_cols = ['amount', 'units', 'time', 'week', 'customerId', 'supermarket', 'basket', 'day']
        for col in numerical_cols:
            if col in self.sales_df.columns:
                self.sales_df[col] = pd.to_numeric(self.sales_df[col], errors='coerce')
        
        # Remove rows with invalid data
        self.sales_df.dropna(subset=['code', 'amount', 'units'], inplace=True)
        
        # Remove negative amounts or units
        self.sales_df = self.sales_df[(self.sales_df['amount'] >= 0) & (self.sales_df['units'] > 0)]
        
        logger.info(f"Sales data cleaned. Shape: {self.sales_df.shape}")
    
    def _clean_promotion_data(self):
        """Clean promotion dataset."""
        # Remove duplicates
        self.promotion_df.drop_duplicates(inplace=True)
        
        # Handle missing values
        self.promotion_df.fillna('Not Specified', inplace=True)
        
        logger.info(f"Promotion data cleaned. Shape: {self.promotion_df.shape}")
    
    def _clean_supermarkets_data(self):
        """Clean supermarkets dataset."""
        # Handle column name variations
        if 'supermarket_No' in self.supermarkets_df.columns:
            self.supermarkets_df.rename(columns={'supermarket_No': 'supermarket'}, inplace=True)
        if 'postal-code' in self.supermarkets_df.columns:
            self.supermarkets_df.rename(columns={'postal-code': 'postal_code'}, inplace=True)
        
        # Remove duplicates
        self.supermarkets_df.drop_duplicates(subset=['supermarket'], inplace=True)
        
        logger.info(f"Supermarkets data cleaned. Shape: {self.supermarkets_df.shape}")
    
    def merge_datasets(self):
        """Merge all datasets into a comprehensive dataset."""
        logger.info("Merging datasets...")
        
        # Start with sales as the main dataset
        merged = self.sales_df.copy()
        
        # Merge with items
        merged = merged.merge(self.items_df, on='code', how='left')
        
        # Merge with supermarkets
        merged = merged.merge(self.supermarkets_df, on='supermarket', how='left')
        
        # Merge with promotions (many-to-many relationship possible)
        merged = merged.merge(self.promotion_df, 
                            left_on=['code', 'supermarket', 'week'], 
                            right_on=['code', 'supermarkets', 'week'], 
                            how='left')
        
        self.merged_df = merged
        logger.info(f"Datasets merged successfully. Final shape: {self.merged_df.shape}")
        
        return self.merged_df
    
    def create_features(self):
        """Create additional features for machine learning."""
        if self.merged_df is None:
            raise ValueError("Must merge datasets first before creating features")
        
        logger.info("Creating additional features...")
        
        # Price per unit
        self.merged_df['price_per_unit'] = self.merged_df['amount'] / self.merged_df['units']
        
        # Revenue per transaction
        self.merged_df['revenue'] = self.merged_df['amount']
        
        # Seasonal features
        self.merged_df['quarter'] = ((self.merged_df['week'] - 1) // 13) + 1
        self.merged_df['month'] = ((self.merged_df['week'] - 1) // 4.33).astype(int) + 1
        
        # Promotional indicators
        self.merged_df['has_feature'] = (self.merged_df['feature'] != 'Not on Feature').astype(int)
        self.merged_df['has_display'] = (self.merged_df['display'] != 'Not on Display').astype(int)
        
        # Customer frequency (number of transactions per customer)
        customer_freq = self.merged_df.groupby('customerId').size().reset_index(name='customer_frequency')
        self.merged_df = self.merged_df.merge(customer_freq, on='customerId', how='left')
        
        logger.info("Feature engineering completed")
        
        return self.merged_df
    
    def save_processed_data(self, output_path: str = "data/processed"):
        """Save processed data to files."""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        if self.merged_df is not None:
            self.merged_df.to_csv(output_dir / "supermarket_data_processed.csv", index=False)
            logger.info(f"Processed data saved to {output_dir / 'supermarket_data_processed.csv'}")
        else:
            logger.warning("No merged dataset to save")
    
    def get_data_summary(self):
        """Get summary statistics of the processed data."""
        if self.merged_df is None:
            return "No processed data available"
        
        summary = {
            'total_transactions': len(self.merged_df),
            'unique_customers': self.merged_df['customerId'].nunique(),
            'unique_items': self.merged_df['code'].nunique(),
            'unique_supermarkets': self.merged_df['supermarket'].nunique(),
            'date_range': f"Week {self.merged_df['week'].min()} to Week {self.merged_df['week'].max()}",
            'total_revenue': self.merged_df['amount'].sum(),
            'avg_transaction_value': self.merged_df['amount'].mean()
        }
        
        return summary

def main():
    """Main function to run data preprocessing."""
    preprocessor = SupermarketDataPreprocessor()
    
    # Load data
    preprocessor.load_data()
    
    # Clean data
    preprocessor.clean_data()
    
    # Merge datasets
    preprocessor.merge_datasets()
    
    # Create features
    preprocessor.create_features()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Print summary
    summary = preprocessor.get_data_summary()
    print("\nData Processing Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()