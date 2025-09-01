"""
Export Visualization Data Module
Generates CSV files containing the underlying data used in visualization charts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationDataExporter:
    """
    Export data frames used in visualization analysis for reviewer inspection.
    """
    
    def __init__(self, data_path: str = "data/processed/supermarket_data_processed.csv"):
        self.data_path = Path(data_path)
        self.data = None
        self.output_dir = Path("data/processed/visualization_data")
        
    def load_data(self):
        """Load processed data for analysis."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def export_all_visualization_datasets(self):
        """Export all datasets used in visualizations."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Sales Performance Data
        self._export_sales_performance_data()
        
        # 2. Promotional Analysis Data
        self._export_promotional_analysis_data()
        
        # 3. Customer Analysis Data
        self._export_customer_analysis_data()
        
        # 4. Product Performance Data
        self._export_product_performance_data()
        
        # 5. Temporal Trends Data
        self._export_temporal_trends_data()
        
        # 6. Store Performance Data
        self._export_store_performance_data()
        
        # 7. Business Intelligence KPIs
        self._export_business_intelligence_data()
        
        logger.info(f"All visualization datasets exported to {self.output_dir}")
    
    def _export_sales_performance_data(self):
        """Export sales performance dashboard data."""
        # Weekly sales trend
        weekly_sales = self.data.groupby('week')['amount'].agg(['sum', 'mean', 'count']).reset_index()
        weekly_sales.columns = ['week', 'total_sales', 'average_transaction', 'transaction_count']
        weekly_sales.to_csv(self.output_dir / 'weekly_sales_trends.csv', index=False)
        
        # Provincial sales analysis
        province_sales = self.data.groupby('province_x')['amount'].agg(['sum', 'mean', 'count']).reset_index()
        province_sales.columns = ['province', 'total_sales', 'average_transaction', 'transaction_count']
        province_sales = province_sales.sort_values('total_sales', ascending=False)
        province_sales.to_csv(self.output_dir / 'provincial_sales_summary.csv', index=False)
        
        # Sales distribution statistics
        sales_stats = pd.DataFrame({
            'metric': ['total_revenue', 'average_transaction', 'median_transaction', 'transaction_count', 
                      'min_transaction', 'max_transaction', 'std_transaction'],
            'value': [
                self.data['amount'].sum(),
                self.data['amount'].mean(),
                self.data['amount'].median(),
                len(self.data),
                self.data['amount'].min(),
                self.data['amount'].max(),
                self.data['amount'].std()
            ]
        })
        sales_stats.to_csv(self.output_dir / 'sales_distribution_statistics.csv', index=False)
        
    def _export_promotional_analysis_data(self):
        """Export promotional impact analysis data."""
        # Promotion effectiveness comparison
        promo_effectiveness = self.data.groupby(['has_feature', 'has_display'])['amount'].agg(['mean', 'sum', 'count']).reset_index()
        promo_effectiveness.columns = ['has_feature', 'has_display', 'avg_transaction', 'total_sales', 'transaction_count']
        
        # Add promotion type labels
        promo_effectiveness['promotion_type'] = promo_effectiveness.apply(
            lambda x: 'No Promotion' if x['has_feature']==0 and x['has_display']==0
            else 'Display Only' if x['has_feature']==0 and x['has_display']==1
            else 'Feature Only' if x['has_feature']==1 and x['has_display']==0
            else 'Both Feature & Display', axis=1
        )
        
        # Calculate uplift percentages
        base_rows = promo_effectiveness[promo_effectiveness['promotion_type'] == 'No Promotion']
        if len(base_rows) > 0:
            base_avg = base_rows['avg_transaction'].iloc[0]
            promo_effectiveness['uplift_percentage'] = ((promo_effectiveness['avg_transaction'] - base_avg) / base_avg * 100).round(2)
        else:
            # If no "No Promotion" baseline, use the minimum value as baseline
            base_avg = promo_effectiveness['avg_transaction'].min()
            promo_effectiveness['uplift_percentage'] = ((promo_effectiveness['avg_transaction'] - base_avg) / base_avg * 100).round(2)
        
        promo_effectiveness.to_csv(self.output_dir / 'promotional_effectiveness.csv', index=False)
        
        # Weekly promotional trends
        weekly_promo = self.data.groupby(['week', 'has_feature'])['amount'].mean().unstack(fill_value=0).reset_index()
        # Dynamically set column names based on actual columns
        if len(weekly_promo.columns) == 3:
            weekly_promo.columns = ['week', 'no_feature_avg_sales', 'with_feature_avg_sales']
            weekly_promo['feature_uplift'] = ((weekly_promo['with_feature_avg_sales'] - weekly_promo['no_feature_avg_sales']) / 
                                            weekly_promo['no_feature_avg_sales'] * 100).round(2)
        else:
            # Handle case with different number of columns
            weekly_promo.columns = ['week'] + [f'feature_level_{i}' for i in range(len(weekly_promo.columns)-1)]
        weekly_promo.to_csv(self.output_dir / 'weekly_promotional_trends.csv', index=False)
        
    def _export_customer_analysis_data(self):
        """Export customer behavior analysis data."""
        # Customer frequency distribution
        customer_frequency = self.data.groupby('customerId').size().reset_index()
        customer_frequency.columns = ['customer_id', 'transaction_frequency']
        customer_frequency_summary = customer_frequency['transaction_frequency'].describe()
        
        # Customer value segmentation
        customer_value = self.data.groupby('customerId').agg({
            'amount': ['sum', 'mean'],
            'units': 'sum'
        }).reset_index()
        customer_value.columns = ['customer_id', 'total_spent', 'avg_transaction', 'total_units']
        
        # Create value segments
        customer_value['value_segment'] = pd.cut(customer_value['total_spent'], 
                                               bins=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
        
        customer_value.to_csv(self.output_dir / 'customer_value_analysis.csv', index=False)
        
        # Daily transaction patterns
        daily_patterns = self.data.groupby('day').agg({
            'amount': ['mean', 'sum', 'count'],
            'customerId': 'nunique'
        }).reset_index()
        daily_patterns.columns = ['day', 'avg_transaction_amount', 'total_sales', 'transaction_count', 'unique_customers']
        daily_patterns.to_csv(self.output_dir / 'daily_transaction_patterns.csv', index=False)
        
    def _export_product_performance_data(self):
        """Export product performance analysis data."""
        # Top performing products
        top_products = self.data.groupby(['code', 'description', 'type', 'brand']).agg({
            'amount': ['sum', 'mean'],
            'units': 'sum'
        }).reset_index()
        top_products.columns = ['product_code', 'description', 'type', 'brand', 'total_revenue', 'avg_transaction', 'total_units']
        top_products = top_products.sort_values('total_revenue', ascending=False)
        top_products.to_csv(self.output_dir / 'product_performance_ranking.csv', index=False)
        
        # Product type performance
        type_performance = self.data.groupby('type').agg({
            'amount': ['sum', 'mean', 'count'],
            'units': 'sum'
        }).reset_index()
        type_performance.columns = ['product_type', 'total_revenue', 'avg_transaction', 'transaction_count', 'total_units']
        type_performance = type_performance.sort_values('total_revenue', ascending=False)
        type_performance.to_csv(self.output_dir / 'product_type_performance.csv', index=False)
        
        # Brand performance
        brand_performance = self.data.groupby('brand').agg({
            'amount': ['sum', 'mean', 'count'],
            'units': 'sum'
        }).reset_index()
        brand_performance.columns = ['brand', 'total_revenue', 'avg_transaction', 'transaction_count', 'total_units']
        brand_performance = brand_performance.sort_values('total_revenue', ascending=False)
        brand_performance.to_csv(self.output_dir / 'brand_performance_ranking.csv', index=False)
        
    def _export_temporal_trends_data(self):
        """Export temporal trends analysis data."""
        # Weekly sales with moving averages
        weekly_trends = self.data.groupby('week').agg({
            'amount': ['sum', 'mean'],
            'customerId': 'nunique'
        }).reset_index()
        weekly_trends.columns = ['week', 'total_sales', 'avg_transaction', 'unique_customers']
        
        # Calculate moving averages
        weekly_trends['sales_4week_ma'] = weekly_trends['total_sales'].rolling(window=4).mean()
        weekly_trends['customers_4week_ma'] = weekly_trends['unique_customers'].rolling(window=4).mean()
        weekly_trends.to_csv(self.output_dir / 'weekly_temporal_trends.csv', index=False)
        
        # Seasonal patterns by quarter
        seasonal_data = self.data.groupby(['quarter', 'type']).agg({
            'amount': ['sum', 'mean'],
            'units': 'sum'
        }).reset_index()
        seasonal_data.columns = ['quarter', 'product_type', 'total_sales', 'avg_transaction', 'total_units']
        seasonal_data.to_csv(self.output_dir / 'seasonal_sales_patterns.csv', index=False)
        
        # Monthly trends
        monthly_trends = self.data.groupby('month').agg({
            'amount': ['sum', 'mean', 'count'],
            'customerId': 'nunique'
        }).reset_index()
        monthly_trends.columns = ['month', 'total_sales', 'avg_transaction', 'transaction_count', 'unique_customers']
        monthly_trends.to_csv(self.output_dir / 'monthly_trends.csv', index=False)
        
    def _export_store_performance_data(self):
        """Export store performance analysis data."""
        # Store performance metrics
        store_performance = self.data.groupby('supermarket').agg({
            'amount': ['sum', 'mean', 'count'],
            'customerId': 'nunique',
            'customer_frequency': 'mean',
            'units': 'sum'
        }).reset_index()
        store_performance.columns = ['store_id', 'total_revenue', 'avg_transaction', 'transaction_count', 
                                   'unique_customers', 'avg_customer_loyalty', 'total_units']
        
        # Calculate efficiency metrics
        store_performance['revenue_per_transaction'] = store_performance['total_revenue'] / store_performance['transaction_count']
        store_performance['revenue_per_customer'] = store_performance['total_revenue'] / store_performance['unique_customers']
        
        # Performance ranking
        store_performance = store_performance.sort_values('total_revenue', ascending=False)
        store_performance['revenue_rank'] = range(1, len(store_performance) + 1)
        store_performance.to_csv(self.output_dir / 'store_performance_ranking.csv', index=False)
        
        # Store efficiency analysis
        efficiency_data = store_performance[['store_id', 'revenue_per_transaction', 'revenue_per_customer', 
                                          'avg_customer_loyalty', 'total_revenue']].copy()
        efficiency_data.to_csv(self.output_dir / 'store_efficiency_metrics.csv', index=False)
        
    def _export_business_intelligence_data(self):
        """Export business intelligence KPIs."""
        # Calculate comprehensive KPIs
        total_revenue = self.data['amount'].sum()
        total_transactions = len(self.data)
        avg_transaction = self.data['amount'].mean()
        unique_customers = self.data['customerId'].nunique()
        unique_products = self.data['code'].nunique()
        unique_stores = self.data['supermarket'].nunique()
        
        # Promotional effectiveness
        promo_effectiveness = ((self.data[self.data['has_feature'] == 1]['amount'].mean() / 
                              self.data[self.data['has_feature'] == 0]['amount'].mean() - 1) * 100)
        
        # Customer metrics
        avg_customer_value = total_revenue / unique_customers
        avg_transactions_per_customer = total_transactions / unique_customers
        
        # Product metrics
        avg_revenue_per_product = total_revenue / unique_products
        avg_units_per_transaction = self.data['units'].mean()
        
        # Time period
        date_range_weeks = self.data['week'].max() - self.data['week'].min() + 1
        
        kpi_summary = pd.DataFrame({
            'kpi_name': [
                'Total Revenue', 'Total Transactions', 'Average Transaction Amount',
                'Unique Customers', 'Unique Products', 'Unique Stores',
                'Promotional Effectiveness (%)', 'Average Customer Value',
                'Avg Transactions per Customer', 'Avg Revenue per Product',
                'Avg Units per Transaction', 'Analysis Period (Weeks)'
            ],
            'value': [
                total_revenue, total_transactions, avg_transaction,
                unique_customers, unique_products, unique_stores,
                promo_effectiveness, avg_customer_value,
                avg_transactions_per_customer, avg_revenue_per_product,
                avg_units_per_transaction, date_range_weeks
            ],
            'formatted_value': [
                f'${total_revenue:,.2f}', f'{total_transactions:,}', f'${avg_transaction:.2f}',
                f'{unique_customers:,}', f'{unique_products:,}', f'{unique_stores:,}',
                f'{promo_effectiveness:.1f}%', f'${avg_customer_value:.2f}',
                f'{avg_transactions_per_customer:.1f}', f'${avg_revenue_per_product:.2f}',
                f'{avg_units_per_transaction:.1f}', f'{date_range_weeks} weeks'
            ]
        })
        kpi_summary.to_csv(self.output_dir / 'business_intelligence_kpis.csv', index=False)
        
        # Top performers summary
        top_performers = pd.DataFrame({
            'category': ['Top Store (Revenue)', 'Top Product (Revenue)', 'Top Brand (Revenue)', 
                        'Top Customer (Spend)', 'Most Loyal Customer', 'Best Promotion Type'],
            'identifier': [
                f"Store {self.data.groupby('supermarket')['amount'].sum().idxmax()}",
                f"Product {self.data.groupby('code')['amount'].sum().idxmax()}",
                f"{self.data.groupby('brand')['amount'].sum().idxmax()}",
                f"Customer {self.data.groupby('customerId')['amount'].sum().idxmax()}",
                f"Customer {self.data.groupby('customerId')['customer_frequency'].mean().idxmax()}",
                "Feature + Display"
            ],
            'value': [
                self.data.groupby('supermarket')['amount'].sum().max(),
                self.data.groupby('code')['amount'].sum().max(),
                self.data.groupby('brand')['amount'].sum().max(),
                self.data.groupby('customerId')['amount'].sum().max(),
                self.data.groupby('customerId')['customer_frequency'].mean().max(),
                self.data[(self.data['has_feature']==1) & (self.data['has_display']==1)]['amount'].mean()
            ]
        })
        top_performers.to_csv(self.output_dir / 'top_performers_summary.csv', index=False)

def main():
    """Main function to export all visualization datasets."""
    exporter = VisualizationDataExporter()
    
    # Load processed data
    exporter.load_data()
    
    # Export all visualization datasets
    exporter.export_all_visualization_datasets()
    
    print("\n" + "="*70)
    print("VISUALIZATION DATA EXPORT COMPLETED")
    print("="*70)
    print("Exported Dataset Files:")
    print("• weekly_sales_trends.csv - Weekly sales performance data")
    print("• provincial_sales_summary.csv - Sales by province analysis")
    print("• promotional_effectiveness.csv - Promotion impact metrics")
    print("• customer_value_analysis.csv - Customer segmentation data")
    print("• product_performance_ranking.csv - Product revenue rankings")
    print("• store_performance_ranking.csv - Store efficiency metrics")
    print("• business_intelligence_kpis.csv - Executive KPI summary")
    print("• seasonal_sales_patterns.csv - Quarterly trend analysis")
    print("• And 8 additional supporting datasets...")
    print("="*70)
    print(f"All files saved to: data/processed/visualization_data/")
    print("="*70)

if __name__ == "__main__":
    main()