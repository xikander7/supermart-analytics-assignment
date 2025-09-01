"""
Comprehensive Data Visualization Module
Creates impressive business analytics visualizations for supermarket data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupermarketVisualizationAnalysis:
    """
    Advanced visualization analysis for supermarket business insights.
    """
    
    def __init__(self, data_path: str = "data/processed/supermarket_data_processed.csv"):
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self):
        """Load processed data for visualization."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_comprehensive_visualizations(self, output_dir: str = "report/figures"):
        """Generate comprehensive business visualization analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set professional style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # 1. Sales Performance Dashboard
        self._create_sales_dashboard(output_path)
        
        # 2. Promotional Impact Analysis
        self._create_promotional_analysis(output_path)
        
        # 3. Customer Behavior Analysis
        self._create_customer_analysis(output_path)
        
        # 4. Product Performance Analysis
        self._create_product_analysis(output_path)
        
        # 5. Temporal Trends Analysis
        self._create_temporal_analysis(output_path)
        
        # 6. Store Performance Comparison
        self._create_store_analysis(output_path)
        
        # 7. Advanced Business Intelligence Dashboard
        self._create_business_intelligence_dashboard(output_path)
        
        logger.info(f"All visualizations saved to {output_path}")
    
    def _create_sales_dashboard(self, output_path):
        """Create comprehensive sales performance dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sales Performance Dashboard', fontsize=20, fontweight='bold')
        
        # Sales distribution
        axes[0, 0].hist(self.data['amount'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].set_title('Transaction Amount Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Transaction Amount ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.data['amount'].mean(), color='red', linestyle='--', 
                          label=f'Mean: ${self.data["amount"].mean():.2f}')
        axes[0, 0].legend()
        
        # Weekly sales trend
        weekly_sales = self.data.groupby('week')['amount'].sum().reset_index()
        axes[0, 1].plot(weekly_sales['week'], weekly_sales['amount'], marker='o', linewidth=2)
        axes[0, 1].set_title('Weekly Sales Trend', fontweight='bold')
        axes[0, 1].set_xlabel('Week')
        axes[0, 1].set_ylabel('Total Sales ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Province comparison - using province_x column
        province_sales = self.data.groupby('province_x')['amount'].agg(['sum', 'mean']).reset_index()
        province_sales.columns = ['province', 'sum', 'mean']
        x_pos = np.arange(len(province_sales))
        bars = axes[1, 0].bar(x_pos, province_sales['sum'], alpha=0.7, color=['skyblue', 'lightcoral'])
        axes[1, 0].set_title('Total Sales by Province', fontweight='bold')
        axes[1, 0].set_xlabel('Province')
        axes[1, 0].set_ylabel('Total Sales ($)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'Province {int(p)}' for p in province_sales['province']])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Units vs Amount correlation
        axes[1, 1].scatter(self.data['units'], self.data['amount'], alpha=0.5, color='green')
        axes[1, 1].set_title('Units vs Transaction Amount', fontweight='bold')
        axes[1, 1].set_xlabel('Units Sold')
        axes[1, 1].set_ylabel('Transaction Amount ($)')
        
        # Add correlation coefficient
        corr = self.data['units'].corr(self.data['amount'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[1, 1].transAxes, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'sales_performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_promotional_analysis(self, output_path):
        """Create detailed promotional impact analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Promotional Impact Analysis', fontsize=20, fontweight='bold')
        
        # Promotion type comparison
        promo_comparison = self.data.groupby(['has_feature', 'has_display'])['amount'].mean().reset_index()
        promo_comparison['promo_type'] = promo_comparison['has_feature'].astype(str) + '_' + promo_comparison['has_display'].astype(str)
        promo_labels = ['No Promo', 'Display Only', 'Feature Only', 'Both']
        
        bars = axes[0, 0].bar(range(len(promo_comparison)), promo_comparison['amount'], 
                             color=['lightgray', 'orange', 'lightblue', 'darkgreen'], alpha=0.8)
        axes[0, 0].set_title('Average Sales by Promotion Type', fontweight='bold')
        axes[0, 0].set_xlabel('Promotion Type')
        axes[0, 0].set_ylabel('Average Transaction Amount ($)')
        axes[0, 0].set_xticks(range(len(promo_labels)))
        axes[0, 0].set_xticklabels(promo_labels, rotation=45)
        
        # Add percentage improvement labels
        base_value = promo_comparison['amount'].iloc[0]
        for i, bar in enumerate(bars):
            height = bar.get_height()
            improvement = ((height - base_value) / base_value * 100) if i > 0 else 0
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'${height:.2f}\n({improvement:+.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # Weekly promotional effectiveness
        weekly_promo = self.data.groupby(['week', 'has_feature'])['amount'].mean().unstack()
        weekly_promo.plot(ax=axes[0, 1], marker='o')
        axes[0, 1].set_title('Weekly Promotional Effectiveness', fontweight='bold')
        axes[0, 1].set_xlabel('Week')
        axes[0, 1].set_ylabel('Average Transaction Amount ($)')
        axes[0, 1].legend(['No Feature', 'With Feature'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Promotion distribution by item type
        promo_by_type = pd.crosstab(self.data['type'], self.data['has_feature'], normalize='index') * 100
        promo_by_type.plot(kind='bar', ax=axes[1, 0], color=['lightcoral', 'steelblue'])
        axes[1, 0].set_title('Promotion Distribution by Item Type', fontweight='bold')
        axes[1, 0].set_xlabel('Item Type')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].legend(['No Promotion', 'With Promotion'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Promotional ROI analysis
        promo_roi = self.data.groupby('has_feature').agg({
            'amount': ['sum', 'mean', 'count']
        }).round(2)
        promo_roi.columns = ['Total_Sales', 'Avg_Sales', 'Transaction_Count']
        promo_roi = promo_roi.reset_index()
        
        roi_data = promo_roi['Total_Sales'].values
        # Adjust labels based on actual data length
        roi_labels = ['No Promotion', 'With Promotion'] if len(roi_data) == 2 else [f'Category {i+1}' for i in range(len(roi_data))]
        colors = ['lightcoral', 'lightgreen'] if len(roi_data) == 2 else sns.color_palette("Set2", len(roi_data))
        explode = (0, 0.1) if len(roi_data) == 2 else tuple([0.05] * len(roi_data))
        
        axes[1, 1].pie(roi_data, labels=roi_labels, colors=colors, autopct='%1.1f%%', 
                      startangle=90, explode=explode)
        axes[1, 1].set_title('Sales Distribution by Promotion Status', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'promotional_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_customer_analysis(self, output_path):
        """Create customer behavior analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Behavior Analysis', fontsize=20, fontweight='bold')
        
        # Customer frequency distribution
        customer_freq = self.data.groupby('customerId').size()
        axes[0, 0].hist(customer_freq, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 0].set_title('Customer Transaction Frequency Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Number of Transactions per Customer')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].axvline(customer_freq.mean(), color='red', linestyle='--', 
                          label=f'Mean: {customer_freq.mean():.1f}')
        axes[0, 0].legend()
        
        # Customer value segmentation
        customer_value = self.data.groupby('customerId')['amount'].sum()
        value_segments = pd.cut(customer_value, bins=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
        segment_counts = value_segments.value_counts()
        
        axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                      colors=sns.color_palette("viridis", len(segment_counts)))
        axes[0, 1].set_title('Customer Value Segmentation', fontweight='bold')
        
        # Average basket size by day
        daily_basket = self.data.groupby('day')['amount'].mean()
        axes[1, 0].bar(daily_basket.index, daily_basket.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('Average Basket Size by Day', fontweight='bold')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Average Transaction Amount ($)')
        
        # Customer lifetime value distribution
        customer_ltv = self.data.groupby('customerId').agg({
            'amount': 'sum',
            'units': 'sum'
        }).reset_index()
        
        axes[1, 1].scatter(customer_ltv['units'], customer_ltv['amount'], alpha=0.6, color='teal')
        axes[1, 1].set_title('Customer Lifetime Value Analysis', fontweight='bold')
        axes[1, 1].set_xlabel('Total Units Purchased')
        axes[1, 1].set_ylabel('Total Amount Spent ($)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'customer_behavior_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_product_analysis(self, output_path):
        """Create product performance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Product Performance Analysis', fontsize=20, fontweight='bold')
        
        # Top performing products
        top_products = self.data.groupby('code')['amount'].sum().sort_values(ascending=False).head(15)
        axes[0, 0].barh(range(len(top_products)), top_products.values, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Top 15 Products by Revenue', fontweight='bold')
        axes[0, 0].set_xlabel('Total Revenue ($)')
        axes[0, 0].set_yticks(range(len(top_products)))
        axes[0, 0].set_yticklabels([f'Product {code}' for code in top_products.index])
        
        # Product type performance
        type_performance = self.data.groupby('type').agg({
            'amount': ['sum', 'mean', 'count']
        }).round(2)
        type_performance.columns = ['Total_Revenue', 'Avg_Revenue', 'Transaction_Count']
        type_performance = type_performance.reset_index()
        
        x_pos = np.arange(len(type_performance))
        width = 0.35
        
        bars1 = axes[0, 1].bar(x_pos - width/2, type_performance['Total_Revenue'], width, 
                              label='Total Revenue', color='lightcoral', alpha=0.8)
        bars2 = axes[0, 1].bar(x_pos + width/2, type_performance['Avg_Revenue']*1000, width, 
                              label='Avg Revenue (×1000)', color='steelblue', alpha=0.8)
        
        axes[0, 1].set_title('Product Type Performance', fontweight='bold')
        axes[0, 1].set_xlabel('Product Type')
        axes[0, 1].set_ylabel('Revenue ($)')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(type_performance['type'])
        axes[0, 1].legend()
        
        # Brand performance
        top_brands = self.data.groupby('brand')['amount'].sum().sort_values(ascending=False).head(10)
        axes[1, 0].bar(range(len(top_brands)), top_brands.values, color='green', alpha=0.7)
        axes[1, 0].set_title('Top 10 Brands by Revenue', fontweight='bold')
        axes[1, 0].set_xlabel('Brand')
        axes[1, 0].set_ylabel('Total Revenue ($)')
        axes[1, 0].set_xticks(range(len(top_brands)))
        axes[1, 0].set_xticklabels(top_brands.index, rotation=45, ha='right')
        
        # Price vs Volume analysis
        product_metrics = self.data.groupby('code').agg({
            'price_per_unit': 'mean',
            'units': 'sum',
            'amount': 'sum'
        }).reset_index()
        
        scatter = axes[1, 1].scatter(product_metrics['price_per_unit'], product_metrics['units'], 
                                   c=product_metrics['amount'], cmap='viridis', alpha=0.6, s=50)
        axes[1, 1].set_title('Price vs Volume Analysis', fontweight='bold')
        axes[1, 1].set_xlabel('Price per Unit ($)')
        axes[1, 1].set_ylabel('Total Units Sold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Total Revenue ($)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(output_path / 'product_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_temporal_analysis(self, output_path):
        """Create temporal trends analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Trends Analysis', fontsize=20, fontweight='bold')
        
        # Weekly sales trend with moving average
        weekly_sales = self.data.groupby('week')['amount'].sum().reset_index()
        weekly_sales['moving_avg'] = weekly_sales['amount'].rolling(window=4).mean()
        
        axes[0, 0].plot(weekly_sales['week'], weekly_sales['amount'], marker='o', 
                       label='Actual Sales', linewidth=2, alpha=0.7)
        axes[0, 0].plot(weekly_sales['week'], weekly_sales['moving_avg'], 
                       label='4-Week Moving Average', linewidth=3, color='red')
        axes[0, 0].set_title('Weekly Sales with Trend Line', fontweight='bold')
        axes[0, 0].set_xlabel('Week')
        axes[0, 0].set_ylabel('Total Sales ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Daily transaction patterns
        daily_pattern = self.data.groupby('day')['amount'].agg(['count', 'sum', 'mean']).reset_index()
        
        ax_twin = axes[0, 1].twinx()
        line1 = axes[0, 1].plot(daily_pattern['day'], daily_pattern['count'], 
                               marker='o', color='blue', label='Transaction Count')
        line2 = ax_twin.plot(daily_pattern['day'], daily_pattern['mean'], 
                            marker='s', color='red', label='Average Amount')
        
        axes[0, 1].set_title('Daily Transaction Patterns', fontweight='bold')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Transaction Count', color='blue')
        ax_twin.set_ylabel('Average Amount ($)', color='red')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[0, 1].legend(lines, labels, loc='upper left')
        
        # Seasonal patterns by quarter
        seasonal_sales = self.data.groupby(['quarter', 'type'])['amount'].sum().unstack()
        seasonal_sales.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Seasonal Sales Patterns by Product Type', fontweight='bold')
        axes[1, 0].set_xlabel('Quarter')
        axes[1, 0].set_ylabel('Total Sales ($)')
        axes[1, 0].legend(title='Product Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=0)
        
        # Time-based customer activity
        customer_activity = self.data.groupby('week')['customerId'].nunique().reset_index()
        axes[1, 1].fill_between(customer_activity['week'], customer_activity['customerId'], 
                               alpha=0.7, color='purple')
        axes[1, 1].plot(customer_activity['week'], customer_activity['customerId'], 
                       marker='o', linewidth=2, color='indigo')
        axes[1, 1].set_title('Weekly Unique Customer Activity', fontweight='bold')
        axes[1, 1].set_xlabel('Week')
        axes[1, 1].set_ylabel('Number of Unique Customers')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'temporal_trends_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_store_analysis(self, output_path):
        """Create store performance comparison analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Store Performance Analysis', fontsize=20, fontweight='bold')
        
        # Top performing stores
        store_performance = self.data.groupby('supermarket').agg({
            'amount': ['sum', 'mean', 'count'],
            'customerId': 'nunique'
        }).round(2)
        store_performance.columns = ['Total_Revenue', 'Avg_Transaction', 'Transaction_Count', 'Unique_Customers']
        store_performance = store_performance.reset_index()
        top_stores = store_performance.nlargest(15, 'Total_Revenue')
        
        bars = axes[0, 0].barh(range(len(top_stores)), top_stores['Total_Revenue'], 
                              color='lightgreen', alpha=0.8)
        axes[0, 0].set_title('Top 15 Stores by Total Revenue', fontweight='bold')
        axes[0, 0].set_xlabel('Total Revenue ($)')
        axes[0, 0].set_yticks(range(len(top_stores)))
        axes[0, 0].set_yticklabels([f'Store {int(s)}' for s in top_stores['supermarket']])
        
        # Store efficiency analysis (Revenue per transaction)
        efficiency_data = store_performance.copy()
        efficiency_data['revenue_per_transaction'] = efficiency_data['Total_Revenue'] / efficiency_data['Transaction_Count']
        top_efficient = efficiency_data.nlargest(10, 'revenue_per_transaction')
        
        axes[0, 1].scatter(top_efficient['Transaction_Count'], top_efficient['revenue_per_transaction'], 
                          s=top_efficient['Total_Revenue']/1000, alpha=0.6, color='orange')
        axes[0, 1].set_title('Store Efficiency Analysis', fontweight='bold')
        axes[0, 1].set_xlabel('Total Transactions')
        axes[0, 1].set_ylabel('Revenue per Transaction ($)')
        
        # Store performance distribution
        performance_bins = pd.cut(store_performance['Total_Revenue'], bins=5, 
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        performance_dist = performance_bins.value_counts()
        
        axes[1, 0].pie(performance_dist.values, labels=performance_dist.index, autopct='%1.1f%%',
                      colors=sns.color_palette("RdYlGn", len(performance_dist)))
        axes[1, 0].set_title('Store Performance Distribution', fontweight='bold')
        
        # Customer loyalty by store
        customer_loyalty = self.data.groupby('supermarket')['customer_frequency'].mean().reset_index()
        customer_loyalty = customer_loyalty.sort_values('customer_frequency', ascending=False).head(15)
        
        axes[1, 1].bar(range(len(customer_loyalty)), customer_loyalty['customer_frequency'], 
                      color='steelblue', alpha=0.7)
        axes[1, 1].set_title('Top 15 Stores by Customer Loyalty', fontweight='bold')
        axes[1, 1].set_xlabel('Store Rank')
        axes[1, 1].set_ylabel('Average Customer Frequency')
        axes[1, 1].set_xticks(range(len(customer_loyalty)))
        axes[1, 1].set_xticklabels([f'S{int(s)}' for s in customer_loyalty['supermarket']], rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'store_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_business_intelligence_dashboard(self, output_path):
        """Create advanced business intelligence dashboard."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Business Intelligence Dashboard - Supermarket Analytics', fontsize=24, fontweight='bold', y=0.98)
        
        # KPI Summary
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        # Calculate KPIs
        total_revenue = self.data['amount'].sum()
        total_transactions = len(self.data)
        avg_transaction = self.data['amount'].mean()
        unique_customers = self.data['customerId'].nunique()
        unique_products = self.data['code'].nunique()
        promo_effectiveness = ((self.data[self.data['has_feature'] == 1]['amount'].mean() / 
                              self.data[self.data['has_feature'] == 0]['amount'].mean() - 1) * 100)
        
        kpi_data = [
            ('Total Revenue', f'${total_revenue:,.0f}', 'lightgreen'),
            ('Total Transactions', f'{total_transactions:,}', 'lightblue'),
            ('Avg Transaction', f'${avg_transaction:.2f}', 'lightcoral'),
            ('Unique Customers', f'{unique_customers:,}', 'lightyellow'),
            ('Product Catalog', f'{unique_products}', 'lightpink'),
            ('Promo Effectiveness', f'{promo_effectiveness:.1f}%', 'lightcyan')
        ]
        
        for i, (label, value, color) in enumerate(kpi_data):
            ax1.text(i/6 + 0.08, 0.7, label, fontsize=12, fontweight='bold', ha='center', 
                    transform=ax1.transAxes)
            ax1.text(i/6 + 0.08, 0.3, value, fontsize=14, fontweight='bold', ha='center', 
                    transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        
        # Revenue trend
        ax2 = fig.add_subplot(gs[1, :2])
        weekly_revenue = self.data.groupby('week')['amount'].sum()
        ax2.plot(weekly_revenue.index, weekly_revenue.values, marker='o', linewidth=3, color='darkgreen')
        ax2.fill_between(weekly_revenue.index, weekly_revenue.values, alpha=0.3, color='lightgreen')
        ax2.set_title('Weekly Revenue Trend', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Revenue ($)')
        ax2.grid(True, alpha=0.3)
        
        # Product mix
        ax3 = fig.add_subplot(gs[1, 2:])
        product_mix = self.data.groupby('type')['amount'].sum()
        wedges, texts, autotexts = ax3.pie(product_mix.values, labels=product_mix.index, autopct='%1.1f%%', 
                                          startangle=90, colors=sns.color_palette("Set3"))
        ax3.set_title('Revenue by Product Type', fontweight='bold', fontsize=14)
        
        # Customer segmentation
        ax4 = fig.add_subplot(gs[2, :2])
        customer_segments = pd.cut(self.data.groupby('customerId')['amount'].sum(), 
                                  bins=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
        segment_counts = customer_segments.value_counts()
        bars = ax4.bar(segment_counts.index, segment_counts.values, 
                      color=['#CD7F32', '#C0C0C0', '#FFD700', '#E5E4E2'], alpha=0.8)
        ax4.set_title('Customer Segmentation', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Customer Tier')
        ax4.set_ylabel('Number of Customers')
        
        # Promotional impact
        ax5 = fig.add_subplot(gs[2, 2:])
        promo_impact = self.data.groupby(['has_feature', 'has_display'])['amount'].mean().reset_index()
        promo_impact['label'] = promo_impact.apply(
            lambda x: 'No Promo' if x['has_feature']==0 and x['has_display']==0
            else 'Display Only' if x['has_feature']==0 and x['has_display']==1
            else 'Feature Only' if x['has_feature']==1 and x['has_display']==0
            else 'Both', axis=1
        )
        
        bars = ax5.bar(promo_impact['label'], promo_impact['amount'], 
                      color=['gray', 'orange', 'blue', 'green'], alpha=0.7)
        ax5.set_title('Promotional Strategy Effectiveness', fontweight='bold', fontsize=14)
        ax5.set_xlabel('Promotion Type')
        ax5.set_ylabel('Average Transaction Amount ($)')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # Store performance heatmap
        ax6 = fig.add_subplot(gs[3, :])
        
        # Create store performance matrix
        store_metrics = self.data.groupby(['supermarket', 'week'])['amount'].sum().unstack(fill_value=0)
        
        # Select top 20 stores for visualization
        top_stores_for_heatmap = self.data.groupby('supermarket')['amount'].sum().nlargest(20).index
        store_heatmap_data = store_metrics.loc[top_stores_for_heatmap]
        
        im = ax6.imshow(store_heatmap_data.values, cmap='YlOrRd', aspect='auto')
        ax6.set_title('Store Performance Heatmap (Top 20 Stores by Week)', fontweight='bold', fontsize=14)
        ax6.set_xlabel('Week')
        ax6.set_ylabel('Store ID')
        ax6.set_yticks(range(len(top_stores_for_heatmap)))
        ax6.set_yticklabels([f'Store {int(s)}' for s in top_stores_for_heatmap])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
        cbar.set_label('Weekly Revenue ($)', rotation=270, labelpad=15)
        
        plt.savefig(output_path / 'business_intelligence_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate all visualizations."""
    visualizer = SupermarketVisualizationAnalysis()
    
    # Load processed data
    visualizer.load_data()
    
    # Create comprehensive visualizations
    visualizer.create_comprehensive_visualizations()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE VISUALIZATION ANALYSIS COMPLETED")
    print("="*60)
    print("Generated Visualizations:")
    print("1. Sales Performance Dashboard")
    print("2. Promotional Impact Analysis") 
    print("3. Customer Behavior Analysis")
    print("4. Product Performance Analysis")
    print("5. Temporal Trends Analysis")
    print("6. Store Performance Analysis")
    print("7. Business Intelligence Dashboard")
    print("="*60)

if __name__ == "__main__":
    main()