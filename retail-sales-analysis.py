import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

class RetailSalesAnalyzer:
    """
    A comprehensive sales analysis tool for retail data 
    with advanced insights and visualizations.
    """
    
    def __init__(self, sales_file: str):
        """
        Initialize the analyzer with sales data.
        
        Args:
            sales_file (str): Path to the sales CSV file
        """
        # Read large CSV file with optimized parsing
        self.sales_df = pd.read_csv(sales_file, 
                                    parse_dates=['date'],
                                    dtype={
                                        'store_id': 'category',
                                        'product_type': 'category',
                                        'product_name': 'category',
                                        'location': 'category'
                                    })
        
        # Ensure consistent column names
        self.sales_df.columns = [col.lower().replace(' ', '_') for col in self.sales_df.columns]
    
    def basic_sales_overview(self) -> Dict[str, float]:
        """
        Provide basic sales overview.
        
        Returns:
            Dict with key sales metrics
        """
        return {
            'total_sales': self.sales_df['sales_amount'].sum(),
            'average_transaction': self.sales_df['sales_amount'].mean(),
            'total_transactions': len(self.sales_df),
            'date_range': (
                self.sales_df['date'].min(), 
                self.sales_df['date'].max()
            )
        }
    
    def top_performing_stores(self, n: int = 10) -> pd.DataFrame:
        """
        Identify top performing stores.
        
        Args:
            n (int): Number of top stores to return
        
        Returns:
            DataFrame of top stores by total sales
        """
        store_performance = self.sales_df.groupby('store_id').agg({
            'sales_amount': ['sum', 'mean'],
            'product_name': 'count'
        }).reset_index()
        
        store_performance.columns = ['store_id', 'total_sales', 'avg_sale', 'total_transactions']
        return store_performance.nlargest(n, 'total_sales')
    
    def struggling_stores(self, threshold_percentile: float = 25) -> pd.DataFrame:
        """
        Identify stores with declining or low performance.
        
        Args:
            threshold_percentile (float): Percentile to consider as struggling
        
        Returns:
            DataFrame of struggling stores
        """
        # Monthly sales trend for each store
        monthly_store_sales = self.sales_df.groupby([
            pd.Grouper(key='date', freq='M'), 
            'store_id'
        ])['sales_amount'].sum().reset_index()
        
        # Calculate sales variability and trend
        store_volatility = monthly_store_sales.groupby('store_id').agg({
            'sales_amount': ['mean', 'std']
        }).reset_index()
        store_volatility.columns = ['store_id', 'avg_monthly_sales', 'sales_volatility']
        
        # Identify stores below threshold
        struggling = store_volatility[
            store_volatility['avg_monthly_sales'] <= 
            store_volatility['avg_monthly_sales'].quantile(threshold_percentile/100)
        ]
        
        return struggling
    
    def product_performance_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Comprehensive product performance analysis.
        
        Returns:
            Tuple of top and bottom performing product categories
        """
        # Product category performance
        product_performance = self.sales_df.groupby('product_type').agg({
            'sales_amount': ['sum', 'mean'],
            'product_name': 'count'
        }).reset_index()
        
        product_performance.columns = [
            'product_type', 'total_sales', 'avg_sale_per_product', 'total_transactions'
        ]
        
        # Top and bottom performing categories
        top_products = product_performance.nlargest(5, 'total_sales')
        bottom_products = product_performance.nsmallest(5, 'total_sales')
        
        return top_products, bottom_products
    
    def seasonal_trend_analysis(self) -> pd.DataFrame:
        """
        Analyze seasonal sales trends.
        
        Returns:
            DataFrame with monthly and quarterly sales trends
        """
        # Monthly sales trend
        monthly_sales = self.sales_df.groupby(
            pd.Grouper(key='date', freq='M')
        )['sales_amount'].sum().reset_index()
        
        # Quarterly sales trend
        quarterly_sales = self.sales_df.groupby(
            pd.Grouper(key='date', freq='Q')
        )['sales_amount'].sum().reset_index()
        
        monthly_sales['month'] = monthly_sales['date'].dt.month
        quarterly_sales['quarter'] = quarterly_sales['date'].dt.quarter
        
        return {
            'monthly_trend': monthly_sales,
            'quarterly_trend': quarterly_sales
        }
    
    def location_based_insights(self) -> pd.DataFrame:
        """
        Analyze sales performance by location.
        
        Returns:
            DataFrame of location performance
        """
        location_performance = self.sales_df.groupby('location').agg({
            'sales_amount': ['sum', 'mean'],
            'store_id': 'nunique',
            'product_name': 'count'
        }).reset_index()
        
        location_performance.columns = [
            'location', 'total_sales', 'avg_sale', 'unique_stores', 'total_transactions'
        ]
        
        return location_performance
    
    def visualize_sales_trends(self, output_dir: str = 'sales_visualizations'):
        """
        Create visualizations of sales trends.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Monthly Sales Trend
        plt.figure(figsize=(12, 6))
        monthly_sales = self.seasonal_trend_analysis()['monthly_trend']
        plt.plot(monthly_sales['date'], monthly_sales['sales_amount'])
        plt.title('Monthly Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_sales_trend.png'))
        plt.close()
        
        # Top Product Categories
        top_products, _ = self.product_performance_analysis()
        plt.figure(figsize=(10, 6))
        plt.bar(top_products['product_type'], top_products['total_sales'])
        plt.title('Top 5 Product Categories by Sales')
        plt.xlabel('Product Type')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_product_categories.png'))
        plt.close()
    
    def generate_comprehensive_report(self, output_file: str = 'sales_analysis_report.md'):
        """
        Generate a comprehensive markdown report.
        
        Args:
            output_file (str): Path to save the markdown report
        """
        # Collect insights
        overview = self.basic_sales_overview()
        top_stores = self.top_performing_stores()
        struggling_stores = self.struggling_stores()
        top_products, bottom_products = self.product_performance_analysis()
        location_insights = self.location_based_insights()
        
        # Create markdown report
        with open(output_file, 'w') as f:
            f.write("# Comprehensive Sales Analysis Report\n\n")
            
            # Sales Overview
            f.write("## Sales Overview\n")
            f.write(f"- **Total Sales:** ${overview['total_sales']:,.2f}\n")
            f.write(f"- **Average Transaction:** ${overview['average_transaction']:,.2f}\n")
            f.write(f"- **Total Transactions:** {overview['total_transactions']:,}\n")
            f.write(f"- **Date Range:** {overview['date_range'][0]} to {overview['date_range'][1]}\n\n")
            
            # Top Performing Stores
            f.write("## Top Performing Stores\n")
            f.write(top_stores.to_markdown(index=False))
            f.write("\n\n")
            
            # Struggling Stores
            f.write("## Stores of Concern\n")
            f.write(struggling_stores.to_markdown(index=False))
            f.write("\n\n")
            
            # Product Performance
            f.write("## Top Product Categories\n")
            f.write(top_products.to_markdown(index=False))
            f.write("\n\n")
            
            # Location Insights
            f.write("## Location Performance\n")
            f.write(location_insights.to_markdown(index=False))
    
    def detect_anomalies(self, z_threshold: float = 3) -> pd.DataFrame:
        """
        Detect sales anomalies using statistical methods.
        
        Args:
            z_threshold (float): Z-score threshold for anomaly detection
        
        Returns:
            DataFrame of anomalous sales records
        """
        # Group by store and product type
        grouped = self.sales_df.groupby(['store_id', 'product_type'])
        
        # Calculate z-scores for sales amounts
        self.sales_df['sales_zscore'] = grouped['sales_amount'].transform(
            lambda x: np.abs((x - x.mean()) / x.std())
        )
        
        # Return anomalies beyond z-score threshold
        return self.sales_df[self.sales_df['sales_zscore'] > z_threshold]

# Example usage
def main():
    # Replace with your actual sales data file path
    analyzer = RetailSalesAnalyzer('spencer_retail_sales.csv')
    
    # Generate comprehensive insights
    analyzer.visualize_sales_trends()
    analyzer.generate_comprehensive_report()
    
    # Print key insights
    print(analyzer.top_performing_stores())
    print(analyzer.struggling_stores())

if __name__ == '__main__':
    main()

# README for GitHub Repository
"""
# Retail Sales Analysis Tool

## Overview
This Python script provides a comprehensive analysis of retail sales data, offering deep insights into:
- Store performance
- Product category trends
- Seasonal sales patterns
- Location-based insights
- Anomaly detection

## Features
- Detailed sales overview
- Top and struggling store identification
- Product performance analysis
- Seasonal trend visualization
- Location-based performance metrics
- Sales anomaly detection
- Markdown report generation

## Requirements
- pandas
- numpy
- matplotlib
- seaborn

## Usage
1. Install required libraries
2. Replace 'spencer_retail_sales.csv' with your sales data file
3. Run the script to generate insights and visualizations

## Output
- Markdown report
- Sales trend visualizations
- Comprehensive performance metrics
"""
