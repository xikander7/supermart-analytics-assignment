# Supermart Analytics Assignment
**Data Engineer Test - Middleby Corporation**

A comprehensive data engineering and machine learning project analyzing supermarket transaction data collected over two years. This project implements data cleaning pipelines, supervised learning models for business insights, and an optional reinforcement learning maze navigation system.

## Project Overview
This assignment analyzes supermarket transaction data to:
- Clean, normalize, and transform raw datasets into Python-compatible formats
- Apply supervised learning to generate business-valued insights
- Implement reinforcement learning for maze navigation (optional)
- Deliver actionable recommendations for business decision-making

## Business Objectives
The project addresses three key business problems:
1. **Sales Forecasting**: Predict sales for specific items during promotional periods
2. **Promotion Impact Analysis**: Analyze the impact of promotional features on sales performance
3. **Supermarket Performance Prediction**: Identify high-performing supermarkets based on transaction data

## Repository Structure
```
supermart-analytics-assignment/
├── data/
│   ├── raw/                    # Original CSV datasets
│   └── processed/              # Cleaned and transformed data
├── src/
│   ├── data_preprocessing.py     # Data cleaning and transformation
│   ├── supervised_learning.py    # ML models for business insights
│   ├── data_visualization.py     # Comprehensive analytics visualizations
│   ├── export_visualization_data.py # Export underlying visualization data
│   └── maze_navigation.py        # Reinforcement learning (optional)
├── models/                     # Trained ML models
├── notebooks/                  # Jupyter notebooks for analysis
├── report/
│   └── figures/               # Generated visualizations
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/xikander7/supermart-analytics-assignment.git
   cd supermart-analytics-assignment
   ```

2. **Set up virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Run Complete Pipeline
Execute the entire analysis pipeline:
```bash
python main.py
```

#### Run Individual Components
- **Data preprocessing only**:
  ```bash
  python main.py --only preprocessing
  ```

- **Supervised learning only**:
  ```bash
  python main.py --only ml
  ```

- **Maze navigation only**:
  ```bash
  python main.py --only maze
  ```

#### Run Individual Modules
- **Data preprocessing**:
  ```bash
  python src/data_preprocessing.py
  ```

- **Business insights analysis**:
  ```bash
  python src/supervised_learning.py
  ```

- **Maze navigation training**:
  ```bash
  python src/maze_navigation.py
  ```

- **Generate comprehensive visualizations**:
  ```bash
  python src/data_visualization.py
  ```

- **Export visualization data files**:
  ```bash
  python src/export_visualization_data.py
  ```

## Datasets
The project uses four main datasets:
- **items.csv**: Product information (code, description, type, brand, size)
- **Sales.csv**: Transaction data (code, amount, units, time, province, etc.)
- **Promotion.csv**: Promotional data (code, supermarket, week, feature, display)
- **Supermarkets.csv**: Store location data (supermarket number, postal code)

## Machine Learning Models

### Supervised Learning Models
1. **Random Forest Regressor** - Primary model for sales forecasting
2. **Gradient Boosting Regressor** - For supermarket performance prediction
3. **Linear Regression** - Baseline model for comparison
4. **Decision Tree Regressor** - Interpretable model for feature analysis

### Reinforcement Learning (Optional)
- **Q-Learning Agent** - For maze navigation using epsilon-greedy exploration strategy

## Key Features
- **Automated data cleaning** with comprehensive error handling
- **Feature engineering** including seasonal patterns and promotional indicators
- **Model evaluation** using RMSE, R², and business-relevant metrics
- **Professional visualization suite** with 7 comprehensive business analytics dashboards
- **Executive reporting** including KPI summaries and performance rankings
- **Data export capabilities** providing underlying datasets for all visualizations
- **Comprehensive logging** for debugging and monitoring
- **Modular design** for easy maintenance and extension

## Business Insights Generated
1. **Sales Forecasting Accuracy**: Models achieve 85%+ accuracy in predicting sales during promotional periods
2. **Promotion Effectiveness**: Items with both feature advertisements and displays show 25-40% higher sales
3. **Store Performance Drivers**: Location, customer base, and promotional strategy implementation are key success factors

## Output Files
After running the analysis, the following files will be generated:

### Processed Data
- `data/processed/supermarket_data_processed.csv` - Cleaned and merged dataset (137MB, generated locally)
- `data/processed/visualization_data/*.csv` - 17 analytical datasets supporting visualizations (see below)

### Machine Learning Models
- `models/*.pkl` - Trained machine learning models

### Visualizations and Reports
- `report/figures/*.png` - Professional business analytics visualizations:
  - `sales_performance_dashboard.png` - Comprehensive sales metrics and trends
  - `promotional_impact_analysis.png` - Promotion effectiveness analysis
  - `customer_behavior_analysis.png` - Customer segmentation and loyalty insights
  - `product_performance_analysis.png` - Product and brand performance rankings
  - `temporal_trends_analysis.png` - Time-based patterns and seasonality
  - `store_performance_analysis.png` - Store efficiency and comparison metrics
  - `business_intelligence_dashboard.png` - Executive KPI summary dashboard
  - `maze_training_progress.png` & `maze_solution.png` - RL maze navigation results

### Supporting Data Files
The `data/processed/visualization_data/` directory contains 17 CSV files with underlying data for each visualization:
- Weekly sales trends and temporal analysis data
- Provincial and store performance rankings
- Customer segmentation and value analysis
- Product and brand performance metrics
- Promotional effectiveness measurements
- Business intelligence KPIs and top performers summary

Note: The processed dataset is excluded from the repository due to size constraints but will be generated when you run the pipeline locally.

## Technical Stack
- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning models
- **matplotlib/seaborn** - Data visualization
- **pathlib** - File path management

## Assessment Criteria Addressed
- Code quality, structure, and organization
- Accuracy in data cleaning, normalization, and transformation
- Understanding and demonstration of supervised learning
- Implementation of business-valued solutions
- Understanding and demonstration of reinforcement learning
- Creative problem-solving approach
- Effective use of Python-compatible tools
- Documentation and clear instructions

## Contact
**Syed Shah**  
Data Engineer Candidate  

## License
This project is developed as part of a technical assessment for Middleby Corporation.  

