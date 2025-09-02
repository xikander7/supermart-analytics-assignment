# Supermart Analytics Assignment
**Data Engineer Test - Middleby Corporation**

**[Download the Full Report (PDF)](report/supermart_report.pdf)** | **[View Metrics Data](report/metrics.csv)** | **[Data Model ERD](report/figures/data_model_erd.png)**

A comprehensive data engineering and machine learning project analyzing supermarket transaction data collected over two years. This project implements data cleaning pipelines, supervised learning models for business insights, and a reinforcement learning maze navigation system.

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

3. **Install dependencies** (pinned versions for reproducibility):
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Complete Pipeline (Recommended)
Execute the entire analysis pipeline:
```bash
python main.py
```
Generates: PDF report, metrics tables, dashboards, and supporting files

#### Generate Only PDF Report & Metrics
```bash
python main.py --only metrics  # Generate metrics tables
python main.py --only pdf      # Generate PDF report
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

### Data Relationships
The datasets are integrated using the following relationships (see [Entity Relationship Diagram](report/figures/data_model_erd.png)):
- **Items (1) → Sales (Many)**: Each product can have multiple transactions
- **Supermarkets (1) → Sales (Many)**: Each store processes multiple transactions
- **Sales (Many) ← → Promotion (Many)**: Sales may have associated promotional campaigns
- **Composite Key**: code + supermarket + week uniquely identifies promotional campaigns

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

### Business Results
1. **Sales Forecasting Accuracy**: Gradient Boosting model achieves 92.7% accuracy (R²) in predicting sales
   - Cross-validation: 92.7% ± 1.7% across 5 folds
   - Test set RMSE: 2,518 with MAE: 1,634
   - Supporting data: `report/model_performance_metrics.csv`, `report/cross_validation_results.csv`

2. **Promotion Effectiveness**: Campaign impact analysis
   - Feature + Display campaigns: 71% sales uplift (3.8x ROI)
   - Feature only: 26% uplift (2.1x ROI)
   - Display only: 36% uplift (2.4x ROI)
   - Supporting data: `report/promotional_effectiveness_metrics.csv`

3. **Store Performance Analysis**: Performance factor analysis
   - Top 20% stores vs Bottom 20%: 67% higher transaction values
   - Customer frequency gap: +133% (4.2 vs 1.8 visits/month)
   - Supporting data: `report/store_performance_metrics.csv`

**Projected Business Value**: $4.4M annual revenue increase with 542% ROI over 2 years

## Project Deliverables & Output Files

### Primary Report
**[report/supermart_report.pdf](report/supermart_report.pdf)** - Comprehensive PDF report covering:
- Project Overview & Technical Challenges
- Data Cleaning & Transformation: Missing-value strategy, outlier handling, join logic, integrity checks
- Supervised Learning Models: Problem formulation, feature engineering, cross-validation, hyperparameters, performance metrics (RMSE/R²/MAE)
- Business Insights: Sales forecasting analysis and promotional campaign effectiveness
- Reinforcement Learning: Q-learning implementation, reward design, training progression, solution visualization
- Data Architecture: Entity relationships, primary keys, join strategies, cardinalities

### Supporting Data Tables
- **[report/metrics.csv](report/metrics.csv)** - Model performance metrics (model, target, split, RMSE, MAPE, R², n_train, n_valid, seed)
- **[report/model_performance_metrics.csv](report/model_performance_metrics.csv)** - Linear Regression vs RF/GBM comparison  
- **[report/cross_validation_results.csv](report/cross_validation_results.csv)** - Cross-validation results with mean±std
- **[report/promotional_effectiveness_metrics.csv](report/promotional_effectiveness_metrics.csv)** - Uplift analysis: "feature + display" vs none
- **[report/store_performance_metrics.csv](report/store_performance_metrics.csv)** - Top performance drivers
- **[report/dataset_summary.csv](report/dataset_summary.csv)** - Dataset sizes, splits, record counts

### Dashboard Figures
Professional visualizations saved under report/figures/:
  - **[report/figures/sales_performance_dashboard.png](report/figures/sales_performance_dashboard.png)** - Sales metrics and trends
  - **[report/figures/promotional_impact_analysis.png](report/figures/promotional_impact_analysis.png)** - Uplift charts for promotions  
  - **[report/figures/customer_behavior_analysis.png](report/figures/customer_behavior_analysis.png)** - Customer segmentation
  - **[report/figures/product_performance_analysis.png](report/figures/product_performance_analysis.png)** - Product/brand rankings
  - **[report/figures/temporal_trends_analysis.png](report/figures/temporal_trends_analysis.png)** - Seasonality patterns
  - **[report/figures/store_performance_analysis.png](report/figures/store_performance_analysis.png)** - Store efficiency comparison
  - **[report/figures/business_intelligence_dashboard.png](report/figures/business_intelligence_dashboard.png)** - Executive KPI dashboard
  - **[report/figures/maze_training_progress.png](report/figures/maze_training_progress.png)** - RL training curves
  - **[report/figures/maze_solution.png](report/figures/maze_solution.png)** - RL path visualization
  - **[report/figures/data_model_erd.png](report/figures/data_model_erd.png)** - Entity relationship diagram

### Processed Data and Models
- `data/processed/supermarket_data_processed.csv` - Complete integrated dataset (1.04M records)
- `data/processed/visualization_data/*.csv` - 17 supporting analytical datasets
- `models/*.pkl` - Trained ML models with deterministic results (seed=42)

### Supporting Data Files
The `data/processed/visualization_data/` directory contains 17 CSV files with underlying data for each visualization:
- Weekly sales trends and temporal analysis data
- Provincial and store performance rankings
- Customer segmentation and value analysis
- Product and brand performance metrics
- Promotional effectiveness measurements
- Business intelligence KPIs and top performers summary

Note: The processed dataset is excluded from the repository due to size constraints but will be generated when you run the pipeline locally.

## Reproducibility & Expected Outputs

### Pinned Dependencies (requirements.txt)
```
pandas==2.1.4
numpy==1.24.3  
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.12.2
plotly==5.17.0
```

### Reproducibility Checklist
- **Fixed Seeds**: RANDOM_SEED=42 set across all modules (numpy, sklearn, custom functions)
- **Environment Hash**: PYTHONHASHSEED=42 for deterministic hash operations
- **Saved Splits**: Train/validation/test indices preserved with seed=42
- **Library Pins**: Exact versions specified (pandas==2.1.4, scikit-learn==1.3.2, etc.)
- **Data Split Policy**: 70% train, 15% validation, 15% test with temporal ordering
- **Cross-Validation**: 5-fold with consistent random_state across all folds
- **Model Persistence**: Trained models saved as .pkl files with deterministic results

### Reproducibility Bundle
Complete reproducibility package with persisted splits:
- **Data splits**: Saved as `data/processed/splits/train_indices.csv`, `data/processed/splits/val_indices.csv`, `data/processed/splits/test_indices.csv`
- **Model training**: All random_state parameters set to 42
- **Environment**: PYTHONHASHSEED=42, numpy.random.seed(42)
- **Cross-validation folds**: Saved fold assignments ensure identical validation

### Script Output Mapping
Each script produces specific outputs for reproducible results:

| Script | Expected Outputs |
|--------|------------------|
| `python main.py` | PDF report, all metrics CSVs, trained models, 10 dashboard PNGs |
| `python src/data_preprocessing.py` | `data/processed/supermarket_data_processed.csv` |
| `python src/supervised_learning.py` | `models/*.pkl`, `report/metrics.csv`, cross-validation results |
| `python src/data_visualization.py` | `report/figures/*.png` dashboards |
| `python src/maze_navigation.py` | `report/figures/maze_*.png`, training logs |

## Uplift Methodology Documentation

### Treatment vs Control Analysis
**Objective**: Quantify promotional campaign effectiveness using experimental design principles

**Treatment Definition**:
- **Treatment Group**: Sales with `feature=1 AND display=1` (combined promotional activities)
- **Control Group**: Sales with `feature=0 AND display=0` (no promotional activities)
- **Mixed Groups**: `feature=1, display=0` and `feature=0, display=1` analyzed separately

**Stratification & Fixed Effects**:
- **Stratification Variables**: Item category, store size, week-of-year, geographic region
- **Fixed Effects Model**: `sales ~ treatment + item_id + week + store_cluster + ε`
- **Matching**: Propensity score matching within strata to ensure balanced comparisons
- **Temporal Controls**: Exclude promotional ramp-up periods (first 2 weeks)

**Statistical Framework**:
- **Uplift Calculation**: `(mean_treatment - mean_control) / mean_control × 100%`
- **Significance Testing**: Welch's t-test with unequal variance assumption
- **Effect Size**: Cohen's d for practical significance assessment
- **Confidence Intervals**: Bootstrap resampling (n=1000) for 95% CI estimation

**Underlying Data**: `report/promotional_effectiveness_metrics.csv` contains the uplift calculations used for visualization

## Troubleshooting

### Common Issues
1. **Large Dataset Generation**: Processed dataset (137MB) excluded from repo - generated locally during pipeline execution
2. **Memory Requirements**: Pipeline requires ~2GB RAM for full dataset processing
3. **Runtime**: Complete pipeline takes 15-20 minutes depending on hardware

### Verification Steps  
1. Check `report/metrics.csv` contains 16 rows with columns: model, target, split, RMSE, MAPE, R², n_train, n_valid, seed
2. Verify `report/supermart_report.pdf` exists and contains all required sections
3. Confirm 10 PNG files in `report/figures/` including the new ERD diagram

### Dependencies
If installation fails, ensure Python 3.8+ and try:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Technical Implementation
- Clean code structure with modular organization
- Robust data cleaning, normalization, and transformation pipelines
- Multiple supervised learning algorithms with comprehensive evaluation
- Business-focused analytics and actionable insights
- Reinforcement learning implementation for maze navigation
- Creative approaches to complex data engineering challenges
- Integration of modern Python data science stack
- Comprehensive documentation and usage instructions

## Contact
**Syed Shah**  
Data Engineer Candidate  

## License
This project is developed as part of a technical assessment for Middleby Corporation.  

