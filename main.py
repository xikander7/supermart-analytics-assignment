"""
Main execution script for Supermarket Analytics Assignment
Data Engineer Test - Middleby Corporation

This script runs the complete data analysis pipeline including:
1. Data preprocessing and cleaning
2. Supervised learning for business insights
3. Optional maze navigation reinforcement learning
"""

import sys
import logging
from pathlib import Path
import argparse

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_preprocessing import SupermarketDataPreprocessor
from supervised_learning import SupermarketAnalytics
from maze_navigation import MazeTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_preprocessing():
    """Run data preprocessing pipeline."""
    logger.info("="*60)
    logger.info("STEP 1: DATA PREPROCESSING AND CLEANING")
    logger.info("="*60)
    
    try:
        preprocessor = SupermarketDataPreprocessor(data_dir="data/raw")
        
        # Load raw data
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
        print("\n" + "-"*40)
        print("DATA PROCESSING SUMMARY")
        print("-"*40)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("-"*40)
        
        logger.info("Data preprocessing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return False

def run_supervised_learning():
    """Run supervised learning analysis."""
    logger.info("="*60)
    logger.info("STEP 2: SUPERVISED LEARNING FOR BUSINESS INSIGHTS")
    logger.info("="*60)
    
    try:
        # Check if processed data exists
        processed_data_path = Path("data/processed/supermarket_data_processed.csv")
        if not processed_data_path.exists():
            logger.error("Processed data not found. Please run data preprocessing first.")
            return False
        
        analytics = SupermarketAnalytics()
        
        # Load processed data
        analytics.load_data()
        
        # Run business insights analyses
        print("\n" + "-"*50)
        print("Running Business Insight 1: Sales Forecasting...")
        print("-"*50)
        analytics.business_insight_1_sales_forecasting()
        
        print("\n" + "-"*50)
        print("Running Business Insight 2: Promotion Impact Analysis...")
        print("-"*50)
        analytics.business_insight_2_promotion_impact()
        
        print("\n" + "-"*50)
        print("Running Business Insight 3: Supermarket Performance Analysis...")
        print("-"*50)
        analytics.business_insight_3_supermarket_performance()
        
        # Generate insights summary
        insights = analytics.generate_business_insights()
        
        # Save models
        analytics.save_models()
        
        # Generate visualizations
        analytics.generate_visualizations()
        
        # Print business insights
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS SUMMARY")
        print("="*60)
        
        for insight_name, insight_text in insights.items():
            print(f"\n{insight_name.upper().replace('_', ' ')}:")
            print(f"  {insight_text}")
        
        print("\n" + "="*60)
        
        logger.info("Supervised learning analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Supervised learning analysis failed: {e}")
        return False

def run_maze_navigation():
    """Run maze navigation reinforcement learning."""
    logger.info("="*60)
    logger.info("STEP 3: MAZE NAVIGATION REINFORCEMENT LEARNING (OPTIONAL)")
    logger.info("="*60)
    
    try:
        # Create maze trainer
        trainer = MazeTrainer(maze_size=(10, 10))
        
        print("\nTraining maze navigation agent...")
        print("-"*40)
        
        # Train agent
        trainer.train(episodes=1000, max_steps_per_episode=200)
        
        # Test agent performance
        print("\nTesting trained agent...")
        print("-"*40)
        test_results = trainer.test_agent(num_tests=10)
        
        # Create visualizations
        trainer.visualize_training_progress()
        
        # Save trained model
        trainer.save_model()
        
        # Print final results
        success_count = sum(1 for result in test_results if result['success'])
        print("\n" + "="*50)
        print("MAZE NAVIGATION RESULTS")
        print("="*50)
        print(f"Training Episodes: 1000")
        print(f"Test Success Rate: {success_count}/10 ({success_count*10}%)")
        
        if success_count > 0:
            import numpy as np
            successful_tests = [r for r in test_results if r['success']]
            avg_steps = np.mean([r['steps'] for r in successful_tests])
            print(f"Average Steps (Successful): {avg_steps:.1f}")
        
        print("="*50)
        
        logger.info("Maze navigation training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Maze navigation training failed: {e}")
        return False

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Supermarket Analytics Assignment - Data Engineer Test')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip data preprocessing step')
    parser.add_argument('--skip-ml', action='store_true', 
                       help='Skip supervised learning analysis')
    parser.add_argument('--skip-maze', action='store_true', 
                       help='Skip maze navigation training')
    parser.add_argument('--only', choices=['preprocessing', 'ml', 'maze'], 
                       help='Run only the specified component')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SUPERMARKET ANALYTICS ASSIGNMENT - DATA ENGINEER TEST")
    print("Middleby Corporation")
    print("="*80)
    
    success_count = 0
    total_steps = 0
    
    # Determine which steps to run
    if args.only:
        if args.only == 'preprocessing':
            steps_to_run = ['preprocessing']
        elif args.only == 'ml':
            steps_to_run = ['ml']
        elif args.only == 'maze':
            steps_to_run = ['maze']
    else:
        steps_to_run = []
        if not args.skip_preprocessing:
            steps_to_run.append('preprocessing')
        if not args.skip_ml:
            steps_to_run.append('ml')
        if not args.skip_maze:
            steps_to_run.append('maze')
    
    # Run selected steps
    if 'preprocessing' in steps_to_run:
        total_steps += 1
        if run_data_preprocessing():
            success_count += 1
        print("\n")
    
    if 'ml' in steps_to_run:
        total_steps += 1
        if run_supervised_learning():
            success_count += 1
        print("\n")
    
    if 'maze' in steps_to_run:
        total_steps += 1
        if run_maze_navigation():
            success_count += 1
        print("\n")
    
    # Final summary
    print("="*80)
    print("ASSIGNMENT EXECUTION SUMMARY")
    print("="*80)
    print(f"Steps Completed Successfully: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("✓ All components completed successfully!")
        print("\nGenerated Files:")
        print("  - data/processed/supermarket_data_processed.csv")
        print("  - models/[trained_models].pkl")
        print("  - report/figures/[visualizations].png")
    else:
        print(f"⚠ Some components failed. Check logs above for details.")
    
    print("="*80)
    
    return success_count == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)