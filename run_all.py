#!/usr/bin/env python3
"""
Complete Pipeline Runner for Supermarket Analytics
Executes all analysis components in correct order
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    logger.info(f"Starting: {description}")
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        logger.info(f"✓ Completed: {description} ({duration:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Execute complete analysis pipeline"""
    logger.info("="*60)
    logger.info("SUPERMARKET ANALYTICS - COMPLETE PIPELINE")
    logger.info("="*60)
    
    # Define pipeline steps
    steps = [
        ("src/data_preprocessing.py", "Data cleaning and preprocessing"),
        ("src/supervised_learning.py", "Machine learning model training"),
        ("src/data_visualization.py", "Dashboard and chart generation"), 
        ("src/export_visualization_data.py", "Export supporting data tables"),
        ("src/maze_navigation.py", "Reinforcement learning maze (optional)")
    ]
    
    # Execute each step
    total_start = time.time()
    successful_steps = 0
    
    for script, description in steps:
        if Path(script).exists():
            if run_script(script, description):
                successful_steps += 1
            else:
                logger.error(f"Pipeline failed at: {description}")
                break
        else:
            logger.warning(f"Script not found: {script} - Skipping")
    
    # Summary
    total_duration = time.time() - total_start
    logger.info("="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Completed: {successful_steps}/{len(steps)} steps")
    logger.info(f"Total time: {total_duration:.1f} seconds")
    
    if successful_steps == len(steps):
        logger.info("✓ All components executed successfully!")
        logger.info("Check report/ folder for generated outputs:")
        logger.info("  - supermart_report.pdf (main deliverable)")
        logger.info("  - metrics.csv (model performance)")
        logger.info("  - figures/ (visualization dashboards)")
    else:
        logger.error("✗ Pipeline completed with errors")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())