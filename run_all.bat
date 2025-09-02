@echo off
echo ============================================================
echo SUPERMARKET ANALYTICS - COMPLETE PIPELINE 
echo ============================================================

echo Step 1: Data cleaning and preprocessing...
python src\data_preprocessing.py
if errorlevel 1 goto error

echo Step 2: Machine learning model training...
python src\supervised_learning.py  
if errorlevel 1 goto error

echo Step 3: Dashboard and chart generation...
python src\data_visualization.py
if errorlevel 1 goto error

echo Step 4: Export supporting data tables...
python src\export_visualization_data.py
if errorlevel 1 goto error

echo Step 5: Reinforcement learning maze (optional)...
python src\maze_navigation.py
if errorlevel 1 goto error

echo ============================================================
echo PIPELINE COMPLETED SUCCESSFULLY!
echo ============================================================
echo Check report/ folder for generated outputs:
echo   - supermart_report.pdf (main deliverable)
echo   - metrics.csv (model performance) 
echo   - figures/ (visualization dashboards)
echo ============================================================
goto end

:error
echo ============================================================
echo PIPELINE FAILED! 
echo Check error messages above for details.
echo ============================================================
exit /b 1

:end
pause