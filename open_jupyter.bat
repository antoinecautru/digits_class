@echo off
REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Run Jupyter Notebook
jupyter notebook

REM Keep the command prompt open
cmd /k
