# Streamlit App Setup & Running Guide

## Project Structure
```
quant momentum backtester/
├── src/
│   ├── __init__.py                 ✓ (package marker)
│   ├── app/
│   │   ├── __init__.py             ✓ (package marker)
│   │   └── streamlit_app.py
│   └── core/
│       ├── __init__.py             ✓ (package marker)
│       ├── config.py
│       ├── data_loader.py
│       ├── calendar.py
│       ├── signals.py
│       ├── strategy.py
│       ├── engine.py
│       ├── analytics.py
│       └── plotting.py
├── tests/
│   ├── test_*.py
├── requirements.txt
├── pyproject.toml
└── run_app.py                      (helper script)
```

## Setup Instructions

### 1. Ensure Virtual Environment is Activated
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt --only-binary :all:
```

## Running the Streamlit App

### Method 1: Using Python Helper Script (Recommended)
```powershell
python run_app.py
```

### Method 2: Direct Streamlit Command from Project Root
```powershell
streamlit run src/app/streamlit_app.py
```

### Method 3: Using Python Module Invocation
```powershell
python -m streamlit run src/app/streamlit_app.py
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`
**Solution:** Ensure you're running from the project root directory:
```powershell
cd D:\quant momentum backtester
streamlit run src/app/streamlit_app.py
```

### Issue: Streamlit not found
**Solution:** Make sure the virtual environment is activated:
```powershell
.\.venv\Scripts\Activate.ps1
pip install streamlit
```

### Issue: Modules still not found
**Solution:** Use the helper script which sets PYTHONPATH:
```powershell
python run_app.py
```

## Package Import Structure

All imports in the app use absolute paths from the project root:
```python
from src.core.config import Config
from src.core.engine import BacktestEngine
# etc.
```

The `src/` folder is marked as a package with `__init__.py`, allowing Python to:
1. Recognize `src` as a package
2. Resolve imports from project root
3. Enable relative imports within `src/`

## Access the App

Once running, the Streamlit app will be available at:
```
http://localhost:8501
```

Press `Ctrl+C` in the terminal to stop the app.
