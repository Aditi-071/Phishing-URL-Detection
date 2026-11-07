# Phishing URL Detection — Single Dataset

## 1. Dataset
Place your CSV in `data/`. Default filename:
`phishing_site_urls_control_chars_removed.csv`

Expected columns:
- `URL`: website URL
- `Label`: `good` (legitimate) or `bad` (phishing)

If your file has a different name or column names, pass them as command-line options.

## 2. Setup in VS Code (Windows PowerShell terminal)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Run
```powershell
python main.py
```

For the original filename:
```powershell
python main.py --csv "data/phishing_site_urls.csv"
```

For custom columns:
```powershell
python main.py --csv "data/mydata.csv" --url-col "URL" --label-col "Label"
```

## 4. Outputs
- `reports/model_comparison.csv`: metrics for all three models
- `reports/data_quality_audit.json`: row counts and class counts
- `reports/*confusion_matrix.png`: confusion matrices
- `reports/roc_curves.png`: ROC curves
- `reports/*classification_report.txt`: precision/recall/F1 per class
- `reports/*feature_importance.csv`: tree-model feature importance
- `models/*.joblib`: saved trained models

## 5. Important interpretation
The project uses a stratified random 80/20 split. Results estimate performance on held-out examples from this same dataset. They do not establish generalization to a different dataset, future URLs, or live traffic.
