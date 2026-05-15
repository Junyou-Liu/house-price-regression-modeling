# House Price Regression Modeling

This project builds an interpretable housing price prediction workflow using log-linear regression, hedonic feature engineering, regularized linear models, and an XGBoost benchmark. The emphasis is not only prediction accuracy, but also clean feature construction, leakage control, and model interpretation.

## Why This Project Matters

- Turns raw housing transaction records into a modeling-ready dataset of 4,553 observations.
- Uses transparent regression specifications before adding a non-linear benchmark.
- Shows how property age, renovation status, view quality, condition, and zip-code effects change predictive error.
- Fits zip-code target encoding only on the training split, then maps it to the hold-out set to avoid location leakage.
- Reports results with a direct hold-out metric: Mean Absolute Percentage Error (MAPE).

## Data and Method

The input file is `house_dataset.csv`, containing transaction, structure, location, and date fields. The notebook follows this sequence:

1. Clean raw records and standardize usable numeric, date, and location fields.
2. Log-transform sale price and living area to improve linear fit and coefficient interpretation.
3. Add lifecycle, structural, and spatial pricing features.
4. Compare OLS variants with RidgeCV, LassoCV, and XGBoost on the same hold-out split.
5. Report model performance with test-set MAPE.

## Model Results

| Model | Main Specification | Test-Set MAPE |
| --- | --- | ---: |
| Baseline OLS | Log price vs. log living area | 33.5520% |
| Lifecycle OLS | Adds age at sale and renovation status | 32.5106% |
| Structural OLS | Adds layout, view, and condition features | 31.2006% |
| Engineered OLS | Adds train-only zip-code target encoding | 17.8971% |
| RidgeCV | L2-regularized engineered feature set | 17.8952% |
| LassoCV | L1-regularized engineered feature set | 17.8963% |
| XGBoost | Non-linear benchmark | 16.9544% |

The main improvement comes from disciplined feature engineering and train-only location encoding. XGBoost performs best, while the engineered linear models stay close enough to remain useful when interpretability is more important than marginal accuracy.

## Repository Structure

```text
.
|-- Assignment 1 House Price Prediction with Linear Models/
|   |-- Assignment 1 House Price Prediction with Linear Models.ipynb
|   |-- Assignment 1 House Price Prediction with Linear Models.html
|   `-- house_dataset.csv
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Reproduce

The notebook was executed successfully with Python 3.13.7.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
cd "Assignment 1 House Price Prediction with Linear Models"
jupyter notebook "Assignment 1 House Price Prediction with Linear Models.ipynb"
```

Command-line execution:

```powershell
cd "Assignment 1 House Price Prediction with Linear Models"
..\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace "Assignment 1 House Price Prediction with Linear Models.ipynb" --ExecutePreprocessor.timeout=600
```

## Scope and Limits

This is a course modeling assignment and should be read as evidence of regression modeling, feature engineering, and validation discipline. It is not a production valuation system, and the reported MAPE is tied to the provided dataset and split.

## Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, XGBoost, and Jupyter Notebook.
