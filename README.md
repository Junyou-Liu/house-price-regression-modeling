# House Price Regression Modeling

This project builds a reproducible housing price prediction workflow around interpretable regression models. It starts from a log-log OLS baseline, adds domain-informed hedonic features, controls for location effects through target encoding, and benchmarks the final linear models against XGBoost.

## Project Highlights

- Cleaned the housing transaction dataset from raw structural, temporal, and location fields into 4,553 usable observations.
- Applied log transformation to price and living area to improve linear fit and support percentage-based interpretation.
- Engineered lifecycle, structural, and spatial pricing signals, including property age, renovation status, bath-to-bed ratio, view indicators, condition flags, and zip-level target encoding.
- Prevented location target leakage by fitting the zip-code encoding only on the training split before mapping it to the hold-out test set.
- Compared OLS, RidgeCV, LassoCV, and XGBoost using test-set Mean Absolute Percentage Error (MAPE).

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

The results show that disciplined feature engineering closes most of the gap between transparent linear models and a stronger non-linear benchmark. The final linear models remain easier to explain while achieving a hold-out MAPE below 18%.

## Repository Structure

```text
.
├── Assignment 1 House Price Prediction with Linear Models/
│   ├── Assignment 1 House Price Prediction with Linear Models.ipynb
│   ├── Assignment 1 House Price Prediction with Linear Models.html
│   └── house_dataset.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproduce the Analysis

The notebook was executed successfully with Python 3.13.7.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
cd "Assignment 1 House Price Prediction with Linear Models"
jupyter notebook "Assignment 1 House Price Prediction with Linear Models.ipynb"
```

To execute the notebook from the command line:

```powershell
cd "Assignment 1 House Price Prediction with Linear Models"
..\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace "Assignment 1 House Price Prediction with Linear Models.ipynb" --ExecutePreprocessor.timeout=600
```

## Technical Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, and XGBoost.
