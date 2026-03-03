# EcoReport AI — Methodology

> This document explains every statistical and ML choice in plain language for economists and ML practitioners alike.

---

## 1. Data Acquisition & Preprocessing

### 1.1 FRED API vs Local Fallback
The pipeline first attempts to download data from the Federal Reserve Economic Data (FRED) API using series codes `CPIAUCSL`, `UNRATE`, and `FEDFUNDS`. If `FRED_API_KEY` is not set or the request fails, it transparently falls back to `data/sample_macro.csv`.

### 1.2 Frequency Enforcement
All series are resampled to monthly start (`MS`) frequency using:
```python
df.resample("MS").last()
```
This ensures a consistent DateTimeIndex across all series.

### 1.3 Missing Value Strategy
| Scenario | Method | Justification |
|----------|--------|--------------|
| < 5% missing | Forward-fill then back-fill | Time-series safe; preserves most recent observed value |
| > 5% missing | Linear interpolation between known values | Avoids extreme distortion while filling gaps |
| After interpolation | Remaining NaN → drop row | Safety net; documented in report |

**Why not mean imputation?** Mean imputation ignores temporal structure and can introduce discontinuities.

### 1.4 CPI Year-over-Year Computation
```
CPI_YoY(t) = (CPIAUCSL(t) / CPIAUCSL(t-12) - 1) × 100
```
This removes the unit-root (non-stationarity) inherent in the CPI level series, producing a percentage-change target that is more stationary and directly interpretable as inflation.

---

## 2. Feature Engineering

### 2.1 Lag Features
For each series `x`, lags `x_{t-1}, ..., x_{t-k}` where `k = config.features.max_lag` (default: 6) are computed. These capture the autocorrelation structure directly as inputs to ML/DL models.

### 2.2 Rolling Statistics
Rolling mean and standard deviation over windows `[3, 6, 12]` months:
- **Rolling mean**: smooths noise, captures trend
- **Rolling std**: captures volatility regime changes

All rolling computations use `min_periods=1` to avoid NaN at the start.

### 2.3 Train/Test Split (No Leakage)
All scaling (for LSTM) and rolling statistics are fitted exclusively on training data. Test data is transformed using training-data parameters. This mirrors production deployment where future data is unavailable.

---

## 3. Econometric Models

### 3.1 OLS Regression with HAC Standard Errors
**Model**:
```
CPI_YoY_t = β₀ + β₁ × FEDFUNDS_t + β₂ × UNRATE_t + β₃ × CPI_YoY_{t-1} + β₄ × CPI_YoY_{t-3} + ε_t
```

**HAC Correction (Newey-West)**:  
Macroeconomic time series exhibit:
1. **Heteroskedasticity**: variance of ε changes over time (e.g., higher during crises)
2. **Autocorrelation**: ε_t is correlated with ε_{t-1}, ..., ε_{t-k}

Standard OLS standard errors are biased under these conditions. HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors use a weighted sum of sample autocovariances:
```
V_HAC = (X'X)⁻¹ [Σ_j w_j Σ_t ε_t ε_{t-j} x_t x_{t-j}'] (X'X)⁻¹
```
with Bartlett kernel weights `w_j = 1 - j/(L+1)` for `j ≤ L` (Newey-West, 1987), where `L = 6` lags.

**Outputs**: β coefficients, HAC standard errors, t-statistics, p-values, R², Adjusted R².

### 3.2 ARIMA/SARIMAX
**Stationarity test**: Augmented Dickey-Fuller (ADF) test determines differencing order `d`.  
**Order selection**: Grid search over `p ∈ [0,1,2,3]`, `q ∈ [0,1,2]` minimizing AIC.  
**Seasonal component**: Optional `(P,D,Q,s=12)` seasonal ARIMA for CPI.  
**Forecast**: `h`-step ahead forecast with 95% confidence intervals from `statsmodels`.

---

## 4. Machine Learning Model

### 4.1 Gradient Boosting Regressor
Ensemble of decision trees where each tree fits the residuals of the previous ensemble:
```
F_m(x) = F_{m-1}(x) + γ_m h_m(x)
```
where `h_m` minimizes the loss on pseudo-residuals.

**Why gradient boosting over random forests?** Gradient boosting typically achieves lower bias for structured tabular data. Both are available via config.

**Hyperparameter search**: `RandomizedSearchCV` on training folds only (prevents test-set leakage). Parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`.

---

## 5. LSTM Deep Learning Model

### 5.1 Architecture
```
Input: (batch, lookback=12, n_features)
    └─► LSTM Layer 1 (hidden=64, dropout=0.2)
    └─► LSTM Layer 2 (hidden=64, dropout=0.2)
    └─► Linear(64, 1)
Output: (batch, 1)  — CPI YoY prediction
```

### 5.2 Sliding Window Dataset
For a sequence of length `T` and lookback `L`, training samples are:
```
X[i] = data[i : i+L]       shape: (L, n_features)
y[i] = target[i+L]         shape: scalar
```
This creates `T - L` training samples.

### 5.3 Training Protocol
- Loss: MSE
- Optimizer: Adam (lr=1e-3)
- Early stopping: monitor validation loss, patience=10 epochs
- Batch size: 32
- All seeds set from `config.seed` for reproducibility

---

## 6. Evaluation: Rolling-Origin Cross-Validation

```
Fold 1:  Train=[0..T₀]           Test=[T₀+1..T₀+h]
Fold 2:  Train=[0..T₀+step]      Test=[T₀+step+1..T₀+step+h]
...
Fold k:  Train=[0..T₀+(k-1)step] Test=[T₀+(k-1)step+1..T₀+(k-1)step+h]
```
Default: `T₀` is set so at least 60 months remain for evaluation, `step = h = 6`, `k = 5`.

**Key property**: Each fold's test window begins strictly after its training window ends. This is the only correct way to evaluate forecasting models.

### 6.1 Metrics
| Metric | Formula | Units |
|--------|---------|-------|
| MAE | `mean(|y - ŷ|)` | % points |
| RMSE | `sqrt(mean((y - ŷ)²))` | % points |
| MAPE | `mean(|y - ŷ| / max(|y|, 1e-8)) × 100` | % |

---

## 7. Model Selection

```python
best_model = min(models, key=lambda m: mean_rmse_across_folds[m])
```

The best model then generates an out-of-sample forecast for the next `config.evaluation.forecast_horizon` months (default: 12).

---

## 8. Report Generation

### 8.1 Evidence Store
Before narrative generation, all computed statistics are collected into a single `dict`:
```python
evidence = {
    "cpi_yoy_latest": 3.2,
    "cpi_yoy_mean": 2.8,
    "best_model": "ARIMA",
    "best_model_rmse": 0.41,
    "ols_fedfunds_coeff": -0.12,
    "ols_fedfunds_pvalue": 0.03,
    ...
}
```

### 8.2 Template NLG (Offline)
String templates are populated with evidence values:
```python
f"CPI inflation averaged {evidence['cpi_yoy_mean']:.2f}% annually..."
```
Every number in the report corresponds to a key in the evidence store.

### 8.3 LLM NLG (Optional)
If `OPENAI_API_KEY` is present, GPT-4o-mini is called with:
- System role: economic analyst
- Pre-computed evidence store as JSON
- Explicit prohibition on generating numbers not in evidence
- Temperature: 0.3
- `response_format: json_object`

The response is merged with the deterministic table outputs, and all numeric values are validated against the evidence store.

---

## 9. References

- Newey, W.K. & West, K.D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708.
- Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Hyndman, R.J. & Koehler, A.B. (2006). Another Look at Measures of Forecast Accuracy. *International Journal of Forecasting*, 22(4), 679–688.
