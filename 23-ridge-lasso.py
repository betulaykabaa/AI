import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# ==============================================================================
# 1. DATA GENERATION (Synthetic Data)
# ==============================================================================
np.random.seed(42)
n_samples = 50
n_features = 50 

# X: Random features
X = np.random.randn(n_samples, n_features)

# True Coefficients: Only the first 10 are real, the rest (40) are noise (0)
true_coef = 3 * np.random.randn(n_features)
true_coef[10:] = 0  
y = np.dot(X, true_coef) + np.random.normal(0, 2, n_samples)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# 2. TRAIN MODELS
# ==============================================================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1)
}

# Store predictions and coefficients for plotting
predictions = {}
coefficients = {}
scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    coefficients[name] = model.coef_
    scores[name] = r2_score(y_test, predictions[name])

# ==============================================================================
# 3. VISUALIZATION 1: ACTUAL VS PREDICTED (Performance)
# ==============================================================================
# We want to see how close the points are to the perfect diagonal line.

plt.figure(figsize=(15, 5))

for i, (name, y_pred) in enumerate(predictions.items()):
    plt.subplot(1, 3, i+1)
    
    # Scatter plot of predictions
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='k')
    
    # Perfect prediction line (Diagonal)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    plt.title(f"{name}\nR2 Score: {scores[name]:.2f}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==============================================================================
# 4. VISUALIZATION 2: COEFFICIENTS (The "Brain" of Models)
# ==============================================================================
# This chart shows how Ridge and Lasso handle "noise" features (Index 10 to 50).

plt.figure(figsize=(12, 6))

# Plot True Coefficients (The Ground Truth)
plt.plot(true_coef, 'k--', label='True Coefficients (Target)', linewidth=2, alpha=0.5)

# Plot Linear Regression (Often goes crazy/wild)
plt.plot(coefficients['Linear Regression'], 'r^', label='Linear Reg (Noisy)', markersize=4, alpha=0.5)

# Plot Ridge (Shrinks but keeps)
plt.plot(coefficients['Ridge (L2)'], 'bs', label='Ridge (Shrunk)', markersize=4)

# Plot Lasso (Eliminates)
plt.plot(coefficients['Lasso (L1)'], 'go', label='Lasso (Sparse)', markersize=6)

plt.title("Comparison of Model Coefficients (Weights)", fontsize=14)
plt.xlabel("Feature Index (0-9 are Real, 10-49 are Noise)")
plt.ylabel("Coefficient Value")
plt.axhline(0, color='black', linewidth=1) # Zero line
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()