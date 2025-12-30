import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# ==============================================================================
# 1. CREATE NON-LINEAR DATA (Kıvrımlı Veri)
# ==============================================================================
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3       # X between -3 and 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) # Quadratic equation (Parabola)

# ==============================================================================
# 2. LINEAR REGRESSION (The "Bad" Model)
# ==============================================================================
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Calculate R2 Score for Linear Model
lin_r2 = r2_score(y, y_lin_pred)

# ==============================================================================
# 3. POLYNOMIAL REGRESSION (The "Good" Model)
# ==============================================================================
# Transform X into X^2 (Degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Calculate R2 Score for Polynomial Model
poly_r2 = r2_score(y, y_poly_pred)

# ==============================================================================
# 4. PRINT SCORES TO CONSOLE
# ==============================================================================
print("-" * 40)
print(f"Linear Regression R2 Score    : {lin_r2:.4f} (Underfitting)")
print(f"Polynomial Regression R2 Score: {poly_r2:.4f} (Good Fit)")
print("-" * 40)

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(10, 6))

# A) Scatter Plot of Actual Data
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')

# B) Plot Linear Regression (Sort X for smooth line plotting)
# We plot the line using the min and max of X
plt.plot(X, y_lin_pred, color='red', linestyle='--', linewidth=2, 
         label=f'Linear Model (R² = {lin_r2:.2f})')

# C) Plot Polynomial Regression
# To draw a smooth curve, we need a sorted range of X values
X_range = np.linspace(-3, 3, 100).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
y_range_pred = poly_reg.predict(X_range_poly)

plt.plot(X_range, y_range_pred, color='green', linewidth=3, 
         label=f'Polynomial Model (R² = {poly_r2:.2f})')

plt.title('Linear vs Polynomial Regression: R² Score Comparison', fontsize=14)
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()