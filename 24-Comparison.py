import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# ==============================================================================
# 1. CREATE CURVED DATA (Sine Wave)
# ==============================================================================
np.random.seed(42)
n_samples = 30

# X: Between 0 and 10
X = np.sort(np.random.rand(n_samples) * 10).reshape(-1, 1)

# y: Sine wave + Noise
# This is the "True Function" we want to find
y = np.sin(X).ravel() + np.random.normal(0, 0.3, n_samples)

# We create a smooth range for plotting lines later
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)

# ==============================================================================
# 2. DEFINE MODELS (The Overfitting Trap)
# ==============================================================================
# We use Degree 15! This creates 15 features for just 1 variable.
# Linear Regression will try to use ALL of them perfectly.
degree = 15

# Standard Linear Regression (No Brakes)
model_linear = make_pipeline(
    PolynomialFeatures(degree),
    StandardScaler(), # Scaling is crucial for Ridge/Lasso
    LinearRegression()
)

# Ridge (L2 Regularization) - Gentle Brakes
model_ridge = make_pipeline(
    PolynomialFeatures(degree),
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Lasso (L1 Regularization) - Hard Brakes (Feature Selection)
# Increased max_iter because Lasso needs time to converge on complex data
model_lasso = make_pipeline(
    PolynomialFeatures(degree),
    StandardScaler(),
    Lasso(alpha=0.1, max_iter=100000)
)

# ==============================================================================
# 3. TRAIN MODELS
# ==============================================================================
model_linear.fit(X, y)
model_ridge.fit(X, y)
model_lasso.fit(X, y)

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(12, 7))

# A. Plot Actual Data
plt.scatter(X, y, color='black', s=50, label='Actual Data (Noisy Sine)')

# B. Plot Linear Model (The Crazy One)
plt.plot(X_plot, model_linear.predict(X_plot), color='red', linewidth=2, linestyle='--',
         label=f'Linear Reg (Degree {degree}) - OVERFITTING')

# C. Plot Ridge Model (The Balanced One)
plt.plot(X_plot, model_ridge.predict(X_plot), color='blue', linewidth=3, alpha=0.8,
         label='Ridge (L2) - Smooth')

# D. Plot Lasso Model (The Selective One)
plt.plot(X_plot, model_lasso.predict(X_plot), color='green', linewidth=3, alpha=0.8,
         label='Lasso (L1) - Simple')

plt.title('Polynomial Regression: Linear vs Ridge vs Lasso', fontsize=16)
plt.xlabel('X')
plt.ylabel('y')
plt.ylim(-2, 2) # Limit y-axis to ignore extreme Linear Reg outliers
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# ==============================================================================
# 5. CHECK THE SCORES (R2)
# ==============================================================================
print(f"{'MODEL':<20} | {'R2 SCORE (Higher is Better)'}")
print("-" * 50)
print(f"{'Linear Regression':<20} | {model_linear.score(X, y):.4f} (High but misleading!)")
print(f"{'Ridge (L2)':<20} | {model_ridge.score(X, y):.4f}")
print(f"{'Lasso (L1)':<20} | {model_lasso.score(X, y):.4f}")