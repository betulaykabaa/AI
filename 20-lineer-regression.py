import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==============================================================================
# 1. DATA GENERATION (Synthetic Dataset)
# ==============================================================================
# We simulate a relationship between "Years of Experience" (X) and "Salary" (y).
# Formula: Salary = 3000 * Years + 25000 + (Random Noise)

np.random.seed(42) # For reproducible results

# X: Years of Experience (0 to 15 years) - Needs to be 2D array for sklearn
X = 15 * np.random.rand(100, 1)

# y: Salary (Target variable)
# We add randomness (noise) because real life isn't perfectly linear
noise = np.random.randn(100, 1) * 5000
y = (3000 * X) + 25000 + noise

# ==============================================================================
# 2. DATA SPLITTING (Train/Test Split)
# ==============================================================================
# split data into training set (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Points: {len(X_train)}")
print(f"Test Data Points: {len(X_test)}")
print("="*60 + "\n")

# ==============================================================================
# 3. MODEL TRAINING (.fit)
# ==============================================================================
# Initialize the Linear Regression model
model = LinearRegression()

print("Training the model using .fit()...")
# .fit(): The command that makes the model learn the relationship
model.fit(X_train, y_train)
print("Training complete.\n")
print("="*60 + "\n")

# ==============================================================================
# 4. MODEL INSPECTION (.coef_ and .intercept_)
# ==============================================================================
# Linear Regression equation: y = (Coefficient * X) + Intercept
# Coefficient (slope): How much y changes for a 1-unit change in X.
# Intercept (bias): The value of y when X is 0.

coefficient = model.coef_[0][0]
intercept = model.intercept_[0]

print("--- MODEL PARAMETERS (The Math Behind the Line) ---")
print(f"1. Slope (Coefficient / .coef_): {coefficient:.2f}")
print(f"   -> Meaning: For every 1 year of experience, salary increases by approx ${coefficient:.2f}.")
print("-" * 40)
print(f"2. Y-Intercept (Bias / .intercept_): {intercept:.2f}")
print(f"   -> Meaning: A starting salary with 0 experience is approx ${intercept:.2f}.")
print("-" * 40)
print(f"Final Equation Learned: Salary = ({coefficient:.2f} * Years) + {intercept:.2f}")
print("\n" + "="*60 + "\n")

# ==============================================================================
# 5. MODEL EVALUATION (.score)
# ==============================================================================
# Make predictions on the test set
y_pred_test = model.predict(X_test)

# .score() returns the R-squared (R²) value.
# R² tells us how well the regression line approximates the real data points.
# 1.0 is perfect fit, 0.0 is terrible fit.
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("--- MODEL EVALUATION (R-squared Score) ---")
print(f"Training Score (Accuracy on seen data): {train_score:.4f}")
print(f"Test Score (Accuracy on unseen data)  : {test_score:.4f}")
print("(Score closest to 1.0 means a better fit)")
print("\n" + "="*60 + "\n")

# ==============================================================================
# 6. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(12, 8))

# A) Scatter plot of Actual Training Data
plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=50, label='Training Data (Actual)')

# B) Scatter plot of Actual Test Data (The unseen data)
plt.scatter(X_test, y_test, color='red', alpha=0.6, s=70, edgecolor='black', label='Test Data (Actual)')

# C) Plotting the Regression Line
# To draw the smooth line, we predict Y values for the whole X range
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)
plt.plot(X_range, y_range_pred, color='green', linewidth=3, linestyle='--', label=f'Regression Line (Model Prediction)')


# D) Formatting the Plot
plt.title('Linear Regression: Experience vs Salary', fontsize=16, fontweight='bold')
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# Show the coefficient and intercept on the plot for clarity
plt.text(X.min()+1, y.max()-10000, 
         f'Model Equation:\nSalary = {coefficient:.0f}*Years + {intercept:.0f}\nR² Score: {test_score:.2f}', 
         fontsize=12, bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

plt.tight_layout()
plt.show()