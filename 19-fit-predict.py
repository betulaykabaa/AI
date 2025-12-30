import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. Data (Study Hours vs Pass/Fail)
# 0: Fail (Kaldı), 1: Pass (Geçti)
X = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 5.5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# 2. Train Model
model = LogisticRegression()
model.fit(X, y)

# 3. Prepare Visualization Data
X_test_smooth = np.linspace(0, 10, 300).reshape(-1, 1)

# Get probabilities
y_prob = model.predict_proba(X_test_smooth)[:, 1]

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

# A. The Actual Data Points
plt.scatter(X, y, color='black', s=100, label='Actual Data', zorder=3)

# B. The Sigmoid Curve
plt.plot(X_test_smooth, y_prob, color='blue', linewidth=3, label='Sigmoid Curve (Probability)')

# C. The Decision Boundary (The 0.5 Threshold)
plt.axhline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')

# --- FIX IS HERE ---
# We find the index where probability is closest to 0.5
idx = np.abs(y_prob - 0.5).argmin()

# We extract the single value using [0] to avoid TypeError
limit_x = X_test_smooth[idx][0] 

plt.axvline(limit_x, color='green', linestyle=':', linewidth=2, label=f'Critical Limit (~{limit_x:.1f} Hours)')

# Formatting
plt.title('Logistic Regression Visualization: Who Passes?')
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing (0 to 1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()