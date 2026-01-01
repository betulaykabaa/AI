import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Data
# We create 100 data points.
# X: Features (e.g., Study Hours)
# y: Target (e.g., Exam Score)
X = np.arange(100).reshape(-1, 1)  # Numbers from 0 to 99
y = X * 2  # Simple relationship (y = 2x)

# 2. Split the Data (Crucial Step)
# test_size=0.2  -> 20% for Testing, 80% for Training
# random_state=42 -> Ensures the split is the same every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes to verify
print(f"Total Data: {len(X)}")
print(f"Training Data (X_train): {len(X_train)} samples")
print(f"Test Data (X_test): {len(X_test)} samples")

# --- Visualization ---
plt.figure(figsize=(10, 6))

# Plot Training Data
plt.scatter(X_train, y_train, color='blue', label='Training Set (Model Learns from these)', alpha=0.7)

# Plot Test Data
plt.scatter(X_test, y_test, color='red', label='Test Set (Model is tested on these)', s=100, edgecolor='black')

plt.title("Train - Test Split Visualization (80/20 Ratio)")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
