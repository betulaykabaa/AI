import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Required Libraries
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# STEP 1: Generate Data (Moon Shaped)
# ---------------------------------------------------------
# noise=0.3 adds some randomness to make the task harder/realistic.
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# ---------------------------------------------------------
# STEP 2: Split into Train and Test Sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------
# STEP 3: Model Setup (Gradient Boosting)
# ---------------------------------------------------------
# n_estimators=100: Build 100 sequential trees.
# learning_rate=0.1: Each tree contributes 10% to the correction.
# max_depth=3: Limit tree depth to prevent overfitting (weak learners).
gb_model = GradientBoostingClassifier(n_estimators=100, 
                                      learning_rate=0.1, 
                                      max_depth=3, 
                                      random_state=42)
gb_model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 4: Evaluation
# ---------------------------------------------------------
y_pred = gb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(class_report)

# ---------------------------------------------------------
# STEP 5: Visualization
# ---------------------------------------------------------
plt.figure(figsize=(14, 6))

# --- Plot 1: Confusion Matrix ---
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# --- Plot 2: Decision Boundary ---
plt.subplot(1, 2, 2)

# Plot test data points
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette="coolwarm", edgecolor="k")

# Create a grid to visualize the decision boundary background
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Predict for every point on the grid
Z = gb_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Draw the boundary contours (You will see the stepped/jagged structure here)
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.title(f"Gradient Boosting Decision Boundary (Acc: {accuracy*100:.1f}%)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()