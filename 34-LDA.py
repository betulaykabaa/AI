import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Required Scikit-learn libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# STEP 1: Generate Synthetic Dataset
# ---------------------------------------------------------
# Same parameters as the Random Forest example
# n_features=10: 10 columns
# n_informative=5: Only 5 features are useful for separation
X, y = make_classification(n_samples=1000, 
                           n_features=10, 
                           n_informative=5, 
                           n_redundant=2, 
                           random_state=42)

feature_names = [f"Feature_{i+1}" for i in range(10)]

# ---------------------------------------------------------
# STEP 2: Split into Train and Test Sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------
# STEP 3: Initialize and Train LDA Model
# ---------------------------------------------------------
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 4: Prediction and Performance Evaluation
# ---------------------------------------------------------
y_pred = lda_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"LDA Model Accuracy: %{accuracy * 100:.2f}")
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# STEP 5: Visualization
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))

# Plot 1: Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False)
plt.title("Confusion Matrix (LDA)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Plot 2: LDA Coefficients
# This plot shows the weight/influence of each feature on the separation boundary.
# We sort by absolute value to see the magnitude of influence regardless of direction.
plt.subplot(1, 2, 2)

# Get coefficients (flattening the array for binary classification)
coeffs = lda_model.coef_.flatten()
# Sort indices by absolute value (Largest impact to smallest)
indices = np.argsort(np.abs(coeffs))[::-1]

sns.barplot(x=coeffs[indices], y=[feature_names[i] for i in indices], palette="magma")
plt.title("LDA Feature Coefficients")
plt.xlabel("Coefficient Value (Direction & Magnitude)")
plt.ylabel("Features")
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8) # Center line at 0

plt.tight_layout()
plt.show()