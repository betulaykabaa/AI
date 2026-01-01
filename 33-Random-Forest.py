import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Required Scikit-learn libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# STEP 1: Generate Synthetic Dataset
# ---------------------------------------------------------
# n_samples=1000: 1000 data points
# n_features=10: 10 columns (features)
# n_informative=5: Only 5 features are actually useful for prediction
# n_classes=2: Binary classification (0 or 1)
X, y = make_classification(n_samples=1000, 
                           n_features=10, 
                           n_informative=5, 
                           n_redundant=2, 
                           random_state=42)

# Create a DataFrame for better readability
feature_names = [f"Feature_{i+1}" for i in range(10)]
df = pd.DataFrame(X, columns=feature_names)
df['Target_Class'] = y

print(f"Dataset Shape: {df.shape}")
print("-" * 30)

# ---------------------------------------------------------
# STEP 2: Split into Train and Test Sets
# ---------------------------------------------------------
# 70% Training, 30% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------
# STEP 3: Initialize and Train Random Forest Model
# ---------------------------------------------------------
# n_estimators=100: The forest will have 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 4: Prediction and Performance Evaluation
# ---------------------------------------------------------
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: %{accuracy * 100:.2f}")
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# STEP 5: Visualization
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))

# Plot 1: Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Plot 2: Feature Importance
# Shows which features affected the result the most
plt.subplot(1, 2, 2)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1] # Sort descending

sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.show()