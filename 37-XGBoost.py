import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import XGBoost Library
from xgboost import XGBClassifier

# ==============================================================================
# 1. DATA GENERATION (Moons Dataset)
# ==============================================================================
np.random.seed(42)
X, y = make_moons(n_samples=500, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# 2. XGBOOST MODEL SETUP
# ==============================================================================
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    eval_metric='logloss',
    random_state=42
)

print("Training XGBoost Model...")
model.fit(X_train, y_train)

# ==============================================================================
# 3. EVALUATION
# ==============================================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("="*40)
print(f"XGBoost Accuracy: {acc * 100:.2f}%")
print("="*40)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================================================================
# 4. VISUALIZATION PART 1: Boundary & Feature Importance
# ==============================================================================
plt.figure(figsize=(16, 6))

# --- PLOT 1: Decision Boundary ---
plt.subplot(1, 2, 1)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette='coolwarm', edgecolor='k')

plt.title(f'XGBoost Decision Boundary (Acc: {acc*100:.1f}%)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# --- PLOT 2: Feature Importance ---
plt.subplot(1, 2, 2)
importances = model.feature_importances_
features = ['Feature 1 (X-Axis)', 'Feature 2 (Y-Axis)']

sns.barplot(x=importances, y=features, hue=features, palette='viridis', legend=False)
plt.title('Feature Importance (What influenced the decision?)')
plt.xlabel('Importance Score')

plt.tight_layout()
plt.show()

# ==============================================================================
# 5. VISUALIZATION PART 2: Classification Report Heatmap
# ==============================================================================
# We generate the report as a Dictionary (output_dict=True) to make it dynamic
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame and Transpose
df_report = pd.DataFrame(report_dict).T

# We filter to keep only Class 0 and Class 1 for a cleaner look 
# (Removing 'accuracy', 'macro avg', etc.)
df_report_clean = df_report.iloc[:2, :3] # Selecting first 2 rows and first 3 columns

plt.figure(figsize=(8, 5))

# Heatmap
# annot=True: Write numbers inside cells
# cmap='Blues': Color map
sns.heatmap(df_report_clean, annot=True, cmap='Blues', fmt='.2f', linewidths=1, linecolor='black')

plt.title('XGBoost Classification Report (Visual Table)')
plt.show()