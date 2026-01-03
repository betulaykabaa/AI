import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

# ==============================================================================
# 1. DATA GENERATION
# ==============================================================================
# We use the same dataset (Moons) to compare fairly with XGBoost
np.random.seed(42)
X, y = make_moons(n_samples=500, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================================================================
# 2. LIGHTGBM MODEL SETUP
# ==============================================================================
# LightGBM is extremely fast and uses a "Leaf-wise" growth strategy.
# verbosity=-1: Keeps the console clean (suppresses warnings)
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    verbosity=-1
)

print("Training LightGBM Model...")
model.fit(X_train, y_train)

# ==============================================================================
# 3. EVALUATION
# ==============================================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("="*40)
print(f"LightGBM Accuracy: {acc * 100:.2f}%")
print("="*40)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================================================================
# 4. VISUALIZATION PART 1: Decision Boundary & Feature Importance
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

plt.title(f'LightGBM Decision Boundary (Acc: {acc*100:.1f}%)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# --- PLOT 2: Feature Importance ---
plt.subplot(1, 2, 2)
importances = model.feature_importances_
features = ['Feature 1', 'Feature 2']

sns.barplot(x=importances, y=features, hue=features, palette='magma', legend=False)
plt.title('Feature Importance (LightGBM Style)')
plt.xlabel('Importance Score (Split Count)')

plt.tight_layout()
plt.show()

# ==============================================================================
# 5. VISUALIZATION PART 2: Performance Metrics Bar Chart
# ==============================================================================
# Get report as dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Prepare data for plotting
data_list = []
for cls in ['0', '1']:
    data_list.append({'Class': f'Class {cls}', 'Metric': 'Precision', 'Score': report_dict[cls]['precision']})
    data_list.append({'Class': f'Class {cls}', 'Metric': 'Recall', 'Score': report_dict[cls]['recall']})
    data_list.append({'Class': f'Class {cls}', 'Metric': 'F1-Score', 'Score': report_dict[cls]['f1-score']})

df_chart = pd.DataFrame(data_list)

# Draw Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(data=df_chart, x='Metric', y='Score', hue='Class', palette='magma')

plt.ylim(0.8, 1.05)
plt.title('LightGBM Performance Metrics (Bar Chart)')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.3)

for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.2f', padding=3)

plt.show()