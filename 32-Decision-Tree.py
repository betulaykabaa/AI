import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# ==============================================================================
# 1. DATA GENERATION (Scenario: Apple Quality Control)
# ==============================================================================
# Features: [Size, Sweetness]
# Logic: We define specific rules to simulate real-world decision making.

np.random.seed(42)
n_samples = 400

# Generate random data (Uniform distribution)
# Size: 0 to 10, Sweetness: 0 to 10
X = np.random.uniform(0, 10, size=(n_samples, 2))
y = np.zeros(n_samples)

# Apply rules to create classes (0: Bad Apple, 1: Good Apple)
for i in range(n_samples):
    size = X[i, 0]
    sweetness = X[i, 1]
    
    # Rule 1: Big and Sweet -> Good
    if size > 6 and sweetness > 5:
        y[i] = 1 
    # Rule 2: Small but very Sweet (Exception case) -> Good
    elif size < 4 and sweetness > 8:
        y[i] = 1 
    # Everything else -> Bad
    else:
        y[i] = 0

# Add noise (simulating errors in real life)
noise_indices = np.random.choice(n_samples, 20, replace=False)
for i in noise_indices:
    y[i] = 1 - y[i]

# ==============================================================================
# 2. MODEL TRAINING
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the Decision Tree
# max_depth=4 prevents the tree from becoming too complex (overfitting) Pre-pruning
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

# ==============================================================================
# 3. PERFORMANCE EVALUATION
# ==============================================================================
# Make predictions
y_pred = tree_model.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
train_acc = tree_model.score(X_train, y_train)

# Print detailed report
print("\n" + "="*50)
print("DECISION TREE PERFORMANCE REPORT")
print("="*50)

# Check for Overfitting
print(f"Training Score: {train_acc * 100:.2f}%")
print(f"Test Score:     {acc * 100:.2f}%")
print("-" * 50)

# Confusion Matrix
print("Confusion Matrix:")
print(cm)
print("-" * 50)
print("(Row 1: True Negatives | False Positives)")
print("(Row 2: False Negatives | True Positives)")
print("-" * 50)

# Detailed breakdown
tn, fp, fn, tp = cm.ravel()
print(f"Correct Predictions:   {tn + tp}")
print(f"Incorrect Predictions: {fp + fn}")
print("="*50 + "\n")

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(14, 6))

# Plot 1: Decision Boundary (The Map)
plt.subplot(1, 2, 1)
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 0.5, stop=X_set[:, 0].max() + 0.5, step=0.05),
    np.arange(start=X_set[:, 1].min() - 0.5, stop=X_set[:, 1].max() + 0.5, step=0.05)
)
Z = tree_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i),
                label=f'Class {int(j)}',
                edgecolor='black', s=30)
plt.title('Decision Boundary (Test Set)')
plt.xlabel('Size')
plt.ylabel('Sweetness')
plt.legend()

# Plot 2: The Tree Structure (The Logic)
plt.subplot(1, 2, 2)
plot_tree(tree_model, filled=True, feature_names=['Size', 'Sweetness'], class_names=['Bad', 'Good'], rounded=True)
plt.title('Decision Tree Logic Flow')

plt.tight_layout()
plt.show()