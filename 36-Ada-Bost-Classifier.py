import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report # Added classification_report

# 1. DATA GENERATION
# ---------------------------------------------------------
X, y = make_circles(n_samples=400, noise=0.1, factor=0.5, random_state=42)

# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. MODEL SETUP (AdaBoost)
# ---------------------------------------------------------
# We use simple Decision Stumps (max_depth=1)
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_model.fit(X_train, y_train)

# 3. EVALUATION
# ---------------------------------------------------------
y_pred = ada_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Final Model Accuracy: {acc * 100:.2f}%")
print("\n--- Detailed Classification Report ---\n")
print(classification_report(y_test, y_pred))

# 4. VISUALIZATION FUNCTION
# ---------------------------------------------------------
def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Handle prediction for both single estimators and the full ensemble
    if hasattr(model, "predict"):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k', ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xticks(())
    ax.set_yticks(())

# 5. PLOTTING
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: The First Weak Learner (Just one line)
plot_decision_boundary(ada_model.estimators_[0], X_test, y_test, axes[0], 
                       "Step 1: First Weak Learner\n(Single Split)")

# Plot 2: The Second Weak Learner (Focusing on errors)
plot_decision_boundary(ada_model.estimators_[1], X_test, y_test, axes[1], 
                       "Step 2: Second Weak Learner\n(Correcting Errors)")

# Plot 3: Final Combined Model
plot_decision_boundary(ada_model, X_test, y_test, axes[2], 
                       f"Final AdaBoost Model\n(Accuracy: {acc*100:.1f}%)")

plt.tight_layout()
plt.show()