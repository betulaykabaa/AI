import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

# ==============================================================================
# 1. DATA GENERATION (The "Island" Scenario)
# ==============================================================================
np.random.seed(42)
n_samples = 300

X = np.random.randn(n_samples, 2)
# Create circular boundary (Island logic)
y = np.array([1 if x[0]**2 + x[1]**2 < 1.5 else 0 for x in X])

# ==============================================================================
# 2. PREPROCESSING (Scaling)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# ==============================================================================
# 3. TRAIN MODEL (SVM - RBF Kernel)
# ==============================================================================
# C=1.0, gamma=1.0 is a balanced setting.
# Try changing gamma to 100 to see OVERFITTING (Train 100%, Test Drops).
model = SVC(kernel='rbf', random_state=42, gamma=1.0, C=1.0)
model.fit(X_train_scaled, y_train)

# ==============================================================================
# 4. OVERFITTING CHECK (THE CRITICAL PART) 🔍
# ==============================================================================
# Calculate accuracy for both Training (seen) and Test (unseen) data
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print("\n" + "="*50)
print(f"REPORT CARD (KARNE):")
print(f"1. Training Score (Ezber): %{train_acc * 100:.2f}")
print(f"2. Test Score     (Gerçek): %{test_acc * 100:.2f}")
print("-" * 50)

print("DIAGNOSIS (TEŞHİS):")
gap = train_acc - test_acc

if gap > 0.10:
    print("  OVERFITTING DETECTED! (Aşırı Öğrenme)")
    print("    The model memorized the training data but fails on new data.")
    print("    Suggestion: Decrease 'gamma' or 'C'.")
elif train_acc < 0.60:
    print("  UNDERFITTING DETECTED! (Yetersiz Öğrenme)")
    print("    The model is too simple to understand the pattern.")
    print("    Suggestion: Increase 'gamma' or 'C', or use more features.")
else:
    print("  PERFECT FIT! (Mükemmel Uyum)")
    print("    Train and Test scores are high and close to each other.")
    print("    The model has truly learned the logic.")
print("="*50 + "\n")

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(10, 6))

X_set, y_set = X_test_scaled, y_test

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i),
                label=f'Class {j}',
                edgecolor='black', s=50)

plt.title(f'SVM RBF Analysis (Test Acc: %{test_acc*100:.1f})')
plt.legend()

plt.show()
