import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# ==============================================================================
# 1. DATA GENERATION (3 Distinct Clusters)
# ==============================================================================
np.random.seed(42)
n_samples = 300

# Group 1: Young & Low Salary (Class 0 - Red)
X1 = np.random.normal(25, 5, (100, 1)) # Age ~25
Y1 = np.random.normal(30000, 5000, (100, 1)) # Salary ~30k
C1 = np.zeros((100, 1))

# Group 2: Middle Age & High Salary (Class 1 - Green)
X2 = np.random.normal(45, 5, (100, 1)) # Age ~45
Y2 = np.random.normal(80000, 10000, (100, 1)) # Salary ~80k
C2 = np.ones((100, 1))

# Group 3: Old Age & Low Salary (Class 0 - Red)
X3 = np.random.normal(60, 5, (100, 1)) # Age ~60
Y3 = np.random.normal(35000, 5000, (100, 1)) # Salary ~35k
C3 = np.zeros((100, 1))

# Combine Data
X = np.vstack((np.hstack((X1, Y1)), np.hstack((X2, Y2)), np.hstack((X3, Y3))))
y = np.vstack((C1, C2, C3)).ravel()

# ==============================================================================
# 2. ADDING REALISM (NOISE INJECTION) 
# ==============================================================================
# In real life, data is never perfect. Let's spoil %10 of the data.
noise_ratio = 0.10 
num_noise = int(len(y) * noise_ratio)

# Pick random indices
noise_indices = np.random.choice(len(y), num_noise, replace=False)

# Flip labels (0 -> 1, 1 -> 0)
# This creates outliers like "Rich but didn't buy" or "Poor but bought"
for i in noise_indices:
    y[i] = 1 - y[i]

print(f" Added noise to {num_noise} samples to make it realistic.")

# ==============================================================================
# 3. PREPROCESSING (Scaling is MANDATORY for KNN)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# ==============================================================================
# 4. TRAIN MODEL (KNN)
# ==============================================================================
# n_neighbors=5: The standard starting point.
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
model.fit(X_train_scaled, y_train)

# ==============================================================================
# 5. REPORT CARD & DIAGNOSIS
# ==============================================================================
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print("\n" + "="*50)
print(f"REPORT CARD (KARNE):")
print(f"1. Training Score: %{train_acc * 100:.2f}")
print(f"2. Test Score:     %{test_acc * 100:.2f}")
print("-" * 50)

print("DIAGNOSIS:")
if train_acc - test_acc > 0.10:
    print("  OVERFITTING: Try increasing K (e.g., K=10 or K=15).")
elif test_acc > train_acc:
    print("  GOOD FIT: Test score is higher (lucky split).")
else:
    print("  REALISTIC FIT: Scores are balanced and not 100%.")
print("="*50 + "\n")

# ==============================================================================
# 6. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(10, 6))

X_set, y_set = X_test_scaled, y_test

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

# Decision Boundary
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot Data Points
# Notice: You will now see some Red dots in the Green area (Noise/Errors)!
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i),
                label=f'Class {int(j)}',
                edgecolor='black', s=50)

plt.title(f'KNN (K=5) with Noise (Realistic Data)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Salary (Scaled)')
plt.legend()

plt.show()
