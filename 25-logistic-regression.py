import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================================================================
# 1. DATA GENERATION (Synthetic Customer Data)
# ==============================================================================
# Scenario: We have 'Age' and 'Estimated Salary'.
# We want to predict if they purchased the product (0 = No, 1 = Yes).

np.random.seed(0)
n_samples = 200

# Feature 1: Age (Random between 18 and 60)
age = np.random.randint(18, 60, n_samples)

# Feature 2: Estimated Salary (Random between 20k and 150k)
salary = np.random.randint(20000, 150000, n_samples)

# Target: Purchased (0 or 1)
# Logic: Older people with higher salary are more likely to buy (1).
# We create a threshold to separate classes with some noise.
action = []
for a, s in zip(age, salary):
    if (a * s) > 1800000: # Simple logic: Age * Salary > Threshold
        action.append(1) # Buy
    else:
        action.append(0) # No Buy

# Add some noise (make some random people behave unexpectedly)
# This prevents the data from being "perfectly" separable.
for _ in range(20):
    idx = np.random.randint(0, n_samples)
    action[idx] = 1 - action[idx] # Flip 0 to 1 or 1 to 0

X = np.column_stack((age, salary))
y = np.array(action)

# ==============================================================================
# 2. PREPROCESSING (Scaling is Crucial for Visualization)
# ==============================================================================
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling (StandardScaler)
# Logistic Regression visualization works best when features are on the same scale.
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# ==============================================================================
# 3. TRAIN MODEL
# ==============================================================================
model = LogisticRegression(random_state=0)
model.fit(X_train_scaled, y_train)

# ==============================================================================
# 4. EVALUATION (Accuracy instead of R2)
# ==============================================================================
y_pred = model.predict(X_test_scaled)

# Accuracy Score: Percentage of correct predictions
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("-" * 40)
print(f"Model Accuracy: %{acc * 100:.2f}")
print("-" * 40)
print("Confusion Matrix:")
print(cm)
print("(Row 1: True Negatives | False Positives)")
print("(Row 2: False Negatives | True Positives)")
print("-" * 40)

# ==============================================================================
# 5. VISUALIZATION (Decision Boundary)
# ==============================================================================
plt.figure(figsize=(10, 6))

from matplotlib.colors import ListedColormap
X_set, y_set = X_test_scaled, y_test

# Create a grid to paint the background
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

# Predict every pixel on the grid to color the regions
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot the actual data points
for i, j in enumerate(np.unique(y_set)):
# Uyarıyı düzelten satır:
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), # BURASI DEĞİŞTİ (color=)
                label=f'Class {j} ({"Buy" if j==1 else "No Buy"})',
                edgecolor='black', s=50)

plt.title('Logistic Regression (Test Set Classification)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.legend()
plt.grid(False) # Clean look
plt.show()