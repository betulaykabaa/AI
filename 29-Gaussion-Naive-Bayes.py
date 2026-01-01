import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.colors import ListedColormap

# ==============================================================================
# 1. DATA GENERATION (Medical Scenario: Heart Disease)
# ==============================================================================
# Scenario:
# - Class 0 (Green): Healthy Patients (Younger, Normal Cholesterol)
# - Class 1 (Red):   High Risk Patients (Older, High Cholesterol)
# GaussianNB loves this data because biological features often follow a Bell Curve.

np.random.seed(42)
n_samples = 400

# Group 0: Healthy
# Age: Mean 30, Sigma 10
# Cholesterol: Mean 180, Sigma 30
X_healthy_age = np.random.normal(30, 10, (200, 1))
X_healthy_chol = np.random.normal(180, 30, (200, 1))
y_healthy = np.zeros((200, 1))

# Group 1: High Risk (Disease)
# Age: Mean 55, Sigma 10
# Cholesterol: Mean 260, Sigma 40
X_sick_age = np.random.normal(55, 10, (200, 1))
X_sick_chol = np.random.normal(260, 40, (200, 1))
y_sick = np.ones((200, 1))

# Combine and Shuffle
X = np.vstack((np.hstack((X_healthy_age, X_healthy_chol)), 
               np.hstack((X_sick_age, X_sick_chol))))
y = np.vstack((y_healthy, y_sick)).ravel()

# ==============================================================================
# 2. PREPROCESSING
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling is crucial for visualization and model stability
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# ==============================================================================
# 3. TRAIN MODEL (Gaussian Naive Bayes)
# ==============================================================================
# We use GaussianNB because we assume features are normally distributed.
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# ==============================================================================
# 4. EVALUATION & METRICS REPORT
# ==============================================================================
y_pred = model.predict(X_test_scaled)

# Calculate Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*60)
print("🏥 MEDICAL AI DIAGNOSTIC REPORT (Heart Disease Model)")
print("="*60)

# 1. ACCURACY
print(f"1. Accuracy: {acc * 100:.2f}%")
print(f"   -> Overall correctness of the diagnosis.")
print("-" * 30)

# 2. PRECISION
print(f"2. Precision: {prec * 100:.2f}%")
print(f"   -> Reliability: When model says 'SICK', is it true?")
print(f"   -> Low Precision means: False Alarms (Healthy people scared unnecessarily).")
print("-" * 30)

# 3. RECALL (CRITICAL FOR MEDICINE)
print(f"3. Recall (Sensitivity): {rec * 100:.2f}%")
print(f"   -> Detection Power: Did we catch all the sick patients?")
print(f"   -> Low Recall means: DANGER! We sent sick people home.")
print("-" * 30)

# 4. F1-SCORE
print(f"4. F1-Score: {f1 * 100:.2f}%")
print(f"   -> Balance between Precision and Recall.")
print("="*60)

# --- OVERFITTING CHECK ---
train_acc = model.score(X_train_scaled, y_train)
print("\n🔍 OVERFITTING CHECK:")
print(f"   - Training Score: {train_acc * 100:.2f}%")
print(f"   - Test Score:     {acc * 100:.2f}%")

if train_acc - acc > 0.10:
    print("  WARNING: Overfitting detected.")
else:
    print("  STATUS: Healthy Model (Generalizes well).")
print("="*60 + "\n")

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(10, 6))
X_set, y_set = X_test_scaled, y_test

# Create Grid
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

# Plot Decision Boundary (Probabilistic contours of Naive Bayes)
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('green', 'red')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot Data Points
# 0 = Healthy (Green), 1 = Sick (Red)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('green', 'red'))(i),
                label=f'Patient: {"Healthy" if j==0 else "Sick"}',
                edgecolor='black', s=50)

plt.title('Heart Disease Detection (Naive Bayes)\nGreen: Healthy Zone | Red: High Risk Zone')
plt.xlabel('Age (Standardized)')
plt.ylabel('Cholesterol Level (Standardized)')
plt.legend()

plt.show()
