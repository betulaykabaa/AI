import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from matplotlib.colors import ListedColormap

# ==============================================================================
# 1. THE ARENA (HARD MODE - OVERLAPPING DATA) 
# ==============================================================================
np.random.seed(42)
n_samples = 300

# Grupları birbirine çok yaklaştırıyoruz ve dağılımı (5 -> 10, 5000 -> 15000) artırıyoruz.
# Böylece düz çizgiyle ayırmak imkansız hale geliyor.

# Grup 1: Genç & Düşük-Orta Maaş (Kırmızı)
X1 = np.random.normal(30, 10, (100, 1))
Y1 = np.random.normal(40000, 15000, (100, 1))
C1 = np.zeros((100, 1))

# Grup 2: Orta Yaş & Orta-Yüksek Maaş (Yeşil) - Kırmızının içine giriyor!
X2 = np.random.normal(40, 10, (100, 1))
Y2 = np.random.normal(60000, 15000, (100, 1)) # Maaşlar karıştı
C2 = np.ones((100, 1))

# Grup 3: Yaşlı & Düşük Maaş (Kırmızı)
X3 = np.random.normal(55, 10, (100, 1))
Y3 = np.random.normal(45000, 15000, (100, 1))
C3 = np.zeros((100, 1))

X = np.vstack((np.hstack((X1, Y1)), np.hstack((X2, Y2)), np.hstack((X3, Y3))))
y = np.vstack((C1, C2, C3)).ravel()

# Gürültü eklemeye bile gerek yok, verinin kendisi zaten gürültülü (iç içe) oldu.

# ==============================================================================
# 2. PREPARING FIGHTERS
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# ==============================================================================
# 3. CHALLENGERS
# ==============================================================================
classifiers = [
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("SVM (RBF Kernel)", SVC(kernel='rbf', gamma=1.0, C=1.0, random_state=42)),
    ("KNN (K=5)", KNeighborsClassifier(n_neighbors=5))
]

# ==============================================================================
# 4. BATTLE & VISUALIZATION
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

X_set, y_set = X_test_scaled, y_test
X1, X2 = np.meshgrid(
    np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.05),
    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.05)
)

print(f"{'MODEL NAME':<25} | {'ACCURACY':<10} | {'F1-SCORE':<10}")
print("-" * 50)

for i, (name, model) in enumerate(classifiers):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name:<25} | %{acc*100:<9.1f} | %{f1*100:<9.1f}")
    
    ax = axes[i]
    Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    ax.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
    
    for j, val in enumerate(np.unique(y_set)):
        ax.scatter(X_set[y_set == val, 0], X_set[y_set == val, 1],
                   color=ListedColormap(('red', 'green'))(j),
                   label=f'Class {int(val)}', edgecolor='black', s=40)
        
    ax.set_title(f"{name}\nAcc: %{acc*100:.1f}", fontsize=14)

plt.tight_layout()

plt.show()
