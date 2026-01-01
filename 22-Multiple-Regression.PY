import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==============================================================================
# 1. CREATE DATASET (Synthetic House Data)
# ==============================================================================
np.random.seed(42)
n_samples = 200

# Feature 1: Area (80m² to 250m²)
area = np.random.randint(80, 250, n_samples)

# Feature 2: Room Count (2 to 6 rooms)
rooms = np.random.randint(2, 7, n_samples)

# Feature 3: Building Age (0 to 30 years)
age = np.random.randint(0, 31, n_samples)

# TARGET: Price ($)
# The "God Formula" (Real Logic): 
# Price = 2000 * Area + 10000 * Rooms - 3000 * Age + Base(50000)
# We add noise/randomness because real market isn't perfect
price = (2000 * area) + (10000 * rooms) - (3000 * age) + 50000 + np.random.normal(0, 10000, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'Area': area,
    'Rooms': rooms,
    'Age': age,
    'Price': price
})

print("--- First 5 Rows of Data ---")
print(df.head())
print("\n" + "="*60 + "\n")

# ==============================================================================
# 2. DATA SPLITTING
# ==============================================================================
# X (Features): Area, Rooms, Age
# y (Target): Price
X = df[['Area', 'Rooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# 3. MODEL TRAINING (The Code is Identical to Simple Regression!)
# ==============================================================================
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================================================================
# 4. INSPECTING THE "BRAIN" OF THE MODEL
# ==============================================================================
# The model learned 3 coefficients (weights), one for each feature.
coeffs = model.coef_
intercept = model.intercept_

print("--- MODEL COEFFICIENTS (What did AI learn?) ---")
print(f"Base Price (Intercept): ${intercept:.2f}")
print(f"1. Weight for Area    : ${coeffs[0]:.2f} (Real was 2000)")
print(f"2. Weight for Rooms   : ${coeffs[1]:.2f} (Real was 10000)")
print(f"3. Weight for Age     : ${coeffs[2]:.2f} (Real was -3000)")
print("\nNotice how Age is negative! The model understood that older = cheaper.")
print("="*60 + "\n")

# ==============================================================================
# 5. EVALUATION & VISUALIZATION (Actual vs Predicted)
# ==============================================================================
# Since we have 3 dimensions (Area, Rooms, Age), we cannot draw a simple line.
# Best way to visualize Multiple Regression is "Actual vs Predicted" plot.

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print(f"Model Accuracy (R² Score): {score:.4f}")

plt.figure(figsize=(10, 6))

# Scatter plot: X-axis = Real Prices, Y-axis = AI Predictions
plt.scatter(y_test, y_pred, color='purple', alpha=0.6, s=70, edgecolor='black')

# Draw a diagonal "Perfect Match" line
# If a dot is on this line, it means Prediction == Actual
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=3, label='Perfect Prediction Line')

plt.title(f'Multiple Regression: Actual vs Predicted Price (R²={score:.2f})')
plt.xlabel('Actual Prices ($)')
plt.ylabel('Predicted Prices ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==============================================================================
# 6. MAKE A CUSTOM PREDICTION
# ==============================================================================
# House: 120 m², 3 Rooms, 5 Years Old
new_house = np.array([[120, 3, 5]]) 

# Since we trained with a DataFrame, sklearn might warn about feature names if we pass raw numpy array
# but it will still work.
predicted_price = model.predict(new_house)

print(f"\n--- Custom Prediction ---")
print(f"House Features: 120m², 3 Rooms, 5 Years Old")
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
