import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# 1. Create a Dataset with Missing Values (NaN)
data = {
    'Student_ID': range(1, 11),
    'Math_Score': [85, 90, np.nan, 70, 88, np.nan, 95, 60, 78, 82], # 2 Missing
    'Physics_Score': [80, np.nan, 85, 75, 92, 88, np.nan, np.nan, 70, 65] # 3 Missing
}

df = pd.DataFrame(data)

# 2. Visualize the Missing Data (BEFORE)
# Yellow lines represent missing values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('BEFORE: Missing Values (Yellow Lines)')

# 3. Apply Imputation (Filling the holes)
# Strategy='mean': Replaces NaN with the average of the column
imputer = SimpleImputer(strategy='mean')
df_filled_array = imputer.fit_transform(df)

# Convert back to DataFrame (Imputer returns a numpy array)
df_filled = pd.DataFrame(df_filled_array, columns=df.columns)

# 4. Visualize the Data (AFTER)
plt.subplot(1, 2, 2)
sns.heatmap(df_filled.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('AFTER: Clean Data (No Missing Values)')

plt.tight_layout()
plt.show()

# Print Comparison
print("--- Orijinal Veri (İlk 6 Satır) ---")
print(df.head(6))
print("\n--- Doldurulmuş Veri (İlk 6 Satır) ---")
print(df_filled.head(6))