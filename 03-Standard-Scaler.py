import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Create "Raw" (Unscaled) Data
# Mean is 50, Std is 20 (Bad scale for AI)
raw_data = np.random.normal(loc=50, scale=20, size=1000).reshape(-1, 1)

# 2. Apply Standardization
# This transforms data to have Mean=0 and Std=1
scaler = StandardScaler()
scaled_data = scaler.fit_transform(raw_data)

# --- Visualization ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Raw Data
axs[0].hist(raw_data, bins=30, color='orange', alpha=0.7, density=True)
axs[0].set_title(f"Raw Data\nMean ≈ {raw_data.mean():.1f}, Std ≈ {raw_data.std():.1f}")
axs[0].set_xlabel("Value Range (Wide & Shifted)")
axs[0].axvline(raw_data.mean(), color='k', linestyle='dashed', linewidth=1)

# Plot 2: Scaled Data (AI Ready)
axs[1].hist(scaled_data, bins=30, color='purple', alpha=0.7, density=True)
axs[1].set_title(f"Standardized Data (AI Ready)\nMean ≈ {scaled_data.mean():.1f}, Std ≈ {scaled_data.std():.1f}")
axs[1].set_xlabel("Value Range (Centered at 0)")
axs[1].axvline(scaled_data.mean(), color='k', linestyle='dashed', linewidth=1)

plt.tight_layout()
plt.show()
