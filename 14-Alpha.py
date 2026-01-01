import matplotlib.pyplot as plt
import numpy as np

# Generate 1000 random overlapping points
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)

plt.figure(figsize=(10, 5))

# CASE 1: alpha=1.0 (Solid Color)
# Since the dots are solid, we cannot see where the data is clustered.
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', alpha=1.0, s=50)
plt.title("alpha=1.0 (Solid)\nDensity is Unclear")
plt.grid(True)

# CASE 2: alpha=0.3 (Transparent Color)
# Transparency allows us to see the 'Gaussian' cluster in the center.
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='blue', alpha=0.3, s=50) # Transparency applied here
plt.title("alpha=0.3 (Transparent)\nDensity is Visible")
plt.grid(True)

plt.tight_layout()
plt.show()
