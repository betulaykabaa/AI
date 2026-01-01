import matplotlib.pyplot as plt
import numpy as np

# Generate Random Data
x = np.random.rand(10)
y = np.random.rand(10)

plt.figure(figsize=(10, 5))

# CASE 1: Standard (Default Parameters)
# Standard dots can be hard to see or distinguish.
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='red') 
plt.title("Standard Points\n(Small & No Border)")
plt.grid(True, alpha=0.3)

# CASE 2: Customized (s=200, edgecolor='black')
# 's': Increases size to make them pop out.
# 'edgecolor': Adds a black border for high contrast.
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='red', s=200, edgecolor='black', linewidth=2) 
plt.title("Customized Points\n(s=200, edgecolor='black')")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
