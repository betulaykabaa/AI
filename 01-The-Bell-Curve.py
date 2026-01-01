import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Define the X-axis range (from -5 to 15)
x_axis = np.arange(-5, 15, 0.01)

# 2. Standard Normal Distribution (Mean=0, Std=1)
# This is the baseline curve (Blue)
plt.plot(x_axis, norm.pdf(x_axis, 0, 1), label='Mean=0, Std=1 (Standard)', color='blue', linewidth=2)

# 3. Shifting the Mean (Mean=5, Std=1)
# The curve shifts to the right (Red)
plt.plot(x_axis, norm.pdf(x_axis, 5, 1), label='Mean=5, Std=1 (Shifted Right)', color='red', linestyle='--')

# 4. Changing the Standard Deviation (Mean=0, Std=3)
# The curve becomes wider/flatter (Green)
plt.plot(x_axis, norm.pdf(x_axis, 0, 3), label='Mean=0, Std=3 (Wider)', color='green', linestyle=':')

# Graph Formatting
plt.title("Gaussian Distribution: Effect of Mean & Standard Deviation")
plt.xlabel("Data Values")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
