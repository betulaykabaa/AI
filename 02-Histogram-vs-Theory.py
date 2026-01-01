import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Generate Random Data (e.g., Heights of people)
# Mean = 170, Std Dev = 10, Sample Size = 10,000
mu_real = 170
sigma_real = 10
data_points = np.random.normal(mu_real, sigma_real, 10000)

# 2. Plot Histogram (The actual data distribution)
# 'density=True' normalizes the bars to match the probability curve
plt.hist(data_points, bins=50, density=True, alpha=0.6, color='skyblue', label='Real Data (Histogram)')

# 3. Plot Theoretical Gaussian Curve (The Math)
# Create a range for the x-axis based on data limits
xmin, xmax = plt.xlim()
x_axis_theory = np.linspace(xmin, xmax, 100)
# Calculate probability density function (PDF)
p = norm.pdf(x_axis_theory, mu_real, sigma_real)

plt.plot(x_axis_theory, p, 'k', linewidth=3, label='Theoretical Curve')

# Graph Formatting
plt.title("Real Data Histogram vs. Theoretical Gaussian Curve")
plt.xlabel("Values (e.g., Height)")
plt.ylabel("Probability")
plt.legend()
plt.show()
