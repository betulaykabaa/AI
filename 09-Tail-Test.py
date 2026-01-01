import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Setup the standard normal distribution
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)

plt.figure(figsize=(10, 5))
plt.plot(x, y, color='black', linewidth=2)

# Define Critical Regions (Alpha = 0.05, two-tailed)
# We reject H0 if we are in the bottom 2.5% or top 2.5%
critical_value = 1.96 

# Fill the "Rejection Regions" (The tails)
plt.fill_between(x, y, where=(x > critical_value), color='red', alpha=0.5, label='Rejection Region (Significant)')
plt.fill_between(x, y, where=(x < -critical_value), color='red', alpha=0.5)

# Fill the "Acceptance Region" (The middle)
plt.fill_between(x, y, where=((x > -critical_value) & (x < critical_value)), color='lightgreen', alpha=0.3, label='Acceptance Region (Not Significant)')

# Add lines
plt.axvline(critical_value, color='red', linestyle='--')
plt.axvline(-critical_value, color='red', linestyle='--')

plt.title("Visualizing P-Value & Significance Level (Alpha=0.05)")
plt.text(0, 0.15, 'Null Hypothesis ($H_0$)\nis True here', ha='center', fontsize=12)
plt.text(2.5, 0.05, 'Reject $H_0$', ha='center', color='darkred', fontweight='bold')

plt.legend()
plt.show()
