import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# 1. Define Probability of Success (p)
# Let's say p = 0.3 (30% chance of success/1, 70% chance of failure/0)
p = 0.3

# 2. X-axis values (Only 0 and 1 exist in Bernoulli)
x = [0, 1]

# 3. Calculate Probabilities
# P(X=1) = p
# P(X=0) = 1 - p
probabilities = bernoulli.pmf(x, p)

# --- Visualization ---
plt.figure(figsize=(8, 5))
bars = plt.bar(x, probabilities, color=['red', 'green'], alpha=0.7)

# Add labels
plt.xticks([0, 1], ['0 (Failure / No)', '1 (Success / Yes)'])
plt.ylabel('Probability')
plt.title(f'Bernoulli Distribution (p={p})')
plt.ylim(0, 1) # Y-axis limit from 0 to 1

# Add text on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', fontweight='bold')

plt.show()