import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# 1. Simulation Parameters
total_neurons = 1000
p_active = 0.5  # 50% chance for a neuron to be active

# 2. Generate Bernoulli Samples (Simulate Dropout)
# rvs = Random Variates (generating random 0s and 1s)
neuron_states = bernoulli.rvs(p_active, size=total_neurons)

# Count active (1) and inactive (0) neurons
count_zeros = np.sum(neuron_states == 0)
count_ones = np.sum(neuron_states == 1)

# --- Visualization ---
labels = ['Inactive Neurons (0)', 'Active Neurons (1)']
counts = [count_zeros, count_ones]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, counts, color=['gray', 'blue'])

plt.title(f'Neural Network Dropout Simulation\nTotal Neurons: {total_neurons}, Probability(p): {p_active}')
plt.ylabel('Number of Neurons')

# Show exact numbers
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 10, f'{height}', ha='center', fontsize=12)

plt.show()