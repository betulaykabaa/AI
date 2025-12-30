import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Generate Synthetic Data
# Let's assume we tested both models 100 times.
# Model A: Mean score 80, Std Dev 5
# Model B: Mean score 82, Std Dev 5
np.random.seed(42) # For reproducibility
scores_model_A = np.random.normal(80, 5, 100)
scores_model_B = np.random.normal(82, 5, 100)

# 2. Perform T-Test (The Judge)
# This calculates if the difference is statistically significant
t_stat, p_value = stats.ttest_ind(scores_model_A, scores_model_B)

# Decision Rule (Alpha = 0.05)
alpha = 0.05
if p_value < alpha:
    decision = "REJECT Null Hypothesis (Significant Difference)"
    color_code = 'green'
else:
    decision = "FAIL TO REJECT Null Hypothesis (No Significant Difference)"
    color_code = 'red'

# --- Visualization ---
plt.figure(figsize=(10, 6))

# Plot Histograms
plt.hist(scores_model_A, alpha=0.6, label='Model A (Old)', color='blue', density=True)
plt.hist(scores_model_B, alpha=0.6, label='Model B (New)', color='orange', density=True)

# Add Gaussian Curves for better visibility
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p_a = stats.norm.pdf(x, np.mean(scores_model_A), np.std(scores_model_A))
p_b = stats.norm.pdf(x, np.mean(scores_model_B), np.std(scores_model_B))
plt.plot(x, p_a, 'b', linewidth=2)
plt.plot(x, p_b, 'orange', linewidth=2)

# Annotation for P-Value
plt.title(f'Hypothesis Testing: Model A vs Model B\nP-value: {p_value:.4f}')
plt.xlabel('Accuracy Scores')
plt.ylabel('Density')
plt.legend()

# Show the decision on the plot
plt.figtext(0.5, 0.01, f"Decision: {decision}", ha="center", fontsize=12, bbox={"facecolor":color_code, "alpha":0.3, "pad":5})

plt.show()