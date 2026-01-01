import matplotlib.pyplot as plt

# 1. Define Probabilities (The Knowns)
p_disease = 0.01        # Prior: 1% of population has the disease
p_healthy = 1 - p_disease

p_pos_given_disease = 0.99  # Sensitivity: Test is positive if you are sick
p_pos_given_healthy = 0.05  # False Positive: Test is positive even if you are healthy

# 2. Calculate the "Evidence" (Total Probability of Positive Test)
# P(Positive) = (Sick & Pos) + (Healthy & Pos)
p_positive = (p_disease * p_pos_given_disease) + (p_healthy * p_pos_given_healthy)

# 3. Apply Bayes Theorem (Calculate Posterior)
# P(Disease | Positive)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive

print(f"Gerçek Hasta Olma İhtimali: {p_disease_given_pos:.4f}")

# --- Visualization ---
labels = ['Prior Belief\n(Before Test)', 'Posterior Belief\n(After Positive Test)']
values = [p_disease, p_disease_given_pos]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=['gray', 'red'])

plt.title("Bayesian Update: Impact of a Positive Test")
plt.ylabel("Probability of Sickness")
plt.ylim(0, 1) # Probability is between 0 and 1

# Add percentage labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'%{height*100:.1f}', ha='center', fontweight='bold', fontsize=12)

plt.show()
