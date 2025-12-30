import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Generate Synthetic Data (1000 Students)
np.random.seed(42)
n_samples = 1000

# Useful Features (Signal)
study_hours = np.random.normal(5, 2, n_samples)          # Mean 5 hours
prev_grades = np.random.normal(70, 10, n_samples)        # Mean 70 points

# Useless Features (Noise)
shoe_size = np.random.normal(40, 3, n_samples)           # Random shoe size
lucky_number = np.random.randint(1, 100, n_samples)      # Random number

# Create Target Variable (Pass=1, Fail=0)
# Students who study more and have high grades are more likely to pass
# We add some noise (0.8 factor) so it's not perfectly linear
score_formula = (study_hours * 10) + (prev_grades * 0.5) + np.random.normal(0, 10, n_samples)
passed = (score_formula > 85).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Prev_Grades': prev_grades,
    'Shoe_Size': shoe_size,      # Garbage feature
    'Lucky_Number': lucky_number # Garbage feature
})

# 2. Apply Hypothesis Testing for Feature Selection
# We use ANOVA (f_classif) to calculate the F-score and p-value for each feature
# k='all' means we want to see scores for all features first
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(df, passed)

# Get the scores and p-values
scores = selector.scores_
p_values = selector.pvalues_

# --- Visualization ---
features = df.columns
plt.figure(figsize=(10, 6))

# We will plot the "-log(p-value)". 
# Why? Because very small p-values (0.00001) are hard to see. 
# -log(small number) = Huge Number. So, Taller Bar = Better Feature.
log_p_values = -np.log10(p_values)

bars = plt.bar(features, log_p_values, color=['green', 'green', 'red', 'red'])

plt.title('Feature Selection using ANOVA Hypothesis Test\n(Taller Bar = More Important Feature)')
plt.ylabel('Significance Score (-log10 p-value)')
plt.axhline(y=-np.log10(0.05), color='blue', linestyle='--', label='Significance Threshold (p=0.05)')

# Add actual p-values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    # If p-value is extremely small, show 0.000
    p_val_text = f'p={p_values[i]:.3f}' if p_values[i] > 0.001 else 'p<0.001'
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, p_val_text, ha='center', fontweight='bold')

plt.legend()
plt.show()

# Print detailed stats
print("Feature Scores (Higher is better):", scores)
print("P-Values (Lower is better):", p_values)