import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create Sample Data
# We create a dataset with clear relationships to see the colors change.
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7],      # As this increases, Score increases (Positive)
    'Exam_Score': [20, 30, 40, 50, 60, 75, 90],
    'Phone_Usage': [5, 4.5, 4, 3, 2, 1, 0.5],  # As this increases, Score decreases (Negative)
    'Shoe_Size': [36, 38, 42, 40, 37, 39, 41]  # Random data (No Correlation)
}

df = pd.DataFrame(data)

# 2. Calculate Correlation Matrix
# This creates the table of numbers between -1 and 1
correlation_matrix = df.corr()

# 3. Visualization (Heatmap)
plt.figure(figsize=(8, 6))

# sns.heatmap arguments:
# data: The correlation matrix we calculated
# annot=True: Write the actual numbers inside the boxes
# cmap='coolwarm': Color map (Red for positive, Blue for negative)
# fmt='.2f': Format numbers to 2 decimal places
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title("Correlation Matrix Heatmap")
plt.show()
