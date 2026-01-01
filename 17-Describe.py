import pandas as pd
import numpy as np

# Sample Data: Age and Salary
df = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 150], # 150 is an outlier/error
    'Salary': [3000, 4000, 5000, 6000, 7000, 8000]
})

# The X-Ray Command
print(df.describe())
