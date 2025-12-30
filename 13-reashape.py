import numpy as np

# 1. We have 6 numbers (1D Array - Vector)
# Shape: (6,) -> 6 numbers side by side
data = np.array([1, 2, 3, 4, 5, 6])
print(f"Original State:\n{data}")
print(f"Shape: {data.shape}\n")

# 2. Let's make it a Rectangle (2 Rows, 3 Columns)
# Shape: (2, 3)
reshaped_data = data.reshape(2, 3)
print(f"2x3 Table Format:\n{reshaped_data}")
print(f"Shape: {reshaped_data.shape}\n")

# 3. Convert to Column Format for AI (reshape(-1, 1))
# -1: "Calculate the necessary rows automatically" (Here it becomes 6)
#  1: "Force the number of columns to be 1"
ai_ready_data = data.reshape(-1, 1)
print(f"AI Ready Format (-1, 1):\n{ai_ready_data}")
print(f"Shape: {ai_ready_data.shape}")