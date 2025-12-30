import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Raw Data
data = {
    'Student': ['Ali', 'Ayse', 'Mehmet', 'Zeynep'],
    'City': ['Istanbul', 'Ankara', 'Istanbul', 'Izmir'],  # No Order (Nominal)
    'Size': ['S', 'XL', 'M', 'L']                           # Has Order (Ordinal)
}

df = pd.DataFrame(data)

print("--- 1. RAW DATA (The model cannot understand this) ---")
print(df)
print("\n" + "="*40 + "\n")

# ---------------------------------------------------------
# METHOD 1: Label Encoding (For Ordinal Data)
# ---------------------------------------------------------
# We map S, M, L, XL to 0, 1, 2, 3 respectively.
# Note: Automatic LabelEncoder() sorts alphabetically (L, M, S, XL), 
# which is wrong for sizes. So, using a manual 'map' is safer here.
size_mapping = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
df['Size_Encoded'] = df['Size'].map(size_mapping)

print("--- 2. LABEL ENCODING (Check the 'Size_Encoded' Column) ---")
print(df[['Student', 'Size', 'Size_Encoded']])
print("\n" + "="*40 + "\n")

# ---------------------------------------------------------
# METHOD 2: One-Hot Encoding (For Nominal Data)
# ---------------------------------------------------------
# We split cities into separate columns.
# New binary columns like 'City_Istanbul', 'City_Ankara' will be created.
df_one_hot = pd.get_dummies(df, columns=['City'], prefix='City')

# Convert True/False to 1/0 (to ensure numeric format for AI)
df_one_hot = df_one_hot.replace({True: 1, False: 0})

print("--- 3. ONE-HOT ENCODING (Check the City Columns) ---")
# Adjust column names based on the data (alphabetical order by default)
print(df_one_hot[['Student', 'City_Ankara', 'City_Istanbul', 'City_Izmir']])