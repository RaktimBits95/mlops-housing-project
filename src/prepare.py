import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Load dataset
data = pd.read_csv("data/california_housing.csv")

# 🚀 Apply log(1 + x) to reduce skewness
skewed_features = ["population", "households", "median_income", "total_rooms", "total_bedrooms"]
for feature in skewed_features:
    if feature in data.columns:
        data[feature] = np.log1p(data[feature])

# Split into train and test
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# 🔧 Ensure output directory exists
os.makedirs("data/prepared", exist_ok=True)

# Save the processed train and test sets
train_set.to_csv("data/prepared/train.csv", index=False)
test_set.to_csv("data/prepared/test.csv", index=False)
