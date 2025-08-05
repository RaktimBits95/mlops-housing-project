import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load dataset
data = pd.read_csv("data/california_housing.csv")

# Split into train and test
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# 🔧 Ensure output directory exists
os.makedirs("data/prepared", exist_ok=True)

# Save split data
train_set.to_csv("data/prepared/train.csv", index=False)
test_set.to_csv("data/prepared/test.csv", index=False)
