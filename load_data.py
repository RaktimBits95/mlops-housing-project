# load_data.py
from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

data = fetch_california_housing(as_frame=True)
df = data.frame

os.makedirs("data", exist_ok=True)
df.to_csv("data/california_housing.csv", index=False)
