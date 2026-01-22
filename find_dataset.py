from sklearn.datasets import fetch_california_housing
import pandas as pd

# To fetch the California Housing dataset
dataset = fetch_california_housing(as_frame=True)

# Combining features + target into a single DataFrame
df = pd.concat([dataset.data, dataset.target.rename("MedHouseVal")], axis=1)

# Saving to CSV
df.to_csv("california_housing.csv", index=False)

print("Saved california_housing.csv")
