import pandas as pd

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("=" * 50)
print("CUSTOMER SEGMENTATION PROJECT")
print("=" * 50)

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())