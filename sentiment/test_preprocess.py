#load data and preprocess for test three models on small scale of data, take 10,000 raw for test
import pandas as pd

# Load dataset
file_path = "Final Project/Amazon_review.csv"
df = pd.read_csv(file_path, nrows=10000)

# Display basic information
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())
