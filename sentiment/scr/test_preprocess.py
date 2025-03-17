###Load data and preprocess for test three models on small scale of data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
file_path = "Final Project/Amazon_review.csv"

#Take 10,000 rows for test
df = pd.read_csv(file_path, nrows=10000)

# Display basic information
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (if necessary)
df = df.dropna(subset=['Text', 'Score'])
print("\nShape after dropping missing values:", df.shape)


# Convert Score into Sentiment Labels
def label_sentiment(score):
    if score in [4, 5]:
        return 1  # Positive
    elif score == 3:
        return 0  # Neutral
    else:
        return -1  # Negative

df["Sentiment"] = df["Score"].apply(label_sentiment)

# Split data to traub abd test
X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Sentiment"], test_size=0.2, random_state=42)
