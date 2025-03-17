###Load data and preprocess for test three models on small scale of data
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

# Load dataset
file_path = "Final Project/Amazon_review.csv"

#Take 10,000 rows for test
df = pd.read_csv(file_path, nrows=1000)

# Display basic information
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (if necessary)
df = df.dropna(subset=['Text', 'Score'])
print("\nShape after dropping missing values:", df.shape)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    words = nltk.word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Define stopwords
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Define function to remove illegal characters
def remove_illegal_characters(text):
    if isinstance(text, str):
        # Remove all non-printable characters
        return re.sub(r'[^\x20-\x7E\u00A0-\u00FF]', '', text)  # Added more Unicode range
    return text
    
# Preprocessing 
def preprocess_data(df):
    df['Text'] = df['Text'].fillna("")
    df['Cleaned_Text'] = df['Text'].apply(clean_text)  # Apply text cleaning
    df['Cleaned_Text'] = df['Cleaned_Text'].apply(remove_illegal_characters)
    return df

df = prepocess_data(df)

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
