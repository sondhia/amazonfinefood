import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True, max_length=512)

# Convert Pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_function, batched=True)

# Rename "Sentiment" column to "labels" for compatibility with Trainer
dataset = dataset.rename_column("Sentiment", "labels")

# Train-Test Split
dataset = dataset.train_test_split(test_size=0.2)

# Load DistilBERT model for classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./distilbert_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train DistilBERT Model
trainer.train()

# Make predictions on the test dataset
predictions = trainer.predict(dataset["test"])

# Convert logits to predicted class labels
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = dataset["test"]["labels"]

# Generate classification report
eval_report = classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"])
print("Classification Report:\n", eval_report)
