Amazon generates a massive amount of user review data, which reflects amazon customer sentiments, preference on products, and feedback trends. Customer reviews play a vital role in shaping business strategies and customer decisions. In this project, we aim to use Natural Language Processing (NLP) techniques to analyze the dataset of ‘Amazon Fine Food Reviews’ and bring insights for business and consumers.

Consumers and businesses heavily rely on product reviews to make purchasing decisions and for future order strategy. While sentiment analysis alone is valuable, reviews can offer much more potential for deriving insights. Businesses could benefit from predictive capabilities, trend analysis and an understanding of why reviews are helpful or not. 

We have 2 models at the moment , sentiment and summarization model.

## Setup Instructions for Summarization Model

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`

3. Install all required packages
   ```bash
   pip install -r requirements.txt

3. Running the Code
	- Jupyter Notebook
      Open the provided notebook (e.g., t5_summarization.ipynb) and run the cells sequentially.

4. Feel free to change training configuration
   ```bash
     training_args = Seq2SeqTrainingArguments(
    output_dir="my_summarization_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    predict_with_generate=True,
    generation_max_length=64,
    generation_num_beams=1  # Greedy decoding; experiment with higher numbers if needed)
