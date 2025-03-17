This repository contains code to fine-tune models for sentiment classification on the Amazon Fine Food Reviews dataset. The project evaluates multiple NLP techniques and ultimately selects DistilBERT for its balance of performance and efficiency.

## Overview

- **Tested Models:** Logistic Regression + TF-IDF, LSTM, DistilBERT
- **Frameworks:** Scikit-Learn, PyTorch, Hugging Face Transformers
- **Dataset:** 10,000 reviews (Train: 80%, Test: 20%)
- **Metrics:** Accuracy, F1-score, Precision, Recall


## Project Structure

- **Dataset Preparation:**  
  - Preprocessing functions to clean
  - Convert review scores into sentiment labels
  - Preprocess text (lowercasing, tokenization, stopword removal)
  - Split into train (80%) and test (20%)

- **Model Development:**  
  - Baseline Model: TF-IDF + Logistic Regression 
  - Deep Learning: LSTM model for sequence-based sentiment analysis
  - Transformer Model: Fine-tuned DistilBERT

- **Training & Evaluation:**  
  - Hyperparameter tuning (learning rate, batch size, epochs)
  - Compare accuracy, F1-score, precision, recall
 

**Objective**

- Sentiment Classification
    - Predict whether a review is Positive, Neutral, or Negative

**Literature Review**

        BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)
        BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP by introducing deep bidirectional context understanding. BERT’s core principles, useful for classification tasks. Unlike previous models, it reads text both left-to-right and right-to-left simultaneously for better comprehension.
        BERT’s contextual embeddings significantly improve sentiment classification.
        Fine-tuning BERT on sentiment datasets achieves state-of-the-art accuracy.
        Pre-trained model: “bert-base-uncased” is commonly used for text classification tasks.
        
        
        DistilBERT: a distilled version of BERT: smaller, faster, cheaper and lighter (Sanh et al., 2019)
        Lighter and faster version of BERT trained using knowledge distillation. Retains 97% of BERT’s performance while being 60% smaller and 2× faster.
        Key Techniques:
        Removes Next Sentence Prediction (NSP).
        Reduces the number of layers from 12 → 6 while maintaining BERT-like embeddings.
        Trained to mimic BERT’s outputs while using fewer parameters.
        Much faster inference than BERT and RoBERTa. Requires less computational power (can run on CPUs efficiently). Great for real-time sentiment classification in production environments.
        
        
        Long Short-Term Memory (LSTM) Networks: a type of Recurrent Neural Network (RNN) designed to handle long-term dependencies in sequential data. (Ralf C. Staudemeyer, Eric Rothstein Morris 2019) 
        LSTMs use gates (input, forget, output) to control information flow, solving vanishing gradient problems.
        Forget Gate: Decides which past information to discard.
        Input Gate: Determines which new information to store.
        Output Gate: Decides what to output for the next state.
        Impact on Sentiment Classification:
        Captures word dependencies in long reviews (e.g., “not great at all” vs. “great”).
        Better context retention than traditional RNNs.
        Helps classify sequential text effectively.
        
        
        Logistic Regression: a statistical model used for binary classification (positive vs. negative sentiment) (Husna et al., 2019; Cox, 1958)
        Uses sigmoid function to output probabilities between 0 and 1.
        Advantages in Sentiment Analysis:
        Interpretable – Coefficients indicate which words influence sentiment.
        Computationally efficient – Faster training than deep learning models.
        Works well with TF-IDF – Can effectively classify sentiment without needing deep networks.
        TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection (corpus). Converts raw text into numerical feature vectors for machine learning models. Captures important words related to sentiment (e.g., “great”, “terrible”).
        
        
    
**Benchmark and Evaluation**

    Automated Metrics
        -   Accuracy → Overall model correctness
        -   F1-score → Handles imbalanced data better than accuracy
        -   Precision & Recall → Measures false positives/negatives

    Human Evaluation
        -   Misclassification analysis to identify model weaknesses
        -   Qualitative review of difficult-to-classify sentiments
    
