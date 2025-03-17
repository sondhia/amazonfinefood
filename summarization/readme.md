This repository contains code to fine-tune a model for summarization tasks using Hugging Face Transformers and PyTorch. The project focuses on generating summaries.

## Overview

- **Model:** T5-small (with potential to switch to T5-base if memory allows)
- **Frameworks:** PyTorch, Hugging Face Transformers, evaluate, and NLTK
- **Dataset:** 1000 rows split into 700 training and 300 validation samples
- **Metrics:** ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum
- **Decoding:** Greedy decoding with room for experimentation with beam search or other strategies

## Project Structure

- **Dataset Preparation:**  
  - Preprocessing functions to clean and tokenize the data  
  - Dataset split into training and validation sets

- **Model Development:**  
  - Loading T5-small with `T5ForConditionalGeneration` and its tokenizer  
  - Custom training loop using `Seq2SeqTrainer`

- **Training & Evaluation:**  
  - Fine-tuning using `Seq2SeqTrainingArguments`
  - Evaluation based on ROUGE metrics with additional qualitative assessments planned

**Objective**

- Sequence to sequence model
    - The primary NLP task here is summarization, a classic Seq2Seq problem where an input sequence is transformed into a shorter sequence. This problem is typically tackled using encoder-decoder architectures.
    - The encoder compresses the input sequence into a fixed-length vector, and the decoder generates the summary sequence.
    - The encoder-decoder architecture is a powerful tool for sequence-to-sequence tasks, and it has been widely used in various NLP applications, including machine translation, text summarization, and question answering.

**Literature Review**

    Recent Papers

        Text summarization methods broadly fall into two major categories, each with distinct advantages and limitations when applied to product reviews.

        Extractive summarization identifies and concatenates the most important sentences from the original text. This approach preserves the original wording but may result in disjointed summaries [1]

        Extractive techniques include:
            - Frequency-based methods: Ranking sentences based on word frequency to determine importance [2]
            - Graph-based methods: Representing sentences as nodes in a graph, connected by similarity, and extracting those with highest centrality.These are quick to run, easy to implement, often yields high recall for key sentences. (e.g., TextRank) [3]
            - BERT extractive summarizer: Leveraging BERT’s contextual understanding to identify key sentences [1][4]

        Abstractive summarization generates new sentences that capture the essence of the text, essentially “paraphrasing” the original content. This approach typically produces more coherent and concise summaries but is technically more challenging.[1][3]

        Abstractive techniques include:
            - Sequence-to-sequence models with attention: Processing input review through an encoder and generating summary via a decoder with attention mechanism [4]
            - Transformer-based architectures: Leveraging self-attention mechanisms to capture long-range dependencies in text [3]
            - Pre-trained language models: Fine-tuning large pre-trained models for summarization tasks [3]

        Research indicates that abstractive summarization generally outperforms extractive methods for product reviews, particularly when reviews contain diverse opinions on different product aspects [4]
    
    State-Of-The-Art Models and Architectures

        Transformer architectures dominate the summarization models landscape.

        Transformer-Based Models
            - BART : Particularly well-suited for abstractive summarization through its denoising pre-training approach [3]
            - T5 (Text-to-Text Transfer Transformer): A versatile model that frames all NLP tasks, including summarization, as text-to-text problems [3]
            - PEGASUS: Specifically designed for summarization tasks through gap-sentence generation pre-training [3]
	        - PASS (Perturb-and-Select Summarizer): Employs T5 with few-shot fine-tuning and systematic input perturbations to generate multiple candidate summaries before selecting the best one [4]
        
        Training Approaches
            - Few-Shot Fine-Tuning: Rather than training from scratch, fine-tuning pre-trained models using a small set of high-quality examples [3][4]
            - Reinforcement Learning from Human Feedback (RLHF): Training models based on human preferences for summary quality [3]
            - Reinforcement Learning from AI Feedback (RLAIF): Using AI evaluation to improve model performance, a promising alternative to RLHF [3]
    
**Benchmark and Evaluation**

    Automated Metrics
        -   ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures overlap between generated summary and reference summary
        -   BLEU (Bilingual Evaluation Understudy): Measures precision of n-grams
        -   BERTScore: Uses contextual embeddings to measure semantic similarity between generated and reference summaries
        -   METEOR: Evaluates summaries based on alignment between generated and reference summaries

    Human and AI-Based Evaluation
        -   Human Evaluation: Assessing summaries based on informativeness, coherence, consistency, and relevance using Likert scales
        -   AI-as-a-Judge: Using large language models to evaluate summary quality, providing consistent and potentially less biased assessment
    
    Resource Constraints
        -	Memory / VRAM: Large transformers can easily consume 8–16GB or more.
        -	Training Time: With ~570K reviews, training from scratch may be prohibitive. Fine-tuning a pretrained model can be more tractable.

**Preliminary Experiments**

    Data Preparation
        -	Cleaning & Pre-processing: Remove HTML, handle special characters, and lemmatize words.
        -   Sampling: We'll start with 50k reviews to get a sense of feasibility and performance.
        -   Splitting: We'll create a training, validation, and test split.

    Baseline Models
	    -	Extractive Baseline: Implement TextRank or LexRank to get a quick measure.
	    -	Abstractive Baseline: Fine-tune a small or medium-sized T5 or BART on our subset of data. Evaluate with ROUGE on the test set.

    Iterative Refinement
	    -	Hyperparameters: Batch size, Learning rate, Max Sequence length, etc.
	    -	Model Size: Experiment with smaller vs. larger versions of T5/BART to see if performance scales with model size.
	    -	Early Stopping & Validation: Track validation loss and ROUGE scores. Aim to stop when metrics plateau to avoid overfitting.

**Other Notes**

	There is an existing repository for the summarization task on Amazon Fine Food Reviews on GitHub. Here's the link for it,

	https://github.com/Currie32/Text-Summarization-with-Amazon-Reviews/tree/master

	To avoid duplicating that approach and to achieve more modern, state-of-the-art results, we will be focusing on:
	    -   Transformer-based models (BART, T5, PEGASUS)
	    -   Better evaluation and possibly a multi-task or transfer learning approach
	    -   Cleaner, more efficient data pipelines and a modern framework (PyTorch or TensorFlow)

**References**

[1]https://aws.amazon.com/blogs/machine-learning/techniques-for-automatic-summarization-of-documents-using-language-models/

[2]https://www.irjmets.com/uploadedfiles/paper/issue_4_april_2023/37776/final/fin_irjmets1683222722.pdf

[3]https://www.assemblyai.com/blog/text-summarization-nlp-5-best-apis

[4]https://vfast.org/journals/index.php/VTSE/article/view/856
