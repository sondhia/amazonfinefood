What’s Left / Next Steps

1. Further Fine-Tuning & Hyperparameter Optimization
    -   Look to keep fine tuning then model parameters such as batch size, learning rate, etc.

2. Model and Decoding Improvements:
	-	Model Size:
	        -	Moving from t5-small to t5-base model
  		-	Fine-tune GPT to compare results
	-	Decoding Strategies:
	        -	Experiment with beam search, top-k, or top-p sampling to potentially improve summary quality.
	-	Post-Processing:
	        -	Implement any necessary post-processing on generated summaries to improve fluency and coherence.

3. Evaluation Enhancements:
	-	Additional Metrics:
	        -	Adding BLEU, METEOR, and BERTScore along with the current ROUGE numbers and human evaluation to gain a more complete picture of summary quality.
	-	Qualitative Analysis:
	        -	Perform manual reviews of generated summaries to identify specific failure modes or areas for improvement.

⸻
