# Simple-Embeddings
A lightweight library for quickly generating embeddings without training a neural network or using pre-trained models. It leverages simple techniques like tokenization, co-occurrence matrices, Pointwise Mutual Information (PMI), and Singular Value Decomposition (SVD) to create meaningful embeddings.

# Features
- Simple Tokenization: Splits text into words, removes unwanted characters, and further breaks words into smaller chunks.
- Co-occurrence Matrix: Captures word context relationships within a given window size.
- PMI Calculation: Measures statistical association between words.
- SVD for Dimensionality Reduction: Converts high-dimensional word relationships into compact embedding representations.
- Embedding Lookup: Converts sentences into a sequence of embeddings.

# Prerequisites
- Python 3.9.13
- Required Packages:
  - numpy 2.0.2
  - tqdm 4.67.1

# Usage
## 1. Tokenization
The `simple_tokenizer` function tokenizes input text:
  ```python
  data = ["This is an example sentence.", "Testing, testing, one, two, three."]
  tokens = simple_tokenizer(data)
  print(tokens)
  ```
## 2. Generating Embeddings
The `EmbeddingData` class generates word embeddings:
```python
embeddings = EmbeddingData()
embeddings.calculate(tokens, smoothing=3e-4, d_model=128, window_size=3)
```

## 3. Embedding a Sentence
The `embed_sequence` function converts a sentence into an embedding sequence:
```python
sentence = "hello world"
embedded_sentence = embed_sequence(sentence, embed_dataset=embeddinngs.embeddings)
print(embedded_sentence)
```
**WARNING:** It won't work with words out of the scope

# Near Future Updates
- Adding Better Tokenizer
- Word Normalization
- Out Of Vocabulary Handling
