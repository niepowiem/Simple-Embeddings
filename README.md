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
