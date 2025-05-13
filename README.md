# Simple-Embeddings
A lightweight library for quickly generating embeddings without training a neural network or using pre-trained models. It leverages simple techniques like tokenization, co-occurrence matrices, Pointwise Mutual Information (PMI), and Singular Value Decomposition (SVD) to create meaningful embeddings.

# Features
- Simple Tokenization: Splits text into words, removes unwanted characters, and further breaks words into smaller chunks.
- Co-occurrence Matrix: Captures word context relationships within a given window size.
- PMI Calculation: Measures statistical association between words.
- SVD for Dimensionality Reduction: Converts high-dimensional word relationships into compact embedding representations.
- OOV Handling: BPE
- Embedding Lookup: Converts sentences into a sequence of embeddings.

# Prerequisites
- Python 3.9.13
- Required Packages:
  - numpy 2.0.2
  - tqdm 4.67.1
  - regex 2024.11.6

# Usage
## 1. Tokenization
The `simple_tokenizer` function tokenizes input text:
  ```python
  data = ["This is an example sentence.", "Testing, testing, one, two, three."]
  tokens = pre_tokenizer(data)
  print(tokens)
  ```

## 2. Generating Embeddings
The `EmbeddingData` class generates word embeddings:
```python
data = ["This is an example sentence.", "Testing, testing, one, two, three."]
tokens = pre_tokenizer(data, output_type=list)

embeddings = EmbeddingData()
embeddings.calculate(tokens, smoothing=3e-4, d_model=128, window_size=3)
```

## 3. Embedding a Sentence
The `embed_sequence` function converts a sentence into an embedding sequence:
```python
data = ["This is an example sentence.", "Testing, testing, one, two, three."]
tokens = pre_tokenizer(data, output_type=list)

embeddings = EmbeddingData()
embeddings.calculate(tokens, smoothing=3e-4, d_model=128, window_size=3)
    
# Embedding a sentence
data = "This is an example sentence."
tokens = pre_tokenizer(data, output_type=list)

embedded_sentence = embeddings.embed_sequence(tokens)
```
**WARNING:** It won't work with words out of the scope

## 3. OOV Handling - Byte Pair Encoder
The `BytePairEncoder` class allows for oov handling:
```python
data = ['He is fast as lightning', "I'm learning how to run as fast as him"]

bpe = BytePairEncoder(corpus=data)

embeddings = EmbeddingData(vocabulary=bpe.vocabulary)
embeddings.calculate(bpe.tokenize(data), smoothing=3e-4, d_model=128, window_size=3)

embedded_sequence = embeddings.embed_sequence(bpe.tokenize("He's learning how to run"))
```

## 4. OOV Handling - Word Piece
The `WordPiece` class allows for oov handling:
```python
data = ['He is fast as lightning', "I'm learning how to run as fast as him"]

wp = WordPiece(corpus=data)

embeddings = EmbeddingData(vocabulary=wp.vocabulary)
embeddings.calculate(wp.tokenize(data), smoothing=3e-4, d_model=128, window_size=3)

embedded_sequence = embeddings.embed_sequence(wp.tokenize("He's learning how to run"))
```

# History
- 1.1.2:
    - Added WordPiece  
- 1.1.1:
    - Optimized BPE. Up to 60% faster than the previous version
    - Added dunder methods to BPE
    - Updated the pre_tokenizer
    - removed `tokenizer`
    - moved `embed_sequence` to `EmbeddingData`
- 1.1.0:
    - Added BPE
    - Renamed `simple_tokenizer` to `pre_tokenizer`
    - Added `tokenizer`
    - Adjusted Standard Variables
- 1.0.0 Initial Commit

# Near Future Updates
- Word Normalization
- Out Of Vocabulary Handling
  - Byte Pair Encoder ✅
  - Word Piece ✅
  - Unigram
