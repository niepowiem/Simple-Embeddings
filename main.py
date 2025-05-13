from simple_embeddings import *

data = ['He is fast as lightning', "I'm learning how to run as fast as him"]

# === BPE ===

# bpe = BytePairEncoder(corpus=data, white_space=True)
# print(bpe.vocabulary)
# 
# embeddings = EmbeddingData(vocabulary=bpe.vocabulary)
# embeddings.calculate(bpe.tokenize(data), smoothing=3e-4, d_model=128, window_size=3)
# 
# embedded_sequence = embeddings.embed_sequence(bpe.tokenize("He's learning how to run"))
# print(embedded_sequence)

# === WordPiece ===

# wp = WordPiece(corpus=data)
# print(wp.vocabulary)
# 
# embeddings = EmbeddingData(vocabulary=wp.vocabulary)
# embeddings.calculate(wp.tokenize(data), smoothing=3e-4, d_model=128, window_size=3)
# 
# embedded_sequence = embeddings.embed_sequence(wp.tokenize("He's learning how to run"))
# print(embedded_sequence)
