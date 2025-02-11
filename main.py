import string

from tqdm import tqdm
import numpy as np

def simple_tokenizer(data, *, token_size=255, forbidden_chars=string.punctuation):
    """
    Tokenizes input data into smaller chunks of specified size, removing specific characters.

    Args:
        data (str or list of str): The input text or list of texts to tokenize.
        token_size (int, optional): The maximum size of each token chunk. Default is 255.
        forbidden_chars (str, optional): Characters to be replaced with spaces. Default is punctuation.

    Returns:
        list: A list of tokenized chunks for each text input.

    1. Splits each token into chunks of size `token_size`.
    2. Converts text to lowercase.
    3. Replaces forbidden characters with spaces.
    4. Splits the text into words.
    5. Further splits each token into smaller chunks.
    6. Handles both single string and list inputs.
    """

    tokens = [
        [token[i:i + token_size] for token in
         text.lower().translate(str.maketrans(forbidden_chars, ' ' * len(forbidden_chars))).split()
         for i in range(0, len(token), token_size)]
        for text in (data if isinstance(data, list) else [data])
    ]

    return tokens

class EmbeddingData:
    """
    Class for generating word embeddings using co-occurrence matrix (COM), Point wise Mutual Information (PMI),
    and Singular Value Decomposition (SVD).

    Args:
        dataset (list of lists of str): The input dataset, where each list contains tokens (words).
        window_size (int): The size of the context window for co-occurrence calculation.
        pmi (bool): Whether to calculate Point wise Mutual Information (PMI) or not.
        smoothing (float): The smoothing value to avoid zero probabilities in PMI calculation.
        from_zero (bool): If True, sets negative PMI values to zero.
        d_model (int): The dimensionality of the final embedding space after SVD.

    1. Calculates co-occurrence matrix
    2. Calculates or not point-wise mutual information
    3. Calculates singular value decomposition
    """

    def __init__(self, *, embeddings = None, vocabulary = None):
        # Initializing empty placeholders for vocabulary and embeddings
        self.embeddings = embeddings
        self.vocabulary = vocabulary
        self.reversedVocabulary = dict(zip(vocabulary.values(), vocabulary.keys())) if type(vocabulary) is None else None

    # Method to calculate the embedding matrix
    def calculate(self, dataset=None, *, window_size=1,
                  pmi=True, smoothing: float = 0.0, from_zero=False,
                  d_model=128):

        # Step 1: Calculate the co-occurrence matrix from the dataset
        com_matrix = self.co_occurrence_matrix(dataset, window_size=window_size)
        pmi_matrix = com_matrix

        # Step 2: Calculate PMI (Point wise Mutual Information) if enabled
        if pmi:
            pmi_progressbar = tqdm(pmi_matrix, total=1, desc="Calculating Point wise Mutual Information")
            pmi_matrix = self.pointwise_mutual_information(com_matrix, smoothing=smoothing, from_zero=from_zero)

            pmi_progressbar.update(1)
            pmi_progressbar.close()

        # Step 3: Apply Singular Value Decomposition (SVD) to the PMI matrix
        svd_progressbar = tqdm(pmi_matrix, total=1, desc="Calculating Singular Value Decomposition")
        svd_matrix = self.singular_value_decomposition(pmi_matrix, d_model=d_model)

        svd_progressbar.update(1)
        svd_progressbar.close()

        # Step 4: Create embeddings from the SVD results and map tokens to vectors
        self.embeddings = {token: weights.tolist() for token, weights in
                           tqdm(zip(self.vocabulary, svd_matrix), total=len(self.vocabulary), desc="Processing Tokens")}

        # Adding the End of Sentence token to vocabulary
        self.vocabulary['<EOS>'] = len(self.vocabulary)

        # Create reversed vocabulary (from index to token)
        self.reversedVocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

    # Method to calculate co-occurrence matrix from dataset
    def co_occurrence_matrix(self, dataset=None, *, window_size=1):
        # Creating vocabulary from dataset by assigning unique indexes to tokens and initialize co-occurrence matrix
        self.vocabulary = {word: i for i, word in
                           enumerate(sorted(set(token for tokens in dataset for token in tokens)))}
        com_matrix = np.zeros((len(self.vocabulary), len(self.vocabulary)), dtype=int)

        # Iterate over the dataset to fill the co-occurrence matrix
        for tokens in tqdm(dataset, desc="Calculating Co-Occurrence Matrix"):
            for index, token in enumerate(tokens):

                # Get tokens within the specified window size
                tokens_in_window = tokens[max(0, index - window_size): index + window_size + 1]
                tokens_in_window.remove(token)

                # Update co-occurrence matrix for each pair of token and context token
                for context_token in tokens_in_window:
                    com_matrix[self.vocabulary[token], self.vocabulary[context_token]] += 1

        return com_matrix

    # Static method to calculate Point wise Mutual Information (PMI)
    @staticmethod
    def pointwise_mutual_information(com, *, smoothing: float = 0.0, from_zero=False):
        # Total number of co-occurrences across all tokens
        total_co_occurrences = np.sum(com)

        # Prevent division by zero in case of empty co-occurrences
        if total_co_occurrences == 0:
            raise Exception("Number 0 cannot be the value of total_co_occurrences")

        # Calculate token and context probabilities
        token_probability = np.sum(com, axis=1, keepdims=True) / total_co_occurrences
        context_probability = np.sum(com, axis=0, keepdims=True) / total_co_occurrences

        # Calculate the pairwise co-occurrence probability (with smoothing)
        pwc = (com + smoothing) / total_co_occurrences

        # Compute the PMI values with error handling for division by zero or invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(pwc / (token_probability * context_probability))
            pmi[np.isinf(pmi)] = 0

            # Optionally set negative PMI values to zero
            if from_zero:
                pmi[pmi < 0] = 0

        return pmi

    # Static method to apply Singular Value Decomposition (SVD) to PMI matrix
    @staticmethod
    def singular_value_decomposition(pmi, *, d_model=128):
        # Perform SVD decomposition on the PMI matrix
        u, sigma, vt = np.linalg.svd(pmi, full_matrices=False)

        # Reduce dimensionality by keeping only the top 'd_model' components
        u_reduced = u[:, :d_model]
        sigma_reduced = np.diag(sigma[:d_model])

        # Return the reduced representation as the final embedding
        return np.dot(u_reduced, sigma_reduced)

def embed_sequence(sentence, *, embed_dataset=None, forbidden_chars=string.punctuation, token_size=255, max_sequence=12,
                   d_model=128):
    """
    Tokenizes input sentence into smaller chunks (tokens), retrieves embeddings for each token from a provided dataset,
    and returns a padded list of embeddings. If any token is not found in the dataset, an error is raised.

    Args:
        sentence (str): The input sentence to be tokenized and embedded.
        embed_dataset (dict, optional): A dictionary mapping tokens to their corresponding embeddings.
        forbidden_chars (str, optional): A string of characters that should be excluded during tokenization (default is punctuation).
        token_size (int, optional): The maximum size for each token (default is 255).
        max_sequence (int, optional): The maximum length of the sequence to return (default is 12).
        d_model (int, optional): The dimensionality of the embedding vector (default is 128).

    Returns:
        list or None: A list of embeddings corresponding to the tokenized sentence, padded to the specified maximum sequence length,
                        or None if an unknown token is encountered.

    1. The function uses a simple tokenizer to break the input sentence into tokens.
    2. It then attempts to retrieve embeddings for each token from the provided `embed_dataset`.
    3. If the number of embeddings is less than `max_sequence`, zero vectors are used to pad the result.
    4. If any token is not found in the `embed_dataset`, the function prints an error message and returns `None`.
    5. The final embedding list has a length equal to `max_sequence`, with each element being a vector of length `d_model`.
    """

    # Tokenize the input sentence
    tokens = simple_tokenizer(sentence, token_size=token_size, forbidden_chars=forbidden_chars)

    try:

        # Attempt to retrieve embeddings for each token in the tokenized sentence
        # If there are fewer tokens than the maximum allowed sequence length, pad the embeddings with zeros
        embed = [embed_dataset[token] for token in tokens[0]]
        embed += [[0] * d_model] * (max_sequence - len(embed))

        return embed

    except KeyError:

        # In case of an unknown token (not in embed_dataset), print an error message
        print(f"Unknown word has occurred: {sentence}")

    return None
