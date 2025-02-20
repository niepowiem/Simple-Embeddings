import re
import string
from collections import defaultdict

from tqdm import tqdm
import numpy as np

def pre_tokenizer(data, *, token_size=255, forbidden_chars=string.punctuation):
    """
    Tokenizes input data into smaller chunks of specified size, removing specific characters.

    Args:
        data (str or list of str): The input text or list of texts to tokenize.
        token_size (int, optional): The maximum size of each token chunk. Default is 255.
        forbidden_chars (str, optional): Characters to be replaced with spaces. Default is punctuation.

    Returns:
        list: Nested list structure where:
            - Outer list: Input texts
            - Middle list: Words in text (split by whitespace)
            - Inner list: Character chunks for each word (if word length > token_size)

    Process Flow:
        1. Handle both single string and list inputs
        2. Convert to lowercase and remove forbidden characters
        3. Split text into words using whitespace
        4. Split long words into chunks of specified size
    """
    tokens = [
        [
            # Split word into chunks if it's longer than token_size
            [token[i:i + token_size] for i in range(0, len(token), token_size)]
            if len(token) > token_size else token  # Keep short words as-is

            # Process each word: lower-case + remove forbidden chars + split by whitespace
            for token in text.lower().translate(
            str.maketrans(forbidden_chars, ' ' * len(forbidden_chars))
        ).split()
        ]
        # Handle both single text and list of texts
        for text in (data if isinstance(data, list) else [data])
    ]

    return tokens

def tokenize(data, *, vocabulary=None):
    """Convert text into vocabulary tokens using regex pattern matching

    Args:
        data (str or list): Input text(s) to tokenize
        vocabulary (dict): Mapping of tokens to indices

    Returns:
        list: List of token lists (for each input text)

    Process Flow:
        1. Create regex pattern sorted by descending token length
        2. Match longest possible tokens first
        3. Split text into vocabulary tokens using regex
    """
    if not vocabulary:
        raise Exception("No vocabulary passed!")

    # Create regex pattern matching longest tokens first to prevent partial matches
    # Escape special characters and sort by descending length
    pattern = '|'.join(sorted(
        map(re.escape, vocabulary.keys()),
        key=len,
        reverse=True  # Critical for longest-match-first strategy
    ))

    # Find all matching tokens in lowercase text
    return [
        re.findall(pattern, text.lower())
        for text in (data if isinstance(data, list) else [data])
    ]

class BytePairEncoder:
    """
        Initialize the Byte Pair Encoder with a pre-tokenized corpus and configuration parameters.

        Parameters:
            pre_tokenized_corpus (list): A nested list structure where:
                - Outer list: Sentences
                - Middle list: Words per sentence
                - Inner list: Characters/subword tokens per word
            num_merges (int): Number of merge operations to perform during training
            min_frequency (int): Minimum frequency required to perform a merge
    """

    def __init__(self, pre_tokenized_corpus, *, num_merges=10, min_frequency=2):
        """
        Initialize the Byte Pair Encoder with a pre-tokenized corpus and configuration parameters.

        Parameters:
            pre_tokenized_corpus (list): A nested list structure where:
                - Outer list: Sentences
                - Middle list: Words per sentence
                - Inner list: Characters/subword tokens per word
            num_merges (int): Number of merge operations to perform during training
            min_frequency (int): Minimum frequency required to perform a merge
        """
        self.tokenized_corpus = pre_tokenized_corpus
        self.num_merges = num_merges
        self.min_frequency = min_frequency

        # Initialize vocabulary with unique characters from the corpus
        self.vocabulary = {word: i for i, word in enumerate(sorted(set(token for sublist in pre_tokenized_corpus for tokens in sublist for token in tokens)))}
        print(self.vocabulary)
    def train(self):
        """Execute the BPE training process with specified number of merges"""
        # Use tqdm for progress visualization during merges
        merge_progressbar = tqdm(range(self.num_merges), desc="Creating Byte Pair Encoder Merges")

        # Create merges for num_merges
        for _ in range(self.num_merges):
            # 1. Find all adjacent pairs and their counts/positions
            pairs = self.make_pairs()

            if not pairs:
                break  # Stop early if no merges possible

            # 2. Select pair with the highest count (using tuple comparison for tie-breaker)
            most_frequent_pair = max(
                pairs,
                key=lambda x: (pairs[x]['count'], x)  # Prefer lex order for ties
            )

            # 3. Check minimum frequency threshold
            if pairs[most_frequent_pair]['count'] < self.min_frequency:
                break

            # 4. Create new token and add to vocabulary
            new_token = ''.join(most_frequent_pair)
            self.vocabulary[new_token] = len(self.vocabulary)

            # 5. Update corpus by merging selected pairs
            self.update_tokenized_corpus(pairs, most_frequent_pair)

            merge_progressbar.update(1)
        merge_progressbar.close()

    def make_pairs(self):
        """
        Identify all adjacent symbol pairs and track their positions

        Returns:
            defaultdict: Dictionary mapping pairs to their counts and positions
        """
        pairs = defaultdict(lambda: {'count': 0, 'pair_ids': []})

        # Three-level iteration: sentence -> word -> character position
        for sentence_id, sentence in enumerate(self.tokenized_corpus):
            for word_id, word in enumerate(sentence):
                # Iterate through adjacent character pairs
                for char_id in range(len(word) - 1):
                    pair = (word[char_id], word[char_id + 1])

                    # Record occurrence and position
                    pairs[pair]['count'] += 1
                    pairs[pair]['pair_ids'].append((sentence_id, word_id, char_id))

        return pairs

    def update_tokenized_corpus(self, pairs, most_frequent_pair):
        """
        Merge occurrences of a specific pair throughout the corpus

        Parameters:
            pairs (dict): Full pair data from make_pairs()
            most_frequent_pair (tuple): The pair to merge
        """
        # Process all recorded positions for the target pair
        for position in pairs[most_frequent_pair]['pair_ids']:
            sentence_id, word_id, char_id = position
            word = self.tokenized_corpus[sentence_id][word_id]

            # Replace first element with merged pair
            word[char_id] = ''.join(most_frequent_pair)
            # Remove second element of the pair
            word.pop(char_id + 1)

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
    def calculate(self, tokenized_dataset=None, *, window_size=1,
                  pmi=True, smoothing: float = 3e-4, from_zero=False,
                  d_model=128):

        # Step 1: Calculate the co-occurrence matrix from the dataset
        com_matrix = self.co_occurrence_matrix(tokenized_dataset, window_size=window_size)
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
        if not self.vocabulary:
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

def embed_sequence(tokens, *, embed_dataset=None, max_sequence=12, d_model=128):
    """
    Tokenizes input sentence into smaller chunks (tokens), retrieves embeddings for each token from a provided dataset,
    and returns a padded list of embeddings. If any token is not found in the dataset, an error is raised.

    Args:
        tokens (list of lists or list of lists of lists): Tokenized sentence
        embed_dataset (dict, optional): A dictionary mapping tokens to their corresponding embeddings.
        max_sequence (int, optional): The maximum length of the sequence to return (default is 12).
        d_model (int, optional): The dimensionality of the embedding vector (default is 128).

    Returns:
        list or None: A list of embeddings corresponding to the tokenized sentence, padded to the specified maximum sequence length,
                        or None if an unknown token is encountered.

    1. It then attempts to retrieve embeddings for each token from the provided `embed_dataset`.
    2. If the number of embeddings is less than `max_sequence`, zero vectors are used to pad the result.
    3. If any token is not found in the `embed_dataset`, the function prints an error message and returns `None`.
    4. The final embedding list has a length equal to `max_sequence`, with each element being a vector of length `d_model`.
    """

    try:
        # Attempt to retrieve embeddings for each token in the tokenized sentence
        # If there are fewer tokens than the maximum allowed sequence length, pad the embeddings with zeros
        embed = [embed_dataset[token] for token in tokens[0]]
        embed += [[0] * d_model] * (max_sequence - len(embed))

        return embed

    except KeyError:

        # In case of an unknown token (not in embed_dataset), print an error message
        print(f"Unknown token has occurred in: {tokens}")

    return None
