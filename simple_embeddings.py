import re

from collections import Counter
from itertools import chain, pairwise
from typing import Self, Union

from tqdm import tqdm
import numpy as np

def pre_tokenizer(data: str | list[str] | tuple[str], *, token_size: int = 255, forbidden_chars: str = '', lowercase: bool = False,
                  flattened: bool = False, output_type: type = None):
    """
    Tokenizes input data into smaller chunks of specified size, removing specific characters.

    Args:
        data (str, list of str, tuple of str): The input text or collection of texts to tokenize.
        token_size (int, optional): Maximum size of each token chunk. Default is 255.
        forbidden_chars (str, optional): Characters to replace with spaces. Default is empty.
        lowercase (bool, optional): If True, lowercases the text. Default is False.
        flattened (bool, optional): If True, returns a flat iterator of tokens; otherwise,
                                   returns a generator for each text's token list.
        output_type (type, optional): converts chain object into given type.

    Returns:
        An iterator of tokens if flat=True (itertools.chain), or a generator of token lists (one per text).
    """

    # Normalize input to handle str, list, or tuple uniformly
    if isinstance(data, (list, tuple)):
        texts = tuple(data)
    else:
        texts = (data,)

    if len(forbidden_chars) > 0:
        translator = str.maketrans(forbidden_chars, ' ' * len(forbidden_chars))
        texts = [text.translate(translator) for text in texts]

    if lowercase:
        texts = [text.lower() for text in texts]

    def tokenize_each_word(word):
        """Yield token chunks for a word if longer than token_size; otherwise, yield the word."""
        if len(word) > token_size:
            for i in range(0, len(word), token_size):
                yield word[i:i + token_size]
        else:
            yield word

    def split_sequence(text):
        """Process a single text: clean, split, and tokenize each word."""
        for word in text.split(' '):
            yield from tokenize_each_word(word)

    if flattened:
        # Return a flattened iterator (itertools.chain object) over tokens from all texts.
        tokenized_sequence = chain.from_iterable(split_sequence(text) for text in texts)
    else:
        # Return a generator of token lists for each text.
        inner_type = tuple if not output_type else output_type
        tokenized_sequence = (inner_type(split_sequence(text)) for text in texts)

    if output_type:
        # Returns tokens in a given type. For example tuple
        return output_type(tokenized_sequence)

    return tokenized_sequence

class BytePairEncoder:
    """Byte Pair Encoding (BPE) tokenizer that learns merges from a corpus.

    The encoder can be initialized with raw text (split into characters) or pre-tokenized text.
    It learns merge rules up to a specified maximum and builds a vocabulary.

    Attributes:
        vocabulary (Dict[str, int]): Mapping of tokens to unique integer IDs.
        regex (Optional[Pattern]): Compiled regex for tokenization.
        max_merges (int): Maximum number of merge operations during training.
        min_frequency (int): Minimum frequency for merging token pairs.
        pre_tokenized_corpus (Optional[Iterable[str]]): Corpus as pre-split tokens.
    """

    def __init__(self, *, corpus: str | list | tuple | set = None, pre_tokenized_corpus: list | tuple | set = None,
                 max_merges: int = 127, min_frequency=2, vocabulary: dict | list | tuple | set = None, regex: str = None,
                 white_space: bool=True) -> None:
        """Initializes BPE encoder with training data and configuration.

        Args:
            corpus: Raw text input(s) to be split into individual characters.
                Example: ["hello world"] becomes ['h','e','l','l','o',' ','w','o','r','l','d']
            pre_tokenized_corpus: Pre-split tokens that bypass character splitting.
                Takes precedence over `corpus` if both are provided.
            max_merges: Maximum number of merge operations to learn during training.
                Controls vocabulary size growth.
            min_frequency: Frequency threshold for merging token pairs.
                Pairs occurring less than this will not be merged.
            vocabulary: Optional pre-existing vocabulary. Can be either:
                - Dict: {token: id} mapping
                - Iterable: Token list/set (auto-assigns IDs)
            regex: Pre-compiled regex pattern string for tokenization.
                If provided, skips vocabulary initialization and training.
            white_space: Whether to parse text with " " or not

        Raises:
            ValueError: If no valid initialization data provided (corpus,
                pre_tokenized_corpus, vocabulary, or regex).

        Note:
            - Vocabulary initialization order is non-deterministic when using sets
            - Provide either raw text (corpus) or pre-tokenized data, not both
            - Regex pattern takes precedence over all other initialization methods
        """

        self.pre_tokenized_corpus = pre_tokenized_corpus
        self.vocabulary = vocabulary

        self.max_merges = max_merges
        self.min_frequency = min_frequency
        self.regex = regex

        self.white_space = white_space

        if isinstance(regex, str):
            return

        if self.check_for_vocabulary():
            self.build_tokenizer()
            return

        if not self.check_for_corpus(corpus):
            raise Exception('Must provide either corpus, pre_tokenized_corpus, vocabulary, or regex')

        # Initialize vocabulary with unique tokens (order depends on Python's set iteration)
        self.vocabulary = {token: idx for idx, token in enumerate(set(self.pre_tokenized_corpus))}

        self.train()

    def check_for_vocabulary(self) -> bool:
        """Process and validate any provided vocabulary.

        Handles both dictionary and iterable vocabulary formats.
        Builds tokenizer if valid vocabulary found.

        Returns:
            bool: True if vocabulary was processed, False otherwise
        """

        if self.vocabulary is not None:
            if isinstance(self.vocabulary, dict):
                return True

            elif isinstance(self.vocabulary, (list, tuple, set)):
                self.vocabulary = {key: value for value, key in enumerate(self.vocabulary)}

                return True

        return False

    def check_for_corpus(self, corpus: str | list | tuple | set=None) -> bool:
        """Process and validate text corpus inputs.

         Prioritizes pre_tokenized_corpus if available. Processes raw text by
         splitting into individual characters when needed.

         Args:
             corpus: Raw text input(s) to process

         Returns:
             bool: True if valid corpus data processed, False otherwise
         """

        if isinstance(self.pre_tokenized_corpus, (list, tuple, set)):
            self.pre_tokenized_corpus = list(self.pre_tokenized_corpus)
            return True

        elif isinstance(corpus, (list, tuple, set)):
            if not self.white_space:
                idx: int = 0

                while idx < len(corpus):
                    if ' ' in corpus[idx]:
                        corpus[idx] = corpus[idx].replace(' ', '')

                    idx += 1

            self.pre_tokenized_corpus = list(chain(*corpus))  # Flattens and splits raw text into chars
            return True

        elif isinstance(corpus, str):
            if not self.white_space:
                if ' ' in corpus:
                    corpus = corpus.split(' ')

            self.pre_tokenized_corpus = list(chain(*corpus))  # Flattens and splits raw text into chars
            return True

        return False

    def train(self) -> None:
        """Performs BPE training by iteratively merging frequent pairs."""
        if not self.vocabulary:
            raise Exception(
                'No corpus was given. Please include a corpus -> BytePairEncoder(corpus=str) or BytePairEncoder(pre_tokenized_corpus=iter)')

        vocab_size = len(self.vocabulary)
        merge_progressbar = tqdm(range(self.max_merges), desc="Creating BPE Merges")

        for _ in range(self.max_merges):
            # Count all consecutive token pairs
            pairs = Counter(pairwise(chain(self.pre_tokenized_corpus)))
            if not pairs:
                break

            # Get most frequent pair
            most_frequent_pair, frequency = pairs.most_common(1)[0]

            if frequency < self.min_frequency:
                break  # Early stopping

            # Create new token and update vocabulary
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            self.vocabulary[new_token] = vocab_size
            vocab_size += 1

            # Replace all occurrences of the pair in the corpus
            self.update_corpus(most_frequent_pair, new_token)
            merge_progressbar.update(1)

        merge_progressbar.close()

        self.build_tokenizer()

    def build_tokenizer(self) -> None:
        """Compiles tokenization regex, prioritizing longer tokens first."""
        tokens = sorted(self.vocabulary.keys(), key=lambda x: (-len(x), x))  # Length then lex order
        escaped = [re.escape(token) for token in tokens]
        self.regex = re.compile(f"{'|'.join(escaped)}")

    def tokenize(self, corpus: str | list | tuple | set = None) -> tuple:
        """Tokenizes input text using learned BPE merges.

        Args:
            corpus: String or iterable of strings to tokenize

        Returns:
            List of token lists (one per input string)
        """
        if isinstance(corpus, str):
            corpus = [corpus]

        return tuple(self.regex.findall(sentence) for sentence in corpus)

    def update_corpus(self, pattern_to_replace: tuple, replace_with: str) -> None:
        """Replaces all instances of a token pair with merged token (in-place)."""
        idx = 0
        while idx < len(self.pre_tokenized_corpus) - 1:
            current_token, next_token = self.pre_tokenized_corpus[idx], self.pre_tokenized_corpus[idx + 1]
            if current_token == pattern_to_replace[0] and next_token == pattern_to_replace[1]:
                self.pre_tokenized_corpus[idx] = replace_with
                self.pre_tokenized_corpus.pop(idx + 1)
            else:
                idx += 1

    def __str__(self) -> str:
        """User-friendly string representation of vocabulary."""
        return f"BytePairEncoder with {len(self.vocabulary)} tokens\nTokens: {list(self.vocabulary.keys())}" if self.vocabulary else "Untrained BytePairEncoder"

    def __repr__(self) -> str:
        """Technical string representation for debugging."""
        return (
            f"BytePairEncoder(\n"
            f"  token_count={len(self.vocabulary)},\n"
            f"  tokens={list(self.vocabulary.keys())[:10]}...\n"
            f"  max_merges={self.max_merges},\n"
            f"  min_frequency={self.min_frequency}\n"
            f"  pre_tokenized_corpus_length={len(self.pre_tokenized_corpus)}\n"
            f"  pre_tokenized_corpus={self.pre_tokenized_corpus[:10]}...\n)"
        )

    def __add__(self, other: Self | dict | list | tuple | set) -> Self:
        """Combines vocabularies of two encoders (token IDs from right encoder take precedence)."""
        if isinstance(other, BytePairEncoder):
            return BytePairEncoder(vocabulary={**self.vocabulary, **other.vocabulary},
                                   pre_tokenized_corpus=self.pre_tokenized_corpus,
                                   min_frequency=self.min_frequency,
                                   max_merges=self.max_merges)

        elif isinstance(other, dict):
            return BytePairEncoder(
                vocabulary={**self.vocabulary,
                            **{token: len(self.vocabulary) + i for i, token in other if token not in self.vocabulary}},
                pre_tokenized_corpus=self.pre_tokenized_corpus,
                min_frequency=self.min_frequency,
                max_merges=self.max_merges)

        elif isinstance(other, (dict, list, tuple, set)):
            return BytePairEncoder(
                vocabulary={**self.vocabulary,
                            **{token: len(self.vocabulary) + i for i, token in enumerate(other) if token not in self.vocabulary}},
                pre_tokenized_corpus=self.pre_tokenized_corpus,
                min_frequency=self.min_frequency,
                max_merges=self.max_merges)

        raise TypeError(f"Can only merge with iterable or BytePairEncoder, got {type(other)}")

    def __sub__(self, other: Self | dict | list | tuple | set) -> Self:
        """Creates new encoder with tokens present in self but not in other."""
        if isinstance(other, BytePairEncoder):
            return BytePairEncoder(vocabulary={k: v for k, v in self.vocabulary.items() if k not in other.vocabulary},
                                   pre_tokenized_corpus=self.pre_tokenized_corpus,
                                   min_frequency=self.min_frequency,
                                   max_merges=self.max_merges)

        elif isinstance(other, dict):
            return BytePairEncoder(
                vocabulary={key: value for key, value in self.vocabulary.items() if key not in other},
                pre_tokenized_corpus=self.pre_tokenized_corpus,
                min_frequency=self.min_frequency,
                max_merges=self.max_merges)

        elif isinstance(other, (dict, list, tuple, set)):
            return BytePairEncoder(
                vocabulary={key: value for key, value in self.vocabulary.items() if key not in other},
                pre_tokenized_corpus=self.pre_tokenized_corpus,
                min_frequency=self.min_frequency,
                max_merges=self.max_merges)

        raise TypeError(f"Can only subtract iterable or BytePairEncoder, got {type(other)}")

class WordPiece:
    """
    Implements the WordPiece tokenization algorithm.

    WordPiece is a subword tokenization method used in various NLP models, including BERT.
    It breaks down words into smaller subword units to balance vocabulary size and handling of unknown words.

    Attributes:
        tokenized_corpus (tuple[list[str]]): A tuple of lists, where each inner list represents
            a tokenized word from the training corpus.  Tokens are pre-split
            with '##' to indicate suffixation.  For example:
            (('t', '##h', '##e'), ('q', '##u', '##i', '##c', '##k'))
        vocabulary (dict[str, int]): A dictionary mapping tokens to their integer IDs.
        starting_words_count (int): The number of tokens in the vocabulary that
            do not start with '##', i.e., the number of unique starting subwords.

    Args:
        corpus (list[str]): A list of strings representing the training corpus.
        max_merges (int, optional): The maximum number of merge operations to perform
            during training. Defaults to 10.
    """
    def __init__(self, *, corpus: list[str], max_merges: int = 10) -> None:
        """
        Initializes the WordPiece tokenizer.

        The constructor performs the initial tokenization of the corpus, builds the initial vocabulary,
        and runs the WordPiece training loop to learn merge rules.

        Args:
            corpus (list[str]): A list of strings representing the training corpus.
            max_merges (int, optional): The maximum number of merge operations
                to perform. Defaults to 10.
        """
        # 1. Tokenize the corpus into individual characters, marking suffixes with '##'.
        #    For example, "the quick" becomes  (('t', '##h', '##e'), ('q', '##u', '##i', '##c', '##k'))
        self.tokenized_corpus = tuple(".##".join(tuple(word)).split('.') for word in corpus)

        # 2. Create the initial vocabulary from the unique tokens in the corpus.
        #    Assign each unique token an integer ID.
        self.vocabulary = {k:v for v,k in enumerate(set(chain.from_iterable(self.tokenized_corpus)))}

        # 3. Perform the WordPiece training loop to learn merge rules.
        self._training_loop(max_merges)

        # 4. Count the number of tokens that *do not* start with '##'. This is used
        #    to track the number of initial subwords (i.e., words that *start* a
        #    token sequence).
        self.starting_words_count = sum(1 for token in self.vocabulary.keys() if not token.startswith('##'))

    def _training_loop(self, merges: int) -> None:
        """
        Implements the WordPiece training loop.

        This method iteratively merges the most frequent character pairs in the corpus
        to build the vocabulary.  The merging process continues until the specified
        number of merges is reached.

        Args:
            merges (int): The maximum number of merge operations to perform.
        """
        for _ in range(merges):
            # 1. Count the occurrences of each token pair and each individual token.
            pair_counts = Counter()
            element_counts = Counter()

            for tokens in self.tokenized_corpus:
                element_counts.update(tokens)
                pair_counts.update(pairwise(tokens))

            # 2. If no more pairs, stop.
            if not pair_counts:
                break

            # 3. Calculate the "relevance" of each pair.  This is the probability
            #    of the pair occurring, given the individual token frequencies.
            pair_relevance = {
                pair: count / (element_counts[pair[0]] * element_counts[pair[1]])
                for pair, count in pair_counts.items()
            }

            # 4. Find the most relevant pair.
            best_pair = max(pair_relevance, key=pair_relevance.get)
            # 5. Create the new merged token.  If the second part of the pair
            #    starts with '##', remove the '##' to avoid double '##' in the
            #    merged token.
            new_token = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]

            # 6. Merge the best pair in all token sequences in the corpus.  This
            #    updates `self.tokenized_corpus` in place.
            for i, tokens in enumerate(self.tokenized_corpus):
                j = 0
                while j < len(tokens) - 1:
                    if tokens[j] == best_pair[0] and tokens[j + 1] == best_pair[1]:
                        tokens[j] = new_token  # Replace the pair with the new token
                        del tokens[j + 1]      # Remove the second token in the pair
                    else:
                        j += 1  # Move to the next token

            # 7. Add the new merged token to the vocabulary.
            self.vocabulary[new_token] = len(self.vocabulary)

    def _tokenize_single_text(self, text: str, unk_token: str='[UNK]') -> list[str]:
        """
        Tokenizes a single text/word using the learned WordPiece vocabulary.
        Internal helper method.

        This method implements the core WordPiece tokenization algorithm. It iteratively
        finds the longest matching subword in the vocabulary, starting from the
        beginning of the text.

        Args:
            text (str): The text string to tokenize.
            unk_token (str, optional): The token to use for unknown words.
                Defaults to '[UNK]'.

        Returns:
            list[str]: A list of token strings.

        Raises:
            TypeError: If the input `text` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        if not text:
            return []

        tokens = []
        current_index = 0
        text_len = len(text)

        while current_index < text_len:
            best_match_token = ""
            best_match_len = 0

            # Iterate through the text, finding the longest matching subword
            # in the vocabulary.
            for j in range(text_len, current_index, -1):
                sub_string = text[current_index:j]

                # Check for full word at the beginning, add '##' for subwords.
                if current_index == 0:
                    token_to_check = sub_string
                else:
                    token_to_check = "##" + sub_string

                if token_to_check in self.vocabulary:
                    best_match_token = token_to_check
                    best_match_len = len(sub_string)
                    break

            # If a match is found, add it to the tokens list and advance the index.
            if best_match_len > 0:
                tokens.append(best_match_token)
                current_index += best_match_len
            else:
                # If no match is found, add the unknown token and advance the index by 1.
                tokens.append(unk_token)
                current_index += 1

        return tokens

    def tokenize(self, texts: Union[str, list[str]], unk_token: str='[UNK]') -> list[list[str]]:
        """
        Tokenizes a single text string or a list of text strings using the learned WordPiece vocabulary.
        Always returns a list of lists of tokens.

        This is the main public method for tokenizing text using the WordPiece tokenizer.
        It handles both single strings and lists of strings, and ensures that the output
        is always a list of lists.

        Args:
            texts (str or list[str]): A single text string or a list of text strings to tokenize.
            unk_token (str, optional): The token to use for unknown words.
                Defaults to '[UNK]'.

        Returns:
            list[list[str]]: A list of lists of token strings.
                E.g., [["tok1", "tok2"], ["tok3"]].

        Raises:
            TypeError: If the input `texts` is not a string or a list of strings,
                       or if an item in the list is not a string.
        """
        if isinstance(texts, str):
            input_texts = [texts]  # If input is a single string, convert it to a list.
        elif isinstance(texts, list):
            input_texts = texts
        else:
            raise TypeError("Input must be a string or a list of strings.")

        all_tokenized_texts = []
        for text_item in input_texts:
            # _tokenize_single_text will raise TypeError if text_item is not a string
            tokenized_text = self._tokenize_single_text(text_item, unk_token)
            all_tokenized_texts.append(tokenized_text)

        return all_tokenized_texts

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
        self.d_model = None

        self.embeddings = embeddings
        self.vocabulary = vocabulary
        self.reversedVocabulary = dict(zip(vocabulary.values(), vocabulary.keys())) if type(vocabulary) is None else None

    # Method to calculate the embedding matrix
    def calculate(self, tokenized_dataset=None, *, window_size=1,
                  pmi=True, smoothing: float = 3e-4, from_zero=False,
                  d_model=128):

        # Step 0: Save d_model variable
        self.d_model = d_model

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

    def embed_sequence(self, tokenized_sequences: list | tuple | set, *, context: int = 0):
        # Tokenizes input sentence into smaller chunks (tokens),
        # retrieves embeddings for each token from self.embeddings variable,
        # and returns a list of embeddings.
        embed = [self.embeddings[token] for sequence in tokenized_sequences for token in sequence]

        if context > 0:
            embed += [[0] * self.d_model] * (context - len(embed))

        return embed

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
