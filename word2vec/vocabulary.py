from collections import defaultdict
from typing import Dict, Iterable, List


from word2vec.preprocess import tokenize


class Vocabulary:
    """
    Vocabulary of a text corpus.
    """
    _unknown_word = "<UNKNOWN_WORD>"

    def __init__(self):
        self._word_index: Dict[str, int] = {}
        self._word_count: Dict[str, int] = defaultdict(int)

    def __len__(self) -> int:
        """
        Return the length of the vocabulary.
        """
        return len(self._word_index)

    def _fit_word_count(self, tokenized_texts: Iterable[List[str]]):
        """
        Fit the word count dictionary.

        Parameters
        ----------
        tokenized_texts: sequence of tokenized texts
        """
        for tokens in tokenized_texts:
            for token in tokens:
                self._word_count[token] += 1

    def _fit_word_index(self):
        """
        Fit the word index dictionary.

        The word indices are ordered according to word frequency,
        and the index 0 is reserved for unknown words.
        """
        sorted_word_count = sorted(self._word_count.items(), key=lambda x: x[1], reverse=True)
        self._word_index[self._unknown_word] = 0
        for index, item in enumerate(sorted_word_count, start=1):
            self._word_index[item[0]] = index

    def _transform_tokenized_texts(self, tokenized_texts: List[List[str]]) -> List[List[int]]:
        """
        Transform tokenized texts into sequences of integers.

        Parameters
        ----------
        tokenized_texts: sequence of tokenized texts

        Returns
        -------
        index_sequences: lists of word indices
        """
        return [[self.word_index(token) for token in tokens] for tokens in tokenized_texts]

    def fit(self, texts: Iterable[str]):
        """
        Create a dictionary for a text corpus.

        Parameters
        ----------
        texts: sequence of strings
        """
        tokenized_texts = [tokenize(text) for text in texts]
        self._fit_word_count(tokenized_texts)
        self._fit_word_index()

    def transform(self, texts: Iterable[str]) -> List[List[int]]:
        """
        Transform a collection of texts into sequences of integers.

        Parameters
        ----------
        texts: sequence of strings

        Returns
        -------
        index_sequences: lists of word indices
        """
        tokenized_texts = [tokenize(text) for text in texts]
        return self._transform_tokenized_texts(tokenized_texts)

    def fit_transform(self, texts: Iterable[str]) -> List[List[int]]:
        """
        Create a dictionary for a text corpus and transform the texts into sequences of integers.

        Parameters
        ----------
        texts: sequence of strings

        Returns
        -------
        index_sequences: lists of word indices
        """
        tokenized_texts = [tokenize(text) for text in texts]
        self._fit_word_count(tokenized_texts)
        self._fit_word_index()
        return self._transform_tokenized_texts(tokenized_texts)

    def word_index(self, word: str) -> int:
        """
        Get the index of a word in the vocabulary.

        Returns 0 if the word is unknown.
        """
        return self._word_index.get(word, 0)

    def word_count(self, word: str) -> int:
        """
        Get the frequency of a word in the vocabulary.

        Returns 0 if the word is unknown.
        """
        return self._word_count.get(word, 0)
