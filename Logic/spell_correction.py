import pickle
import os

class SpellCorrection:
    def __init__(self, all_documents=None, load_path=None, save_path=None):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str, optional
            The input documents used to build the vocabulary.
        load_path : str, optional
            Path to load precomputed data from.
        save_path : str, optional
            Path to save computed data to.
        """
        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = {}

    def k_gram_word(self, word, k=2):
        """
        Convert a word into a set of k-grams.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each k-gram.

        Returns
        -------
        set
            A set of k-grams.
        """
        #TODO
        pass

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of k-grams.
        second_set : set
            Second set of k-grams.

        Returns
        -------
        float
            Jaccard score.
        """
        #TODO
        pass

    def k_gramming_and_counting(self, all_documents):
        """
        k-grams all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_k_gram_words : dict
            A dictionary from words to their k-grams sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        #TODO
        pass

    def save(self, path):
        """
        Save the k-grams data and word counter to a file.
        """
        data = {
            'all_k_gram_words': self.all_k_gram_words,
            'word_counter': self.word_counter
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        """
        Load the shingle data and word counter from a file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.all_k_gram_words = data['all_k_gram_words']
            self.word_counter = data['word_counter']

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        #TODO
        pass

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        #TODO
        pass