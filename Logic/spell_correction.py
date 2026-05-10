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
        word = f"${word}$"
        if len(word) < k:
            return set([word])
        return set([word[i:i+k] for i in range(len(word) - k + 1)])


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
        if len(first_set) == 0 and len(second_set) == 0:
            return 1.0
        
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        if union == 0:
            return 0.0
        
        return intersection / union
        

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
        all_k_gram_words = {}
        word_counter = {}
        
        for doc in all_documents:
            words = doc.split()
            for word in words:
                if word not in word_counter:
                    word_counter[word] = 0
                    all_k_gram_words[word] = self.k_gram_word(word)
                word_counter[word] += 1
        
        return all_k_gram_words, word_counter

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
        word_k_grams = self.k_gram_word(word)
        
        candidates = []
        for word, k_grams in self.all_k_gram_words.items():
            score = self.jaccard_score(word_k_grams, k_grams)
            if score > 0.4:
                candidates.append((word, score, self.word_counter.get(word, 0)))
            
        candidates.sort(key=lambda x: (x[1], x[2]))
        return [candidate[0] for candidate in candidates[:5]]
        

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
        corrected_words = []
        
        for word in query.split():
            if word in self.word_counter:
                corrected_words.append(word)
            else:
                candidates = self.find_nearest_words(word)
                if candidates:
                    corrected_words.append(candidates[0])
                else:
                    corrected_words.append(word)
                    
        return " ".join(corrected_words)
    
    

def test():
    import json
    
    with open("./Logic/LSHFakeData.json", "r") as f:
        full_docs = json.load(f)
        text_documents = [" ".join(doc.get("descriptions", [])) for doc in full_docs]

    spell_correction = SpellCorrection(text_documents)
    corrected = spell_correction.spell_check(input("Type your query:\n"))
    print(f'Did you mean: "{corrected}"?')
    
    

if __name__ == "__main__":
    test()