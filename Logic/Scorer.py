import math
from collections import Counter


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The inverted index with structure {term: {document_id: tf}}.
        number_of_documents : int
            The number of documents in the collection.
        """
        self.index = index
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.
        """
        docs = set()
        for term in query.split():
            if term in self.index:
                docs.update(self.index[term].keys())
        return docs
    

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.
        """
        if term not in self.idf:
            df = len(self.index.get(term, {}))
            if df == 0:
                self.idf[term] = 0
            else:
                self.idf[term] = math.log10(self.N / df)
        return self.idf[term]
        

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.
        """
        query_terms = query.split()
        return dict(Counter(query_terms))
        

    def compute_scores_with_vector_space_model(self, query, method):
        """
        Compute scores with vector space model.
        """
        #TODO
        pass


    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.
        """
        #TODO
        pass


    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25.
        """
        #TODO
        pass

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.
        """
        #TODO
        pass

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates scores for each document based on the unigram model.
        """
        #TODO
        pass

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the unigram score of a document for a query.
        """
        #TODO
        pass

    def _apply_tf(self, tf, mode):
        """
        Apply term frequency (tf) weighting based on the specified mode.
        mode (str): Weighting scheme:
            - 'n'
            - 'l'

        """
        if mode == "l":
            return 1 + math.log10(tf) if tf > 0 else 0
        return tf
        

    def _cosine_normalize(self, weights):
        """
        Normalize a vector of term weights using cosine normalization.
        """
        norm = math.sqrt(sum(w ** 2 for w in weights.values()))
        if norm == 0:
            return weights
        return {term: w / norm for term, w in weights.items()}
        

    def _prepare_collection_stats(self):
        """
        Compute and cache collection-wide statistics for the index.
        """
        if self._collection_frequencies is not None:
            return
        
        self._collection_frequencies = {}
        self._collection_length = 0
        
        for term, postings in self.index.items():
            total_tf = sum(postings.values())
            self._collection_frequencies[term] = total_tf
            self._collection_length += total_tf