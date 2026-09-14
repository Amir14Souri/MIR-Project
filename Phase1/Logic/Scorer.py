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
        doc_method, query_method = method.split(".")
        query_tfs = self.get_query_tfs(query)
        docs = self.get_list_of_documents(query)
        scores = {}
        
        for doc_id in docs:
            score = self.get_vector_space_model_score(
                query, query_tfs, doc_id, doc_method, query_method
            )
            scores[doc_id] = score
            
        return scores


    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.
        """
        doc_weights = {}
        for term in query_tfs:
            if term in self.index and document_id in self.index[term]:
                tf = self.index[term][document_id]
                tf = self._apply_tf(tf, document_method[0])
                if document_method[1] == "t":
                    tf *= self.get_idf(term)
                doc_weights[term] = tf
        if document_method[2] == "c":
            doc_weights = self._cosine_normalize(doc_weights)
            
        query_weights = {}
        for term, tf in query_tfs.items():
            tf = self._apply_tf(tf, document_method[0])
            if document_method[1] == "t":
                tf *= self.get_idf(term)
            query_weights[term] = tf
        if document_method[2] == "c":
            query_weights = self._cosine_normalize(query_weights)
            
        score = sum(doc_weights.get(term, 0) * query_weights.get(term, 0) for term in query_weights)
        return score


    def compute_scores_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25.
        """
        docs = self.get_list_of_documents(query)
        scores = {}
        
        for doc_id in docs:
            score = self.get_okapi_bm25_score(
                query, doc_id, average_document_field_length, document_lengths
            )
            scores[doc_id] = score
            
        return scores
        

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.
        """
        k1 = 1.5
        b = 0.75
        score = 0
        
        doc_len = document_lengths.get(document_id, 0)
        
        for term in query.split():
            if term in self.index and document_id in self.index[term]:
                tf = self.index[term][document_id]
                df = len(self.index[term])
                idf = self.get_idf(term)
                numerator = tf * (k1 + 1)
                denumerator = tf + k1 * (1 - b + b * (doc_len / average_document_field_length))
                
                score += idf * (numerator / denumerator)
        
        return score
    

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates scores for each document based on the unigram model.
        """
        self._prepare_collection_stats()
        docs = self.get_list_of_documents(query)
        scores = {}
        
        for doc_id in docs:
            score = self.compute_score_with_unigram_model(
                query, doc_id, smoothing_method, document_lengths, alpha, lamda
            )
            scores[doc_id] = score
            
        return scores


    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the unigram score of a document for a query.
        """
        score = 1.0
        doc_len = document_lengths.get(document_id, 1) if document_lengths else 1
        
        for term in query.split():
            tf = self.index.get(term, {}).get(document_id, 0)
            p_doc = tf / doc_len if doc_len > 0 else 0
            
            collection_tf = self._collection_frequencies.get(term, 0)
            p_collection = collection_tf / self._collection_length if self._collection_length > 0 else 0
            
            if smoothing_method == "naive":
                p = p_doc if p_doc > 0 else p_collection
            elif smoothing_method == "bayes":
                mu = alpha * self._collection_length  # Dirichlet
                p = (tf + mu * p_collection) / (doc_len + mu)
            elif smoothing_method == "mixture":
                p = lamda * p_doc + (1 - lamda) * p_collection
                
            score *= p
            
        return scores
        

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