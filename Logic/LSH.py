import numpy as np
import itertools
import random
import hashlib
import json


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.characteristic_matrix = None
        self.signature_matrix = None
        self.shingle_to_idx = None
        self.buckets = None
        

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        words = document.split()
        
        if len(words) < k:
            if words:
                shingles.add(" ".join(words))
            return shingles
        
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i:i+k])
            shingles.add(shingle)
            
        return shingles
        

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        all_shingles = set()
        doc_shingles = []
        
        for doc in self.documents:
            shingles = self.shingle_document(doc)
            doc_shingles.append(shingles)
            all_shingles.update(shingles)
            
        all_shingles = list(all_shingles)
        shingle_to_idx = {shingle: idx for idx, shingle in enumerate(all_shingles)}
        matrix = np.zeros((len(all_shingles), len(self.documents)), dtype=bool)
        
        for doc_idx, shingles in enumerate(doc_shingles):
            for shingle in shingles:
                shingle_idx = shingle_to_idx[shingle]
                matrix[shingle_idx, doc_idx] = True
                
        self.characteristic_matrix = matrix
        self.shingle_to_idx = shingle_to_idx
        return matrix
        
        

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        if self.characteristic_matrix is None:
            self.build_characteristic_matrix()
            
        num_shingles, num_docs = self.characteristic_matrix.shape
        matrix = np.full((self.num_hashes, num_docs), np.inf)
        
        # permutation: h(x) = (a * x + b) % p
        p = self._next_prime(num_shingles)
        a_s = np.random.randint(1, p, size=self.num_hashes).reshape(-1, 1)
        b_s = np.random.randint(0, p, size=self.num_hashes).reshape(-1, 1)
        
        for doc_idx in range(num_docs):
            true_shingle_indices = np.where(self.characteristic_matrix[:, doc_idx] == True)[0]
            if len(true_shingle_indices) == 0:
                continue
            true_shingle_indices = true_shingle_indices.reshape(1, -1)
            
            hashes = (a_s * true_shingle_indices + b_s) % p
            min_hashes = np.min(hashes, axis=1)
            matrix[:, doc_idx] = min_hashes
            
        self.signature_matrix = matrix
        return matrix
        

    def lsh_buckets(self, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        if self.signature_matrix is None:
            self.min_hash_signature()
        num_docs = self.signature_matrix.shape[1]
        buckets = {}
        
        for band_idx in range(bands):
            from_row = band_idx * rows_per_band
            to_row = min(from_row + rows_per_band, self.num_hashes)
            
            for doc_idx in range(num_docs):
                band_segment = self.signature_matrix[from_row:to_row, doc_idx]
                band_tuple = tuple(int(value) if value != np.inf else -1 for value in band_segment)
                bucket_id = hash((band_idx, band_tuple))
                
                if bucket_id not in buckets.keys():
                    buckets[bucket_id] = []
                buckets[bucket_id].append(doc_idx)
                    
        self.buckets = buckets
        return buckets
        

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_bands = 25
        self.min_hash_signature()
        ans = self.lsh_buckets(num_bands, self.num_hashes//num_bands)
        return ans


    def _jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        if 0 in [len(first_set), len(second_set)]:
            return 1.0
        
        intersections = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        if union == 0:
            return 0.0
        
        return intersections / union
        

    def jaccard_similarity_test(self):
        """
        Test your near duplicate detection code based on jaccard similarity.
        """
        if self.buckets is None:
            self.lsh_buckets()
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in self.buckets.keys():
            docs_in_this_bucket = self.buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(self.documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(self.documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self._jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(self.documents) - 1)
                        random_shingled_doc = self.shingle_document(self.documents[random_doc_id], 2)

                        random_jaccard_score = self._jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)
        
    
    def _next_prime(self, n):
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        prime = n + 1
        while not is_prime(prime):
            prime += 1
        return prime



def main():
    with open("./Logic/LSHFakeData.json", "r") as f:
        full_docs = json.load(f)
        text_documents = [" ".join(doc.get("descriptions", [])) for doc in full_docs]

    lsh = MinHashLSH(text_documents, 100)
    buckets = lsh.perform_lsh()
    print("buckets: ", [value if len(value) > 1 else "" for value in buckets.values()])
    lsh.jaccard_similarity_test()


    
if __name__ == '__main__':
    main()
