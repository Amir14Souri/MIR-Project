import string
from collections import deque
from typing import Callable, List, Tuple, Dict

class Snippet:
    """
    A class to generate relevant text snippets from documents based on a search query.
    
    It uses a single-pass sliding window with a 'lag' mechanism to ensure keywords
    at the very beginning and very end of documents are treated as potential centers.
    """

    def __init__(self, normalize_function: Callable, remove_stopword_function: Callable, number_of_words_on_each_side: int = 5):
        """
        Initialize the Snippet generator.

        Args:
            normalize_function (Callable): A function that takes a word and returns its stemmed/normalized version.
            remove_stopword_function (Callable): A function that takes a query string and returns a list of filtered tokens.
            number_of_words_on_each_side (int): Number of words to include to the left and right of a keyword.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.normalize = normalize_function
        self.remove_stopword = remove_stopword_function
        self.win_size = (2 * number_of_words_on_each_side) + 1


    def find_snippet(self, raw_doc: str, query: str) -> Tuple[str, List[str]]:
        """
        Main orchestrator for snippet generation.

        Parameters:
            raw_doc (str): The original document string.
            query (str): The user's search query string.

        Returns:
            final_snippet (str): The formatted snippet with '***' highlighting and '...' separators.
            not_exist_words (list): The list of words from the query that were not found in the document.
        """
        doc_tokens = raw_doc.split()
        normalized_cache = [self.normalize(token) for token in doc_tokens]
        doc_normalized_set = set(normalized_cache)
        
        query_tokens = self.remove_stopword(query)
        query_set = set()
        not_exist_words = []
        for token in query_tokens:
            normalized = self.normalize(token)
            if normalized:
                query_set.add(normalized)
                if normalized not in doc_normalized_set:
                    not_exist_words.append(token)
                    
        windows = self._identify_best_windows(doc_tokens, normalized_cache, query_set)
        merged_windows = self._merge_windows(windows)
        final_snippet = self._create_snippet_text(doc_tokens, normalized_cache, merged_windows, query_set)
        
        return final_snippet, not_exist_words


    def _identify_best_windows(self, doc_tokens: list, normalized_cache: list, query_set: set) -> List[Tuple[int, int]]:
        """
        Uses a sliding window to score the 'density' of query matches.
        
        Parameters:
            doc_tokens (list): List of original words from the document.
            normalized_cache (list): List of the same words, but normalized/stemmed.
            query_set (set): Set of normalized query stems.

        Returns:
            list: A list of (start_index, end_index) for the best windows found.
        """
        n = len(doc_tokens)
        if n == 0:
            return []
        
        padded_tokens = [''] * self.number_of_words_on_each_side + doc_tokens + [''] * self.number_of_words_on_each_side
        padded_normalized = [''] * self.number_of_words_on_each_side + normalized_cache + [''] * self.number_of_words_on_each_side
        
        best_windows = []
        used_centers = set()
        
        for i in range(self.number_of_words_on_each_side, len(padded_tokens) - self.number_of_words_on_each_side):
            if padded_normalized[i] in query_set and padded_normalized[i] not in used_centers:
                start = max(0, i - self.number_of_words_on_each_side)
                end = min(len(padded_tokens) - 1, i + self.number_of_words_on_each_side)
                
                score = sum(1 for j in range(start, end + 1) if padded_normalized[j] in query_set)
                orig_start = start - self.number_of_words_on_each_side
                orig_end = end - self.number_of_words_on_each_side
                
                best_windows.append((orig_start, orig_end))
                used_centers.add(padded_normalized[i])
        
        return best_windows
        

    def _merge_windows(self, windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Combines window ranges that overlap or touch.
        
        Parameters:
            windows (list): List of (start, end) index tuples.

        Returns:
            list: List of merged (start, end) index tuples.
        """
        if not windows:
            return []
        
        sorted_windows = sorted(windows, key=lambda x: x[0])
        merged = [sorted_windows[0]]
        
        for current_start, current_end in sorted_windows[1:]:
            last_start, last_end = merged[-1]
            
            if current_start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged


    def _create_snippet_text(self, doc_tokens: list, normalized_cache: list, 
                             merged_windows: List[Tuple[int, int]], query_set: set) -> str:
        """
        Constructs the final formatted snippet string.
        
        Parameters:
            doc_tokens (list): Original document tokens.
            normalized_cache (list): Stemmed document tokens.
            merged_windows (list): Merged (start, end) indices.
            query_set (set): Normalized query stems.

        Returns:
            str: The final snippet with highlights and ellipses. 
                example: "The ***wizard*** went to ***Hogwarts.*** The ***wizard*** loved magic."

        """
        if not merged_windows:
            return " ... "
        
        snippet_parts = []
        last_end = -2
        
        for start, end in merged_windows:
            start = max(0, start)
            end = min(len(doc_tokens) - 1, end)
            
            if snippet_parts and start > last_end + 1:
                snippet_parts.append("...")
            
            window_tokens = []
            for i in range(start, end + 1):
                token = doc_tokens[i]
                if normalized_cache[i] in query_set:
                    window_tokens.append(f"***{token}***")
                else:
                    window_tokens.append(token)
            
            snippet_parts.append(" ".join(window_tokens))
            last_end = end
        
        return " ".join(snippet_parts)