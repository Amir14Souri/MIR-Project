import re
import string
import json
import csv



class Preprocessor:
    def __init__(self, custom_stopwords_path='./Logic/stopwords.txt'):
        """
        Initialize the preprocessor, compile patterns, load components, etc.
        """
        pattern = r'\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*'
        #TODO



    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to a single text document.
        """
        #TODO

    def remove_stopwords(self, text: str) -> list:
        """
        Remove stopwords from the text.
        """
        
    
    def normalize(self, word: str) -> str:
        """
        Normalize the text by stemming, lemmatization, etc.

        Parameters
        ----------
        word : str
            The word to be normalized.

        Returns
        ----------
        list
            The normalized word.
        """

        #TODO

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing pipeline to a list of documents.
        """
        #TODO
    


def preprocess_docs(docs: list):
    """
    Apply preprocessing to specific fields in a list of documents in-place.
    
    Args:
        docs (list): List of document dictionaries to preprocess
        
    Returns:
        None: Modifies the input list in-place
    
    Notes:
        Preprocesses the following fields: title, description, author
        Handles both string and list field types
    """
    pass


def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format with specific field mapping.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        json_file_path (str): Path where the output JSON file will be saved
        
    Returns:
        None: Writes output directly to JSON file
    
    Notes:
        Maps CSV fields to JSON structure including:
        - id (from bookId)
        - title, author, description
        - genres, characters, languages (split by commas)
        - publish_date, num_pages, avg_rating
    """
    pass


if __name__ == '__main__':


    csv_to_json('top_3000_rated_books.csv','crawled.json')

    
    json_file_path = 'crawled.json'
    with open(json_file_path, "r") as file:
        docs = json.load(file)

    preprocess_docs(docs)

    with open('preprocessed.json', "w") as file:
        file.write(json.dumps(docs))
