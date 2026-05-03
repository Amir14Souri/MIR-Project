import re
import string
import json
import csv
from nltk.stem import PorterStemmer, WordNetLemmatizer


class Preprocessor:
    def __init__(self, custom_stopwords_path='./Logic/stopwords.txt'):
        """
        Initialize the preprocessor, compile patterns, load components, etc.
        """
        pattern = r'\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*'
        self.url_email_pattern = re.compile(pattern, re.IGNORECASE)
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        self.stopwords = set()
        with open(custom_stopwords_path, "r") as f:
            self.stopwords = set(f.readlines().split("\n").strip().lower())



    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to a single text document.
        """
        text = self.url_email_pattern.sub(" ", text)      # Remove URLs and emails
        text = text.lower()                               # Case Folding
        text = self.punctuation_pattern.sub(" ", text)    # Remove punctuations
        
        tokens = text.split()
        tokens = [self.normalize(token) for token in tokens]
        return "".join(tokens)


    def remove_stopwords(self, text: str) -> list:
        """
        Remove stopwords from the text.
        """
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token not in self.stopwords]
        return " ".join(filtered_tokens)
        
    
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
        return self.lemmatizer.lemmatize(word)
        

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing pipeline to a list of documents.
        """
        processed_docs = []
        for doc in documents:
            processed_docs.append(self.preprocess_text(doc))
            


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
    preprocessor = Preprocessor()
    keys = ["title", "description", "author"]
    
    for key in keys:
        for doc in docs:
            if key in doc:
                if isinstance(doc[key], str):
                    doc[key] = preprocessor.preprocess_text(doc[key])
                elif isinstance(doc[key], list):
                    doc[key] = [preprocessor.preprocess_text(item) for item in doc[key]]



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
    books = []
    
    with open(csv_file_path, "r") as csv:
        reader = csv.DictReader(csv)
        
        for row in reader:
            book = {
                "id": row.get("bookId", ""),
                "title": row.get("title", ""),
                "author": row.get("author", ""),
                "description": row.get("description", ""),
                "genres": [g.strip() for g in row.get("genres", "").split(",")],
                "characters": [c.strip() for c in row.get("characters", "").split(",")],
                "languages": [l.strip() for l in row.get("languages", "").split(",")],
                "publish_date": row.get("publish_date", ""),
                "num_pages": int(row.get("num_pages")) if row.get("num_pages", "").isdigit() else 0,
                "avg_rating": float(row.get("avg_rating")) if row.get("avg_rating", "") else 0.0,
            }
            books.append(book)
        
        with open(json_file_path, "w") as json:
            json.dump(books, json, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    csv_to_json('top_3000_rated_books.csv','crawled.json')

    json_file_path = 'crawled.json'
    with open(json_file_path, "r") as file:
        docs = json.load(file)

    preprocess_docs(docs)
    with open('preprocessed.json', "w") as file:
        file.write(json.dumps(docs))
