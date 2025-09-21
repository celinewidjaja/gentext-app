import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class WordEmbeddingModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
    
    def calculate_embedding(self, input_word):
        """Calculate word embeddings based on input word"""
        word = self.nlp(input_word)
        return word.vector
    
    def calculate_similarity(self, word1, word2):
        """Compute similarity between two words based on their embeddings"""
        similarity = self.nlp(word1).similarity(self.nlp(word2))
        return float(similarity)  # Convert to Python float
    
    def calculate_query_similarities(self, query, responses):
        """Calculate similarity between a query and multiple responses"""
        similarities = []
        for i, response in enumerate(responses):
            similarity = self.nlp(query).similarity(self.nlp(response))
            similarities.append({
                f"response_{i+1}": response,
                "similarity": float(similarity)  # Convert to Python float
            })
        return similarities
    
    def calculate_cosine_similarity(self, word1, word2):
        """Calculate cosine similarity using sklearn"""
        embedding1 = self.calculate_embedding(word1)
        embedding2 = self.calculate_embedding(word2)
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return float(similarity)  # Convert to Python float
