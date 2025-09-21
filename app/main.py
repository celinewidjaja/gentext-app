from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel
from .bigram_model import BigramModel
from .word_embeddings import WordEmbeddingModel

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "this is another example sentence", 
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = WordEmbeddingModel()

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class WordEmbeddingRequest(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

class QuerySimilarityRequest(BaseModel):
    query: str
    responses: List[str]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_word_embedding(request: WordEmbeddingRequest):
    embedding = embedding_model.calculate_embedding(request.word)
    return {"word": request.word, "embedding": embedding[:10].tolist()}

@app.post("/similarity")
def calculate_word_similarity(request: SimilarityRequest):
    similarity = embedding_model.calculate_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": similarity}

@app.post("/query-similarity")
def calculate_query_similarity(request: QuerySimilarityRequest):
    similarities = embedding_model.calculate_query_similarities(request.query, request.responses)
    return {"query": request.query, "similarities": similarities}

@app.post("/cosine-similarity")
def calculate_cosine_similarity(request: SimilarityRequest):
    similarity = embedding_model.calculate_cosine_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "cosine_similarity": similarity}
