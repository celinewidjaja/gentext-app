import random
from collections import defaultdict

class BigramModel:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bigrams = defaultdict(list)
        self._build_model()
    
    def _build_model(self):
        for sentence in self.corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                self.bigrams[words[i]].append(words[i + 1])
    
    def generate_text(self, start_word, length):
        if start_word.lower() not in self.bigrams:
            return f"Start word '{start_word}' not found in corpus"
        
        result = [start_word.lower()]
        current_word = start_word.lower()
        
        for _ in range(length - 1):
            if current_word in self.bigrams and self.bigrams[current_word]:
                next_word = random.choice(self.bigrams[current_word])
                result.append(next_word)
                current_word = next_word
            else:
                break
        
        return " ".join(result)
