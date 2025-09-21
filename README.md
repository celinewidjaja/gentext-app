# GenText FastAPI Application

This application combines text generation using bigram models with word embeddings using Spacy.

## Running the Application

### Docker Deployment
```bash
docker build -t gentext-app .
docker run -p 8000:80 gentext-app
