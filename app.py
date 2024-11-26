from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from sentence_transformers import SentenceTransformer
import spacy
import pandas as pd
from collections import Counter

# Initialize FastAPI app
app = FastAPI()

# Load Pre-trained K-Means Model and Embedding Model
try:
    with open('./models/model.pkl', 'rb') as file:
        kmeans = pickle.load(file)
    
    df = pd.read_parquet('./models/df.parquet')
    
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

custom_stopwords = [
    'experience', 'work', 'opportunity', 'strong', 'ability', 'include', 'responsible', 'skill', 'knowledge', 'ability', 
    'company', 'good', 'world', 'year', 'understand'
]

# Define input data model
class TextInput(BaseModel):
    text: str

# Preprocessing function
def preprocess_text(text: str) -> str:
    text = ' '.join(text.split()).lower()
    text = ' '.join([word for word in text.split() if word.isalpha()])
    doc = nlp(text)
    text = ' '.join(token.lemma_ for token in doc if not token.is_stop)
    text = ' '.join([word for word in text.split() if word not in custom_stopwords])
    return text

def keywords_and_teams(cluster_df):
    # join all the text of the cleaned text into one string
    combined_string = " ".join([sentence for sentence in cluster_df['cleaned_text'].values])
    word_counter = Counter(combined_string.split())
    most_common_words = word_counter.most_common(20)
    top_teams = cluster_df['Team'].value_counts(normalize=True)[0:3].to_dict()

    return most_common_words, top_teams

# Endpoint to predict cluster
@app.post("/predict/")
async def predict_cluster(input: TextInput):
    try:
        # Preprocess input text
        cleaned_text = preprocess_text(input.text)

        # Generate embeddings
        embedding = model.encode([cleaned_text])
        print(embedding)
        
        # Predict the cluster
        cluster = kmeans.predict(embedding)[0]
        cluster_df = df[df['cluster'] == cluster]

        keywords, teams = keywords_and_teams(cluster_df)

        keywords = [keyword[0] for keyword in keywords]

        print(keywords, list(teams.keys()))

        return {
            "cluster_id": int(cluster),
            "top_keywords": keywords,
            "predicted_teams": teams
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")
