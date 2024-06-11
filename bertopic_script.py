# Import necessary libraries
import spacy
import numpy as np
import pandas as pd
from bertopic import BERTopic

# Load English tokenizer
nlp = spacy.load("en_core_web_sm")

# Load dataset
data = pd.read_csv("IMDB Dataset.csv")

# Preprocess data
def preprocess(text):
    # Tokenize text
    doc = nlp(text)

    # Remove stopwords and punctuation
    tokens = [token.lemma_.lower().strip() for token in doc if token.is_alpha and not token.is_stop]

    # Return cleaned tokens
    return tokens


data["clean_text"] = data["review"].apply(preprocess)

# Convert list of lists to list of strings
data["clean_text"] = data["clean_text"].apply(lambda x: " ".join(x))

# Create BERTopic model
model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    verbose=True,
)

# Generate topics
topics, probs = model.fit_transform(data["clean_text"])

# Print topics
for topic in model.get_topics():
    print(topic)