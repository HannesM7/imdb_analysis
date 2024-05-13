import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import HDBSCAN
from bertopic import BERTopic
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Download NLTK resources if not already downloaded
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocess stopwords
stop_words = set(stopwords.words('english'))
specific_words = {"br", "positive", "negative"}
stop_words.update(specific_words)

# Define function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# Read data from CSV file
csv_file = r'C:\Users\A1D5688\Desktop\imdb_analysis\IMDB Dataset.csv'
df = pd.read_csv(csv_file, sep=";;;;;;")

# Assume the column name containing text data is 'text'
texts = df["review,sentiment"].tolist()

# Preprocess documents
preprocessed_documents = [preprocess_text(text) for text in texts]

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the vectorizer
X = vectorizer.fit_transform(preprocessed_documents)

# Initialize the HDBSCAN model
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='cosine', cluster_selection_method='eom')

# Fit the HDBSCAN model
hdbscan_model.fit(X.toarray())

# Create a BERTopic model using the HDBSCAN clusters
model = BERTopic(language='english', verbose=True, calculate_probabilities=True, random_state=42)

# Fit the BERTopic model using the HDBSCAN clusters
model.fit_transform(preprocessed_documents, hdbscan_model.labels_)

# Display topics
topics = model.get_topics()
for topic_index, topic in enumerate(topics):
    print(f"Topic {topic_index}:")
    word_weights = [(model.get_topic_words(topic_index)[word_index][0], model.get_topic_words(topic_index)[word_index][1]) for word_index in topic]
    words = [(word, weight) for word, weight in word_weights]
    sorted_words = sorted(words, key=lambda x: x[1], reverse=True)
    for word, weight in sorted_words[:10]:
        print(f"\t{word}: {weight}")
    print()

# Visualize the topics using PyLDAvis
vis_data = gensimvis.prepare(model, preprocessed_documents)
pyLDAvis.display(vis_data)