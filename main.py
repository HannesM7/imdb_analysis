import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap

# Sample data
data = {
    'text': [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
        "This is another document.",
        "Yet another document."
    ]
}

# Create a pandas DataFrame
csv_file = r'C:\Users\hmuen\PycharmProjects\imdb_analysis\data\IMDB Dataset.csv'
df = pd.read_csv(csv_file)

# Preprocess the documents
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])

# Perform truncated SVD
svd = TruncatedSVD(n_components=10)
decomposed_matrix = svd.fit_transform(X)

# Get topics
terms = vectorizer.get_feature_names_out()
topic_weights = svd.components_
topics = []
for i, topic_weights_ in enumerate(topic_weights):
    top_terms = [terms[idx] for idx in topic_weights_.argsort()[:-10:-1]]
    topics.append(top_terms)

# Print topics
for i, topic in enumerate(topics):
    print(f"Topic {i+1}:")
    print(", ".join(topic))

# Perform UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(decomposed_matrix)

# Plot the documents
plt.figure(figsize=(10, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], s=50)
for i, doc in enumerate(df['text']):
    plt.annotate(doc, (embedding[i, 0], embedding[i, 1]))
plt.title("Document Embedding")
plt.show()