import pandas as pd
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources if not already downloaded
#import nltk

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

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

    return tokens


# Read data from CSV file
csv_file = r'C:\Users\hmuen\PycharmProjects\imdb_analysis\data\imdb_dataset.csv'
df = pd.read_csv(csv_file, sep=";;;;;;")

# Assume the column name containing text data is 'text'
texts = df["review,sentiment"].tolist()

# Preprocess documents
preprocessed_documents = [preprocess_text(text) for text in texts]

# Create dictionary and corpus
dictionary = corpora.Dictionary(preprocessed_documents)
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]

# Train LDA model
num_topics = 5  # You can adjust the number of topics as per your requirement
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print topics
for topic in lda_model.print_topics():
    print(topic)
