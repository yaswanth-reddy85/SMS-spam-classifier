import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
import pickle
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Read the dataset
dataset = pd.read_csv('spam (1).csv', encoding='latin-1')
sent = dataset.iloc[:, [1]]['v2']

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess the text
sentences = []
for text in sent:
    # Remove special characters and convert to lowercase
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    
    # Join words back together
    text = ' '.join(words)
    sentences.append(text)

# Create and fit tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences)

# Save both tokenizer and lemmatizer
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

with open('lemmatizer.pkl', 'wb') as file:
    pickle.dump(lemmatizer, file)

print("Tokenizer and Lemmatizer saved successfully!")