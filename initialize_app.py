import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import re
from keras.models import load_model
import os

def check_file_exists(filepath):
    if os.path.exists(filepath):
        print(f"{filepath} exists and its size is {os.path.getsize(filepath)} bytes")
    else:
        print(f"{filepath} does not exist")

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Check if files exist
print("\nChecking existing files:")
check_file_exists('spam_lstm_model.h5')
check_file_exists('tokenizer.pkl')
check_file_exists('lemmatizer.pkl')

# Create and save lemmatizer
print("\nCreating and saving lemmatizer...")
lemmatizer = WordNetLemmatizer()
with open('lemmatizer.pkl', 'wb') as file:
    pickle.dump(lemmatizer, file)

# Check if we need to create tokenizer
if not os.path.exists('tokenizer.pkl'):
    print("\nCreating and saving tokenizer...")
    try:
        # Read the dataset
        dataset = pd.read_csv('spam (1).csv', encoding='latin-1')
        sent = dataset.iloc[:, [1]]['v2']

        # Preprocess the text
        sentences = []
        for text in sent:
            text = re.sub('[^A-Za-z]', ' ', text)
            text = text.lower()
            words = word_tokenize(text)
            words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
            text = ' '.join(words)
            sentences.append(text)

        # Create and fit tokenizer
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(sentences)

        # Save tokenizer
        with open('tokenizer.pkl', 'wb') as file:
            pickle.dump(tokenizer, file)
        print("Tokenizer created and saved successfully!")
    except Exception as e:
        print(f"Error creating tokenizer: {e}")

# Verify files after creation
print("\nVerifying files after creation:")
check_file_exists('spam_lstm_model.h5')
check_file_exists('tokenizer.pkl')
check_file_exists('lemmatizer.pkl')

print("\nInitialization complete!")