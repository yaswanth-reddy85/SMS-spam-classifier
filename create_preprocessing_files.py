import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import re

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('all')  # Download all NLTK data

try:
    # Read the dataset
    print("Reading dataset...")
    dataset = pd.read_csv('spam (1).csv', encoding='latin-1')
    messages = dataset.iloc[:, 1].values  # Get the 'v2' column (messages)
    
    # Initialize lemmatizer
    print("Creating lemmatizer...")
    lemmatizer = WordNetLemmatizer()
    
    # Save lemmatizer
    print("Saving lemmatizer...")
    with open('lemmatizer.pkl', 'wb') as file:
        pickle.dump(lemmatizer, file)
    
    # Preprocess text
    print("Preprocessing text...")
    stop_words = set(stopwords.words('english'))
    processed_texts = []
    
    for text in messages:
        # Clean and tokenize
        text = re.sub('[^A-Za-z]', ' ', str(text))
        text = text.lower()
        words = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        processed_text = ' '.join(words)
        processed_texts.append(processed_text)
    
    # Create and fit tokenizer
    print("Creating and fitting tokenizer...")
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(processed_texts)
    
    # Save tokenizer
    print("Saving tokenizer...")
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)
    
    print("Successfully created and saved both lemmatizer and tokenizer!")

except Exception as e:
    print(f"An error occurred: {str(e)}")