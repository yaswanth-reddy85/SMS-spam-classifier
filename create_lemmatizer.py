import nltk
from nltk.stem import WordNetLemmatizer
import pickle

# Download required NLTK data
nltk.download('wordnet', quiet=True)

# Create lemmatizer
lemmatizer = WordNetLemmatizer()

# Save the lemmatizer
with open('lemmatizer.pkl', 'wb') as file:
    pickle.dump(lemmatizer, file)

print("Lemmatizer saved successfully!")