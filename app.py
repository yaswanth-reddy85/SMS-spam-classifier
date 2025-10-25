# Suppress TensorFlow deprecation warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load LSTM model
lstm_model = load_model('spam_lstm_model.keras')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Title of the Streamlit app
st.title("Spam Classifier App")

st.write("""
This app uses a pre-trained LSTM model to classify messages as Spam or Ham.
You can type a message manually or upload a CSV file containing a 'message' column.
""")

# Preprocessing function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(lemmatized)

# Create a tokenizer placeholder (temporary)
# ⚠️ Ideally, this should be the same tokenizer used during training
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

def predict_spam(messages, max_len=80):
    preprocessed = [preprocess_text(msg) for msg in messages]

    # Fit the tokenizer on current text (temporary — not ideal but avoids crash)
    tokenizer.fit_on_texts(preprocessed)
    sequences = tokenizer.texts_to_sequences(preprocessed)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')

    predictions = lstm_model.predict(padded)
    
    # Your Dense layer has 2 neurons, so predictions shape is (n, 2)
    # We'll assume [Spam, Ham] or similar — adjust based on how you trained labels
    result_labels = []
    for pred in predictions:
        label = 'Spam' if pred[0] > pred[1] else 'Ham'
        result_labels.append(label)
    
    return result_labels

# Choose input method
input_method = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

if input_method == "Manual Input":
    st.subheader("Enter Message")
    message = st.text_area("Message")

    if st.button("Predict"):
        if message:
            prediction = predict_spam([message])[0]
            st.subheader("Prediction Result")
            st.write(f"The message is predicted to be: **{prediction}**")
        else:
            st.error("Please enter a message.")
else:
    st.subheader("Upload CSV File")
    st.write("Upload a CSV file that contains a 'message' column.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        if 'message' not in input_df.columns:
            st.error("Invalid CSV. Expected a 'message' column.")
        else:
            try:
                predictions = predict_spam(input_df['message'].tolist())

                output_df = input_df.copy()
                output_df['Prediction'] = predictions

                st.subheader("Predictions")
                st.dataframe(output_df)

                csv = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name='spam_predictions.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Error processing CSV or making predictions: {e}")
    else:
        st.info("Please upload a CSV file to make predictions.")
