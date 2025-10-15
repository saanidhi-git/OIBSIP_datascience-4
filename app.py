import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- NLTK Data Setup and Caching ---
# We use st.cache_resource to ensure these downloads only happen once.
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data (stopwords and wordnet) only once."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

download_nltk_data()

# --- Global NLP objects initialized here ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
DATASET_PATH = 'spam.csv' 

# --- 1. Text Cleaning Function (from your script) ---
def text_cleaning(text):
    """Cleans text by removing punctuation, stop words, and applying lemmatization."""
    if not isinstance(text, str):
        return "" 
        
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.lower().split()
    
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words and word.isalpha():
            cleaned_tokens.append(lemmatizer.lemmatize(word))
            
    return " ".join(cleaned_tokens)


# --- 2. Model Training and Caching ---
@st.cache_resource
def train_model(data_path):
    """
    Loads data, trains the TF-IDF Vectorizer and Multinomial Naive Bayes model.
    This expensive operation runs only once when the app is launched.
    """
    with st.spinner('Training Model... This may take a moment.'):
        try:
            # Data Loading (from your script)
            df = pd.read_csv(data_path, sep=',', encoding='latin-1', header=None)
            df = df.iloc[:, :2] 
            df.columns = ['label', 'message']
            df['label'] = df['label'].astype(str).str.strip().str.lower()
            df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
            df.dropna(subset=['label_encoded'], inplace=True) # Drop any unmapped rows
            
            # Text Cleaning
            df['cleaned_message'] = df['message'].apply(text_cleaning)
            
            # Feature Extraction (TF-IDF)
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(df['cleaned_message'])
            y = df['label_encoded']
            
            # Model Training (Training on full dataset for app stability)
            model = MultinomialNB()
            model.fit(X, y)
            
            return vectorizer, model
            
        except FileNotFoundError:
            st.error(f"DATASET ERROR: The file '{DATASET_PATH}' was not found. Please ensure it is in the same directory as the app.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred during model training: {e}")
            st.stop()


# --- 3. Prediction Function ---
def predict_message(message, vectorizer, model):
    """Runs the full pipeline (Clean -> Vectorize -> Predict) on a new message."""
    
    cleaned_msg = text_cleaning(message)
    
    if not cleaned_msg:
        return None 
        
    # Vectorize the cleaned message using the *fitted* vectorizer
    X_new = vectorizer.transform([cleaned_msg])
    
    # Predict
    prediction = model.predict(X_new)[0]
    
    return "SPAM 🚨" if prediction == 1 else "HAM ✅"

# --------------------------------------------------------------------------
# STREAMLIT UI LAYOUT
# --------------------------------------------------------------------------

# Load the model and vectorizer only once
VECTORIZER, MODEL = train_model(DATASET_PATH)

st.set_page_config(
    page_title="SMS Spam Detector  App",
    layout="centered",
)

st.title("✉️ SMS Spam Detector ")
st.markdown("---")

st.markdown("""
    This application uses a Multinomial Naive Bayes model trained on the SMS Spam Collection Dataset
    to classify messages as **Legitimate (HAM)** or **Junk (SPAM)**.
""")

# Input Area
message_input = st.text_area(
    "Enter the SMS Message:", 
    height=150,
    placeholder="Example: Congrats! Your number has been selected to win a FREE cruise ship ticket! Text 'YES' to claim.",
    key="sms_input"
)

# Prediction Button
if st.button("Classify Message", type="primary"):
    
    if not message_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Run prediction
        prediction_result = predict_message(message_input, VECTORIZER, MODEL)

        st.markdown("### Prediction Result")

        if prediction_result == "SPAM 🚨":
            st.error(f"## {prediction_result}")
            st.write("🔴 **Warning:** This message contains keywords and characteristics highly associated with spam.")
        elif prediction_result == "HAM ✅":
            st.success(f"## {prediction_result}")
            st.write("🟢 **Safe:** This message appears to be legitimate.")
        else:
             st.warning("Could not process the message. It might be too short or contain no meaningful content.")

st.markdown("---")
st.caption("Model Type: Multinomial Naive Bayes with TF-IDF Vectorization.")
