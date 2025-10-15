import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK stopwords not found. Downloading NLTK stopwords...")
    nltk.download('stopwords')

# We download 'wordnet' for the lemmatizer to look up root words.
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK wordnet not found. Downloading NLTK wordnet and omw-1.4...")
    nltk.download('wordnet')
    # 'omw-1.4' contains additional multilingual WordNet data needed by some lemmatizer setups
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

lemmatizer=WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

print("All imports and dependencies are ready!")

#Data Loading 

DATASET_PATH = 'spam.csv' 

try:
    # Load the dataset. The file uses commas (,) as separators and latin-1 encoding.
    df = pd.read_csv(DATASET_PATH, sep=',', encoding='latin-1', header=None)
    
    # The original dataset has 5 columns, but we only need the first two.
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']  # Rename the columns

    # Normalize labels (strip whitespace and lowercase) to avoid mismatches like ' Spam' or 'HAM'
    df['label'] = df['label'].astype(str).str.strip().str.lower()

    # Label Encoding: Convert 'ham'/'spam' into 0/1 for the model
    mapping = {'ham': 0, 'spam': 1}
    df['label_encoded'] = df['label'].map(mapping)

    # Detect any unmapped/invalid labels that produced NaN in label_encoded
    unmapped = df[df['label_encoded'].isna()]
    if not unmapped.empty:
        problematic = unmapped['label'].unique()
        print("\nWARNING: Found label values that could not be mapped to 0/1:", problematic)
        print(f"These rows will be dropped. Count: {unmapped.shape[0]}")
        # Drop the problematic rows so y contains only valid 0/1 values
        df = df.drop(unmapped.index).reset_index(drop=True)

    print("\n Data Loading Complete.")
    spam_count = (df['label'] == 'spam').sum()
    print(f"Total messages loaded: {df.shape[0]}. Spam messages: {spam_count}")
    
except FileNotFoundError:
    print(f"\nERROR: File not found. Please ensure the dataset is saved as '{DATASET_PATH}' in the same directory.")
    exit()


# Data Preprocessing


def text_cleaning(text):
   
    # 1. Remove punctuation (e.g., '!', '?', ',', '$')
    # str.maketrans creates a translation table to remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 2. Tokenize (split the text into a list of words) and convert to lowercase
    tokens = text.lower().split()
    
    cleaned_tokens = []
    for word in tokens:
        # 3. Remove stopwords (e.g., 'the', 'is', 'a')
        # 4. Check if the word is purely alphabetic (removes stray numbers/symbols)
        if word not in stop_words and word.isalpha():
            # 5. Apply Lemmatization (e.g., 'running' becomes 'run')
            cleaned_tokens.append(lemmatizer.lemmatize(word))
            
    # 6. Join the cleaned words back into a single string
    return " ".join(cleaned_tokens)

# Apply the cleaning function to the entire 'message' column
df['cleaned_message'] = df['message'].apply(text_cleaning)
print("\nText Cleaning and Lemmatization Complete.")



# PART 4: FEATURE EXTRACTION (TF-IDF)


# TF-IDF (Term Frequency-Inverse Document Frequency) is used here.
# It converts the cleaned text messages into a numerical matrix where each column 
# is a word, and each cell contains a score representing the importance of that 
# word in that specific message. 
vectorizer = TfidfVectorizer(max_features=5000) # We limit the vocabulary to the 5000 most frequent/important words
X = vectorizer.fit_transform(df['cleaned_message']).toarray()
y = df['label_encoded'] # Our target variable (0s and 1s)

print("\n TF-IDF Vectorization Complete.")
print(f"Feature matrix (X) shape: {X.shape} (Your messages are now vectors of numbers!)")

# PART 5: MODEL TRAINING AND EVALUATION


# 5.1 Split the data into Training and Testing sets.
# We use 80% for training the model and 20% for testing its performance on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("\n Data Split Complete.")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 5.2 Train the Multinomial Naive Bayes model.
# Naive Bayes is a probabilistic algorithm that works exceptionally well for text classification 
# due to its simplicity and effectiveness with word frequency features. [Image of the Bayes Theorem formula]
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model Training Complete (Multinomial Naive Bayes).")

# 5.3 Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'])

print("\n Model Evaluation Results")
print(f"Overall Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix (Mistake Analysis):")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)



# PART 6: LIVE PREDICTION

def predict_message(message):
   
    # 1. Clean the new message using the identical function used during training
    cleaned_msg = text_cleaning(message)
    
    # 2. Vectorize the message using the *fitted* vectorizer
    # Note: We use .transform(), NOT .fit_transform(), as we only convert, not re-fit the vocabulary
    X_new = vectorizer.transform([cleaned_msg])
    
    # 3. Predict the label
    prediction = model.predict(X_new)[0]
    
    return "SPAM 🚨" if prediction == 1 else "HAM ✅"

# Test cases
test_messages = [
    "URGENT! You have won $1000 cash prize! Claim now by texting back!", 
    "Hey, what time are we meeting for coffee tomorrow?",                
    "Congrats! Get your FREE entry to the final draw. Call 0800-456-789.",
    "Did you remember to send the final report to the manager?",          
    "Your account has been suspended. Click this link to verify.",
]

print("\n" + "="*50)
print("             LIVE PREDICTION TEST")
print("="*50)

for msg in test_messages:
    result = predict_message(msg)
    print(f"MESSAGE: '{msg}'")
    print(f"CLASSIFICATION: {result}\n")

