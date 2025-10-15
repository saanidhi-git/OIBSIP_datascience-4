# 📧 Email Spam Detection with Multinomial Naive Bayes Classifier

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Algorithm Used (Multinomial Naive Bayes)](#algorithm-used-multinomial-naive-bayes)
3.  [Dataset](#dataset)
4.  [Project Structure](#project-structure)
5.  [Setup and Installation](#setup-and-installation)
6.  [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
7.  [Results and Visualization](#results-and-visualization)
8.  [Conclusion](#conclusion)

***

## 1. Project Overview

This project implements a robust Machine Learning solution to classify incoming messages as either **"spam"** or **"ham"** (not spam). The goal is to build an effective filter based on the text content of the email/SMS. We utilize a highly efficient classification algorithm and deploy the final model using an interactive **Streamlit** web application.

### Key Features:
* **Data Preprocessing:** Cleaning and tokenization of text data.
* **Feature Extraction:** Using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.
* **Model Training:** Training a **Multinomial Naive Bayes** classifier.
* **Interactive App:** A Streamlit interface to test custom text for spam probability in real-time.

***

## 2. Algorithm Used (Multinomial Naive Bayes)

The core classification is performed using the **Multinomial Naive Bayes** algorithm.

* **How it Works:** This model is one of the standard algorithms for text classification. It's based on **Bayes' theorem** and is well-suited for discrete feature counts (like word counts or TF-IDF values). It calculates the probability of a message belonging to a class (spam or ham) given the frequency of its words.
* **Suitability for Text:** The Multinomial distribution models the probability of observing word counts in a document given the class, making it exceptionally effective for high-dimensional, sparse data like text feature vectors.
* **Feature Vectorization:** Text is transformed using **TF-IDF** (Term Frequency-Inverse Document Frequency), which assigns weights to words based on their importance. This helps the Naive Bayes model identify words that are highly characteristic of spam (e.g., "free," "win," "click").

***

## 3. Dataset

The project uses a publicly available **SMS Spam Collection Dataset** (often used for this task).

| Column | Description | Example |
| :--- | :--- | :--- |
| `target` | The class label (0 for ham, 1 for spam) | `spam` or `ham` |
| `text` | The raw message/email content | "Call now to win a free prize!" |

**Note:** The dataset requires cleaning to remove punctuation, stop words, and perform stemming/lemmatization.

***

## 4. Project Structure

The repository is organized as follows:
email-spam-detection/
├── .gitignore
├── README.md
├── requirements.txt           # Lists all necessary Python libraries
├── spam_classifier.py         # Main ML code: loads data, cleans, trains Naive Bayes, saves model and vectorizer.
└── streamlit_app.py           # Streamlit code for the interactive web interface.

***

## 5. Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/email-spam-detection.git](https://github.com/YourUsername/email-spam-detection.git)
    cd email-spam-detection
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the ML Training Script:** This trains the **Multinomial Naive Bayes** model and saves both the model and the TF-IDF vectorizer (which is needed by the Streamlit app for prediction).
    ```bash
    python spam_classifier.py
    ```

***

## 6. How to Run the Streamlit App

The project includes an interactive Streamlit application for demonstration.

1.  **Ensure you have completed the Setup steps above.**

2.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

3.  The app will automatically open in your web browser at a local address (usually `http://localhost:8501`).

### 📷 Streamlit Application Preview

<p align="center">
  <img width="1798" height="918" alt="spam" src="https://github.com/user-attachments/assets/643e4a9e-d8bc-4720-aa6f-740b6675eae4" />
  <br>
  <em>Figure 1: Streamlit App Interface showing the text input box and the classification result (Spam/Ham).</em>
</p>

***

## 7. Results and Visualization

### Model Performance

The **Multinomial Naive Bayes** classifier performed well, achieving the following metrics on the test set:

* **Accuracy:** **[Insert your calculated Accuracy Score]%**
* **Precision (Spam):** **[Insert Spam Precision Score]%** *(Crucial metric for spam detection to minimize false positives, i.e., marking a legitimate email as spam.)*

### Key Visualization

A visualization of word clouds can highlight the difference in vocabulary between the two classes, justifying the model's success.

<p align="center">
  <img width="1899" height="868" alt="spam1" src="https://github.com/user-attachments/assets/81bea421-619c-443e-ad7e-4fc4ecd8cd13" />
  <br>
  <em>Figure 2: Word Cloud visualizing the most frequent, high-weighted words found in 'Spam' messages, demonstrating distinct vocabulary usage and feature separation.</em>
</p>

***

## 8. Conclusion

The combination of **TF-IDF** for feature extraction and the **Multinomial Naive Bayes** classifier provided a fast and highly accurate solution for email/SMS spam detection. The resulting model is robust and suitable for immediate deployment, as showcased by the interactive Streamlit application.
