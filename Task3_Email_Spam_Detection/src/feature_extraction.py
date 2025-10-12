from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# TF-IDF Vectorization
def get_tfidf_features(X_train, X_test, logger, max_features=5000):
    """
    Convert text to TF-IDF features
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    if logger:
        logger.info(f"TF-IDF features created with {X_train_tfidf.shape[1]} features")
    return X_train_tfidf, X_test_tfidf, vectorizer

# Save vectorizer
def save_vectorizer(vectorizer, path, logger):
    """
    Save trained TF-IDF vectorizer to disk
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the vectorizer
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    if logger:
        logger.info(f"TF-IDF vectorizer saved to {path}")
    print(f"Vectorizer saved to {path}")

# Load vectorizer
def load_vectorizer(path, logger):
    """
    Load saved TF-IDF vectorizer from disk
    """
    with open(path, 'rb') as f:
        vectorizer = pickle.load(f)
    logger.info(f"Vectorizer loaded from {path}")
    return vectorizer