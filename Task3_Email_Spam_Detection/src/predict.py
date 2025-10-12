import pickle

def load_model(path, logger):
    """Load a saved model from disk"""
    with open(path, "rb") as f:
        model = pickle.load(f)
    if logger:
        logger.info(f"Model loaded from {path}")
    return model 

def load_vectorizer(path, logger):
    """Load a saved TF-IDF vectorizer from disk"""
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    if logger:
        logger.info(f"Model loaded from {path}")
    return vectorizer

def predict_message(text, model_path, vectorizer_path, logger):
    """
    Predict if a message is spam or ham
    """
    # Load model and vectorizer
    model = load_model(model_path, logger)
    vectorizer = load_vectorizer(vectorizer_path, logger)

    # Transform text
    text_tfidf = vectorizer.transform([text])

    # Predict
    pred = model.predict(text_tfidf)[0]
    label = 'Spam' if pred == 1 else 'Ham'
    if logger:
        logger.info(f"Predicted message: {text} -> {label}")
    return label