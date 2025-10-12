import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Save model
def save_model(model, path, logger):
    """
    Save trained model safely
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    if logger:
        logger.info(f"Model saved to {path}")
    print(f" Best model saved to {path}")
    
# Train & Evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, logger):
    """
    Train 3 models, evaluate, and save only the best one
    """
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    
    results = {}
    best_acc = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        if logger:
            logger.info(f"Training {name}...")
        print(f"\n Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        if logger:
            logger.info(f"{name} Accuracy: {acc:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Ham','Spam'])}")
            logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        results[name] = acc
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
    
    # Save only best model
    if best_model is not None and logger:
        save_model(best_model, f"../models/{best_name.replace(' ','_').lower()}_spam_classifier.pkl", logger)

    return results, best_name, best_acc

