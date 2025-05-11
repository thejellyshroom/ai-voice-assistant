import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
import joblib
import os
from datasets import load_dataset

# config copied from bert_finetune_multiple.ipynb
EMOTIONS_ID2LABEL = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
    5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
    10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement',
    14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love',
    19: 'nervousness', 20: 'optimism', 21: 'pride', 22: 'realization',
    23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}
NUM_CLASSES = 28

DEFAULT_TRAIN_SIZE = 20000
DEFAULT_TEST_SIZE = 1000

SAVE_DIR = "main/training/"
MNB_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "trained_mnb_model.joblib")
LR_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "trained_lr_model.joblib")
VECTORIZER_SAVE_PATH = os.path.join(SAVE_DIR, "tfidf_vectorizer.joblib")
MLB_SAVE_PATH = os.path.join(SAVE_DIR, "mlb_binarizer.joblib")

def load_raw_data(train_size=DEFAULT_TRAIN_SIZE, test_size=DEFAULT_TEST_SIZE):
    print(f"Loading GoEmotions dataset...")
    emotion_dataset = load_dataset("jellyshroom/go_emotions_augmented")
    
    small_train_dataset = emotion_dataset['train'].select(range(train_size))
    _ = emotion_dataset['test'].select(range(test_size))

    train_texts = [item['text'] for item in small_train_dataset]
    train_labels_raw = [item['labels'] for item in small_train_dataset]
    
    print(f"Loaded {len(train_texts)} texts for training.")
    return train_texts, train_labels_raw

def train_and_save_sklearn_model(
    model_type,
    X_train, y_train_raw,
    mlb, vectorizer
):
    model_save_path = MNB_MODEL_SAVE_PATH if model_type == 'mnb' else LR_MODEL_SAVE_PATH
    model_name_display = "Multinomial Naive Bayes" if model_type == 'mnb' else "Logistic Regression"

    print(f"\n--- Training {model_name_display} ---")

    y_train_bin = mlb.transform(y_train_raw)
    X_train_tfidf = vectorizer.transform(X_train)

    print(f"Transformed label shape for training: {y_train_bin.shape}")
    print(f"Transformed TF-IDF shape for training: {X_train_tfidf.shape}")
        
    base_estimator = None
    param_grid = {}

    if model_type == 'mnb':
        base_estimator = MultinomialNB()
        param_grid = {
            'estimator__alpha': [0.1, 0.5, 1.0],
            'estimator__fit_prior': [True, False]
        }
    elif model_type == 'lr':
        base_estimator = LogisticRegression(solver='liblinear', max_iter=2000, class_weight='balanced', random_state=42)
        param_grid = {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__penalty': ['l1', 'l2']
        }
    
    ovr_classifier = OneVsRestClassifier(base_estimator)
    print(f"Running GridSearchCV for {model_name_display} (OneVsRestClassifier)...")
    grid_search = GridSearchCV(ovr_classifier, param_grid, cv=3, scoring='f1_micro', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_tfidf, y_train_bin)
    
    print(f"Best parameters for {model_name_display}: {grid_search.best_params_}")
    trained_pipeline = grid_search.best_estimator_

    print(f"Saving tuned {model_name_display} model to {model_save_path}")
    joblib.dump(trained_pipeline, model_save_path)
    print(f"{model_name_display} model training complete and saved to {model_save_path}")
    return trained_pipeline

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    train_texts, train_labels_raw = load_raw_data()

    print("\n--- Fitting MultiLabelBinarizer ---")
    mlb = MultiLabelBinarizer(classes=list(range(NUM_CLASSES)))
    mlb.fit(train_labels_raw)
    print(f"Saving MultiLabelBinarizer to {MLB_SAVE_PATH}")
    joblib.dump(mlb, MLB_SAVE_PATH)
    print("MultiLabelBinarizer fitted and saved.")

    print("\n--- Fitting TF-IDF Vectorizer ---")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    vectorizer.fit(train_texts)
    print(f"Saving TF-IDF vectorizer to {VECTORIZER_SAVE_PATH}")
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    print("TF-IDF vectorizer fitted and saved.")

    train_and_save_sklearn_model('mnb', train_texts, train_labels_raw, mlb, vectorizer)
    train_and_save_sklearn_model('lr', train_texts, train_labels_raw, mlb, vectorizer)

    print("\nScikit-learn model training and saving finished.")
    print(f"Models and preprocessors saved in: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main()