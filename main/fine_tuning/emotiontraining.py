import tensorflow as tf
import numpy as np
from collections import Counter
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, hf_hub_download
import os
from roberta_classification import RoBERTaForClassification
from utils_emotiontraining import (emotions_id2label, emotions_label2id, manage_dataset_columns, 
                                   pipe as augmentation_pipe, 
                                   run_examples, TEST_TEXTS)
from dataset_augmentation import load_or_augment_dataset
from functools import partial


NUM_CLASSES = 28
BATCH_SIZE = 32
PROBABILITY_THRESHOLD = 0.5
EPOCHS = 20
MINORITY_THRESHOLD_PERCENT = 1.5

TEST_TRAIN_RANGE = 2000
TEST_TEST_RANGE = 500
TEST_VALIDATE_RANGE = 100

HF_USERNAME = "jellyshroom"
AUGMENTED_DATASET_NAME = "go_emotions_augmented"
HF_DATASET_ID = f"{HF_USERNAME}/{AUGMENTED_DATASET_NAME}"
LOCAL_SAVE_PATH = "augmented_emotion_dataset"
FORCE_REAUGMENT = False

model_base_name = "distilroberta-base"
model = TFAutoModel.from_pretrained(model_base_name)
tokenizer = AutoTokenizer.from_pretrained(model_base_name)

print("--- Attempting to load or augment dataset ---")
use_emotion_dataset = load_or_augment_dataset(
    hf_dataset_id=HF_DATASET_ID,
    local_save_path=LOCAL_SAVE_PATH,
    force_reaugment=FORCE_REAUGMENT,
    minority_threshold_percent=MINORITY_THRESHOLD_PERCENT,
    augmentation_pipe=augmentation_pipe,
    emotions_id2label=emotions_id2label
)

if use_emotion_dataset is None:
    raise RuntimeError("Fatal Error: Failed to load or create the augmented dataset.")

print("--- Dataset ready for use ---")

print("\n--- Final Dataset Statistics ---")
print(f"Train samples: {len(use_emotion_dataset['train'])}")
print(f"Test samples: {len(use_emotion_dataset['test'])}")
print(f"Validation samples: {len(use_emotion_dataset['validation'])}")
print("----------------------------------")

print("\nAnalyzing final training data distribution...")
final_train_labels = use_emotion_dataset['train']['labels']
all_final_labels = [label for sublist in final_train_labels for label in sublist]
label_counts_final = Counter(all_final_labels)
total_samples_final = len(use_emotion_dataset['train'])

print(f"Final Total Samples: {total_samples_final}")
minority_threshold_count_final = total_samples_final * (MINORITY_THRESHOLD_PERCENT / 100.0)
print(f"Minority Threshold Count (final): {minority_threshold_count_final:.0f}")
print("Final Class Counts:")
for label_id in range(NUM_CLASSES):
    count = label_counts_final.get(label_id, 0)
    label_name = emotions_id2label.get(label_id, f"Unknown({label_id})")
    below_threshold_flag = "*" if count < minority_threshold_count_final else ""
    print(f"  - {label_name} (ID: {label_id}): {count} samples {below_threshold_flag}")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, return_tensors='tf')

emotions_encoded = {}
emotions_encoded['train'] = use_emotion_dataset['train'].map(tokenize, batched=True, batch_size=None)
emotions_encoded['test'] = use_emotion_dataset['test'].map(tokenize, batched=True, batch_size=None)
emotions_encoded['validation'] = use_emotion_dataset['validation'].map(tokenize, batched=True, batch_size=None)

def create_multi_hot_labels(data):
    multi_hot_label = np.zeros(NUM_CLASSES, dtype=np.float32)
    if 'labels' in data and isinstance(data['labels'], list) and len(data['labels']) > 0:
        for label_id in data['labels']:
            if isinstance(label_id, int) and 0 <= label_id < NUM_CLASSES:
                multi_hot_label[label_id] = 1.0
    data['multi_hot_labels'] = multi_hot_label
    return data

emotions_encoded['train'] = emotions_encoded['train'].map(create_multi_hot_labels)
emotions_encoded['test'] = emotions_encoded['test'].map(create_multi_hot_labels)
emotions_encoded['validation'] = emotions_encoded['validation'].map(create_multi_hot_labels)

emotions_encoded = manage_dataset_columns(
    datasets=emotions_encoded,
    columns_to_remove=['label_int', 'labels'],
    column_renames={'multi_hot_labels': 'labels'},
    verbose=False
)

train_labels_np = np.array(emotions_encoded['train']['labels'])
label_counts = np.sum(train_labels_np, axis=0)
total_samples = len(train_labels_np)

class_weights_calc = {}
for i in range(NUM_CLASSES):
    count = label_counts[i] if label_counts[i] > 0 else 1
    class_weights_calc[i] = total_samples / (NUM_CLASSES * count)

sample_weights_np = np.zeros(total_samples, dtype=np.float32)
for i in range(total_samples):
    sample_label_indices = np.where(train_labels_np[i] == 1.0)[0]
    if len(sample_label_indices) > 0:
        sample_weights_np[i] = max(class_weights_calc[idx] for idx in sample_label_indices)
    else:
        sample_weights_np[i] = 1.0

feature_cols = ["input_ids", "attention_mask"]
label_col = "labels"
cols_to_set_format = feature_cols + [label_col]

actual_train_cols = list(emotions_encoded['train'].features)
actual_test_cols = list(emotions_encoded['test'].features)
actual_validation_cols = list(emotions_encoded['validation'].features)
final_train_cols = [col for col in cols_to_set_format if col in actual_train_cols]
final_test_cols = [col for col in cols_to_set_format if col in actual_test_cols]
final_validation_cols = [col for col in cols_to_set_format if col in actual_validation_cols]

train_features_np = {col: np.array(emotions_encoded['train'][col]) for col in feature_cols}
train_labels_np = np.array(emotions_encoded['train']['labels'])

test_features_np = {col: np.array(emotions_encoded['test'][col]) for col in feature_cols}
test_labels_np = np.array(emotions_encoded['test']['labels'])

validate_features_np = {col: np.array(emotions_encoded['validation'][col]) for col in feature_cols}
validate_labels_np = np.array(emotions_encoded['validation']['labels'])

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_features_np, train_labels_np, sample_weights_np)
)
train_dataset = train_dataset.shuffle(len(sample_weights_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_features_np, test_labels_np)
)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validate_features_np, validate_labels_np)
)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print("Datasets created successfully.")


def predict_emotion(text, model, threshold=PROBABILITY_THRESHOLD):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    predictions = model(inputs=inputs)

    predicted_labels_indices = tf.where(predictions[0] > threshold).numpy().flatten()
    predicted_emotions = []
    confidences = []
    if len(predicted_labels_indices) > 0:
        for index in predicted_labels_indices:
            predicted_emotions.append(emotions_id2label[index])
            confidences.append(float(predictions[0][index]))
    else:
        highest_prob_index = tf.argmax(predictions, axis=1).numpy()[0]
        predicted_emotions.append(emotions_id2label[highest_prob_index])
        confidences.append(float(predictions[0][highest_prob_index]))


    return {
        'text': text,
        'emotions': predicted_emotions,
        'confidences': confidences
    }

untrained_classifier = RoBERTaForClassification(model, num_classes=NUM_CLASSES)

untrained_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)

print("Predictions with UNTRAINED model (random weights - multi-label):")
print("-------------------------------------------------------------")

run_examples(
    classifier=untrained_classifier, 
    predict_emotion_func=predict_emotion, 
    threshold=PROBABILITY_THRESHOLD,
    test_texts=TEST_TEXTS
)

classifier = RoBERTaForClassification(model, num_classes=NUM_CLASSES)

print("Compiling model with AUC, Precision, Recall...")
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(multi_label=True, name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
        ]
)
print("Model compiled.")

print("Starting multi-label model training with sample weights...")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("--- GPU Check ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs detected and configured.")
    except RuntimeError as e:
        print(f"GPU Memory Growth Error: {e}")
else:
    print("No GPU detected by TensorFlow. Running on CPU.")
print("-----------------")

history = classifier.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)
print("Training finished.")

print("Evaluating model on test set...")
results = classifier.evaluate(test_dataset, verbose=1)

print("\nTest Set Evaluation Results:")
for name, value in zip(classifier.metrics_names, results):
    print(f"- {name}: {value:.4f}")

model_save_path = "./custom_models/emotion_classifier.keras"
print(f"\nSaving trained model to: {model_save_path}")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
classifier.save(model_save_path)
print("Model saved successfully.")

print("\n--- Loading and Testing Saved Model --- ")
try:
    loaded_classifier = tf.keras.models.load_model(model_save_path)
    print(f"Model loaded successfully from {model_save_path}")

    tokenizer_for_inference = AutoTokenizer.from_pretrained("distilroberta-base")

    def predict_with_loaded_model(text, loaded_model, tokenizer, threshold=PROBABILITY_THRESHOLD):
        inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
        predictions = loaded_model(inputs)
        predictions_tensor = predictions[0]
        predicted_labels_indices = tf.where(predictions_tensor > threshold).numpy().flatten()
        predicted_emotions = []
        confidences = []

        if len(predicted_labels_indices) > 0:
            for index in predicted_labels_indices:
                predicted_emotions.append(emotions_id2label.get(index, f"Unknown({index})"))
                confidences.append(float(predictions_tensor[index]))
        else:
            highest_prob_index = tf.argmax(predictions_tensor, axis=0).numpy()
            predicted_emotions.append(emotions_id2label.get(highest_prob_index, f"Unknown({highest_prob_index})"))
            confidences.append(float(predictions_tensor[highest_prob_index]))
        return {
            'text': text,
            'emotions': predicted_emotions,
            'confidences': confidences
        }

    test_text = "This is fantastic news, I feel so relieved!"
    result = predict_with_loaded_model(test_text, loaded_classifier, tokenizer_for_inference)
    print(f"\nPrediction using loaded model for: '{result['text']}'")
    print(f"Predicted emotions: {result['emotions']}")
    emotion_confidence_pairs = list(zip(result['emotions'], result['confidences']))
    print(f"Confidences: {emotion_confidence_pairs}")

except Exception as e:
    print(f"Error loading or predicting with saved model: {e}")
    print("Loading Keras models with custom objects might require registering the custom class.")
    print("See: https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects")

print("\nPredictions with TRAINED multi-label model (using original classifier object):")
print("----------------------------------------")

run_examples(
    classifier=classifier, 
    predict_emotion_func=predict_emotion, 
    threshold=PROBABILITY_THRESHOLD,
    test_texts=TEST_TEXTS
)