import tensorflow as tf
import numpy as np
from collections import Counter
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
from roberta_classification import RoBERTaForClassification
from utils import emotions_id2label, emotions_label2id, manage_dataset_columns, iterative_augment_minority_classes, pipe as augmentation_pipe
from functools import partial


NUM_CLASSES = 28
BATCH_SIZE = 32
PROBABILITY_THRESHOLD = 0.5
EPOCHS = 5
MINORITY_THRESHOLD_PERCENT = 1.0

TEST_TRAIN_RANGE = 2000
TEST_TEST_RANGE = 500
TEST_VALIDATE_RANGE = 100

model = TFAutoModel.from_pretrained("distilroberta-base")
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

emotion_dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# Select a subset of data using .select() for specific ranges
use_train_dataset = emotion_dataset['train'].select(range(TEST_TRAIN_RANGE))
use_test_dataset = emotion_dataset['test'].select(range(TEST_TEST_RANGE))
use_validation_dataset = emotion_dataset['validation'].select(range(TEST_VALIDATE_RANGE))

print("Analyzing original training data for minority classes...")
original_train_labels = use_train_dataset['labels']
all_labels = [label for sublist in original_train_labels for label in sublist]
label_counts_original = Counter(all_labels)
total_samples_original = len(use_train_dataset)

# --- Identify minority classes (e.g., frequency < 1% of total samples) ---
minority_threshold_count = total_samples_original * (MINORITY_THRESHOLD_PERCENT / 100.0)
minority_classes = {
    label for label, count in label_counts_original.items()
    if count < minority_threshold_count
}
if minority_classes:
    print(f"Identified {len(minority_classes)} minority classes (frequency < {MINORITY_THRESHOLD_PERCENT}%, count < {minority_threshold_count:.0f}):")
    minority_class_names = {emotions_id2label.get(idx, f"Unknown({idx})") for idx in minority_classes}
    print(f"Minority Classes Indices: {minority_classes}")
    print(f"Minority Classes Names: {minority_class_names}")
else:
    print(f"No minority classes found with frequency < {MINORITY_THRESHOLD_PERCENT}%.")

# --- Perform Iterative Augmentation on Training Data ---
print("Starting iterative data augmentation...")
augmented_train_dataset = iterative_augment_minority_classes(
    train_dataset=use_train_dataset,
    minority_threshold_percent=MINORITY_THRESHOLD_PERCENT,
    emotions_id2label_map=emotions_id2label,
    augmentation_pipe=augmentation_pipe
    # Optional: add max_iterations or target_augmentation_factor if needed
)
print(f"Finished iterative augmentation. Final training set size: {len(augmented_train_dataset)}")

# --- Calculate final distribution after augmentation (optional but informative) ---
print("\nAnalyzing final training data distribution after augmentation...")
final_train_labels = augmented_train_dataset['labels']
all_final_labels = [label for sublist in final_train_labels for label in sublist]
label_counts_final = Counter(all_final_labels)
total_samples_final = len(augmented_train_dataset)

print(f"Final Total Samples: {total_samples_final}")
minority_threshold_count_final = total_samples_final * (MINORITY_THRESHOLD_PERCENT / 100.0)
print(f"Minority Threshold Count (final): {minority_threshold_count_final:.0f}")
print("Final Class Counts:")
for label_id in range(NUM_CLASSES):
    count = label_counts_final.get(label_id, 0)
    label_name = emotions_id2label.get(label_id, f"Unknown({label_id})")
    below_threshold_flag = "*" if count < minority_threshold_count_final else ""
    print(f"  - {label_name} (ID: {label_id}): {count} samples {below_threshold_flag}")


# --- Prepare datasets dictionary with augmented training data ---
use_emotion_dataset = {
    'train': augmented_train_dataset, # Use augmented data for training
    'test': use_test_dataset,
    'validation': use_validation_dataset
}

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
    # Avoid division by zero for labels that might not appear in the subset
    count = label_counts[i] if label_counts[i] > 0 else 1
    class_weights_calc[i] = total_samples / (NUM_CLASSES * count)

sample_weights_np = np.zeros(total_samples, dtype=np.float32)
for i in range(total_samples):
    sample_label_indices = np.where(train_labels_np[i] == 1.0)[0]
    if len(sample_label_indices) > 0:
        sample_weights_np[i] = max(class_weights_calc[idx] for idx in sample_label_indices)
    else:
        sample_weights_np[i] = 1.0

# Define feature columns required by the model (excluding token_type_ids for RoBERTa)
feature_cols = ["input_ids", "attention_mask"]
label_col = "labels"
cols_to_set_format = feature_cols + [label_col]

actual_train_cols = list(emotions_encoded['train'].features)
actual_test_cols = list(emotions_encoded['test'].features)
actual_validation_cols = list(emotions_encoded['validation'].features)
final_train_cols = [col for col in cols_to_set_format if col in actual_train_cols]
final_test_cols = [col for col in cols_to_set_format if col in actual_test_cols]
final_validation_cols = [col for col in cols_to_set_format if col in actual_validation_cols]

# Extract features and labels as numpy arrays before creating dataset
train_features_np = {col: np.array(emotions_encoded['train'][col]) for col in feature_cols}
train_labels_np = np.array(emotions_encoded['train']['labels']) #already have, resetting

test_features_np = {col: np.array(emotions_encoded['test'][col]) for col in feature_cols}
test_labels_np = np.array(emotions_encoded['test']['labels'])

validate_features_np = {col: np.array(emotions_encoded['validation'][col]) for col in feature_cols}
validate_labels_np = np.array(emotions_encoded['validation']['labels'])

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_features_np, train_labels_np, sample_weights_np)
)
train_dataset = train_dataset.shuffle(len(sample_weights_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) # Add prefetch

# Test dataset remains (features, labels)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_features_np, test_labels_np)
)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) # Add prefetch

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validate_features_np, validate_labels_np)
)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) # Add prefetch
print("Datasets created successfully.")


def predict_emotion(text, model, threshold=PROBABILITY_THRESHOLD):
    """Predict multiple emotions for a given text using the provided model and threshold"""
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

test_texts = [
    "I'm so happy today!",
    "This makes me really angry.",
    "I'm feeling very sad and disappointed.",
    "That's really interesting, tell me more.",
    "I am both excited and nervous about the presentation.", # data with multiple emotions
]

untrained_classifier = RoBERTaForClassification(model, num_classes=NUM_CLASSES)

untrained_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    # Use BinaryCrossentropy for multi-label with sigmoid activation
    loss=tf.keras.losses.BinaryCrossentropy(),
    # Use BinaryAccuracy for multi-label evaluation
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)

print("Predictions with UNTRAINED model (random weights - multi-label):")
print("-------------------------------------------------------------")

for text in test_texts:
    result = predict_emotion(text, untrained_classifier, threshold=PROBABILITY_THRESHOLD)
    print(f"Text: {result['text']}")
    print(f"Predicted emotions: {result['emotions']}")
    emotion_confidence_pairs = list(zip(result['emotions'], result['confidences']))
    print(f"Confidences: {emotion_confidence_pairs}")
    print()

classifier = RoBERTaForClassification(model, num_classes=NUM_CLASSES)

print("Compiling model with AUC, Precision, Recall...")
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), # Consider trying AdamW later
    loss=tf.keras.losses.BinaryCrossentropy(), # Correct loss for multi-label sigmoid
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(multi_label=True, name='auc'), # Good overall multi-label metric
        tf.keras.metrics.Precision(name='precision'), # How many selected items are relevant?
        tf.keras.metrics.Recall(name='recall') # How many relevant items are selected?
        ]
)
print("Model compiled.")

# Train the model
print("Starting multi-label model training with sample weights...")

history = classifier.fit(
    train_dataset,
    epochs=EPOCHS, 
    validation_data=validation_dataset
)
print("Training finished.")

print("Evaluating model on test set...")
results = classifier.evaluate(test_dataset, verbose=1)

# Print evaluation results dynamically based on compiled metrics
print("\nTest Set Evaluation Results:")
for name, value in zip(classifier.metrics_names, results):
    print(f"- {name}: {value:.4f}")

print("Predictions with TRAINED multi-label model:")
print("----------------------------------------")

for text in test_texts:
    result = predict_emotion(text, classifier, threshold=PROBABILITY_THRESHOLD)
    print(f"Text: {result['text']}")
    print(f"Predicted emotions: {result['emotions']}")
    # Zip confidences with emotions for clarity
    emotion_confidence_pairs = list(zip(result['emotions'], result['confidences']))
    print(f"Confidences: {emotion_confidence_pairs}")
    print()