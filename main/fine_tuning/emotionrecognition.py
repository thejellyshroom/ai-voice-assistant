import tensorflow as tf
import numpy as np
import tensorflow as tf
from collections import Counter
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
from bert_classification import BERTForClassification
from utils import emotions_id2label, emotions_label2id, manage_dataset_columns

NUM_CLASSES = 28
BATCH_SIZE = 32
PROBABILITY_THRESHOLD = 0.5
EPOCHS = 10

TEST_TRAIN_RANGE = 2000
TEST_TEST_RANGE = 500

model = TFAutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

emotion_dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

use_train_dataset = emotion_dataset['train'].select(range(TEST_TRAIN_RANGE))
use_test_dataset = emotion_dataset['test'].select(range(TEST_TEST_RANGE))
use_emotion_dataset = {
    'train': use_train_dataset,
    'test': use_test_dataset
}

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, return_tensors='tf')

emotions_encoded = {}
emotions_encoded['train'] = use_emotion_dataset['train'].map(tokenize, batched=True, batch_size=None)
emotions_encoded['test'] = use_emotion_dataset['test'].map(tokenize, batched=True, batch_size=None)

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

emotions_encoded = manage_dataset_columns(
    datasets=emotions_encoded,
    columns_to_remove=['label_int', 'labels'],
    column_renames={'multi_hot_labels': 'labels'},
    verbose=True
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

feature_cols = ["input_ids", "token_type_ids", "attention_mask"]
label_col = "labels"
cols_to_set_format = feature_cols + [label_col]

actual_train_cols = list(emotions_encoded['train'].features)
actual_test_cols = list(emotions_encoded['test'].features)
final_train_cols = [col for col in cols_to_set_format if col in actual_train_cols]
final_test_cols = [col for col in cols_to_set_format if col in actual_test_cols]

if all(col in final_train_cols for col in cols_to_set_format) and \
   all(col in final_test_cols for col in cols_to_set_format):
    print("Setting dataset format to TensorFlow")
    # Don't set format yet, extract numpy arrays first, then create dataset
else:
     raise ValueError(f"Error: Could not find all necessary columns. Train has: {actual_train_cols}, Test has: {actual_test_cols}. Needed: {cols_to_set_format}")


# Extract features and labels as numpy arrays before creating dataset
train_features_np = {col: np.array(emotions_encoded['train'][col]) for col in feature_cols}
train_labels_np = np.array(emotions_encoded['train']['labels']) #already have, resetting

test_features_np = {col: np.array(emotions_encoded['test'][col]) for col in feature_cols}
test_labels_np = np.array(emotions_encoded['test']['labels'])

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_features_np, train_labels_np, sample_weights_np)
)
train_dataset = train_dataset.shuffle(len(sample_weights_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) # Add prefetch

# Test dataset remains (features, labels)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_features_np, test_labels_np)
)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) # Add prefetch
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

untrained_classifier = BERTForClassification(model, num_classes=NUM_CLASSES)

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

classifier = BERTForClassification(model, num_classes=NUM_CLASSES)

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
    validation_data=test_dataset
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