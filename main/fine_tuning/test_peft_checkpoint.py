import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel
import os
import numpy as np

# Import necessary items from the utils script
try:
    from utils_emotiontraining import emotions_id2label, TEST_TEXTS
    NUM_CLASSES = len(emotions_id2label)
except ImportError:
    print("Error: Could not import from utils_emotiontraining.py.")
    print("Please ensure the script is in the same directory or adjust the import path.")
    # Define fallbacks if import fails, though they might be incomplete
    emotions_id2label = { # Example fallback
        0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
        5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
        10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement',
        14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
        20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse',
        25: 'sadness', 26: 'surprise', 27: 'neutral'
    }
    NUM_CLASSES = 28
    TEST_TEXTS = ["I'm happy.", "I'm angry."]

# --- Configuration ---
BASE_MODEL_NAME = "distilroberta-base" # Assuming this was the base model for the checkpoint
CHECKPOINT_PATH = "main/fine_tuning/debug_output/checkpoint-8" # Corrected path relative to workspace root
PROBABILITY_THRESHOLD = 0.5 # Adjust as needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

print(f"Using device: {DEVICE}")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_PATH}")

# --- Load Tokenizer ---
# Can load from checkpoint or base model, should be compatible
try:
    print(f"Loading tokenizer from checkpoint: {CHECKPOINT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
except Exception as e:
    print(f"Could not load tokenizer from checkpoint ({e}). Loading from base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# --- Load Base Model ---
# Load the base model architecture FOR SEQUENCE CLASSIFICATION
print(f"Loading base model for sequence classification: {BASE_MODEL_NAME}")
# Make sure to specify the number of labels for the classification head
model_config = AutoConfig.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=NUM_CLASSES,
    # Add mappings if the dataset labels don't match default (optional but good practice)
    id2label=emotions_id2label,
    label2id={v: k for k, v in emotions_id2label.items()}
)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    config=model_config
)

# --- Load PEFT Adapter ---
print(f"Loading PEFT adapter from checkpoint: {CHECKPOINT_PATH}")
# Load the PEFT model by specifying the base model and the checkpoint path
# Ensure the PEFT adapter config (adapter_config.json) is present in the checkpoint dir
peft_model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
print("PEFT adapter loaded.")

# --- Merge Adapter Weights ---
# Merge the adapter weights into the base model for easier inference
print("Merging PEFT adapter weights into the base model...")
merged_model = peft_model.merge_and_unload()
print("Adapter weights merged.")

# Move model to the appropriate device
merged_model.to(DEVICE)
merged_model.eval() # Set model to evaluation mode

# --- Prediction Function ---
def predict_emotion_peft(text, model, tokenizer, device, threshold=PROBABILITY_THRESHOLD):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device

    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply sigmoid to logits for multi-label probabilities
    probabilities = torch.sigmoid(logits).squeeze() # Remove batch dim if batch size is 1

    # Ensure probabilities is on CPU for numpy conversion
    probabilities_cpu = probabilities.cpu().numpy()

    # Get indices where probability exceeds threshold
    predicted_labels_indices = np.where(probabilities_cpu > threshold)[0]

    predicted_emotions = []
    confidences = []

    if len(predicted_labels_indices) > 0:
        for index in predicted_labels_indices:
            predicted_emotions.append(emotions_id2label.get(index, f"Unknown({index})"))
            confidences.append(float(probabilities_cpu[index]))
    else:
        # Fallback: predict the single most likely label if none exceed threshold
        highest_prob_index = np.argmax(probabilities_cpu)
        predicted_emotions.append(emotions_id2label.get(highest_prob_index, f"Unknown({highest_prob_index})"))
        confidences.append(float(probabilities_cpu[highest_prob_index]))

    return {
        'text': text,
        'emotions': predicted_emotions,
        'confidences': confidences
    }

# --- Run Predictions ---
print(f"Running predictions on test texts using PEFT checkpoint {CHECKPOINT_PATH} ---")
for text in TEST_TEXTS:
    result = predict_emotion_peft(text, merged_model, tokenizer, DEVICE, threshold=PROBABILITY_THRESHOLD)
    print(f"Text: {result['text']}")
    print(f"Predicted emotions: {result['emotions']}")
    emotion_confidence_pairs = list(zip(result['emotions'], result['confidences']))
    print(f"Confidences: {emotion_confidence_pairs}")

print("\n--- Prediction complete ---") 