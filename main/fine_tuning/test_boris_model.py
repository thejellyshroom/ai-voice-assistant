from transformers import pipeline
import torch # Or tensorflow if you prefer that backend

# Import test texts from the utils script
try:
    from utils_emotiontraining import TEST_TEXTS, emotions_id2label # Assuming this model uses the same first 28 labels
except ImportError:
    print("Error: Could not import from utils_emotiontraining.py.")
    print("Please ensure the script is in the same directory or adjust the import path.")
    # Define fallback test texts
    TEST_TEXTS = [
        "I'm so happy today!",
        "This makes me really angry.",
        "I'm feeling very sad and disappointed.",
        "That's really interesting, tell me more.",
        "I am both excited and nervous about the presentation."
    ]

# --- Configuration ---
# MODEL_NAME = "bhadresh-savani/bert-base-go-emotion" #not as confident
# MODEL_NAME = "borisn70/bert-43-multilabel-emotion-detection" #best for single emotion
# MODEL_NAME = "SamLowe/roberta-base-go_emotions" # seems to be most sophisticated for multiple emotions, but not very confident
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base" # less accurate
CONFIDENCE_THRESHOLD = 0.70
TOP_N = 3
# Attempt to use GPU if available (mps for Mac, cuda for Nvidia)
DEVICE = -1 # Default to CPU
if torch.cuda.is_available():
    DEVICE = 0 # Use first CUDA device
elif torch.backends.mps.is_available():
     DEVICE = 0 # Use MPS device (will be mapped correctly by pipeline)
     print("Using MPS device (Apple Silicon GPU)")
else:
    print("No GPU detected, using CPU.")
# ---------------------

print(f"Loading pipeline for model: {MODEL_NAME}")
# Use text-classification task and return_all_scores=True
try:
    nlp = pipeline(
        'text-classification',
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        return_all_scores=True,
        device=DEVICE # Pass device index to pipeline
    )
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    exit()

# Get results for all texts at once
print("\nRunning inference...")
all_results = nlp(TEST_TEXTS)
print("Inference complete.")

# Process and print results for each text
print(f"\n--- Processing Results (Top {TOP_N} and Confidence > {CONFIDENCE_THRESHOLD}) ---")

for i, text in enumerate(TEST_TEXTS):
    print(f"\nText: '{text}'")
    results_for_one_text = all_results[i]

    # --- Get Top N Results ---
    # Sort by score descending
    sorted_results = sorted(results_for_one_text, key=lambda x: x['score'], reverse=True)
    top_n_results = sorted_results[:TOP_N]

    print(f"  Top {TOP_N} Emotions:")
    for result in top_n_results:
        print(f"    - {result['label']}: {result['score']:.4f}")

    # --- Get Results Above Threshold (with fallback) ---
    above_threshold_results = [res for res in results_for_one_text if res['score'] > CONFIDENCE_THRESHOLD]

    print(f"  Emotions > {CONFIDENCE_THRESHOLD * 100}% Confidence:")
    if above_threshold_results:
        # Sort them for consistent output order (optional)
        above_threshold_results_sorted = sorted(above_threshold_results, key=lambda x: x['score'], reverse=True)
        for result in above_threshold_results_sorted:
            print(f"    - {result['label']}: {result['score']:.4f}")
    else:
        # Fallback: If none are above threshold, show the single highest one
        if sorted_results: # Ensure there are results to show
             top_result = sorted_results[0]
             print(f"    (No labels > {CONFIDENCE_THRESHOLD * 100}%, showing highest) {top_result['label']}: {top_result['score']:.4f}")
        else:
             print("    (No labels found)")

print("\n--- Processing complete --- ") 