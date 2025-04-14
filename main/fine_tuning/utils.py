import random
import nltk
import torch
import re
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from collections import Counter
import datasets
import time

emotions_id2label = {
    0: 'admiration',
    1: 'amusement',
    2: 'anger',
    3: 'annoyance',
    4: 'approval',
    5: 'caring',
    6: 'confusion',
    7: 'curiosity',
    8: 'desire',
    9: 'disappointment',
    10: 'disapproval',
    11: 'disgust',
    12: 'embarrassment',
    13: 'excitement',
    14: 'fear',
    15: 'gratitude',
    16: 'grief',
    17: 'joy',
    18: 'love',
    19: 'nervousness',
    20: 'optimism',
    21: 'pride',
    22: 'realization',
    23: 'relief',
    24: 'remorse',
    25: 'sadness',
    26: 'surprise',
    27: 'neutral'  # Last entry (no comma)
}
emotions_label2id = {v: k for k, v in emotions_id2label.items()}

def manage_dataset_columns(datasets, 
                          columns_to_remove=None, 
                          column_renames=None, 
                          verbose=True):
    modified_datasets = {}
    
    for split, dataset in datasets.items():
        modified_dataset = dataset
        
        # Remove columns if they exist
        if columns_to_remove:
            actual_columns_to_remove = [col for col in columns_to_remove if col in modified_dataset.features]
            if actual_columns_to_remove:
                if verbose:
                    print(f"[{split}] Removing columns: {actual_columns_to_remove}")
                modified_dataset = modified_dataset.remove_columns(actual_columns_to_remove)
        
        # Rename columns
        if column_renames:
            for source_col, target_col in column_renames.items():
                if source_col in modified_dataset.features:
                    if verbose:
                        print(f"[{split}] Renaming '{source_col}' to '{target_col}'")
                    modified_dataset = modified_dataset.rename_column(source_col, target_col)
        modified_datasets[split] = modified_dataset
    
    return modified_datasets

augmentation_model = "huihui-ai/Llama-3.2-1B-Instruct-abliterated"
augmentation_tokenizer = AutoTokenizer.from_pretrained(augmentation_model)
augmentation_generator = AutoModelForCausalLM.from_pretrained(
    augmentation_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

pipe = pipeline(
    "text-generation",
    model=augmentation_generator,
    tokenizer=augmentation_tokenizer,
    max_length=512,
    truncation=True,
)
if pipe.tokenizer.pad_token_id is None:
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

def generate_augmented_text(original_text, target_label_id, augmentation_pipe):
    emotional_label_str = emotions_id2label[target_label_id]
    prompt = f"Rephrase the following text to strongly express the emotion '{emotional_label_str}'. Output ONLY the rephrased sentence(s), nothing else. Do not explain or add notes. Original text: '{original_text}'. Rephrased text:"
    augmented_text = ""

    # List of forbidden substrings often found in bad augmentations
    forbidden_phrases = [
        "rephrased text:", "original text:", "note:", "explanation:",
        "step ", "option ", "disclaimer:", "cannot assist", "unable to",
        "i cannot", "i'm unable", "my purpose is", "as an ai",
        "this is a test", "rephrasing:", "augmented:", "sentiment:",
        "emotion label:", "context:",
        # more?
    ]

    max_words = 50 
    min_words = 3  

    while not augmented_text:
        outputs = augmentation_pipe(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=1.0,
            pad_token_id=augmentation_pipe.tokenizer.pad_token_id,
            num_return_sequences=1, 
        )
        if outputs and outputs[0]['generated_text']:
            generated_full_text = outputs[0]['generated_text']
            # Find the prompt end and take the text after it
            prompt_end_index = generated_full_text.find(prompt)
            if prompt_end_index != -1:
                potential_augmented_text = generated_full_text[prompt_end_index + len(prompt):].strip()

            # --- Start Enhanced Filtering ---
            if potential_augmented_text:
                clean_text = re.sub(r"^[^\w\(\)]+", "", potential_augmented_text).strip()
                if not clean_text.endswith(('.', '!', '?')):
                    last_punc = max(clean_text.rfind('.'), clean_text.rfind('!'), clean_text.rfind('?'))
                    if last_punc != -1:
                        clean_text = clean_text[:last_punc+1]
                    else:
                        last_space = clean_text.rfind(' ')
                        if last_space != -1:
                            clean_text = clean_text[:last_space].strip()

                # Check for forbidden phrases (case-insensitive)
                lower_clean_text = clean_text.lower()
                if any(phrase in lower_clean_text for phrase in forbidden_phrases):
                    print(f"Filtered (forbidden phrase): '{clean_text}'")
                    potential_augmented_text = "" # Discard
                else:
                    # Check word count
                    word_count = len(clean_text.split())
                    if not (min_words <= word_count <= max_words):
                        print(f"Filtered (word count {word_count}): '{clean_text}'")
                        potential_augmented_text = "" # Discard
                    else:
                         # Final check against original text
                         if clean_text.lower() == original_text.lower():
                              print(f"Filtered (same as original): '{clean_text}'")
                              potential_augmented_text = "" #Discard
                         else:
                             potential_augmented_text = clean_text # Keep the cleaned text
            if (potential_augmented_text):
                augmented_text = potential_augmented_text
                break
            else:
                 print(f"Discarded generated text for '{original_text}' after filtering.")
    
    print(f"Original: {original_text}")
    print(f"Augmented: {augmented_text}")
    return augmented_text

def iterative_augment_minority_classes(
    train_dataset,
    minority_threshold_percent,
    emotions_id2label_map,
    augmentation_pipe,
    max_iterations=50,
    target_augmentation_factor=2.0 # Aim to increase minority samples by this factor per iteration
    ):

    print("\n--- Starting Iterative Data Augmentation ---")
    current_train_dataset = train_dataset
    num_classes = len(emotions_id2label_map)

    for iteration in range(max_iterations):
        print(f"\nAugmentation Iteration {iteration + 1}/{max_iterations}")

        all_labels = [label for sublist in current_train_dataset['labels'] for label in sublist]
        label_counts = Counter(all_labels)
        total_samples = len(current_train_dataset)
        minority_threshold_count = total_samples * (minority_threshold_percent / 100.0)

        minority_classes = {
            label: count for label, count in label_counts.items()
            if count < minority_threshold_count
        }

        if not minority_classes:
            print("No minority classes found below the threshold. Augmentation complete.")
            break

        print(f"Identified {len(minority_classes)} minority classes (threshold count < {minority_threshold_count:.0f}):")
        for label_id, count in minority_classes.items():
             label_name = emotions_id2label_map.get(label_id, f"Unknown({label_id})")
             print(f"  - {label_name} (ID: {label_id}): {count} samples")

        newly_augmented_data = {feature: [] for feature in current_train_dataset.features}
        processed_texts_in_iter = set() # Avoid re-augmenting same text for different minority labels in one go

        sorted_minority_classes = sorted(minority_classes.items(), key=lambda item: item[1])

        for label_id, current_count in sorted_minority_classes:
             samples_needed = int((minority_threshold_count - current_count) * target_augmentation_factor)
             samples_to_generate = max(1, samples_needed) # Ensure we generate at least one if needed
             print(f"  Targeting label {emotions_id2label_map.get(label_id)} (ID: {label_id}). Need ~{samples_needed} more. Generating up to {samples_to_generate}.")

             # Find original samples containing this minority label
             candidate_indices = [
                 i for i, sample_labels in enumerate(current_train_dataset['labels'])
                 if label_id in sample_labels and current_train_dataset['text'][i] not in processed_texts_in_iter
             ]

             generated_count = 0
             random.shuffle(candidate_indices)

             for idx in candidate_indices:
                 if generated_count >= samples_to_generate:
                     break

                 original_sample = current_train_dataset[idx]
                 original_text = original_sample['text']

                 # Generate augmented text specifically for this minority label
                 augmented_text = generate_augmented_text(
                     original_text,
                     label_id,
                     augmentation_pipe
                 )

                 if augmented_text:
                     newly_augmented_data['text'].append(augmented_text)
                     newly_augmented_data['labels'].append(original_sample['labels'])
                     new_id = f"{original_sample['id']}_aug_{iteration+1}_{generated_count}"
                     newly_augmented_data['id'].append(new_id)

                     processed_texts_in_iter.add(original_text) # Mark base text as processed for this iteration
                     processed_texts_in_iter.add(augmented_text) # Avoid augmenting the newly created text immediately
                     generated_count += 1

             print(f"    Generated {generated_count} new samples for label {label_id}.")


        # 4. Create and Concatenate Augmented Dataset
        if newly_augmented_data['text']:
            final_augmented_data = {k: v for k, v in newly_augmented_data.items() if v}

            augmented_dataset_chunk = datasets.Dataset.from_dict(
                final_augmented_data, # Use the dict with all required keys
                features=current_train_dataset.features
            )
            print(f"Created augmented chunk with {len(augmented_dataset_chunk)} samples.")

            current_train_dataset = datasets.concatenate_datasets([current_train_dataset, augmented_dataset_chunk])
            print(f"New training dataset size: {len(current_train_dataset)}")
    else: # This else block executes if the loop completes without break (i.e., max_iterations reached)
        print(f"Reached maximum augmentation iterations ({max_iterations}). Stopping.")

    print("--- Finished Iterative Data Augmentation ---")
    return current_train_dataset