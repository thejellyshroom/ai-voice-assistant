import os
from collections import Counter
from datasets import load_dataset, DatasetDict, load_from_disk
from huggingface_hub import HfApi, hf_hub_download
from typing import Union

# Assuming utils_emotiontraining is in the same directory or accessible via PYTHONPATH
from utils_emotiontraining import iterative_augment_minority_classes

def load_or_augment_dataset(
    hf_dataset_id: str,
    local_save_path: str,
    emotions_id2label: dict,
    augmentation_pipe,
    force_reaugment: bool = False,
    base_dataset_name: str = "google-research-datasets/go_emotions",
    base_dataset_config: str = "simplified",
    minority_threshold_percent: float = 1.5,
    test_train_range: Union[int, None] = None,
    test_test_range: Union[int, None] = None,
    test_validate_range: Union[int, None] = None,
) -> Union[DatasetDict, None]:
    """
    Attempts to load the augmented dataset first from the Hugging Face Hub,
    then from a local path. If not found or force_reaugment is True,
    it performs the augmentation process.

    Args:
        hf_dataset_id: The repository ID on Hugging Face Hub (e.g., "username/dataset_name").
        local_save_path: The local directory path to save/load the dataset.
        emotions_id2label: Mapping from label ID to emotion name.
        augmentation_pipe: The pre-initialized text generation pipeline for augmentation.
        force_reaugment: If True, skip loading and perform augmentation.
        base_dataset_name: Name of the original dataset on Hugging Face Hub.
        base_dataset_config: Configuration of the original dataset (e.g., "simplified").
        minority_threshold_percent: Threshold to identify minority classes.
        test_train_range: Max number of samples for the training split (for testing).
        test_test_range: Max number of samples for the test split (for testing).
        test_validate_range: Max number of samples for the validation split (for testing).

    Returns:
        The loaded or augmented DatasetDict, or None if loading fails.
    """
    loaded_dataset = None

    if not force_reaugment:
        print(f"--- Dataset Loading Phase ---")
        # 1. Try loading from Hugging Face Hub
        print(f"Attempting to load augmented dataset from Hugging Face Hub: {hf_dataset_id}")
        try:
            loaded_dataset = load_dataset(hf_dataset_id)
            print("Successfully loaded augmented dataset from the Hub.")
            return loaded_dataset
        except Exception as e:
            print(f"Could not load dataset from Hub: {e}")

        # 2. Try loading from local disk
        if os.path.exists(local_save_path):
            print(f"Attempting to load augmented dataset from local path: {local_save_path}")
            try:
                loaded_dataset = load_from_disk(local_save_path)
                print("Successfully loaded augmented dataset from local disk.")
                return loaded_dataset
            except Exception as e:
                print(f"Could not load dataset from local disk: {e}")
        else:
             print(f"Local dataset path not found: {local_save_path}")

    # 3. If loading failed or forced, perform augmentation
    if loaded_dataset is None:
        print("--- Augmentation Phase --- ")
        print(f"Loading original dataset: {base_dataset_name} ({base_dataset_config}) ...")
        try:
            emotion_dataset = load_dataset(base_dataset_name, base_dataset_config)
        except Exception as e:
            print(f"Fatal Error: Could not load base dataset {base_dataset_name}. {e}")
            return None

        use_train_dataset = emotion_dataset['train']
        use_test_dataset = emotion_dataset['test']
        use_validation_dataset = emotion_dataset['validation']

        # Apply slicing if ranges are specified
        if test_train_range is not None:
            print(f"Selecting {test_train_range} samples for training.")
            use_train_dataset = use_train_dataset.select(range(test_train_range))
        if test_test_range is not None:
            print(f"Selecting {test_test_range} samples for testing.")
            use_test_dataset = use_test_dataset.select(range(test_test_range))
        if test_validate_range is not None:
            print(f"Selecting {test_validate_range} samples for validation.")
            use_validation_dataset = use_validation_dataset.select(range(test_validate_range))

        print("Analyzing original training data for minority classes...")
        original_train_labels = use_train_dataset['labels']
        all_labels = [label for sublist in original_train_labels for label in sublist]
        label_counts_original = Counter(all_labels)
        total_samples_original = len(use_train_dataset)

        minority_threshold_count = total_samples_original * (minority_threshold_percent / 100.0)
        minority_classes = {
            label for label, count in label_counts_original.items()
            if count < minority_threshold_count
        }
        if minority_classes:
            print(f"Identified {len(minority_classes)} minority classes (frequency < {minority_threshold_percent}%, count < {minority_threshold_count:.0f}):")
            minority_class_names = {emotions_id2label.get(idx, f"Unknown({idx})") for idx in minority_classes}
            print(f"Minority Classes Indices: {minority_classes}")
            print(f"Minority Classes Names: {minority_class_names}")
        else:
            print("No minority classes identified below threshold.")

        print("Starting iterative data augmentation...")
        augmented_train_dataset = iterative_augment_minority_classes(
            train_dataset=use_train_dataset,
            minority_threshold_percent=minority_threshold_percent,
            emotions_id2label_map=emotions_id2label,
            augmentation_pipe=augmentation_pipe
        )
        print(f"Finished iterative augmentation. Final training set size: {len(augmented_train_dataset)}")

        # Combine augmented train with original test/validation
        augmented_emotion_dataset = DatasetDict({
            'train': augmented_train_dataset,
            'test': use_test_dataset,
            'validation': use_validation_dataset
        })
        print("Combined augmented train data with original test/validation splits.")

        # Save locally
        print(f"Saving augmented dataset to local path: {local_save_path}")
        try:
            # Ensure parent directory exists if path includes directories
            os.makedirs(os.path.dirname(local_save_path), exist_ok=True)
            augmented_emotion_dataset.save_to_disk(local_save_path)
            print("Successfully saved dataset locally.")
        except Exception as e:
            print(f"Error saving dataset locally: {e}")

        # Push to Hub
        print(f"Attempting to push augmented dataset to Hugging Face Hub: {hf_dataset_id}")
        try:
            augmented_emotion_dataset.push_to_hub(hf_dataset_id, private=False) # Set private=True if needed
            print("Successfully pushed dataset to the Hub.")
        except Exception as e:
            print(f"Error pushing dataset to Hub: {e}")
            print("Ensure you are logged in (`huggingface-cli login`) and the repository name is valid.")

        return augmented_emotion_dataset

    # Should not happen if logic is correct, but return loaded if it exists
    return loaded_dataset
