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
    loaded_dataset = None

    if not force_reaugment:
        print(f"Attempting to load augmented dataset from Hugging Face Hub: {hf_dataset_id}")
        try:
            loaded_dataset = load_dataset(hf_dataset_id)
            return loaded_dataset
        except Exception as e:
            print(f"Could not load dataset from Hub: {e}")

        if os.path.exists(local_save_path):
            try:
                loaded_dataset = load_from_disk(local_save_path)
                return loaded_dataset
            except Exception as e:
                print(f"Could not load dataset from local disk: {e}")
        else:
             print(f"Local dataset path not found: {local_save_path}")

    if loaded_dataset is None:
        print(f"Loading original dataset: {base_dataset_name} ({base_dataset_config}) ...")
        try:
            emotion_dataset = load_dataset(base_dataset_name, base_dataset_config)
        except Exception as e:
            print(f"Fatal Error: Could not load base dataset {base_dataset_name}. {e}")
            return None

        use_train_dataset = emotion_dataset['train']
        use_test_dataset = emotion_dataset['test']
        use_validation_dataset = emotion_dataset['validation']

        print(f"Selecting {test_train_range} samples for training.")
        use_train_dataset = use_train_dataset.select(range(test_train_range))
        print(f"Selecting {test_test_range} samples for testing.")
        use_test_dataset = use_test_dataset.select(range(test_test_range))
        print(f"Selecting {test_validate_range} samples for validation.")
        use_validation_dataset = use_validation_dataset.select(range(test_validate_range))

        print("Analyzing for minority classes...")
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

        augmented_train_dataset = iterative_augment_minority_classes(
            train_dataset=use_train_dataset,
            minority_threshold_percent=minority_threshold_percent,
            emotions_id2label_map=emotions_id2label,
            augmentation_pipe=augmentation_pipe
        )
        print(f"Finished iterative augmentation. Final training set size: {len(augmented_train_dataset)}")

        augmented_emotion_dataset = DatasetDict({
            'train': augmented_train_dataset,
            'test': use_test_dataset,
            'validation': use_validation_dataset
        })
        print("Combined augmented train data with original test/validation splits.")

        # Save locally
        try:
            os.makedirs(os.path.dirname(local_save_path), exist_ok=True)
            augmented_emotion_dataset.save_to_disk(local_save_path)
        except Exception as e:
            print(f"Error saving dataset locally: {e}")

        # Push to Hub
        try:
            augmented_emotion_dataset.push_to_hub(hf_dataset_id, private=False) # Set private=True if needed
            print("Successfully pushed dataset to the Hub.")
        except Exception as e:
            print(f"Error pushing dataset to Hub: {e}")
            print("Ensure you are logged in (`huggingface-cli login`) and the repository name is valid.")

        return augmented_emotion_dataset

    return loaded_dataset
