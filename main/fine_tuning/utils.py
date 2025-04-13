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

