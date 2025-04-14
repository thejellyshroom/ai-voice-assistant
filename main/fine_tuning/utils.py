import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

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

common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 'for', 'with', 'about',
                   'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                   'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                   'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                   'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                   'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                   'can', 'will', 'just', 'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our',
                   'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                   'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
                   'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                   'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'}

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



# --- Augmentation Functions ---
# Define a mapping from NLTK POS tags to WordNet POS tags
def get_wordnet_pos(nltk_tag):
    """Convert NLTK POS tag to WordNet POS tag for better synonym matching."""
    if nltk_tag.startswith('J'):  # Adjective
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):  # Verb
        return wordnet.VERB
    elif nltk_tag.startswith('N'):  # Noun
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):  # Adverb
        return wordnet.ADV
    else:
        return None  # No matching WordNet POS tag

def get_synonyms(word, pos=None):
    """Fetches synonyms for a word using WordNet, filtered by part of speech."""
    synonyms = set()
    
    if word.lower() in common_words or len(word) <= 2 or not word.isalpha():
        return list(synonyms)  # Return empty list for these words
    
    if pos:
        # Use specific POS
        synsets = wordnet.synsets(word, pos=pos)
    else:
        synsets = wordnet.synsets(word)
        
    for syn in synsets:
        for lemma in syn.lemmas():
            # Filter only single-word synonyms that are different from original
            synonym = lemma.name().replace('_', ' ').lower()
            if ' ' not in synonym and synonym != word.lower() and synonym.isalpha():
                synonyms.add(synonym)
                
    return list(synonyms)

def augment_text_synonym_replacement(text, augmentation_rate=0.6):
    """Replaces words with synonyms at a given rate."""
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    new_words = words.copy()
    
    # Target only longer content words (nouns, verbs, adjectives, adverbs)
    content_word_indices = [i for i, (word, tag) in enumerate(tagged_words) 
                            if len(word) > 3 and get_wordnet_pos(tag) is not None]
    
    if not content_word_indices:
        return text
        
    num_words_to_replace = max(1, min(int(len(content_word_indices) * augmentation_rate), 5))
    # Don't replace more than 3 words to avoid excessive changes
    
    # Shuffle the indices to pick random content words
    random.shuffle(content_word_indices)
    replaced_count = 0

    for i in content_word_indices:
        if replaced_count >= num_words_to_replace:
            break
            
        word = words[i]
        pos = get_wordnet_pos(tagged_words[i][1])  # Get WordNet POS from NLTK tag
        synonyms = get_synonyms(word, pos)
        
        if synonyms:
            synonym = random.choice(synonyms)
            # Preserve capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
            new_words[i] = synonym
            replaced_count += 1

    # Handle cases where tokenization might add spaces around punctuation
    result = ' '.join(new_words)
    for fix in [(" 's", "'s"), (" n't", "n't"), (" .", "."), (" ,", ","), 
               (" ?", "?"), (" !", "!"), (" ;", ";"), (" :", ":")]:
        result = result.replace(*fix)
    
    return result


def augment_data(example, minority_classes_set=None):
    """Applies augmentation, targeting minority classes with higher probability."""
    should_augment = False

    if minority_classes_set:
        example_labels = set(example.get('labels', []))
        # Check for intersection between example labels and minority classes
        if not example_labels.isdisjoint(minority_classes_set):
            should_augment = True

    if should_augment:
        original_text = example['text']
        augmented_text = augment_text_synonym_replacement(original_text)
        # Only update if augmentation actually changed the text
        if augmented_text != original_text:
            example['text'] = augmented_text

    return example