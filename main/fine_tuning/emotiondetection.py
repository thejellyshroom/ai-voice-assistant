from datasets import load_dataset, DatasetDict, Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

import evaluate
import torch
import numpy as np
import gc

emotion_dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# model_checkpoint = 'prithivMLmods/Llama-Sentient-3.2-3B-Instruct'
# model_checkpoint = 'ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g'
model_checkpoint = 'unsloth/Llama-3.2-1B-Instruct'

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

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=28, id2label=emotions_id2label, 
    label2id=emotions_label2id, 
    problem_type="multi_label_classification"
)

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(list(emotions_id2label.values()))
    model.resize_token_embeddings(len(tokenizer))

# create tokenize function
def tokenize_function(examples):
    # tokenize and truncate text
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )
    
    # Convert labels to one-hot encoding for each example in the batch
    batch_size = len(examples["labels"])
    batch_one_hot_labels = []
    
    for i in range(batch_size):
        # Create a one-hot encoded array for this example
        one_hot_labels = np.zeros(len(emotions_id2label), dtype=np.float32)
        # Set the corresponding positions to 1 for each label in this example
        for label in examples["labels"][i]:
            one_hot_labels[label] = 1.0
        batch_one_hot_labels.append(one_hot_labels)
    
    tokenized_inputs["labels"] = batch_one_hot_labels
    
    return tokenized_inputs

# tokenize training and validation datasets
tokenized_dataset = emotion_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=4  # Reduced batch size for processing
)
# create data collator
@dataclass
class MultiLabelDataCollator:
    """
    Data collator that properly handles multi-label classification datasets
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        
        # Handle input_ids, attention_mask, token_type_ids
        for k in ['input_ids', 'attention_mask', 'token_type_ids']:
            if k in features[0]:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
        
        # Special handling for labels
        if "labels" in features[0]:
            # Ensure labels are float tensors for multi-label classification
            labels = [f["labels"] for f in features]
            if isinstance(labels[0], torch.Tensor):
                batch["labels"] = torch.stack(labels).float()
            else:
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
        
        return batch

# Initialize the custom data collator
data_collator = MultiLabelDataCollator()

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = (predictions > 0.5).astype(int)  # Convert logits to binary predictions
    
    # Calculate accuracy for multi-label classification
    accuracy_value = (predictions == labels).mean()
    return {"accuracy": accuracy_value}


# PEFT configuration for sequence classification (not causal language modeling)
peft_config = LoraConfig(
    task_type="SEQ_CLS",  # Use sequence classification task type instead of CAUSAL_LM
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=['q_proj', 'v_proj'],
    modules_to_save=['score']  # Keep score module for classification head
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 2  # Reduce batch size to decrease memory usage
num_epochs = 10

# define training arguments
training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    label_names=["labels"],  # Explicitly define label names
    remove_unused_columns=False,  # Required for custom datasets
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    gradient_accumulation_steps=4,  # Accumulate gradients over multiple steps
    fp16=False,  # Disable half-precision as MPS may have issues with it
    optim="adamw_torch"  # Use torch's native optimizer for better compatibility
)


# Clear memory before training
gc.collect()
torch.mps.empty_cache()
torch.mps.set_per_process_memory_fraction(0.95)  # Further reduced memory fraction

# Add these before training
tokenizer.padding_side = "right"  # Required for Llama models
model.config.pad_token_id = tokenizer.pad_token_id

# Apply manual cleanup to dataset format - we don't need the ID field
def clean_dataset(dataset):
    # Remove 'id' field if present
    if 'id' in dataset.column_names:
        dataset = dataset.remove_columns(['id'])
    return dataset

tokenized_dataset = clean_dataset(tokenized_dataset)

# Convert labels to float tensors in small batches
def convert_to_tensor_format(batch):
    return {"labels": [torch.tensor(item, dtype=torch.float32) for item in batch["labels"]]}

tokenized_dataset = tokenized_dataset.map(
    convert_to_tensor_format,
    batched=True,
    batch_size=4
)

# Fix tuple issue in subset selection and reduce dataset size further
print("Using a smaller subset of data due to memory constraints")
train_subset = tokenized_dataset["train"].select(range(min(200, len(tokenized_dataset["train"]))))
val_subset = tokenized_dataset["validation"].select(range(min(50, len(tokenized_dataset["validation"]))))

# Make sure we're using CPU for training
device = torch.device("cpu")
model = model.to(device)

# Print a sample from the dataset to verify format
print("Sample input_ids shape:", train_subset[0]["input_ids"].shape if hasattr(train_subset[0]["input_ids"], "shape") else len(train_subset[0]["input_ids"]))
print("Sample labels shape/type:", train_subset[0]["labels"].shape if hasattr(train_subset[0]["labels"], "shape") else type(train_subset[0]["labels"]))
print("Sample labels:", train_subset[0]["labels"])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=val_subset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # Use our custom multi-label collator
    compute_metrics=compute_metrics,
)

# train model
try:
    print("Starting training with:", len(train_subset), "training samples")
    print("and", len(val_subset), "validation samples")
    trainer.train()
except Exception as e:
    print(f"Error during training: {str(e)}")
    import traceback
    traceback.print_exc()
    # If it fails, try to debug the issue
    try:
        # Check one batch of data
        batch = data_collator([train_subset[i] for i in range(2)])
        for k, v in batch.items():
            print(f"{k} shape: {v.shape if hasattr(v, 'shape') else type(v)}")
    except Exception as e2:
        print(f"Error during debugging: {str(e2)}")


# generate prediction
model.to('mps') # moving to mps for Mac (can alternatively do 'cpu')



# examples test
# define list of examples
text_list = ["It was good.", "Not a fan, don't recommed.", "Better than the first one.", "This is not worth watching even once.", "This one is a pass."]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # compute logits
    logits = model(inputs).logits
    # convert logits to label
    predictions = torch.argmax(logits)

    print(text + " - " + emotions_id2label[predictions.tolist()])

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("mps") # moving to mps for Mac (can alternatively do 'cpu')

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + emotions_id2label[predictions.tolist()[0]])


# save model
# model.save_pretrained(model_checkpoint + "-lora-text-classification")
# tokenizer.save_pretrained(model_checkpoint + "-lora-text-classification")