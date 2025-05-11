import tensorflow as tf
from transformers import TFAutoModel
import os

BERT_MODEL_NAME = "distilroberta-base"
BERT_WEIGHTS_PATH = "main/fine_tuning/best_emotion_model_weights.h5"
NUM_CLASSES = 28

class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__(name="BERTForClassification_Custom")
        self.bert = bert_model
        self.num_classes = num_classes
        self.fc = tf.keras.layers.Dense(num_classes, activation='sigmoid', name="classifier_head")

    def call(self, inputs):
        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

def debug_load_bert_weights_by_name():
    print("--- Starting BERT Weight Loading Debug (by_name=True) ---")
    print(f"Attempting to load model: {BERT_MODEL_NAME}")
    print(f"Attempting to load weights from: {BERT_WEIGHTS_PATH}")
    
    if not os.path.exists(BERT_WEIGHTS_PATH):
        print(f"!!! FATAL ERROR: Weights file not found at {BERT_WEIGHTS_PATH} !!!")
        return

    try:
        print(f"Loading base model: {BERT_MODEL_NAME}")
        base_model = TFAutoModel.from_pretrained(BERT_MODEL_NAME)
        print("Instantiating BERTForClassification with named layers...")
        classifier = BERTForClassification(base_model, num_classes=NUM_CLASSES)
    
        print("Building model with dummy input...")
        dummy_input_ids = tf.zeros([1, 10], dtype=tf.int32)
        dummy_attention_mask = tf.zeros([1, 10], dtype=tf.int32)
        dummy_inputs = {'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask}
        _ = classifier(dummy_inputs) # Build step   
        print("Model built.")
        
        print("Model Summary (for checking layer names):")
        classifier.summary()
    
        print(f"Loading weights from: {BERT_WEIGHTS_PATH} BY NAME")
        classifier.load_weights(BERT_WEIGHTS_PATH, by_name=True)
        print("*******************************************************")
        print("*** BERT model weights loaded SUCCESSFULLY (by_name=True)! ***")
        print("*******************************************************")

    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! ERROR during BERT weight loading (by_name=True): {e} !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    debug_load_bert_weights_by_name() 