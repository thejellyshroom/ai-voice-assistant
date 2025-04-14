import tensorflow as tf

class RoBERTaForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        # Change activation to 'sigmoid' for multi-label classification
        self.fc = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        # Prepare input dictionary for the BERT model
        bert_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        # Only include token_type_ids if present in the input dictionary
        if 'token_type_ids' in inputs:
            bert_inputs['token_type_ids'] = inputs['token_type_ids']

        # Pass the prepared dictionary to the BERT model
        outputs = self.bert(**bert_inputs, return_dict=True)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)