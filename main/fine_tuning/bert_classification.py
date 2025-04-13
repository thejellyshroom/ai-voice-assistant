import tensorflow as tf

class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        # Change activation to 'sigmoid' for multi-label classification
        self.fc = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        # Make sure we handle the case when inputs is a dictionary
        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'],
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)