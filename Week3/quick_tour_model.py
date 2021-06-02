from transformers import BertModel, BertConfig

model_name = 'bert-base-uncased'
# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig.from_pretrained(model_name)
# Initializing a model from the bert-base-uncased style configuration
model = BertModel(configuration)
# Accessing the model configuration
configuration = model.config
print("stop here")