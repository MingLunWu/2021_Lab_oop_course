from transformers import BertModel, BertConfig, BertTokenizer

model_name = 'bert-base-uncased'

# Build Tokenizer 
tokenizer = BertTokenizer.from_pretrained(model_name)

# Build BERT Model
# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig.from_pretrained(model_name)
# Initializing a model from the bert-base-uncased style configuration
model = BertModel(configuration)
# Accessing the model configuration
configuration = model.config
print("stop here")

# Get Word Embedding

text = "I wonder how, I wonder why, yesterday you told me about the blue blue sky."

inputs = tokenizer(text, return_tensors="pt")
outputs = model(
    input_ids = inputs['input_ids'],
    attention_mask = inputs['attention_mask'],
    token_type_ids = inputs['token_type_ids']
)

# 等同於 
# outputs = model(**outputs)

# 取出 Last Hidden State (Word Embedding)
word_emb = outputs.last_hidden_state

print("stop here!")



