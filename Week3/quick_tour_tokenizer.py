from transformers import AutoTokenizer
from transformers import BertTokenizer

model_name = 'bert-base-uncased'
# 初次執行會自動下載相關檔案
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 基本用法
encoded_text = tokenizer.encode("We are very happy to show you the 🤗 Transformers library.")
reverse_text = tokenizer.decode(encoded_text)

# 進階用法 (根據需要加入參數)
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=100
)

print("Stop Here!")