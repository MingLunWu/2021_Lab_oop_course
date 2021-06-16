import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from datasets import load_dataset # Open Source Package
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm

# 自行定義Bert Module
class BertForSequenceClassification(nn.Module):
    # 定義模型架構
    def __init__(self) -> None:
        super().__init__()
        self.config = BertConfig.from_pretrained("bert-base-cased")

        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    # 定義 Forward Propagation的流程
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits



if __name__ == "__main__":
    
    # 下載範例資料集，並且進行處理
    raw_datasets = load_dataset("imdb")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    # Debug 用
    train_feature = next(iter(train_dataloader))

    # 定義模型
    model = BertForSequenceClassification()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # 設定 GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(1)
    model.to(device)    

    # 設定 Loss Function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    # 訓練流程
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # 把 Value 放進去 GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            # outputs = model(**batch)
            outputs = model.forward(batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
            loss = loss_fn(outputs, batch['label'].double().view(-1,1).to(device))
            # 進行 Backward Propagation
            loss.backward()
            # 使用優化器修正模型
            optimizer.step()
            optimizer.zero_grad()
            # 使用進度條顯示訓練狀況
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': loss.item()})
    
    # 訓練完成後，準備進行 Evaluate: 
    model.eval()

    predict_result = model(model.forward(batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids']))

    print("test")