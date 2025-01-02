import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import pandas as pd

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 自定义数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 加载数据
train_df = pd.read_csv(r'data\train.tsv', delimiter='\t')
dev_df = pd.read_csv(r'data\dev.tsv', delimiter='\t')
test_df = pd.read_csv(r'data\test.tsv', delimiter='\t')

# 提取文本和标签
train_texts = train_df['text_a'].tolist()
train_labels = train_df['label'].tolist()
dev_texts = dev_df['text_a'].tolist()
dev_labels = dev_df['label'].tolist()
test_texts = test_df['text_a'].tolist()
test_labels = test_df['label'].tolist()

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 将模型移动到GPU
model.to(device)

# 数据预处理
max_len = 128
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
dev_dataset = SentimentDataset(dev_texts, dev_labels, tokenizer, max_len)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练模型
best_accuracy = 0.0
best_model_state_dict = None

model.train()
for epoch in range(3):  # 训练3个epoch
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 在每个epoch结束后使用验证集评估模型性能
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    print(f'Validation Accuracy after epoch {epoch}: {accuracy}')

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state_dict = model.state_dict()

    model.train()  # 切换回训练模式

# 保存最佳模型
model_save_path = 'cold_emotion_model.pth'
torch.save(best_model_state_dict, model_save_path)
print(f'Best model saved to {model_save_path} with validation accuracy: {best_accuracy}')

# 在测试集上评估模型
model.load_state_dict(best_model_state_dict)
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f'Test Accuracy: {accuracy}')
