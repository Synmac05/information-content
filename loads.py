import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from collections import Counter
from datetime import datetime

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载模型
model_load_path = 'emotion_model.pth'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.load_state_dict(torch.load(model_load_path))
model.to(device)
model.eval()  # 切换到评估模式
print(f'Model loaded from {model_load_path}')

# 从TSV文件中读取输入
input_tsv_path = 'siben.tsv'
input_df = pd.read_csv(input_tsv_path, delimiter='\t')

input_texts = input_df['TEXT'].tolist()

# 统计正面和负面评论
positive_texts = []
negative_texts = []

# 对每一行进行情感预测并分类
for text in input_texts:
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'].to(device), attention_mask=encoding['attention_mask'].to(device))
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    
    if pred == 1:
        positive_texts.append(text)
    else:
        negative_texts.append(text)

# 输出好评和差评的条数
print(f'好评条数: {len(positive_texts)}')
print(f'差评条数: {len(negative_texts)}')

# 将好评和差评分别写入CSV文件
positive_df = pd.DataFrame(positive_texts, columns=['TEXT'])
negative_df = pd.DataFrame(negative_texts, columns=['TEXT'])

positive_df.to_csv('positive_reviews.csv', index=False, encoding='utf-8')
negative_df.to_csv('negative_reviews.csv', index=False, encoding='utf-8')

# # 加载停用词表
# stop_words = set()
# with open('stopwords.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         stop_words.add(line.strip())

# # 文本预处理函数
# def preprocess_text(text):
#     text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号和非文字字符
#     words = jieba.lcut(text)  # 使用jieba进行中文分词
#     words = [word for word in words if word not in stop_words]  # 去除停用词
#     return words

# # 统计词频
# positive_word_counter = Counter()
# negative_word_counter = Counter()

# for text in positive_texts:
#     words = preprocess_text(text)
#     positive_word_counter.update(words)

# for text in negative_texts:
#     words = preprocess_text(text)
#     negative_word_counter.update(words)

# # 打印出现频率最高的前10个单词
# print("正面评论中出现频率最高的前10个单词：")
# print(positive_word_counter.most_common(50))

# print("\n负面评论中出现频率最高的前10个单词：")
# print(negative_word_counter.most_common(50))
