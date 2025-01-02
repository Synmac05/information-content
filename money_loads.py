import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载模型
model_load_path = 'gupiao_emotion_model.pth'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)
model.load_state_dict(torch.load(model_load_path))
model.to(device)
model.eval()  # 切换到评估模式
print(f'Model loaded from {model_load_path}')

# 从TSV文件中读取输入
input_tsv_path = 'shangzheng.tsv'
input_df = pd.read_csv(input_tsv_path, delimiter='\t')

# 提取时间和文本列
input_times = input_df['TIME'].astype(str).tolist()
input_texts = input_df['TEXT'].tolist()

# 创建一个字典来存储每天的评论数量和各类评论数量
daily_stats = {}

# 对每一行进行情感预测
for idx, (time, text) in enumerate(zip(input_times, input_texts)):
    # 提取日期部分
    try:
        date = datetime.strptime(time, '%m-%d').strftime('%m-%d')
    except ValueError:
        print(f"Skipping invalid date format: {time}")
        continue
    
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'].to(device), attention_mask=encoding['attention_mask'].to(device))
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    
    # 打印前100条记录的预测结果
    if idx < 100:
        sentiment = '看空' if pred == 0 else '中性' if pred == 1 else '看多'
        print(f'记录 {idx + 1}: 时间: {time}, 文本: {text}, 预测情感: {sentiment}')
    
    # 更新每日统计数据
    if date not in daily_stats:
        daily_stats[date] = {'total': 0, 'negative': 0, 'neutral': 0, 'positive': 0}
    daily_stats[date]['total'] += 1
    if pred == 0:
        daily_stats[date]['negative'] += 1
    elif pred == 1:
        daily_stats[date]['neutral'] += 1
    elif pred == 2:
        daily_stats[date]['positive'] += 1

# 准备数据用于绘图
dates = sorted(daily_stats.keys())
negative_rates = [(daily_stats[date]['negative'] / daily_stats[date]['total']) * 100 for date in dates]
neutral_rates = [(daily_stats[date]['neutral'] / daily_stats[date]['total']) * 100 for date in dates]
positive_rates = [(daily_stats[date]['positive'] / daily_stats[date]['total']) * 100 for date in dates]
bi_indices = [(daily_stats[date]['positive'] / (daily_stats[date]['positive'] + daily_stats[date]['negative'])) if (daily_stats[date]['positive'] + daily_stats[date]['negative']) > 0 else 0 for date in dates]

# 输出每日的看空率、中性率和看多率
for date in dates:
    print(f'日期: {date}, 看空率: {negative_rates[dates.index(date)]:.2f}%, 中性率: {neutral_rates[dates.index(date)]:.2f}%, 看多率: {positive_rates[dates.index(date)]:.2f}%')

# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 请确保路径正确

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(dates, bi_indices, label='BI指数', marker='o', linestyle='--')
plt.xlabel('日期', fontproperties=font)
plt.ylabel('百分比', fontproperties=font)
plt.title('每日评论情感分析', fontproperties=font)
plt.legend(prop=font)

# 设置横坐标每个月标一次
month_ticks = [date for date in dates if date.endswith('-01')]
plt.xticks(month_ticks, rotation=45, fontproperties=font)

plt.tight_layout()
plt.show()