import json
import pandas as pd
import re
from datetime import datetime

# 定义转换时间戳为日期的函数
def convert_timestamp_to_date(timestamp_ms):
    # 时间戳是毫秒单位，转换为秒
    timestamp_s = timestamp_ms / 1000
    # 将时间戳转换为日期格式
    date_time = datetime.utcfromtimestamp(timestamp_s)
    return date_time.strftime('%Y-%m-%d')

# 去除#后的部分、[]中的内容，并去除非中文字符和符号
def clean_text(text):
    # 去除#后面的内容
    text = re.sub(r'#.*', '', text)
    # 去除[]中的内容
    text = re.sub(r'\[.*?\]', '', text)
    # 去除表情符号和特殊字符，只保留中文字符及常见中文符号
    text = re.sub(r'[^\u4e00-\u9fa5，。！？；：、]', '', text)
    # 去除多余的空格
    text = text.strip()
    return text

# 读取JSON文件
file_path = 'search_comments_2024-12-31.json'  # 使用提供的文件路径
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 用于存储处理后的数据
result = []

# 遍历每个数据项
for item in data:
    content = item.get('content', '')
    create_time_ms = item.get('create_time', None)

    # 转换时间戳为日期
    if create_time_ms:
        date = convert_timestamp_to_date(create_time_ms)
    else:
        date = 'Unknown'

    # 清理content中的内容
    cleaned_content = clean_text(content)

    # 只将content长度大于等于3的行添加到结果
    if len(cleaned_content) >= 3:  # 只添加长度大于等于3个字的content
        result.append((cleaned_content, date))

# 创建DataFrame并写入Excel
df = pd.DataFrame(result, columns=['Content', 'Date'])

# 保存到Excel文件
output_file = '西贝_2.xlsx'
df.to_excel(output_file, index=False)

print(f"数据已处理并保存为 {output_file}")
