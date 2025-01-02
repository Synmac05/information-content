import pandas as pd
import re

# 读取Excel文件
file_path = '上证指数.xlsx'
df = pd.read_excel(file_path)

# 只保留“字段1_文本”和“时间”列
df_filtered = df[['字段1_文本', '时间']]

# 定义正则表达式，匹配以符号开头的文本
symbol_start_pattern = r'^[^\w\s]'

# 使用正则表达式去除以符号开头的行
df_filtered = df_filtered[~df_filtered['字段1_文本'].str.match(symbol_start_pattern)]

# 定义正则表达式，匹配只包含符号（包括中文符号）的文本
symbol_pattern = r'^[^\w\s]+$'

# 使用正则表达式去除只包含符号的行
df_filtered = df_filtered[~df_filtered['字段1_文本'].str.match(symbol_pattern)]

# 去除字段1_文本长度小于3的行
df_filtered = df_filtered[df_filtered['字段1_文本'].str.len() >= 3]

# 补充时间信息，假设缺失的日期信息使用当前年份
df_filtered['时间'] = pd.to_datetime(df_filtered['时间'], errors='coerce', format='%m-%d %H:%M')

# 格式化时间列为 MM-DD 格式（去掉年份）
df_filtered['时间'] = df_filtered['时间'].dt.strftime('%m-%d')

# 将结果写入新的Excel文件
output_file = '上证指数_new.xlsx'
df_filtered.to_excel(output_file, index=False)

print(f"处理完成，结果已保存至 {output_file}")
