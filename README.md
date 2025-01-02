# 信息内容安全

第四组


此工程主要实现了课程作业所使用的数据获取代码，模型训练代码，以及模型加载使用代码。

结构目录：

```
--BERT-MODULE
 |-bert-base-chinese -> 所用的预训练模型和分词法文件等
 |-crawler -> 使用的爬虫
 |-data -> 第一子项目的训练数据
 |-emotion_model.pth -> 第一子项目训练后的模型
 |-emotionl_new.py -> 第二子项目训练代码
 |-emotionl.py -> 第一子项目训练代码
 |-gupiao_motion_model.pth -> 第二子项目训练后的模型
 |-gupiao.tsv -> 第二子项目训练数据集
 |-loads.py -> 第一子项目模型加载使用
 |-money_loads.py -> 第二子项目模型加载使用
 |-README.md -> 文件说明
 |-requirements.txt -> 使用的相关库依赖
 |-shangzheng.tsv -> 第二子项目应用数据
 |-siben.tsv -> 第一子项目应用数据
 |-stopwords.txt -> 停用词表
```
