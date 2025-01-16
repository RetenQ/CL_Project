import sys, re
from collections import Counter
import random
cnt = Counter()


f_ci = "./data/ci3.txt"
f_cipai = "./data/cipai3.txt"
cipai = Counter()
with open(f_cipai) as f:
    for line in f:
        line = line.strip()
        fs = line.split()
        cipai.update(fs)

cipai = cipai.keys()

docs = {}
# with open(f_ci) as f:
#     for line in f:
#         line = line.strip()
#         fs = line.split("<s1>")
#         author = fs[0]
#         topic, content = fs[1].split("<s2>")
#         if "・" in topic:
#             t1, t2 = topic.split("・")
#             if t1 == t2:
#                 topic = t1
#             else:
#                 if t1 in cipai:
#                     topic = t1
#                 elif t2 in cipai:
#                     topic = t2
#                 else:
#                     topic = t1
#         content = content.replace("、", "，")
#         sents = content.split("</s>")
#         ws = [w for w in author + topic + ''.join(sents)]
#         cnt.update(ws)
#         if topic not in docs:
#             docs[topic] = []
#         docs[topic].append(author + "<s1>" + topic + "<s2>" + '</s>'.join(sents))
# 修改12.22：(作者、主题、正文)进行解析和组织，同时统计词汇的出现频率，并将数据分为不同的部分（训练集、验证集、测试集）
# 修改部分为：<s1> 分隔符用于分隔“作者”，<s2> 分隔符用于分隔“主题”和“内容”。,如果该行不符合格式（例如没有 <s1> 或 <s2>），则跳过该行
with open(f_ci, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        fs = line.split("<s1>")
        if len(fs) < 2:  # 检查是否有足够的元素
            continue  # # 检查分割后的列表长度是否小于2，如果是，则说明数据格式不正确，没有足够的元素，就一个元素需要删除。
        author = fs[0]
        try:
            topic_content = fs[1].split("<s2>")
            if len(topic_content) < 2:  # 再次检查是否有足够的元素
                continue  # 同上
            topic, content = topic_content
            if "・" in topic:
                t1, t2 = topic.split("・")
                if t1 == t2:
                    topic = t1
                else:
                    #gyx
                    if t1 in cipai:
                        topic = t1
                    elif t2 in cipai:
                        topic = t2
                    else:
                        topic = t1
            content = content.replace("、", "，")
            sents = content.split("</s>")
            ws = [w for w in author + topic + ''.join(sents)]
            cnt.update(ws)#这里是为了统计字数
            if topic not in docs:
                docs[topic] = []
            docs[topic].append(author + "<s1>" + topic + "<s2>" + '</s>'.join(sents))
        except IndexError:
            print(f"Error processing line: {line}")

# #纳兰需要
# with open(f_ci, 'r', encoding='utf-8') as f:
#     for line in f:
#         line = line.strip()
#         fs = line.split("<s1>")
#         if len(fs) < 2:  # 检查是否有足够的元素
#             continue  # # 检查分割后的列表长度是否小于2，如果是，则说明数据格式不正确，没有足够的元素，就一个元素需要删除。
#         author = fs[0]
#         try:
#             topic_content = fs[1].split("<s2>")
#             if len(topic_content) < 2:  # 再次检查是否有足够的元素
#                 continue  # 同上
#             topic, content = topic_content
#             if "・" in topic:
#                 t1, t2 = topic.split("・")
#                 if t1 == t2:
#                     topic = t1
#                 else:
#                     #gyx
#                     if t1 in cipai:
#                         topic = t1
#                     elif t2 in cipai:
#                         topic = t2
#                     else:
#                         topic = t1
#             topic = topic
#             content = content.replace("、", "，")
#             sents = content.split("</s>")
#             ws = [w for w in author + topic + ''.join(sents)]
#             cnt.update(ws)#这里是为了统计字数
#             if topic not in docs:
#                 docs[topic] = []
#             docs[topic].append(author + "<s1>" + topic + "<s2>" + '</s>'.join(sents))
#         except IndexError:
#             print(f"Error processing line: {line}")



topics = list(docs.keys())
print(len(topics))#统计主题数目，前文已经细分，同种主题在一个键下

random.shuffle(topics)
#除了最后50个主题之外的所有主题被分配给训练集，50个主题对半分给验证集和测试集
topics_train = topics[:len(topics)-10]
topics_dev_test = topics[-10:]
topics_dev = topics_dev_test[:5]
topics_test = topics_dev_test[-5:]
# topics_train = topics[:len(topics)-10]
# topics_dev_test = topics[-10:]
# topics_dev = topics_dev_test[:5]
# topics_test = topics_dev_test[-5:]

docs_train = []
docs_dev = []
docs_test = []

for t in topics_train:
    docs_train.extend(docs[t])

for t in topics_dev:
    docs_dev.extend(docs[t])

for t in topics_test:
    docs_test.extend(docs[t])

random.shuffle(docs_train)
random.shuffle(docs_dev)
random.shuffle(docs_test)

print(len(docs_train), len(docs_dev), len(docs_test))
train_cps = []
dev_cps = []
test_cps = []


with open('./data/train.txt', 'w', encoding ='utf8') as f:
    for x in docs_train:
        s = x.split("<s2>")[0]
        train_cps.append(s.split("<s1>")[1])
        f.write(x + '\n')
    print(len(set(train_cps)))
with open('./data/dev.txt', 'w', encoding ='utf8') as f:
    for x in docs_dev:
        s = x.split("<s2>")[0]
        dev_cps.append(s.split("<s1>")[1])
        f.write(x + '\n')
    print(len(set(dev_cps)))
with open('./data/test.txt', 'w', encoding ='utf8') as f:
    for x in docs_test:
        s = x.split("<s2>")[0]
        test_cps.append(s.split("<s1>")[1])
        f.write(x + '\n')
    print(len(set(test_cps)))

print("vocab")
with open('./data/vocabtest.txt', 'w', encoding ='utf8') as f:
    for x, y in cnt.most_common():
        f.write(x + '\t' + str(y) + '\n')
print("done")
