import torch
import xml.etree.ElementTree as ET
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 下载并加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')


# 解析 XML 文件并提取 ground_truth 数据
def parse_dependency_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    ground_truth = []

    # 假设 XML 中每个 <sentence> 标签下有一组 <word> 标签
    for sentence in root.findall(".//sentence"):
        sentence_data = []
        for word in sentence.findall(".//word"):
            word_text = word.text
            head = word.get('head')
            label = word.get('label')

            # 保存每个词的依存关系
            sentence_data.append({
                'word': word_text,
                'head': head,
                'label': label
            })

        ground_truth.append(sentence_data)

    return ground_truth


# 假设数据集路径
file_path = '/home/data/GYX/llm/work/nlp_style.xsl'
ground_truth = parse_dependency_data(file_path)

# 打印前10个词以确保正确
for sentence in ground_truth[:1]:  # 打印第一句话的内容
    print(sentence)


# 定义数据集类
class DependencyDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        # 转换标签
        head_labels = torch.tensor(label['head'], dtype=torch.long)
        dep_labels = torch.tensor(label['label'], dtype=torch.long)

        return encoding, head_labels, dep_labels


# 假设每个样本是一句话，并包含依存关系标注
# 示例数据：句子和对应的依存关系标签（head 和 label）
sentences = ['我喜欢学习。', 'BERT 是一个强大的模型。']
labels = [
    {'head': [0, 1, 1], 'label': [0, 2, 2]},  # 对应的 head 和 label
    {'head': [1, 2, 3], 'label': [0, 1, 2]},  # 对应的 head 和 label
]

# 创建 Dataset 和 DataLoader
dataset = DependencyDataset(sentences, labels, tokenizer)
train_loader = DataLoader(dataset, batch_size=2)

# 模型训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

model.train()
for epoch in range(3):
    for batch in train_loader:
        inputs, head_labels, dep_labels = batch
        outputs = model(**inputs)
        logits = outputs.logits  # 模型输出的 logits

        # 假设 logits 的形状为 [batch_size, seq_length, num_labels]
        head_preds = logits[:, :, 0]  # 假设第一个 label 维度为 head
        dep_preds = logits[:, :, 1]  # 假设第二个 label 维度为依存关系标签

        # 计算损失
        head_loss = F.cross_entropy(head_preds.view(-1, head_preds.size(-1)), head_labels.view(-1))
        dep_loss = F.cross_entropy(dep_preds.view(-1, dep_preds.size(-1)), dep_labels.view(-1))

        # 总损失
        loss = head_loss + dep_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 测试和评估
model.eval()

# 假设测试数据集是哈工大的依存句库测试集
test_sentences = ['我爱编程。']
test_labels = [{'head': [0, 1], 'label': [0, 1]}]  # 模拟一个测试数据集


# 评估模型
def evaluate_model(model, sentences, ground_truth):
    predictions = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits
        head_preds = logits[:, :, 0].argmax(dim=-1)  # 预测的 head
        dep_preds = logits[:, :, 1].argmax(dim=-1)  # 预测的依存关系标签
        predictions.append((head_preds, dep_preds))

    return predictions


# 评估模型的性能
predictions = evaluate_model(model, test_sentences, ground_truth)


# 计算UAS和LAS
def calculate_uas(predictions, ground_truth):
    correct_dependencies = 0
    total_dependencies = 0
    for pred, truth in zip(predictions, ground_truth):
        pred_heads, pred_labels = pred
        true_heads, true_labels = truth['head'], truth['label']

        for p_head, t_head in zip(pred_heads, true_heads):
            if p_head == t_head:
                correct_dependencies += 1
            total_dependencies += 1

    return correct_dependencies / total_dependencies


def calculate_las(predictions, ground_truth):
    correct_dependencies = 0
    total_dependencies = 0
    for pred, truth in zip(predictions, ground_truth):
        pred_heads, pred_labels = pred
        true_heads, true_labels = truth['head'], truth['label']

        for p_head, t_head, p_label, t_label in zip(pred_heads, true_heads, pred_labels, true_labels):
            if p_head == t_head and p_label == t_label:
                correct_dependencies += 1
            total_dependencies += 1

    return correct_dependencies / total_dependencies


# 计算评估指标
uas = calculate_uas(predictions, test_labels)
las = calculate_las(predictions, test_labels)

print(f'UAS: {uas}, LAS: {las}')
