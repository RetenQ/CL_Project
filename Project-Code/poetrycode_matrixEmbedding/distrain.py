import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from poetry import TTSWithDiscriminator, DiscriminatorModel
from transformers import BertTokenizer
import torch.nn.functional as F
import json

def load_data_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 使用tokenizer将文本转为模型输入
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # 获取token ids
            'attention_mask': inputs['attention_mask'].squeeze(0),  # 获取attention mask
            'label': torch.tensor(label, dtype=torch.float)
        }

# 设置训练参数
num_epochs = 20
target_length = 16000 * 4  # 4秒音频
batch_size = 2
learning_rate = 1e-4

tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
discriminator_model = DiscriminatorModel(input_dim=512)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts, labels = load_data_from_json('data.json')

dataset = TextDataset(texts, labels, tokenizer)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TTSWithDiscriminator(tts_model=None, discriminator_model=None, tokenizer=tokenizer, target_length=target_length)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        # 生成音频
        text = input_ids
        optimizer.zero_grad()

        outputs, loss = model(text, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    # 保存模型
    model_save_path = f"discriminator_{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for epoch {epoch+1} saved as {model_save_path}")
