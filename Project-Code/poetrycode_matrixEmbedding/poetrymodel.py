import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np


# 矩阵编码
def reshape_embedding_to_tensor(embedding, num_rows):
    
    bs, num_elements = embedding.shape
    num_columns = num_elements
    matrix = torch.zeros((bs*num_rows, num_columns), dtype=torch.int)
    
    # 填充矩阵
    for b in range(bs):
        for i in range(num_columns):
            # 计算在新矩阵中的行和列
            row = (b * num_rows) + (i // (num_columns // num_rows))
            col = i % num_columns
            matrix[row, col] = embedding[b, i]
    
    return matrix

# # 示例使用
# embedding = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
# num_rows = 3
# tensor = reshape_embedding_to_tensor(embedding, num_rows)
# print(tensor)

# 矩阵解码还原
def sum_rows_in_groups_tensor(tensor, num_rows):
    bs, num_columns = tensor.shape
    output_rows = bs // num_rows
    
    reshaped_tensor = tensor.view(output_rows, num_rows, num_columns)
    summed_tensor = reshaped_tensor.sum(dim=1)
    
    return summed_tensor

# # 示例使用
# num_rows = 3
# summed_tensor = sum_rows_in_groups_tensor(tensor, num_rows)

# print(summed_tensor)


# 正交化LoRA微调
class Orthogonal_LoRA_Finetune(nn.Module):
    def __init__(self, model, rank=8):
        super(Orthogonal_LoRA_Finetune, self).__init__()
        
        self.model = model
        self.rank = rank

        self.lora_layers = nn.ModuleList()
        for name, module in model.named_modules():
            if 'attn' in name and isinstance(module, nn.MultiheadAttention):
                self.lora_layers.append(self._apply_lora_to_qkv(module))
    
    def _apply_lora_to_qkv(self, attention_module):
        """
        对自注意力模块中的QKV权重矩阵应用LoRA
        :param attention_module: 原始自注意力模块
        :return: 包含LoRA微调的QKV权重
        """
        # 获取原始的Q, K, V矩阵
        W_q = attention_module.in_proj_weight[:attention_module.embed_dim, :]
        W_k = attention_module.in_proj_weight[attention_module.embed_dim:2 * attention_module.embed_dim, :]
        W_v = attention_module.in_proj_weight[2 * attention_module.embed_dim:, :]
        
        # 定义低秩矩阵
        A_q = nn.Parameter(torch.randn(W_q.shape[0], self.rank))
        B_q = nn.Parameter(torch.randn(self.rank, W_q.shape[1]))
        
        A_k = nn.Parameter(torch.randn(W_k.shape[0], self.rank))
        B_k = nn.Parameter(torch.randn(self.rank, W_k.shape[1]))
        
        A_v = nn.Parameter(torch.randn(W_v.shape[0], self.rank))
        B_v = nn.Parameter(torch.randn(self.rank, W_v.shape[1]))
        
        # 初始化低秩矩阵
        nn.init.normal_(A_q, mean=0.0, std=0.02)
        nn.init.normal_(B_q, mean=0.0, std=0.02)
        nn.init.normal_(A_k, mean=0.0, std=0.02)
        nn.init.normal_(B_k, mean=0.0, std=0.02)
        nn.init.normal_(A_v, mean=0.0, std=0.02)
        nn.init.normal_(B_v, mean=0.0, std=0.02)
        
        # 返回包含LoRA调整的QKV模块
        return {
            'A_q': A_q,
            'B_q': B_q,
            'A_k': A_k,
            'B_k': B_k,
            'A_v': A_v,
            'B_v': B_v,
            'W_q': W_q,
            'W_k': W_k,
            'W_v': W_v
        }

    def forward(self, *input, **kwargs):
        for lora_layer in self.lora_layers:
            # 提取低秩矩阵
            A_q, B_q, A_k, B_k, A_v, B_v, W_q, W_k, W_v = lora_layer.values()

            # 使用低秩矩阵调整QKV
            W_q_new = W_q + torch.matmul(A_q, B_q)
            W_k_new = W_k + torch.matmul(A_k, B_k)
            W_v_new = W_v + torch.matmul(A_v, B_v)
            
            # 更新模型中对应层的QKV权重
            self.model.attn.in_proj_weight[:W_q_new.shape[0], :] = W_q_new
            self.model.attn.in_proj_weight[W_q_new.shape[0]:2*W_q_new.shape[0], :] = W_k_new
            self.model.attn.in_proj_weight[2*W_q_new.shape[0]:, :] = W_v_new

        return self.model(*input, **kwargs)

    def calculate_loss(self):
        """
        计算LoRA微调的损失： ||(W_·_new)(W_·_new)^T - I||_2
        """
        total_loss = 0.0
        identity_matrix = torch.eye(self.rank)

        # 遍历每个自注意力层进行损失计算
        for lora_layer in self.lora_layers:
            A_q, B_q, A_k, B_k, A_v, B_v, W_q, W_k, W_v = lora_layer.values()

            # 计算每个修改后的矩阵的损失
            loss_q = torch.norm(torch.matmul(W_q + torch.matmul(A_q, B_q), (W_q + torch.matmul(A_q, B_q)).T) - identity_matrix, p=2)
            loss_k = torch.norm(torch.matmul(W_k + torch.matmul(A_k, B_k), (W_k + torch.matmul(A_k, B_k)).T) - identity_matrix, p=2)
            loss_v = torch.norm(torch.matmul(W_v + torch.matmul(A_v, B_v), (W_v + torch.matmul(A_v, B_v)).T) - identity_matrix, p=2)

            # 累加损失
            total_loss += loss_q + loss_k + loss_v

        return total_loss

# # 示例：如何使用LoRA微调大模型中的QKV，并计算损失
# from transformers import BertModel

# bert_model = BertModel.from_pretrained('bert-base-uncased')
# lora_finetune_model = Orthogonal_LoRA_Finetune(bert_model, rank=8)

# input_data = torch.randint(0, 100, (8, 128))  # 示例输入
# output = lora_finetune_model(input_data)

# loss = lora_finetune_model.calculate_loss()
# print(f"Calculated loss: {loss.item()}")

# TTS模型与判别器结合
class TTSWithDiscriminator(nn.Module):
    def __init__(self, tts_model, discriminator_model, target_length):
        super(TTSWithDiscriminator, self).__init__()
        self.tts_model = tts_model  # 预训练TTS模型
        self.discriminator_model = discriminator_model  # 判别器模型
        self.target_length = target_length  # 目标音频长度

    def forward(self, text):
        generated_audio = self.tts_model(text)  # tts_model返回一个音频波形（tensor）

        generated_audio = self._adjust_audio_length(generated_audio)

        discriminator_output = self.discriminator_model(generated_audio)
        return discriminator_output

    def _adjust_audio_length(self, audio_tensor):
        current_length = audio_tensor.size(-1)  # 获取音频的当前长度

        # 如果当前音频长度大于目标长度，进行从前截断
        if current_length > self.target_length:
            audio_tensor = audio_tensor[..., -self.target_length:]
        
        # 如果当前音频长度小于目标长度，进行填充
        elif current_length < self.target_length:
            padding = self.target_length - current_length
            audio_tensor = torch.cat([audio_tensor, torch.zeros((audio_tensor.size(0), padding))], dim=-1)
        
        return audio_tensor
    
class DiscriminatorModel(nn.Module):
    def __init__(self, input_dim):
        super(DiscriminatorModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# tts_model = PretrainedTTSModel()
# discriminator_model = DiscriminatorModel(input_dim=512)  # 输入维度为512

# target_length = 16000*4   # 4秒音频

# tts_with_discriminator = TTSWithDiscriminator(tts_model, discriminator_model, target_length)
# text = "Hello, this is an example sentence for TTS."

# output = tts_with_discriminator(text)
# print(output)  # 输出音频质量评分，0到1之间的浮动数


# 投影MLP模型
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(ProjectionMLP, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class AlingmentLoss(nn.Module):
    def __init__(self, text, audio):
        super(AlingmentLoss, self).__init__()
        self.text = text
        self.audio = audio
    
    def forward(self, predicted, target):
        loss = torch.norm(predicted - target, p=2)
        return loss