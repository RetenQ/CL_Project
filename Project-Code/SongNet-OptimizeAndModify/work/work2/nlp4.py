import torch
import torch.nn as nn
import torch.optim as optim


# TreeRNN模型
class TreeRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # 使用 Xavier 初始化
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_xh = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_hy = nn.Parameter(torch.randn(hidden_dim, input_dim))

        # 初始化权重
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.xavier_uniform_(self.W_hy)

    def forward(self, left, right):
        # 递归处理左右子树
        hidden_left = torch.tanh(torch.matmul(left, self.W_xh))
        hidden_right = torch.tanh(torch.matmul(right, self.W_xh))

        # 合并子树
        combined_hidden = torch.matmul(hidden_left + hidden_right, self.W_hh)
        return torch.matmul(combined_hidden, self.W_hy)


# 假设我们有一个数据输入
input_dim = 50  # 假设词向量维度为 50
hidden_dim = 100  # 隐藏层维度
model = TreeRNN(input_dim, hidden_dim)

# 假设输入的左右子树
sample_left = torch.randn(1, input_dim)  # 左子树的输入
sample_right = torch.randn(1, input_dim)  # 右子树的输入

# 前向传播
output = model(sample_left, sample_right)

# 假设目标是连续的词向量（例如PP-attachment的关系）
target = torch.randn(1, input_dim)  # 目标值是一个连续的向量

# 使用均方误差损失（MSELoss），适用于回归任务
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00001)

# 训练步骤
optimizer.zero_grad()
loss = criterion(output, target)
loss.backward()

# 梯度裁剪，防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

optimizer.step()

print(f'Loss: {loss.item()}')
