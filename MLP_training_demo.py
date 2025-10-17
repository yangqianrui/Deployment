import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from quant.replace import replace_linear_layer
# 定义一个简单的MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1, 64)  # 输入1维，输出64维
        self.layer2 = nn.Linear(64, 256)  # 中间层64到256
        self.layer3 = nn.Linear(256, 1)  # 输出层1维

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
# 确定设备（GPU或CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 生成训练数据
x_train = np.linspace(-10, 10, 1000).reshape(-1, 1)  # 1000个点，x从-10到10
y_train = np.sin(x_train)  # 对应的y值是sin(x)

# 转换为PyTorch的tensor并移动到指定设备
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

# 实例化MLP模型并移动到指定设备
model = MLP().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练过程
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    
    # 正向传播
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数
    
    # 打印每100个epoch的损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 替换中间线性层为RobuQLinear
model.layer2 = replace_linear_layer(model.layer2, nbits=2, w_bits=2, if_hadamard=True, if_lora=True)

# QAT训练
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    
    # 正向传播
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数
    
    # 打印每100个epoch的损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 保存模型
torch.save(model.state_dict(), "mlp_model.pt")
print("模型保存为 mlp_model.pt")
