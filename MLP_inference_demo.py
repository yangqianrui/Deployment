import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from quant.replace import replace_linear_layer
# 定义一个简单的MLP模型（与训练时相同的结构）
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 确定设备（GPU或CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建
model = MLP().to(device)
# 量化模型
model.layer2 = replace_linear_layer(model.layer2, nbits=4, w_bits=4, if_hadamard=False, if_lora=False)
# 加载训练好的模型参数
model.load_state_dict(torch.load("mlp_model.pt"))
if hasattr(model.layer2, "prepare_for_inference"):
    model.layer2.prepare_for_inference()
model.eval()  # 设置为推理模式

# 生成测试数据（我们可以使用训练数据，也可以生成新的数据进行推理）
x_test = np.linspace(-10, 10, 1000).reshape(-1, 1)  # 1000个点，x从-10到10
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
# 推理过程
with torch.no_grad():  # 不需要计算梯度
    y_pred = model(x_test_tensor).cpu().numpy()  # 获取模型输出

# 绘制拟合结果图
plt.figure(figsize=(8, 6))
plt.plot(x_test, np.sin(x_test), label='y=sin(x)', color='blue')
plt.plot(x_test, y_pred, label='Quantized MLP', color='red', linestyle='--')
plt.legend()
plt.title("MLP Fitting of y = sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig("mlp_fitting_result.png")
