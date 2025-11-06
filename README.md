## 环境安装

以我本地环境为例：

1.创建conda环境；

2.装cuda（以12.8为例：[CUDA Toolkit 12.8 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)）

3.设置环境变量：

```
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

4.装nunchaku：

```
https://nunchaku.tech/docs/nunchaku/installation/installation.html
```

```
pip install -v -e ".[dev,docs]"
```

5.改算子：在nunchaku中替换掉`/nunchaku/src/kernels/zgemm/gemm_w4a4_launch_impl.cuh`

再到nunchaku目录中

```
python setup.py develop
```

6.装fast-hadamard-transform

```
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
```

# 应用

将 `RobuQ` 应用到您自己的模型分为两个阶段：

### 阶段一：量化感知训练 (QAT)

此阶段的目标是使用模拟量化（STE）对预训练模型进行微调。

**参考脚本:** `MLP_training_demo.py`

1. **导入 API:**

   Python

   ```python
   from quant.replace import replace_linear_layer
   ```

2. 加载预训练模型:

   加载您已经训练好的原始浮点模型。

   ```python
   model = YourModelClass().to(device)
   # 加载您预训练好的权重
   # model.load_state_dict(torch.load("your_pretrained_model.pth")) 
   ```

3. 替换线性层:

   遍历您的模型，将需要量化的 nn.Linear 层替换为 RobuQLinear。

   ```python
   # 示例：替换模型中的 layer2
   model.layer2 = replace_linear_layer(
       model.layer2, 
       nbits=4,       # 激活 (A) 位宽
       w_bits=4,      # 权重 (W) 位宽 (W4A4)
       if_hadamard=False,
       if_lora=False
   )
   #
   ```

   - **参数说明:**
     - `nbits` / `w_bits`: 激活和权重的位宽。
     - 要使用 CUDA 核，`nbits` 和 `w_bits` 必须相等，且为 2, 3 或 4。
     - `w_bits=1.58` 可用于训练三值权重。

4. 执行 QAT 微调:

   设置优化器，像正常训练一样对模型进行微调。

   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.MSELoss()
   
   for epoch in range(num_epochs):
       model.train()
       # ... 训练循环 ...
       loss.backward()  # STE 会处理梯度
       optimizer.step()
   ```

5. 保存 QAT 模型:

   保存训练完成的量化模型的状态字典（state_dict）。

   Python

   ```
   torch.save(model.state_dict(), "mlp_model_quantized.pt")
   ```

### 阶段二：高性能推理

此阶段加载 QAT 模型，并激活 `nunchaku` CUDA 核进行高速推理。

**参考脚本:** `MLP_inference_demo.py`

1. 重建模型结构:

   [关键步骤] 您必须实例化一个与 QAT 阶段 完全相同的模型结构，包括再次调用 replace_linear_layer。

   ```python
   from quant.replace import replace_linear_layer
   
   model = YourModelClass().to(device)
   
   # 必须使用与 QAT 阶段完全相同的参数替换相同的层
   model.layer2 = replace_linear_layer(
       model.layer2, 
       nbits=4, 
       w_bits=4, 
       if_hadamard=False, 
       if_lora=False
   )
   #
   ```

2. 加载 QAT 权重:

   加载在阶段一中保存的 state_dict。

   Python

   ```
   model.load_state_dict(torch.load("mlp_model_quantized.pt"))
   ```

3. 准备推理 (激活 CUDA 核):

   [关键步骤] 这是启用 nunchaku 高性能核的核心。您必须遍历所有 RobuQLinear 模块（或您已知已替换的层），并调用 .prepare_for_inference() 方法。

   此方法会将浮点权重预先量化和打包（包括折叠 LoRA），为 CUDA 核做好准备。

   Python

   ```python
   if hasattr(model.layer2, "prepare_for_inference"):
       model.layer2.prepare_for_inference()
   
   # ... (对其余所有被替换的 RobuQLinear 层执行此操作) ...
   ```

4. 执行推理:

   将模型设置为评估模式 (.eval()) 并在 torch.no_grad() 上下文中运行。

   ```python
   model.eval()
   
   with torch.no_grad():
       # 现在 model(x) 将自动调用 _forward_w4a4 (如果条件满足)
       y_pred = model(x_test_tensor) 
   #
   ```

------