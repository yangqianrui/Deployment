conda create --name robuq python=3.10

查看一下机器上的cuda环境，去pytorch官网https://pytorch.org/get-started/previous-versions 得到一个匹配的torch链接
如：我的cuda是12.4，下载链接为：conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

借着安装fast-hadamard-transform库：
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamara-transform
用你当前环境里的 torch 来编译
pip install -e . --no-build-isolation --config-settings editable_mode=compat

报错原因：
1.cuda与pytorch不匹配 2.不要直接用pip install fast-hadamard-transform

接下来，直接顺着报错把剩下的库都安装了，没啥特别的