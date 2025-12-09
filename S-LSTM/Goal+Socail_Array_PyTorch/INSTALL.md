# 安装说明

## Python 版本要求
- Python 3.7 或更高版本

## 必需的库

### 1. PyTorch
**最低版本：1.8.0**

代码中使用的 PyTorch 功能：
- `torch.nn.Module` - 模型基类
- `torch.nn.Linear` - 线性层
- `torch.nn.LSTMCell` - LSTM单元
- `torch.nn.functional` - 函数式API
- `torch.optim` - 优化器
- `torch.nn.utils.clip_grad_norm_` - 梯度裁剪
- `torch.cuda.is_available()` - GPU检测
- `torch.device` - 设备管理
- `torch.save/torch.load` - 模型保存/加载

**安装方式：**

**CPU版本：**
```bash
pip install torch>=1.8.0
```

**GPU版本（CUDA）：**
```bash
# CUDA 11.1
pip install torch>=1.8.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111

# CUDA 11.3
pip install torch>=1.8.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 11.6
pip install torch>=1.8.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# 最新版本（推荐）
pip install torch torchvision torchaudio
```

查看官方安装指南：https://pytorch.org/get-started/locally/

### 2. NumPy
**最低版本：1.19.0**

用于数据处理和数组操作。

```bash
pip install numpy>=1.19.0
```

### 3. 标准库（无需安装）
- `pickle` - Python标准库，用于序列化
- `os`, `time`, `argparse` - Python标准库

## 快速安装

### 方式1：使用 requirements.txt
```bash
cd Goal+Socail_Array_PyTorch
pip install -r requirements.txt
```

### 方式2：手动安装
```bash
pip install torch>=1.8.0 numpy>=1.19.0
```

## 验证安装

运行以下命令验证安装：
```python
import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## 常见问题

### 1. 如果使用 CPU 训练
代码会自动检测并使用 CPU，无需修改。

### 2. 如果使用 GPU 训练
确保安装了对应 CUDA 版本的 PyTorch，代码会自动使用 GPU。

### 3. 版本兼容性
- PyTorch 1.8.0+ 支持所有使用的功能
- 推荐使用 PyTorch 1.10.0 或更高版本以获得更好的性能
- Python 3.7-3.11 都支持

## 环境建议

推荐使用虚拟环境：
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

