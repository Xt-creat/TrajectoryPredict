import kagglehub
import os
import shutil

# 指定下载目录为 CS-LSTM/data
download_dir = os.path.join(os.path.dirname(__file__), 'data')

# 创建目录（如果不存在）
os.makedirs(download_dir, exist_ok=True)
print(f"目标下载目录: {download_dir}")

# Download latest version
print("正在下载数据集...")
path = kagglehub.dataset_download("nigelwilliams/ngsim-vehicle-trajectory-data-us-101")

print(f"数据集下载到临时位置: {path}")

# 将文件移动到指定目录
if os.path.exists(path):
    # 检查目标目录是否已有内容
    if os.path.exists(download_dir) and os.listdir(download_dir):
        print(f"提示: {download_dir} 目录已存在内容，将合并文件")
    
    # 移动所有文件到目标目录
    moved_count = 0
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(download_dir, item)
        
        try:
            if os.path.isdir(src):
                # 如果是目录，使用 copytree（如果目标存在则先删除）
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                # 如果是文件，直接复制（如果目标存在则覆盖）
                shutil.copy2(src, dst)
            moved_count += 1
        except Exception as e:
            print(f"警告: 移动 {item} 时出错: {e}")
    
    print(f"✓ 成功移动 {moved_count} 个项目到: {download_dir}")
    print(f"数据集文件列表: {os.listdir(download_dir)}")
else:
    print(f"错误: 下载路径不存在: {path}")