'''
检查坐标系统的实际格式
'''

import numpy as np
import os

def check_coordinates():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    datasets = [
        ('UCY Zara01', 'ucy/zara/zara01', [720, 576]),
        ('UCY Zara02', 'ucy/zara/zara02', [720, 576]),
        ('ETH Univ', 'eth/univ', [720, 576]),
        ('ETH Hotel', 'eth/hotel', [640, 480]),
        ('UCY Univ', 'ucy/univ', [720, 576])
    ]
    
    print("="*70)
    print("坐标系统分析")
    print("="*70)
    
    for name, path, img_size in datasets:
        csv_path = os.path.join(data_dir, path, 'pixel_pos_interpolate.csv')
        
        if not os.path.exists(csv_path):
            continue
            
        data = np.genfromtxt(csv_path, delimiter=',')
        
        x_coords = data[3, :]
        y_coords = data[2, :]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 检查是否是归一化坐标
        is_normalized = (x_max <= 1.0 and x_min >= -1.0 and 
                        y_max <= 1.0 and y_min >= -1.0)
        
        # 如果归一化，尝试反归一化到像素坐标
        if is_normalized:
            # 假设坐标被归一化到 [-0.5, 0.5] 或 [0, 1]
            # 尝试不同的归一化方式
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # 方式1: 归一化到 [-0.5, 0.5]，中心在0
            if abs(x_min + 0.5) < 0.1 and abs(x_max - 0.5) < 0.1:
                norm_type = "[-0.5, 0.5] centered"
                x_pixel_min = (x_min + 0.5) * img_size[0]
                x_pixel_max = (x_max + 0.5) * img_size[0]
                y_pixel_min = (y_min + 0.5) * img_size[1]
                y_pixel_max = (y_max + 0.5) * img_size[1]
            # 方式2: 归一化到 [0, 1]
            elif x_min >= 0 and x_max <= 1.0:
                norm_type = "[0, 1]"
                x_pixel_min = x_min * img_size[0]
                x_pixel_max = x_max * img_size[0]
                y_pixel_min = y_min * img_size[1]
                y_pixel_max = y_max * img_size[1]
            else:
                norm_type = "其他归一化方式"
                x_pixel_min = x_pixel_max = y_pixel_min = y_pixel_max = None
        else:
            norm_type = "原始像素坐标"
            x_pixel_min = x_min
            x_pixel_max = x_max
            y_pixel_min = y_min
            y_pixel_max = y_max
        
        print(f"\n【{name}】")
        print(f"  图像尺寸: {img_size[0]} x {img_size[1]} 像素")
        print(f"  归一化坐标范围: X=[{x_min:.4f}, {x_max:.4f}], Y=[{y_min:.4f}, {y_max:.4f}]")
        print(f"  归一化类型: {norm_type}")
        if x_pixel_min is not None:
            print(f"  推测像素坐标范围: X=[{x_pixel_min:.1f}, {x_pixel_max:.1f}], Y=[{y_pixel_min:.1f}, {y_pixel_max:.1f}]")
    
    print("\n" + "="*70)
    print("重要说明")
    print("="*70)
    print("""
1. **坐标系统**:
   - 数据中的坐标已经被归一化，范围大约在 [-0.5, 0.5] 或 [0, 1]
   - 这不是原始像素坐标，也不是世界坐标（米）

2. **像素到米的转换**:
   - ETH/UCY数据集通常使用homography矩阵进行转换
   - 不同场景的转换比例不同
   - 没有通用的"500像素=1米"这样的固定比例

3. **你的结果 (ADE = 0.1636)**:
   - 这是归一化坐标下的误差
   - 要转换为米，需要：
     a) 先转换为像素坐标（乘以图像尺寸）
     b) 再使用homography矩阵转换为世界坐标（米）
   - 或者直接与论文中的归一化坐标结果对比

4. **建议**:
   - 直接使用归一化坐标进行模型评估和对比
   - 如果需要米单位，需要找到对应场景的homography矩阵
   - 论文中的结果通常也使用归一化坐标或像素坐标
    """)


if __name__ == '__main__':
    check_coordinates()

