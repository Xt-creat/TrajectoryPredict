'''
解释 Homography 矩阵及其在行人轨迹预测中的应用
'''

import numpy as np
import matplotlib.pyplot as plt

def explain_homography():
    print("="*70)
    print("Homography 矩阵详解")
    print("="*70)
    
    print("""
1. **什么是 Homography 矩阵？**
   
   Homography（单应性）矩阵是一个 3×3 的矩阵，用于描述两个平面之间的投影变换关系。
   
   在行人轨迹预测中，它用于：
   - 将图像平面（像素坐标）映射到世界平面（真实世界坐标，单位：米）
   - 或者反过来，将世界坐标映射到像素坐标
   
   数学表示：
   [x']   [h11  h12  h13] [x]
   [y'] = [h21  h22  h23] [y]
   [1 ]   [h31  h32  h33] [1]
   
   其中 (x, y) 是源坐标，(x', y') 是目标坐标
   
2. **为什么需要 Homography 矩阵？**
   
   在 ETH/UCY 数据集中：
   - 视频是从鸟瞰视角拍摄的（从屋顶向下看）
   - 图像中的像素坐标 (u, v) 需要转换为真实世界坐标 (x, y) 单位：米
   - 不同场景的相机位置、角度不同，所以每个场景需要不同的 homography 矩阵
   
3. **Homography 矩阵的格式**
   
   通常保存在文本文件中，格式如下：
   
   2.8128700e-02  2.0091900e-03 -4.6693600e+00
   8.0625700e-04  2.5195500e-02 -5.0608800e+00
   3.4555400e-04  9.2512200e-05  4.6255300e-03
   
   这是一个 3×3 矩阵，用于将像素坐标转换为世界坐标（米）
   
4. **如何使用 Homography 矩阵？**
   
   步骤：
   a) 读取 homography 矩阵 H (3×3)
   b) 将像素坐标转换为齐次坐标 [u, v, 1]
   c) 应用矩阵变换: [x', y', w'] = H × [u, v, 1]
   d) 归一化: x = x'/w', y = y'/w'
   e) 得到世界坐标 (x, y)，单位：米
   
5. **在行人轨迹预测中的应用**
   
   - 训练时：通常使用归一化坐标（已经处理过的数据）
   - 评估时：如果需要与论文结果对比，需要转换为米单位
   - 可视化时：可以转换为真实世界坐标进行展示
   
6. **为什么你的数据集中可能没有 Homography 矩阵？**
   
   - 数据可能已经预处理过，坐标已经归一化
   - Homography 矩阵通常在原始 ETH/UCY 数据集中提供
   - 你的项目可能直接使用了归一化后的坐标，跳过了转换步骤
    """)
    
    print("\n" + "="*70)
    print("示例：像素坐标转换为世界坐标")
    print("="*70)
    
    # 示例 homography 矩阵（ETH Hotel 的典型值，仅作演示）
    H_example = np.array([
        [2.8128700e-02,  2.0091900e-03, -4.6693600e+00],
        [8.0625700e-04,  2.5195500e-02, -5.0608800e+00],
        [3.4555400e-04,  9.2512200e-05,  4.6255300e-03]
    ])
    
    print("\n示例 Homography 矩阵 (ETH Hotel):")
    print(H_example)
    
    def pixel_to_world(pixel_coords, H):
        """将像素坐标转换为世界坐标"""
        # 转换为齐次坐标
        pixel_homo = np.array([pixel_coords[0], pixel_coords[1], 1.0])
        
        # 应用 homography 变换
        world_homo = H @ pixel_homo
        
        # 归一化
        world_coords = world_homo[:2] / world_homo[2]
        return world_coords
    
    # 示例：转换几个像素点
    pixel_points = [
        [320, 240],  # 图像中心 (640×480)
        [100, 100],  # 左上区域
        [540, 380]   # 右下区域
    ]
    
    print("\n像素坐标 → 世界坐标（米）转换示例：")
    print("-" * 70)
    for pixel in pixel_points:
        world = pixel_to_world(pixel, H_example)
        print(f"像素坐标 ({pixel[0]:3d}, {pixel[1]:3d}) → 世界坐标 ({world[0]:.2f}, {world[1]:.2f}) 米")
    
    print("\n" + "="*70)
    print("你的情况")
    print("="*70)
    print("""
1. **你的数据格式**:
   - 坐标已经归一化到 [-0.5, 0.5] 范围
   - 这是归一化坐标，不是像素坐标，也不是世界坐标
   
2. **你的结果 (ADE = 0.1636)**:
   - 这是归一化坐标下的误差
   - 要转换为米，需要：
     a) 归一化坐标 → 像素坐标（乘以图像尺寸）
     b) 像素坐标 → 世界坐标（使用 homography 矩阵）
   
3. **如何获取 Homography 矩阵**:
   - 从原始 ETH/UCY 数据集下载
   - 通常在每个场景的文件夹中有 homography.txt 文件
   - 或者从论文的 GitHub 仓库获取
   
4. **如果找不到 Homography 矩阵**:
   - 可以直接使用归一化坐标进行模型评估
   - 与使用相同坐标系统的结果对比
   - 或者估算转换比例（但不够精确）
    """)


if __name__ == '__main__':
    explain_homography()

