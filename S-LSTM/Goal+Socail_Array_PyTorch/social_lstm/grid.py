'''
社交数组（Social Array）计算模块

功能说明：
本模块实现了社交LSTM中的"社交数组"（Social Array）功能。
社交数组用于编码每个行人周围其他行人的空间分布信息，帮助模型理解行人之间的社交交互。

核心思想：
- 为每个行人找到最近的 grid_size 个其他行人的位置
- 按照距离从近到远排序
- 如果周围行人少于 grid_size，则重复最近的行人位置来填充

Modified by: Simone Zamboni
'''
import numpy as np
import math


def getGridMask(frame, dimensions, neighborhood_size, grid_size):
    """
    为单个帧计算社交数组（Social Array）
    
    功能：
    为帧中的每个行人计算其社交数组，包含该行人周围最近的 grid_size 个其他行人的位置。
    社交数组按照距离从近到远排序。
    
    输入参数：
    - frame: [maxNumPeds, 3] 数组，每行包含 [pedID, x, y]
              pedID=0 表示该位置没有行人
    - dimensions: 图像尺寸 ⚠️ 注意：此参数在函数体内完全未使用，仅保留以兼容接口
    - neighborhood_size: 邻域大小 ⚠️ 注意：此参数在函数体内完全未使用，仅保留以兼容接口
    - grid_size: 社交数组中包含的其他行人数量（实际使用的参数）
    
    说明：
    dimensions 和 neighborhood_size 参数在当前实现中未被使用。
    可能的原因：
    1. 原始实现中可能用于根据图像尺寸归一化坐标
    2. 或者用于根据neighborhood_size限制搜索范围
    3. 但在当前版本中这些功能被移除，为了保持接口兼容性保留了参数
    
    输出：
    - my_array: [maxNumPeds, grid_size*2] 数组
                每行代表一个行人的社交数组
                格式：[x1, y1, x2, y2, ..., x_grid_size, y_grid_size]
                其中 (x1, y1) 是最近的行人位置，(x2, y2) 是第二近的，以此类推
    
    算法流程：
    1. 对每个行人，计算到所有其他行人的距离
    2. 按距离排序，选择最近的 grid_size 个行人
    3. 如果周围行人少于 grid_size，则重复最近的行人位置来填充
    """

    # 获取帧中的最大行人数
    mnp = frame.shape[0]

    # 初始化社交数组：为每个行人创建一个大小为 grid_size*2 的数组
    # grid_size*2 是因为每个其他行人需要存储 (x, y) 两个坐标
    my_array = np.zeros((mnp, grid_size*2))

    # 为帧中的每个行人计算其社交数组
    for pedindex in range(mnp):

        # 跳过无效的行人（ID=0 表示该位置没有行人）
        if(frame[pedindex, 0] == 0):
           continue

        # 获取当前行人的位置
        current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]
        other_peds_with_position = []  # 存储其他行人的信息：[ID, x, y, 距离]

        # 遍历帧中的所有其他行人
        for otherpedindex in range(mnp):

            # 跳过无效的行人
            if frame[otherpedindex, 0] == 0:
                continue

            # 跳过自己（一个行人不能出现在自己的社交数组中）
            if frame[otherpedindex, 0] == frame[pedindex, 0]:
                continue

            # 计算当前行人与其他行人之间的欧氏距离
            current_distance = math.sqrt(
                math.pow((current_x - frame[otherpedindex][1]), 2) + 
                math.pow((current_y - frame[otherpedindex][2]), 2)
            )

            # 保存其他行人的信息：[ID, x, y, 距离]
            other_peds_with_position.append([
                frame[otherpedindex][0],
                frame[otherpedindex][1],
                frame[otherpedindex][2],
                current_distance
            ])

        # 如果当前行人周围没有其他行人，插入一个虚拟的远距离行人
        # 这样可以保证社交数组始终有数据，避免空数组
        if (len(other_peds_with_position) == 0):
            # 虚拟行人的位置设置为 (x-2, y-2)，距离约为 2.828（√8）
            other_peds_with_position.append([0, frame[pedindex, 1]-2, frame[pedindex, 2]-2, 2.828427125])
            
        # 记录找到的其他行人数量
        num_other_peds = len(other_peds_with_position)

        # 填充社交数组：按距离从近到远选择 grid_size 个行人
        j = 0  # j 表示社交数组中的第 j/2 个其他行人
               # j*2 是 x 坐标，j*2+1 是 y 坐标
        while j < len(my_array[pedindex]):
            x = 0  # 用于遍历 other_peds_with_position 数组

            # 寻找距离最近的行人
            # min_distance[0]: 最小距离值
            # min_distance[1]: 最近行人在数组中的索引
            min_distance = [1000000, 0]
            update = False

            # 遍历所有其他行人，找到距离最近的那个
            while x < len(other_peds_with_position):
                if(other_peds_with_position[x][3] < min_distance[0]):
                    min_distance[0] = other_peds_with_position[x][3]
                    min_distance[1] = x
                    update = True  # 找到了更近的行人
                x += 1

            # 如果找到了更近的行人，将其坐标保存到社交数组中
            if(update == True):
                # 保存最近行人的 x 坐标
                my_array[pedindex][j] = other_peds_with_position[min_distance[1]][1]
                # 保存最近行人的 y 坐标
                my_array[pedindex][j+1] = other_peds_with_position[min_distance[1]][2]
                # 从列表中移除该行人，避免重复选择
                other_peds_with_position.remove(other_peds_with_position[min_distance[1]])
            j += 2

        # 此时社交数组中已经按距离从近到远存储了其他行人的坐标
        # 如果周围行人数量 >= grid_size，数组已经填满，可以结束
        # 如果周围行人数量 < grid_size，需要填充剩余位置

        # 计算还需要填充多少个位置
        num_peds_missing = (len(my_array[0])//2) - num_other_peds

        # 如果还有空位，用已存在的最近行人的位置重复填充
        # 这样可以保证社交数组始终是固定大小，便于模型处理
        if(num_peds_missing > 0):
            i = 0
            # 从第一个已填充的位置开始，重复填充到剩余的空位
            while i < num_peds_missing:
                # 复制第 i 个行人的位置到剩余位置
                my_array[pedindex][(len(my_array[0])//2 - num_peds_missing + i) * 2] = my_array[pedindex][i * 2]
                my_array[pedindex][(len(my_array[0])//2 - num_peds_missing + i) * 2 + 1] = my_array[pedindex][i * 2 + 1]
                i += 1

    return my_array


def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size):
    """
    为整个序列计算社交数组（Social Array）
    
    功能：
    对序列中的每一帧调用 getGridMask，为整个序列生成社交数组。
    这是训练和推理时使用的主要函数。
    
    输入参数：
    - sequence: [seq_length, maxNumPeds, 3] 数组
                包含整个序列的所有帧数据
                每帧的形状为 [maxNumPeds, 3]，每行是 [pedID, x, y]
    - dimensions: 图像尺寸 ⚠️ 注意：此参数会被传递给 getGridMask，但在 getGridMask 中未使用
    - neighborhood_size: 邻域大小 ⚠️ 注意：此参数会被传递给 getGridMask，但在 getGridMask 中未使用
    - grid_size: 社交数组中包含的其他行人数量（实际使用的参数）
    
    输出：
    - sequence_mask: [seq_length, maxNumPeds, grid_size*2] 数组
                    每帧的社交数组，格式与 getGridMask 的输出相同
    
    使用示例：
    在训练时，sequence 是输入序列 [seq_length, maxNumPeds, 5]（包含goal信息）
    但这里只需要位置信息，所以使用前3列 [:, :, 1:4] 或 [:, :, :3]
    """
    sl = sequence.shape[0]  # 序列长度（sequence_length）
    mnp = sequence.shape[1]  # 最大行人数（maxNumPeds）

    # 初始化序列的社交数组
    # 形状：[序列长度, 最大行人数, grid_size*2]
    sequence_mask = np.zeros((sl, mnp, grid_size*2))

    # 对序列中的每一帧计算社交数组
    for i in range(sl):
        # 为第 i 帧计算社交数组
        # 注意：sequence[i, :, :] 的形状是 [maxNumPeds, 3]，每行是 [pedID, x, y]
        sequence_mask[i, :, :] = getGridMask(sequence[i, :, :], dimensions, neighborhood_size, grid_size)

    return sequence_mask

