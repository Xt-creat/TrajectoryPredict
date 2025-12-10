"""
将数据集处理为 CSV 文件
从 preprocess_data.m 转换而来
"""

import numpy as np
import pandas as pd
import os
import pickle
from collections import defaultdict
from tqdm import tqdm

# 输入文件路径
# us101_files = [
#     'raw/us101-0750-0805.txt',
#     'raw/us101-0805-0820.txt',
#     'raw/us101-0820-0835.txt'
# ]

# i80_files = [
#     'raw/i80-1600-1615.txt',
#     'raw/i80-1700-1715.txt',
#     'raw/i80-1715-1730.txt'
# ]
# all_files = us101_files + i80_files
us101_files = [
    'data/trajectories-0750am-0805am.txt'
]
all_files = us101_files


# 字段说明:
# 1: 数据集ID
# 2: 车辆ID
# 3: 帧号
# 4: 局部X坐标
# 5: 局部Y坐标
# 6: 车道ID
# 7: 横向机动
# 8: 纵向机动
# 9-47: 网格位置的邻居车辆ID

print("Loading data...")

traj = []
for idx, file_path in enumerate(all_files, 1):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found, skipping...")
        continue
    
    print(f"正在加载 {file_path}...")
    # 加载数据（空格分隔的值）
    data = np.loadtxt(file_path)
    
    # 在第一列添加数据集ID
    dataset_id = np.ones((data.shape[0], 1), dtype=np.float32) * idx
    data_with_id = np.hstack([dataset_id, data])
    
    # 从原始数据中提取所需列
    # 原始数据列（0索引）:
    # 0:车辆ID, 1:帧ID, 2:总帧数, 3:全局时间, 4:局部X, 5:局部Y,
    # 6:全局X, 7:全局Y, 8:长度, 9:宽度, 10:类型, 11:速度, 
    # 12:加速度, 13:车道ID, 14:前车, 15:后车, ...
    # 提取: 数据集ID, 车辆ID, 帧号, 局部X, 局部Y, 车道ID, 速度
    selected_cols = np.column_stack([
        data_with_id[:, 0],  # 数据集ID（已添加）
        data[:, 0],          # 车辆ID（原始列0）
        data[:, 1],          # 帧ID（原始列1） 
        data[:, 4],          # 局部X（原始列4）
        data[:, 5],          # 局部Y（原始列5）
        data[:, 13],         # 车道ID（原始列13）
        data[:, 11]          # 速度（原始列11）
    ])
    
    # 对于US-101数据集（idx <= 3），将车道ID上限设为6
    if idx <= 3:
        selected_cols[selected_cols[:, 5] >= 6, 5] = 6
    
    traj.append(selected_cols.astype(np.float32))

print("正在解析字段...")

# 初始化车辆轨迹和时间帧的字典
vehTrajs = [defaultdict(list) for _ in range(6)]
vehTimes = [defaultdict(list) for _ in range(6)]

# 处理每个数据集
for ii in range(len(traj)):
    print(f"正在处理数据集 {ii+1}...")
    
    # 按车辆ID组织
    vehIds = np.unique(traj[ii][:, 1])
    for vehId in vehIds:
        mask = traj[ii][:, 1] == vehId
        vehTrajs[ii][int(vehId)] = traj[ii][mask]
    
    # 按时间帧组织
    timeFrames = np.unique(traj[ii][:, 2])
    for time in timeFrames:
        mask = traj[ii][:, 2] == time
        vehTimes[ii][int(time)] = traj[ii][mask]
    
    # 初始化机动和网格列（列7-47）
    # 当前列: [0:数据集ID, 1:车辆ID, 2:帧号, 3:局部X, 4:局部Y, 5:车道ID, 6:速度]
    # 列7: 横向机动
    # 列8: 纵向机动
    # 列9-47: 邻居车辆ID（39个网格位置）
    n_rows = traj[ii].shape[0]
    traj[ii] = np.hstack([
        traj[ii],
        np.zeros((n_rows, 1)),  # 横向机动（列7）
        np.zeros((n_rows, 1)),  # 纵向机动（列8）
        np.zeros((n_rows, 39))  # 网格邻居（列9-47）
    ])
    
    # 处理每个轨迹点
    for k in tqdm(range(traj[ii].shape[0]), desc=f"Dataset {ii+1}"):
        time = traj[ii][k, 2]
        dsId = int(traj[ii][k, 0])
        vehId = int(traj[ii][k, 1])
        vehtraj = vehTrajs[ii][vehId]
        
        # 在车辆轨迹中查找索引
        time_indices = np.where(vehtraj[:, 2] == time)[0]
        if len(time_indices) == 0:
            continue
        ind = time_indices[0]
        lane = traj[ii][k, 5]  # 车道ID在索引5
        
        # 获取横向机动
        ub = min(vehtraj.shape[0] - 1, ind + 40)  # 确保不越界
        lb = max(0, ind - 40)
        
        if ub <= ind or lb >= ind:
            traj[ii][k, 7] = 1  # 默认: 保持车道
        elif vehtraj[ub, 5] > vehtraj[ind, 5] or vehtraj[ind, 5] > vehtraj[lb, 5]:
            traj[ii][k, 7] = 3  # 向右变道
        elif vehtraj[ub, 5] < vehtraj[ind, 5] or vehtraj[ind, 5] < vehtraj[lb, 5]:
            traj[ii][k, 7] = 2  # 向左变道
        else:
            traj[ii][k, 7] = 1  # 保持车道
        
        # 获取纵向机动
        ub = min(vehtraj.shape[0] - 1, ind + 50)  # 确保不越界
        lb = max(0, ind - 30)
        
        if ub <= ind or lb >= ind:
            traj[ii][k, 8] = 1  # 默认: 保持/加速
        else:
            vHist = (vehtraj[ind, 4] - vehtraj[lb, 4]) / (ind - lb)  # 局部Y在索引4
            vFut = (vehtraj[ub, 4] - vehtraj[ind, 4]) / (ub - ind)
            if vHist == 0 or vFut / vHist < 0.8:
                traj[ii][k, 8] = 2  # 减速
            else:
                traj[ii][k, 8] = 1  # 保持/加速
        
        # 获取网格位置
        t = vehTimes[ii][int(time)]
        frameEgo = t[t[:, 5] == lane]  # 车道ID在索引5
        frameL = t[t[:, 5] == lane - 1] if lane > 1 else np.empty((0, t.shape[1]))
        frameR = t[t[:, 5] == lane + 1] if lane < 5 else np.empty((0, t.shape[1]))
        
        # 左车道
        if len(frameL) > 0:
            for l in range(frameL.shape[0]):
                y = frameL[l, 4] - traj[ii][k, 4]  # 局部Y在索引4
                if abs(y) < 90:
                    gridInd = int(1 + round((y + 90) / 15))
                    if 1 <= gridInd <= 13:
                        traj[ii][k, 9 + gridInd - 1] = frameL[l, 1]  # 车辆ID在索引1
        
        # 本车道
        for l in range(frameEgo.shape[0]):
            y = frameEgo[l, 4] - traj[ii][k, 4]  # 局部Y在索引4
            if abs(y) < 90 and y != 0:
                gridInd = int(14 + round((y + 90) / 15))
                if 14 <= gridInd <= 26:
                    traj[ii][k, 9 + gridInd - 1] = frameEgo[l, 1]  # 车辆ID在索引1
        
        # 右车道
        if len(frameR) > 0:
            for l in range(frameR.shape[0]):
                y = frameR[l, 4] - traj[ii][k, 4]  # 局部Y在索引4
                if abs(y) < 90:
                    gridInd = int(27 + round((y + 90) / 15))
                    if 27 <= gridInd <= 39:
                        traj[ii][k, 9 + gridInd - 1] = frameR[l, 1]  # 车辆ID在索引1

print("正在划分训练集、验证集和测试集...")

# 合并所有轨迹
trajAll = np.vstack(traj)

# 划分训练集、验证集、测试集
trajTr = []
trajVal = []
trajTs = []

for k in range(1, 7):  # 6个数据集
    mask = trajAll[:, 0] == k
    if np.sum(mask) == 0:
        continue
    
    subset = trajAll[mask]
    max_veh_id = np.max(subset[:, 1])
    ul1 = round(0.7 * max_veh_id)
    ul2 = round(0.8 * max_veh_id)
    
    mask_tr = (subset[:, 0] == k) & (subset[:, 1] <= ul1)
    mask_val = (subset[:, 0] == k) & (subset[:, 1] > ul1) & (subset[:, 1] <= ul2)
    mask_ts = (subset[:, 0] == k) & (subset[:, 1] > ul2)
    
    trajTr.append(subset[mask_tr])
    trajVal.append(subset[mask_val])
    trajTs.append(subset[mask_ts])

trajTr = np.vstack(trajTr) if trajTr else np.empty((0, trajAll.shape[1]))
trajVal = np.vstack(trajVal) if trajVal else np.empty((0, trajAll.shape[1]))
trajTs = np.vstack(trajTs) if trajTs else np.empty((0, trajAll.shape[1]))

# 构建轨迹字典
def build_tracks(traj_set):
    tracks = {}
    for k in range(1, 7):
        mask = traj_set[:, 0] == k
        if np.sum(mask) == 0:
            continue
        
        subset = traj_set[mask]
        carIds = np.unique(subset[:, 1])
        
        for carId in carIds:
            car_mask = subset[:, 1] == carId
            car_data = subset[car_mask]
            # 提取: 帧号（列2）, 局部X（列3）, 局部Y（列4）
            # 轨迹应为: [帧号, 局部X, 局部Y]
            vehtrack = car_data[:, [2, 3, 4]].T  # 帧号, 局部X, 局部Y
            tracks[(k, int(carId))] = vehtrack
    return tracks

print("正在构建轨迹...")
tracksTr = build_tracks(trajTr)
tracksVal = build_tracks(trajVal)
tracksTs = build_tracks(trajTs)

print("正在过滤边缘情况...")

# 过滤边缘情况 - 需要至少30帧历史数据和足够的未来帧
def filter_edge_cases(traj_set, tracks_dict):
    valid_indices = []
    for k in range(traj_set.shape[0]):
        dsId = int(traj_set[k, 0])
        vehId = int(traj_set[k, 1])
        t = traj_set[k, 2]
        
        key = (dsId, vehId)
        if key not in tracks_dict:
            continue
        
        track = tracks_dict[key]
        if track.shape[1] < 31:
            continue
        
        # 检查第31帧是否存在以及是否有足够的未来帧
        frame_indices = np.where(track[0, :] == t)[0]
        if len(frame_indices) == 0:
            continue
        
        frame_idx = frame_indices[0]
        if frame_idx >= 30 and frame_idx < track.shape[1] - 1:
            valid_indices.append(k)
    
    return traj_set[valid_indices] if valid_indices else np.empty((0, traj_set.shape[1]))

trajTr = filter_edge_cases(trajTr, tracksTr)
trajVal = filter_edge_cases(trajVal, tracksVal)
trajTs = filter_edge_cases(trajTs, tracksTs)

print("正在保存CSV文件...")

# 如果数据目录不存在则创建
os.makedirs('data', exist_ok=True)

# 将轨迹保存为CSV
def save_traj_csv(traj_data, filename):
    # 创建列名
    # 列: 数据集ID, 车辆ID, 帧号, 局部X, 局部Y, 车道ID, 速度,
    #     横向机动, 纵向机动, 网格邻居（39列）
    cols = ['DatasetId', 'VehicleId', 'FrameNumber', 'LocalX', 'LocalY', 'LaneId', 'Velocity',
            'LateralManeuver', 'LongitudinalManeuver']
    # 添加网格邻居列
    for i in range(39):
        cols.append(f'GridNeighbor_{i+1}')
    
    df = pd.DataFrame(traj_data, columns=cols)
    df.to_csv(filename, index=False)
    print(f"已保存 {filename}，共 {len(df)} 行")

# 将轨迹保存为pickle（因为是嵌套字典结构）
def save_tracks_pickle(tracks_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tracks_dict, f)
    print(f"已保存 {filename}")

save_traj_csv(trajTr, 'data/TrainSet_traj.csv')
save_tracks_pickle(tracksTr, 'data/TrainSet_tracks.pkl')

save_traj_csv(trajVal, 'data/ValSet_traj.csv')
save_tracks_pickle(tracksVal, 'data/ValSet_tracks.pkl')

save_traj_csv(trajTs, 'data/TestSet_traj.csv')
save_tracks_pickle(tracksTs, 'data/TestSet_tracks.pkl')

print("预处理完成！")
print(f"训练集: {len(trajTr)} 个样本")
print(f"验证集: {len(trajVal)} 个样本")
print(f"测试集: {len(trajTs)} 个样本")

