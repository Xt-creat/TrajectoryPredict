from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import pandas as pd
import pickle
import os

#___________________________________________________________________________________________________________________________

### NGSIM数据集类
class ngsimDataset(Dataset):


    def __init__(self, data_file, t_h=30, t_f=50, d_s=2, enc_size = 64, grid_size = (13,3)):
        # 支持 .mat 和 CSV 两种格式
        if data_file.endswith('.mat'):
            # 从MATLAB .mat文件加载
            self.D = scp.loadmat(data_file)['traj']
            self.T = scp.loadmat(data_file)['tracks']
        elif data_file.endswith('_traj.csv'):
            # 从CSV文件加载
            traj_df = pd.read_csv(data_file)
            self.D = traj_df.values.astype(np.float32)
            
            # 加载对应的轨迹pickle文件
            tracks_file = data_file.replace('_traj.csv', '_tracks.pkl')
            if os.path.exists(tracks_file):
                with open(tracks_file, 'rb') as f:
                    tracks_dict = pickle.load(f)
                # 将字典转换为类似MATLAB的结构
                # MATLAB结构: T[dsId-1][vehId-1] = track
                # 需要创建一个列表的列表
                max_ds = int(np.max(self.D[:, 0]))
                max_veh = int(np.max(self.D[:, 1]))
                self.T = [[None] * (max_veh + 1) for _ in range(max_ds)]
                
                for (dsId, vehId), track in tracks_dict.items():
                    if dsId <= max_ds and vehId <= max_veh:
                        self.T[dsId - 1][vehId - 1] = track
            else:
                raise FileNotFoundError(f"未找到轨迹文件: {tracks_file}")
        else:
            raise ValueError(f"不支持的文件格式: {data_file}")
        
        self.t_h = t_h  # 轨迹历史长度
        self.t_f = t_f  # 预测轨迹长度
        self.d_s = d_s  # 所有序列的下采样率
        self.enc_size = enc_size # 编码器LSTM的大小
        self.grid_size = grid_size # 社会上下文网格的大小



    def __len__(self):
        return len(self.D)



    def __getitem__(self, idx):

        dsId = int(self.D[idx, 0])
        vehId = int(self.D[idx, 1])
        t = self.D[idx, 2]
        # 网格邻居在处理后的数据中从第9列（索引8）开始
        # 列: 0:数据集ID, 1:车辆ID, 2:帧号, 3:局部X, 4:局部Y, 
        #     5:车道ID, 6:速度, 7:横向机动, 8:纵向机动,
        #     9-47:网格邻居（39列）
        grid = self.D[idx, 9:9+39]  # 39个网格位置
        neighbors = []

        # 获取轨迹历史 'hist' = ndarray，以及未来轨迹 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId)
        fut = self.getFuture(vehId,t,dsId)

        # 获取所有邻居的轨迹历史 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId))

        # 机动 'lon_enc' = one-hot向量, 'lat_enc' = one-hot向量
        # 横向机动在第7列（索引7），纵向机动在第8列（索引8）
        lon_enc = np.zeros([2])
        lon_man = int(self.D[idx, 8]) - 1  # 纵向机动 (1或2) -> 0或1
        if 0 <= lon_man < 2:
            lon_enc[lon_man] = 1
        lat_enc = np.zeros([3])
        lat_man = int(self.D[idx, 7]) - 1  # 横向机动 (1, 2, 或 3) -> 0, 1, 或 2
        if 0 <= lat_man < 3:
            lat_enc[lat_man] = 1

        return hist,fut,neighbors,lat_enc,lon_enc



    ## 获取轨迹历史的辅助函数
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            # 处理MATLAB单元数组结构和Python列表结构
            if isinstance(self.T, np.ndarray):
                # MATLAB结构
                if self.T.shape[1] <= vehId-1:
                    return np.empty([0,2])
                refTrack = self.T[dsId-1][refVehId-1].transpose()
                vehTrack = self.T[dsId-1][vehId-1].transpose()
            else:
                # Python列表结构（来自pickle）
                if dsId-1 >= len(self.T) or refVehId-1 >= len(self.T[dsId-1]) or self.T[dsId-1][refVehId-1] is None:
                    return np.empty([0,2])
                if dsId-1 >= len(self.T) or vehId-1 >= len(self.T[dsId-1]) or self.T[dsId-1][vehId-1] is None:
                    return np.empty([0,2])
                refTrack = self.T[dsId-1][refVehId-1].transpose()
                vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist



    ## 获取未来轨迹的辅助函数
    def getFuture(self, vehId, t,dsId):
        # 处理MATLAB单元数组结构和Python列表结构
        if isinstance(self.T, np.ndarray):
            vehTrack = self.T[dsId-1][vehId-1].transpose()
        else:
            if dsId-1 >= len(self.T) or vehId-1 >= len(self.T[dsId-1]) or self.T[dsId-1][vehId-1] is None:
                return np.empty([0,2])
            vehTrack = self.T[dsId-1][vehId-1].transpose()
        
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut



    ## 数据加载器的整理函数
    def collate_fn(self, samples):

        # 初始化邻居和邻居长度批次:
        nbr_batch_size = 0
        for _,_,nbrs,_,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
        maxlen = self.t_h//self.d_s + 1
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)


        # 初始化社会掩码批次:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size)
        mask_batch = mask_batch.bool()


        # 初始化历史、历史长度、未来、输出掩码、横向机动和纵向机动批次:
        hist_batch = torch.zeros(maxlen,len(samples),2)
        fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
        lat_enc_batch = torch.zeros(len(samples),3)
        lon_enc_batch = torch.zeros(len(samples), 2)


        count = 0
        for sampleId,(hist, fut, nbrs, lat_enc, lon_enc) in enumerate(samples):

            # 设置历史、未来、横向机动和纵向机动批次:
            hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut),sampleId,:] = 1
            lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            # 设置邻居、邻居序列长度和掩码批次:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).bool()
                    count+=1

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch

#________________________________________________________________________________________________________________________________________





## 输出层的自定义激活函数 (Graves, 2015)
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out

## 批次NLL损失，使用掩码处理可变输出长度
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho = y_pred[:,:,4]
    ohr = torch.pow(1-torch.pow(rho,2),-0.5)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    # 如果以英尺^(-1)表示似然:
    out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # 如果以米^(-1)表示似然:
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## 序列的NLL，为每个时间步输出NLL值序列，使用掩码处理可变输出长度，用于评估
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = True, avg_along_time = False):
    if use_maneuvers:
        # 根据输入张量的设备决定输出设备
        device = op_mask.device
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes, device=device)
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k]
                wts = wts.repeat(len(fut_pred[0]),1)
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                # 如果以英尺^(-1)表示似然:
                out = -(0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + 0.5*torch.pow(sigY, 2)*torch.pow(y-muY, 2) - rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379)
                # 如果以米^(-1)表示似然:
                # out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160)
                acc[:, :, count] =  out + torch.log(wts)
                count+=1
        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        # 根据输入张量的设备决定输出设备
        device = op_mask.device
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1, device=device)
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        # 如果以英尺^(-1)表示似然:
        out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2 * rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
        # 如果以米^(-1)表示似然:
        # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## 批次MSE损失，使用掩码处理可变输出长度
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal

## 完整序列的MSE损失，输出MSE值序列，使用掩码处理可变输出长度，用于评估
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## 对数求和指数的辅助函数:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


# 如果直接运行此脚本，打印一个训练样本的详细信息
if __name__ == "__main__":
    print("="*80)
    print("加载训练数据集并打印一个样本的详细信息")
    print("="*80)
    
    # 尝试加载训练集
    train_file = 'data/TrainSet_traj.csv'
    if not os.path.exists(train_file):
        print(f"错误: 未找到训练数据文件 {train_file}")
        print("请先运行 preprocess_data.py 生成预处理数据")
    else:
        # 创建数据集实例
        print(f"\n正在加载数据集: {train_file}")
        dataset = ngsimDataset(train_file, t_h=30, t_f=50, d_s=2)
        print(f"数据集大小: {len(dataset)} 个样本")
        
        # 获取第一个有效样本（跳过可能为空的样本）
        print("\n正在获取样本...")
        sample_idx = 0
        max_tries = min(100, len(dataset))
        
        for i in range(max_tries):
            try:
                hist, fut, neighbors, lat_enc, lon_enc = dataset[i]
                if len(hist) > 0 and len(fut) > 0:
                    sample_idx = i
                    break
            except:
                continue
        
        print(f"使用样本索引: {sample_idx}")
        hist, fut, neighbors, lat_enc, lon_enc = dataset[sample_idx]
        
        # 获取原始数据信息
        dsId = int(dataset.D[sample_idx, 0])
        vehId = int(dataset.D[sample_idx, 1])
        frameNum = dataset.D[sample_idx, 2]
        localX = dataset.D[sample_idx, 3]
        localY = dataset.D[sample_idx, 4]
        laneId = dataset.D[sample_idx, 5]
        velocity = dataset.D[sample_idx, 6]
        lat_man = int(dataset.D[sample_idx, 7])
        lon_man = int(dataset.D[sample_idx, 8])
        grid = dataset.D[sample_idx, 9:9+39]
        
        # 打印样本信息
        print("\n" + "="*80)
        print("样本基本信息:")
        print("="*80)
        print(f"数据集ID: {dsId}")
        print(f"车辆ID: {vehId}")
        print(f"帧号: {frameNum}")
        print(f"当前位置: LocalX={localX:.2f}, LocalY={localY:.2f}")
        print(f"车道ID: {laneId}")
        print(f"速度: {velocity:.2f} 英尺/秒")
        print(f"横向机动: {lat_man} (1=保持车道, 2=向左变道, 3=向右变道)")
        print(f"纵向机动: {lon_man} (1=保持/加速, 2=减速)")
        print(f"横向机动one-hot编码: {lat_enc}")
        print(f"纵向机动one-hot编码: {lon_enc}")
        
        print("\n" + "="*80)
        print("历史轨迹 (hist):")
        print("="*80)
        print(f"形状: {hist.shape}")
        print(f"长度: {len(hist)} 个时间步 (下采样后，原始为30帧)")
        print("前5个时间步:")
        for i in range(min(5, len(hist))):
            print(f"  时间步 {i}: X={hist[i, 0]:.2f}, Y={hist[i, 1]:.2f}")
        if len(hist) > 5:
            print("  ...")
            print(f"  时间步 {len(hist)-1}: X={hist[-1, 0]:.2f}, Y={hist[-1, 1]:.2f}")
        
        print("\n" + "="*80)
        print("未来轨迹 (fut) - 预测目标:")
        print("="*80)
        print(f"形状: {fut.shape}")
        print(f"长度: {len(fut)} 个时间步 (下采样后，原始为50帧)")
        print("前5个时间步:")
        for i in range(min(5, len(fut))):
            print(f"  时间步 {i}: X={fut[i, 0]:.2f}, Y={fut[i, 1]:.2f}")
        if len(fut) > 5:
            print("  ...")
            print(f"  时间步 {len(fut)-1}: X={fut[-1, 0]:.2f}, Y={fut[-1, 1]:.2f}")
        
        print("\n" + "="*80)
        print("邻居车辆信息:")
        print("="*80)
        print(f"网格大小: {dataset.grid_size[0]} x {dataset.grid_size[1]} = {len(neighbors)} 个位置")
        non_empty_neighbors = sum([len(nbr) > 0 for nbr in neighbors])
        print(f"有邻居的位置数: {non_empty_neighbors}")
        
        # 显示有邻居的网格位置
        print("\n有邻居车辆的网格位置:")
        neighbor_count = 0
        for i, nbr in enumerate(neighbors):
            if len(nbr) > 0:
                neighbor_count += 1
                grid_x = i % dataset.grid_size[0]
                grid_y = i // dataset.grid_size[0]
                vehicle_id = grid[i] if i < len(grid) else 0
                print(f"  网格位置 [{grid_y}, {grid_x}] (索引{i}): 车辆ID={vehicle_id:.0f}, 轨迹长度={len(nbr)}")
                if neighbor_count >= 5:  # 只显示前5个
                    remaining = non_empty_neighbors - 5
                    if remaining > 0:
                        print(f"  ... 还有 {remaining} 个位置有邻居车辆")
                    break
        
        print("\n" + "="*80)
        print("样本统计信息:")
        print("="*80)
        print(f"历史轨迹范围: X=[{hist[:, 0].min():.2f}, {hist[:, 0].max():.2f}], Y=[{hist[:, 1].min():.2f}, {hist[:, 1].max():.2f}]")
        print(f"未来轨迹范围: X=[{fut[:, 0].min():.2f}, {fut[:, 0].max():.2f}], Y=[{fut[:, 1].min():.2f}, {fut[:, 1].max():.2f}]")
        print(f"历史轨迹平均速度: {np.sqrt(np.diff(hist[:, 0])**2 + np.diff(hist[:, 1])**2).mean()*10:.2f} 英尺/秒 (假设10Hz采样)")
        print(f"未来轨迹平均速度: {np.sqrt(np.diff(fut[:, 0])**2 + np.diff(fut[:, 1])**2).mean()*10:.2f} 英尺/秒")
        
        print("\n" + "="*80)
        print("样本数据结构总结:")
        print("="*80)
        print("输入:")
        print("  - hist: 历史轨迹 [16, 2] - 目标车辆过去3秒的轨迹")
        print("  - neighbors: 39个邻居车辆的历史轨迹列表")
        print("  - lat_enc: 横向机动one-hot编码 [3]")
        print("  - lon_enc: 纵向机动one-hot编码 [2]")
        print("输出:")
        print("  - fut: 未来轨迹 [25, 2] - 目标车辆未来5秒的轨迹（预测目标）")
        print("="*80)
