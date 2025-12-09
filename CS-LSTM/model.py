from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation

"""
CS-LSTM (Convolutional Social Pooling LSTM) 模型
用于车辆轨迹预测，结合了LSTM的时序建模和卷积神经网络的空间建模
"""

class highwayNet(nn.Module):

    ## 初始化函数
    def __init__(self,args):
        super(highwayNet, self).__init__()

        ## 解包参数
        self.args = args

        ## GPU使用标志
        self.use_cuda = args['use_cuda']

        # 机动识别标志：True=基于机动的多模态解码器，False=单模态解码器
        self.use_maneuvers = args['use_maneuvers']

        # 训练模式标志：True=训练模式，False=测试模式
        self.train_flag = args['train_flag']

        ## 网络层大小参数
        self.encoder_size = args['encoder_size']  # 编码器LSTM隐藏层大小，默认64
        self.decoder_size = args['decoder_size']  # 解码器LSTM隐藏层大小，默认128
        self.in_length = args['in_length']  # 输入历史轨迹长度（下采样后），默认16（当前未使用，仅用于文档说明）
        self.out_length = args['out_length']  # 输出未来轨迹长度（下采样后），默认25（在decode函数中使用）
        self.grid_size = args['grid_size']  # 社会交互网格大小，默认(13,3)，表示13列3行
        self.soc_conv_depth = args['soc_conv_depth']  # 社会卷积层深度，默认64
        self.conv_3x1_depth = args['conv_3x1_depth']  # 3x1卷积层深度，默认16
        self.dyn_embedding_size = args['dyn_embedding_size']  # 车辆动态嵌入大小，默认32
        self.input_embedding_size = args['input_embedding_size']  # 输入嵌入大小，默认32
        self.num_lat_classes = args['num_lat_classes']  # 横向机动类别数，默认3（保持/左变/右变）
        self.num_lon_classes = args['num_lon_classes']  # 纵向机动类别数，默认2（保持加速/减速）
        
        # 计算社会嵌入大小：经过卷积和池化后的特征维度
        # 公式：((grid_width-4)+1)//2 * conv_3x1_depth
        # 对于grid_size=(13,3)，soc_embedding_size = ((13-4)+1)//2 * 16 = 5 * 16 = 80
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth

        ## 定义网络层权重

        # 输入嵌入层：将2D坐标(x,y)映射到高维特征空间
        # 输入：2维坐标，输出：input_embedding_size维特征（默认32）
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # 编码器LSTM：处理历史轨迹序列，提取时序特征
        # 输入：input_embedding_size维，输出：encoder_size维隐藏状态（默认64）
        # 单层LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # 车辆动态嵌入层：将LSTM编码的特征进一步压缩
        # 输入：encoder_size维，输出：dyn_embedding_size维（默认32）
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

        # 卷积社会池化层：处理邻居车辆的空间分布
        # soc_conv: 第一层卷积，kernel=3，将encoder_size(64)映射到soc_conv_depth(64)
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        # conv_3x1: 第二层卷积，kernel=(3,1)，将64映射到conv_3x1_depth(16)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
        # soc_maxpool: 最大池化层，kernel=(2,1)，padding=(1,0)，用于降维
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

        # 全连接社会池化层（用于对比实验，当前代码中未使用）:
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # 解码器LSTM：基于编码特征预测未来轨迹
        if self.use_maneuvers:
            # 如果使用机动识别：输入维度 = 社会嵌入 + 动态嵌入 + 横向机动类别数 + 纵向机动类别数
            # 80 + 32 + 3 + 2 = 117
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
        else:
            # 如果不使用机动识别：输入维度 = 社会嵌入 + 动态嵌入
            # 80 + 32 = 112
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)

        # 输出层：
        # op: 轨迹预测输出层，输出5个参数（muX, muY, sigX, sigY, rho）用于构建高斯分布
        self.op = torch.nn.Linear(self.decoder_size,5)
        # op_lat: 横向机动分类层，输出3个类别的概率
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
        # op_lon: 纵向机动分类层，输出2个类别的概率
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # 激活函数：
        self.leaky_relu = torch.nn.LeakyReLU(0.1)  # LeakyReLU，负斜率0.1
        self.relu = torch.nn.ReLU()  # ReLU激活函数
        self.softmax = torch.nn.Softmax(dim=1)  # Softmax，在类别维度上归一化


    ## 前向传播函数
    def forward(self,hist,nbrs,masks,lat_enc,lon_enc):
        """
        前向传播
        
        参数:
            hist: [16, batch_size, 2] - 目标车辆的历史轨迹
            nbrs: [16, nbr_batch_size, 2] - 所有邻居车辆的历史轨迹
            masks: [batch_size, 3, 13, 64] - 社会掩码，标记哪些网格位置有邻居
            lat_enc: [batch_size, 3] - 横向机动one-hot编码（训练时使用真实标签）
            lon_enc: [batch_size, 2] - 纵向机动one-hot编码（训练时使用真实标签）
        
        返回:
            训练模式: (fut_pred, lat_pred, lon_pred)
            测试模式: (fut_pred_list, lat_pred, lon_pred) - fut_pred_list包含6个轨迹分布
        """

        ## 处理目标车辆历史轨迹 (hist)
        # 输入: hist [16, batch_size, 2]
        # 1. 输入嵌入：将坐标(x,y)映射到32维特征空间
        hist_embedded = self.leaky_relu(self.ip_emb(hist))  # [16, batch_size, 32]
        # 2. LSTM编码：提取时序特征，取最后一个时间步的隐藏状态
        _,(hist_enc,_) = self.enc_lstm(hist_embedded)  # hist_enc: [1, batch_size, 64]
        # 3. 调整维度并应用动态嵌入
        hist_enc = hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])  # [batch_size, 64]
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc))  # [batch_size, 32]
        # 最终 hist_enc: [batch_size, 32] - 目标车辆的动态特征嵌入

        ## 处理邻居车辆轨迹 (nbrs)
        # 输入: nbrs [16, nbr_batch_size, 2]
        # nbr_batch_size是所有样本中邻居车辆的总数（可变）
        # 1. 输入嵌入
        nbrs_embedded = self.leaky_relu(self.ip_emb(nbrs))  # [16, nbr_batch_size, 32]
        # 2. LSTM编码：提取每个邻居车辆的时序特征
        _, (nbrs_enc,_) = self.enc_lstm(nbrs_embedded)  # nbrs_enc: [1, nbr_batch_size, 64]
        # 3. 调整维度
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])  # [nbr_batch_size, 64]
        # 最终 nbrs_enc: [nbr_batch_size, 64] - 所有邻居车辆的编码特征

        ## 掩码散射：将邻居车辆特征按空间位置填充到网格中
        # 
        # 网格位置是如何确定的？
        #   网格位置在预处理阶段（preprocess_data.py）根据以下信息计算：
        #   1. 车道信息：左车道、本车道、右车道（3行）
        #   2. 纵向距离：邻居车辆相对于目标车辆的Y坐标差（13列，每列约15英尺）
        #   3. 网格索引：1-13（左车道），14-26（本车道），27-39（右车道）
        #   详细说明请参考：网格位置说明.md
        #
        # masks 的含义：
        #   - 形状: [batch_size, 3, 13, 64]
        #   - 这是一个布尔掩码，标记了哪些网格位置有邻居车辆
        #   - 网格大小: 13列(纵向) × 3行(横向车道)
        #   - 每个网格位置有64维（编码器大小）
        #   - 如果某个网格位置有邻居车辆，该位置的64维全部为True(1)，否则为False(0)
        #   - 在 collate_fn 中构建：当发现某个网格位置有邻居时，设置 mask_batch[sampleId, row, col, :] = 1
        #   - 网格索引到行列位置的转换：
        #     * row = id // 13  (id是neighbors列表的索引，0-38)
        #     * col = id % 13
        #
        # masked_scatter_ 操作说明：
        #   - 功能：根据掩码将源张量的值"散射"到目标张量的对应位置
        #   - 输入：
        #     * soc_enc: 目标张量，初始全0，形状 [batch_size, 3, 13, 64]
        #     * masks: 布尔掩码，形状 [batch_size, 3, 13, 64]，标记哪些位置需要填充
        #     * nbrs_enc: 源张量，形状 [nbr_batch_size, 64]，包含所有邻居车辆的编码特征
        #   - 工作原理：
        #     1. 按行优先顺序遍历 masks 中所有为True的位置
        #     2. 从 nbrs_enc 中按顺序取出对应的64维特征向量
        #     3. 将特征向量填充到 soc_enc 的对应位置
        #   - 结果：soc_enc 中，有邻居的网格位置被填充了邻居特征，没有邻居的位置保持为0
        #
        # 示例：
        #   假设 batch_size=1，某个样本有2个邻居：
        #   - 邻居1在网格位置 (row=1, col=5)
        #   - 邻居2在网格位置 (row=2, col=8)
        #   - masks[0, 1, 5, :] = [1,1,1,...,1] (64个1)
        #   - masks[0, 2, 8, :] = [1,1,1,...,1] (64个1)
        #   - nbrs_enc[0, :] 是邻居1的64维特征
        #   - nbrs_enc[1, :] 是邻居2的64维特征
        #   - masked_scatter_ 后：
        #     * soc_enc[0, 1, 5, :] = nbrs_enc[0, :]  (邻居1的特征)
        #     * soc_enc[0, 2, 8, :] = nbrs_enc[1, :]  (邻居2的特征)
        #     * 其他位置保持为0
        
        # 初始化社会编码张量，形状与masks相同
        soc_enc = torch.zeros_like(masks).float()  # [batch_size, 3, 13, 64]
        # 使用掩码将nbrs_enc中的特征填充到对应网格位置
        # masks标记了哪些位置有邻居车辆，nbrs_enc包含这些邻居的特征
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)  # [batch_size, 3, 13, 64]
        # 调整维度顺序，准备进行卷积操作：(batch, height, width, channels) -> (batch, channels, width, height)
        soc_enc = soc_enc.permute(0,3,2,1)  # [batch_size, 64, 13, 3]

        ## 应用卷积社会池化：提取空间特征
        # 1. 第一层卷积：soc_conv，kernel=3
        soc_enc = self.leaky_relu(self.soc_conv(soc_enc))  # [batch_size, 64, 11, 1] (13-3+1=11)
        # 2. 第二层卷积：conv_3x1，kernel=(3,1)
        soc_enc = self.leaky_relu(self.conv_3x1(soc_enc))  # [batch_size, 16, 9, 1] (11-3+1=9)
        # 3. 最大池化：降维
        soc_enc = self.soc_maxpool(soc_enc)  # [batch_size, 16, 5, 1]
        # 4. 展平为向量
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)  # [batch_size, 80]
        # 最终 soc_enc: [batch_size, 80] - 社会交互特征嵌入

        ## 全连接社会池化（用于对比实验，当前代码中未使用）
        # soc_enc = soc_enc.contiguous()
        # soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        # soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        ## 拼接编码特征：将社会交互特征和车辆动态特征合并
        # soc_enc: [batch_size, 80]
        # hist_enc: [batch_size, 32]
        enc = torch.cat((soc_enc,hist_enc),1)  # [batch_size, 112] (80+32)


        if self.use_maneuvers:
            ## 机动识别：预测横向和纵向机动类别
            # 横向机动分类：3个类别（保持车道/向左变道/向右变道）
            lat_pred = self.softmax(self.op_lat(enc))  # [batch_size, 3]
            # 纵向机动分类：2个类别（保持加速/减速）
            lon_pred = self.softmax(self.op_lon(enc))  # [batch_size, 2]

            if self.train_flag:
                ## 训练模式：使用真实的机动标签
                # 拼接真实机动编码到特征向量
                enc = torch.cat((enc, lat_enc, lon_enc), 1)  # [batch_size, 117] (112+3+2)
                # 解码生成未来轨迹预测
                fut_pred = self.decode(enc)  # [25, batch_size, 5]
                return fut_pred, lat_pred, lon_pred
            else:
                ## 测试模式：为每个机动组合生成轨迹分布（多模态预测）
                fut_pred = []
                # 遍历所有机动组合：2个纵向 × 3个横向 = 6种组合
                for k in range(self.num_lon_classes):  # k: 0或1（纵向）
                    for l in range(self.num_lat_classes):  # l: 0,1或2（横向）
                        # 创建当前机动组合的one-hot编码
                        lat_enc_tmp = torch.zeros_like(lat_enc)  # [batch_size, 3]
                        lon_enc_tmp = torch.zeros_like(lon_enc)  # [batch_size, 2]
                        lat_enc_tmp[:, l] = 1  # 设置横向机动
                        lon_enc_tmp[:, k] = 1  # 设置纵向机动
                        # 拼接特征和机动编码
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)  # [batch_size, 117]
                        # 解码生成该机动组合的轨迹分布
                        fut_pred.append(self.decode(enc_tmp))  # [25, batch_size, 5]
                # 返回6个轨迹分布（对应6种机动组合）和机动预测概率
                return fut_pred, lat_pred, lon_pred
        else:
            ## 不使用机动识别：单模态预测
            fut_pred = self.decode(enc)  # [25, batch_size, 5]
            return fut_pred


    def decode(self,enc):
        """
        解码器函数：基于编码特征生成未来轨迹分布
        
        参数:
            enc: [batch_size, feature_dim] - 编码特征向量
                - 使用机动时: feature_dim = 117 (112+3+2)
                - 不使用机动时: feature_dim = 112
        
        返回:
            fut_pred: [25, batch_size, 5] - 未来轨迹分布参数
                5个参数：muX(均值X), muY(均值Y), sigX(标准差X), sigY(标准差Y), rho(相关系数)
        """
        # 将编码特征重复out_length次，为每个未来时间步提供输入
        # 输入: enc [batch_size, feature_dim]
        enc = enc.repeat(self.out_length, 1, 1)  # [25, batch_size, feature_dim]
        
        # LSTM解码：生成每个时间步的隐藏状态
        h_dec, _ = self.dec_lstm(enc)  # h_dec: [25, batch_size, 128]
        
        # 调整维度顺序：(seq_len, batch, hidden) -> (batch, seq_len, hidden)
        h_dec = h_dec.permute(1, 0, 2)  # [batch_size, 25, 128]
        
        # 输出层：将隐藏状态映射到5个分布参数
        fut_pred = self.op(h_dec)  # [batch_size, 25, 5]
        
        # 调整维度顺序：(batch, seq_len, features) -> (seq_len, batch, features)
        fut_pred = fut_pred.permute(1, 0, 2)  # [25, batch_size, 5]
        
        # 应用输出激活函数：确保分布参数的有效性
        # - sigX, sigY: 通过exp确保为正数
        # - rho: 通过tanh确保在[-1,1]范围内
        fut_pred = outputActivation(fut_pred)  # [25, batch_size, 5]
        
        return fut_pred





