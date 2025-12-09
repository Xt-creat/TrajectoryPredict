'''
PyTorch implementation of Social LSTM with Goal and Social Array

模型结构说明：
===========
本模型实现了带Goal和社交数组（Social Array）的社交LSTM，用于行人轨迹预测。

模型架构：
1. 空间嵌入层（Spatial Embedding）
   - 输入：[x, y, goal_x, goal_y] (4维)
   - 输出：embedding_size 维向量
   
2. 社交数组嵌入层（Social Array Embedding）⭐ 社交数组在这里作用
   - 输入：社交数组 [grid_size*2] (包含最近grid_size个其他行人的x,y坐标)
   - 输出：1维标量
   
3. 特征融合
   - 将空间嵌入和社交数组嵌入拼接：[embedding_size + 1]
   
4. LSTM层
   - 输入：融合后的特征 [embedding_size + 1]
   - 输出：隐藏状态 [rnn_size]
   
5. 输出层
   - 输入：LSTM隐藏状态 [rnn_size]
   - 输出：5个参数 [mux, muy, sx, sy, corr]
     用于定义二维高斯分布，预测下一帧的位置

社交数组的作用：
===============
社交数组在模型的第2步（社交数组嵌入层）起作用，具体位置：
- 第99行：提取当前行人的社交数组
- 第103行：通过线性层将社交数组嵌入为1维特征
- 第106行：与空间嵌入拼接，一起输入LSTM

这样，模型能够同时考虑：
- 行人自身的位置和目标（空间嵌入）
- 周围其他行人的位置分布（社交数组嵌入）

Converted from TensorFlow version
Original: Modified by Simone Zamboni
PyTorch conversion: 2025
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from grid import getSequenceGridMask


class SocialModel(nn.Module):
    """
    社交LSTM模型（带Goal和社交数组）
    
    模型结构：
    [x, y, goal_x, goal_y] → 空间嵌入层 → [embedding_size]
    [社交数组 grid_size*2] → 社交数组嵌入层 → [1]  ⭐ 社交数组在这里
    [embedding_size + 1] → LSTM → [rnn_size] → 输出层 → [mux, muy, sx, sy, corr]
    """
    
    def __init__(self, args, infer=False):
        super(SocialModel, self).__init__()
        
        if infer:
            args.batch_size = 1
            args.seq_length = 1
        
        self.args = args
        self.infer = infer
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.maxNumPeds = args.maxNumPeds
        self.output_size = 5
        
        # ========== 模型组件定义 ==========
        
        # 1. 空间嵌入层（Spatial Embedding）
        # 输入：4维 [x, y, goal_x, goal_y]
        # 输出：embedding_size 维向量
        self.embedding_w = nn.Linear(4, args.embedding_size)
        self.embedding_b = nn.Parameter(torch.ones(args.embedding_size) * 0.1)
        
        # 2. 社交数组嵌入层（Social Array Embedding）⭐ 社交数组在这里处理
        # 输入：grid_size*2 维（包含最近grid_size个其他行人的x,y坐标）
        # 输出：1维标量
        # 作用：将社交数组编码为单个特征值，表示周围行人的影响
        self.embedding_t_w = nn.Linear(args.grid_size * 2, 1)
        self.embedding_t_b = nn.Parameter(torch.ones(1) * 0.1)
        
        # 3. LSTM层
        # 输入：embedding_size + 1（空间嵌入 + 社交数组嵌入）
        # 输出：rnn_size 维隐藏状态
        self.lstm = nn.LSTMCell(args.embedding_size + 1, args.rnn_size)
        
        # 4. 输出层
        # 输入：rnn_size 维隐藏状态
        # 输出：5个原始值 [mux_raw, muy_raw, sx_raw, sy_raw, corr_raw]
        #   这些原始值需要经过转换才能成为有效的分布参数：
        #   - mux, muy: 预测位置的均值（x, y坐标），直接使用原始值
        #   - sx, sy: 预测位置的标准差（不确定性），通过 exp() 转换
        #   - corr: x和y之间的相关系数，通过 tanh() 转换
        #
        # 方差（sx, sy）的含义：
        # ===================
        # sx, sy 是标准差（standard deviation），不是方差（variance）
        # - 方差 = 标准差的平方：variance = std²
        # - 标准差表示预测的不确定性
        # - sx 小 → x坐标预测很确定（集中在均值附近）
        # - sx 大 → x坐标预测不确定（可能偏离均值较远）
        #
        # 取值范围：
        # ========
        # - 原始输出 sx_raw, sy_raw: 无限制（可以是任意实数）
        # - 转换后 sx = exp(sx_raw), sy = exp(sy_raw)
        # - 取值范围：(0, +∞)，即所有正数
        # - 没有上限，但实际训练中通常不会太大
        # - 下限接近0（当 sx_raw → -∞ 时，sx → 0）
        #
        # 实际意义：
        # ========
        # - sx = 1.0: 预测位置在均值±1像素范围内的概率约为68%（1个标准差）
        # - sx = 5.0: 预测位置在均值±5像素范围内的概率约为68%
        # - sx = 0.1: 预测非常确定，位置几乎就是均值
        # - sx = 100.0: 预测非常不确定，位置可能在很大范围内
        self.output_w = nn.Linear(args.rnn_size, self.output_size)
        self.output_b = nn.Parameter(torch.zeros(self.output_size))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights similar to TensorFlow version"""
        # Embedding weights
        nn.init.normal_(self.embedding_w.weight, std=0.1)
        nn.init.normal_(self.embedding_t_w.weight, std=0.1)
        nn.init.normal_(self.output_w.weight, std=0.1)
    
    def forward(self, input_data, grid_data, target_data=None):
        """
        前向传播
        
        数据流向：
        1. 输入数据 → 空间嵌入 → [embedding_size]
        2. 社交数组 → 社交数组嵌入 → [1]  ⭐ 社交数组在这里作用
        3. 拼接 → [embedding_size + 1] → LSTM → [rnn_size] → 输出层 → [5]
        
        参数：
            input_data: [seq_length, maxNumPeds, 5] - 输入序列，每行 [ID, x, y, goal_x, goal_y]
            grid_data: [seq_length, maxNumPeds, grid_size*2] - 社交数组序列 ⭐ 关键输入
            target_data: [seq_length, maxNumPeds, 5] - 目标数据（用于计算损失）
        
        返回：
            如果提供了target_data: (loss, (h_states, c_states))
            否则: (h_states, c_states)
        """
        seq_length = input_data.shape[0]
        device = input_data.device
        
        # 为每个行人初始化LSTM状态（使用列表避免原地操作问题）
        h_states = [torch.zeros(self.rnn_size, device=device) for _ in range(self.maxNumPeds)]
        c_states = [torch.zeros(self.rnn_size, device=device) for _ in range(self.maxNumPeds)]
        
        total_loss = 0.0
        total_counter = 0.0
        
        # ========== 逐帧处理序列 ==========
        # 重要说明：为什么每个时间步都要计算输出？
        # ============================================
        # 这是一个序列到序列（Seq2Seq）的训练方式，使用 Teacher Forcing 策略
        # 
        # 数据关系：
        # - input_data[0] 是第0帧，target_data[0] 是第1帧（下一个位置）
        # - input_data[1] 是第1帧，target_data[1] 是第2帧（下一个位置）
        # - ...
        # - input_data[t] 是第t帧，target_data[t] 是第t+1帧（下一个位置）
        #
        # 为什么每个时间步都要计算输出和损失？
        # 1. 学习每一步的预测能力：模型需要学会从任意时刻预测下一时刻
        # 2. 充分利用梯度信息：每个时间步的损失都会产生梯度，加速训练
        # 3. 提高训练稳定性：多步损失的平均比单步损失更稳定
        # 4. 模拟实际预测场景：推理时需要逐步预测，训练时也要逐步学习
        #
        # 如果只在最后一步计算输出会怎样？
        # - 模型只能学习"从整个序列预测最后位置"，而不是"每一步预测下一步"
        # - 无法利用中间步骤的梯度，训练效率低
        # - 推理时表现会差，因为模型没有学习逐步预测
        # ============================================
        for seq in range(seq_length):
            current_frame = input_data[seq]  # [maxNumPeds, 5] - 当前帧的所有行人数据
            current_grid = grid_data[seq]    # [maxNumPeds, grid_size*2] - 当前帧的社交数组 ⭐
            
            # ========== 处理每个行人 ==========
            for ped in range(self.maxNumPeds):
                pedID = current_frame[ped, 0]
                
                # 跳过无效的行人（ID=0表示该位置没有行人）
                if pedID == 0:
                    continue
                
                # ========== 步骤1: 提取空间输入 ==========
                # 提取当前行人的位置和目标：[x, y, goal_x, goal_y]
                spatial_input = current_frame[ped, 1:5].unsqueeze(0)  # [1, 4]
                
                # ========== 步骤2: 提取社交数组输入 ⭐ 社交数组在这里使用 ==========
                # 提取当前行人的社交数组：[x1, y1, x2, y2, ..., x_grid_size, y_grid_size]
                # 这个数组包含该行人周围最近的grid_size个其他行人的位置
                tensor_input = current_grid[ped].unsqueeze(0)  # [1, grid_size*2]
                
                # ========== 步骤3: 嵌入层 ==========
                # 空间嵌入：将位置和目标编码为embedding_size维向量
                embedded_spatial = F.relu(self.embedding_w(spatial_input) + self.embedding_b)
                
                # 社交数组嵌入：将社交数组编码为1维标量 ⭐ 社交数组在这里被处理
                # 这个标量表示周围行人对当前行人的影响程度
                embedded_tensor = F.relu(self.embedding_t_w(tensor_input) + self.embedding_t_b)
                
                # ========== 步骤4: 特征融合 ==========
                # 将空间嵌入和社交数组嵌入拼接
                # 这样LSTM可以同时考虑行人自身信息和周围行人的影响
                complete_input = torch.cat([embedded_spatial, embedded_tensor], dim=1)  # [1, embedding_size+1]
                
                # ========== 步骤5: LSTM处理 ==========
                # LSTM更新隐藏状态，融合了空间信息和社交信息
                h_states[ped], c_states[ped] = self.lstm(
                    complete_input.squeeze(0), 
                    (h_states[ped], c_states[ped])
                )
                
                # ========== 步骤6: 输出层（每个时间步都计算） ==========
                # 从LSTM隐藏状态生成预测参数
                # 注意：每个时间步都要计算输出，用于预测下一个时间步的位置
                output = self.output_w(h_states[ped]) + self.output_b
                
                # ========== 步骤7: 计算损失（每个时间步都计算） ==========
                # 使用 Teacher Forcing：用真实的下一个位置作为目标
                if target_data is not None:
                    target_frame = target_data[seq]  # target_data[seq] 是 input_data[seq] 的下一个位置
                    target_pedID = target_frame[ped, 0]
                    
                    if target_pedID != 0:
                        x_target = target_frame[ped, 1]  # 真实的下一个x坐标
                        y_target = target_frame[ped, 2]  # 真实的下一个y坐标
                        
                        # 从输出中提取分布参数
                        mux, muy, sx, sy, corr = self.get_coef(output)
                        
                        # 计算负对数似然损失（基于二维高斯分布）
                        # 这个损失会反向传播，更新模型参数
                        loss = self.get_lossfunc(mux, muy, sx, sy, corr, x_target, y_target)
                        total_loss += loss
                        total_counter += 1.0
        
        # ========== 后处理：准备返回值 ==========
        
        # 步骤1: 将列表转换为张量
        # 前面使用列表存储每个行人的LSTM状态（避免原地操作问题）
        # 现在需要转换为张量以便返回
        # h_states: [maxNumPeds] 个 [rnn_size] 的列表 → [maxNumPeds, rnn_size] 张量
        # c_states: [maxNumPeds] 个 [rnn_size] 的列表 → [maxNumPeds, rnn_size] 张量
        h_states_tensor = torch.stack(h_states)
        c_states_tensor = torch.stack(c_states)
        
        # 步骤2: 计算平均损失并添加正则化
        if target_data is not None and total_counter > 0:
            # 计算所有时间步和所有行人的平均损失
            # total_loss: 所有有效预测的损失总和
            # total_counter: 有效预测的数量（每个时间步的每个有效行人计数1次）
            mean_loss = total_loss / total_counter
            
            # 步骤3: 添加L2正则化（防止过拟合）
            # L2正则化 = lambda_param * sum(所有参数的平方)
            # 作用：惩罚过大的参数值，使模型更简单，提高泛化能力
            l2_reg = 0
            for param in self.parameters():
                l2_reg += torch.sum(param ** 2)
            # 将正则化项加到损失中
            mean_loss = mean_loss + self.args.lambda_param * l2_reg
            
            # 返回：平均损失（含正则化）+ LSTM状态
            # 训练时使用：loss.backward() 会基于这个损失进行反向传播
            return mean_loss, (h_states_tensor, c_states_tensor)
            
        elif target_data is not None:
            # 情况2: 提供了target_data但没有有效的行人（total_counter=0）
            # 返回虚拟损失0.0，避免除零错误
            return torch.tensor(0.0, device=h_states_tensor.device), (h_states_tensor, c_states_tensor)
        else:
            # 情况3: 没有提供target_data（推理模式）
            # 只返回LSTM状态，不计算损失
            # 用于预测未来轨迹时，不需要计算损失
            return (h_states_tensor, c_states_tensor)
    
    def get_coef(self, output):
        """
        从模型输出中提取分布参数
        
        模型输出5个原始值，需要转换为有效的分布参数：
        - mux, muy: 直接使用（预测位置的均值）
        - sx, sy: 通过exp确保为正数（标准差）
        - corr: 通过tanh限制在[-1, 1]之间（相关系数）
        
        参数：
            output: [5] 张量，包含 [mux_raw, muy_raw, sx_raw, sy_raw, corr_raw]
        
        返回：
            mux, muy, sx, sy, corr - 用于定义二维高斯分布的参数
        
        方差（sx, sy）的取值范围：
        ========================
        转换公式：sx = exp(sx_raw)
        
        取值范围：
        - sx_raw → -∞: sx → 0（预测非常确定）
        - sx_raw = 0: sx = 1.0（标准情况）
        - sx_raw = 2: sx ≈ 7.4（预测较不确定）
        - sx_raw = 5: sx ≈ 148（预测非常不确定）
        - sx_raw → +∞: sx → +∞（理论上无上限）
        
        实际训练中的表现：
        - 训练初期：sx可能较大（模型不确定）
        - 训练后期：sx通常较小（模型更确定）
        - 典型值范围：0.1 ~ 50（取决于数据规模和模型训练程度）
        """
        mux = output[0]  # x坐标的均值（可以是任意实数）
        muy = output[1]  # y坐标的均值（可以是任意实数）
        
        # x坐标的标准差：通过exp确保为正数
        # sx_raw可以是任意实数，exp后得到正数
        # 取值范围：(0, +∞)
        sx = torch.exp(output[2])
        
        # y坐标的标准差：通过exp确保为正数
        # 取值范围：(0, +∞)
        sy = torch.exp(output[3])
        
        # x和y的相关系数：通过tanh限制在[-1, 1]之间
        corr = torch.tanh(output[4])
        
        return mux, muy, sx, sy, corr
    
    def tf_2d_normal(self, x, y, mux, muy, sx, sy, rho):
        """
        计算二维高斯分布的概率密度函数（PDF - Probability Density Function）
        
        PDF是什么？
        ===========
        PDF（概率密度函数）描述了一个连续随机变量在某个值附近的"可能性密度"。
        
        在这个模型中：
        - 模型预测的不是一个确定的位置，而是一个概率分布（二维高斯分布）
        - PDF值表示：在预测位置(mux, muy)附近，真实位置(x, y)出现的"密度"
        - PDF值越大，说明预测越准确（真实位置更接近预测位置）
        
        二维高斯分布PDF公式：
        PDF(x,y) = (1 / (2π·σx·σy·√(1-ρ²))) × exp(-z / (2(1-ρ²)))
        其中 z = ((x-μx)/σx)² + ((y-μy)/σy)² - 2ρ(x-μx)(y-μy)/(σx·σy)
        
        参数说明：
        - mux, muy: 预测位置的均值（最可能的位置）
        - sx, sy: 标准差（预测的不确定性）
        - rho (corr): x和y的相关系数
        - x, y: 真实位置
        
        返回值：
        - PDF值：一个正数，表示在预测分布下，真实位置的概率密度
        - PDF值可以大于1（这是正常的，因为它是密度而不是概率）
        - PDF值越大，说明预测越准确
        """
        normx = x - mux
        normy = y - muy
        sxsy = sx * sy
        
        z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * rho * normx * normy / sxsy
        negRho = 1 - rho ** 2
        
        result = torch.exp(-z / (2 * negRho))
        denom = 2 * np.pi * sxsy * torch.sqrt(negRho)
        result = result / denom
        
        return result
    
    def get_lossfunc(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        """
        计算负对数似然损失（Negative Log Likelihood Loss）
        
        损失计算过程：
        ============
        1. 计算PDF值：在预测分布下，真实位置的概率密度
           PDF = tf_2d_normal(x_data, y_data, mux, muy, sx, sy, corr)
        
        2. 计算负对数似然：
           Loss = -log(PDF)
        
        为什么使用负对数似然？
        ====================
        - 我们希望最大化PDF值（预测越准确，PDF越大）
        - 但优化器是"最小化"损失，所以用负对数
        - -log(PDF) 的值：PDF越大 → 损失越小
        
        损失值的含义：
        ============
        - 如果PDF = 0.1（预测不太准确），Loss = -log(0.1) ≈ 2.3（正数）
        - 如果PDF = 1.0（预测较准确），Loss = -log(1.0) = 0
        - 如果PDF = 10.0（预测很准确），Loss = -log(10.0) ≈ -2.3（负数）
        
        所以负数损失是正常的！表示预测非常准确。
        """
        result0 = self.tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
        epsilon = 1e-20  # 防止log(0)导致无穷大
        result1 = -torch.log(torch.clamp(result0, min=epsilon))
        return result1
    
    def sample_gaussian_2d(self, mux, muy, sx, sy, rho):
        """
        Sample from 2D normal distribution (returns mean for deterministic prediction)
        """
        # Modified version: return mean instead of sampling
        return mux, muy
    
    def sample(self, traj, grid, dimensions, true_traj, num=10):
        """
        预测未来轨迹（自回归预测）
        
        功能说明：
        ==========
        这个方法用于预测行人的未来轨迹。它分为两个阶段：
        1. 处理观察到的轨迹：使用真实的历史轨迹初始化LSTM状态
        2. 预测未来轨迹：基于LSTM状态，逐步预测未来的每一帧
        
        参数：
        - traj: 观察到的轨迹 [obs_length, maxNumPeds, 5]
               每帧包含 [pedID, x, y, goal_x, goal_y]
        - grid: 观察到的社交数组 [obs_length, maxNumPeds, grid_size*2]
        - dimensions: 图像尺寸（用于计算社交数组）
        - true_traj: 完整轨迹（包含真实未来位置，用于计算误差，可选）
        - num: 要预测的未来帧数
        
        返回：
        - complete_traj: 完整轨迹 [obs_length + num, maxNumPeds, 5]
                         包含观察部分 + 预测部分
        """
        # ========== 步骤1: 初始化 ==========
        self.eval()  # 设置为评估模式（关闭dropout等）
        device = next(self.parameters()).device  # 获取模型所在的设备
        
        # 转换为张量（如果还不是张量）
        if not isinstance(traj, torch.Tensor):
            traj = torch.FloatTensor(traj).to(device)
        if not isinstance(grid, torch.Tensor):
            grid = torch.FloatTensor(grid).to(device)
        if not isinstance(true_traj, torch.Tensor):
            true_traj = torch.FloatTensor(true_traj).to(device)
        
        # ========== 步骤2: 处理观察到的轨迹 ==========
        # 目的：使用真实的历史轨迹初始化LSTM的隐藏状态
        # 这样LSTM能够"记住"观察到的轨迹信息，用于后续预测
        
        # 为每个行人初始化LSTM状态（使用列表避免原地操作问题）
        h_states = [torch.zeros(self.rnn_size, device=device) for _ in range(self.maxNumPeds)]
        c_states = [torch.zeros(self.rnn_size, device=device) for _ in range(self.maxNumPeds)]
        
        obs_length = traj.shape[0]
        # 遍历观察轨迹的每一帧（除了最后一帧）
        # 例如：obs_length=8，则处理帧0-6，用它们来预测帧1-7
        for index in range(obs_length - 1):
            frame = traj[index:index+1]  # 当前帧 [1, maxNumPeds, 5]
            grid_frame = grid[index:index+1]  # 当前帧的社交数组 [1, maxNumPeds, grid_size*2]
            target_frame = traj[index+1:index+2]  # 下一帧（作为目标）[1, maxNumPeds, 5]
            
            # 前向传播：更新LSTM状态
            # 注意：这里使用真实的下一个位置作为目标，用于更新LSTM状态
            # 但不会计算损失（因为target_data只是用于更新状态）
            _, (h_states_tensor, c_states_tensor) = self.forward(frame, grid_frame, target_frame)
            
            # 将张量转换回列表，用于下一次迭代
            h_states = [h_states_tensor[i] for i in range(self.maxNumPeds)]
            c_states = [c_states_tensor[i] for i in range(self.maxNumPeds)]
        
        # ========== 步骤3: 准备预测 ==========
        # 初始化返回结果：包含观察到的轨迹
        ret = traj.clone()  # 复制观察到的轨迹
        
        # 从最后一帧开始预测
        last_frame = traj[-1:].clone()  # 最后一帧 [1, maxNumPeds, 5]
        prev_grid = grid[-1:].clone()  # 最后一帧的社交数组 [1, maxNumPeds, grid_size*2]
        
        # ========== 步骤4: 预测未来步骤（自回归预测） ==========
        # 重要说明：预测帧数与误差的关系
        # ============================================
        # 预测是自回归的（autoregressive）：
        # - 每一步的预测都基于前一步的预测结果
        # - 如果第1步有误差，这个误差会传播到第2步
        # - 第2步的误差会传播到第3步，以此类推
        #
        # 误差累积效应：
        # - 预测1帧：只预测下一个位置，误差通常较小（例如：5像素）
        # - 预测5帧：需要预测未来5个位置，误差会累积
        #   第1帧误差：5像素
        #   第2帧误差：8像素（基于第1帧的预测，误差累积）
        #   第3帧误差：12像素（基于第2帧的预测，误差进一步累积）
        #   ...
        #   平均误差：可能达到10-15像素
        #
        # 所以理论上：预测帧数越小，平均误差越小
        # ============================================
        for t in range(num):
            # 步骤4.1: 获取真实目标（如果可用，用于更新LSTM状态）
            # 注意：即使提供了true_traj，也只是用于更新LSTM状态，不影响预测结果
            # 预测结果完全基于模型的前一步预测
            if t < true_traj.shape[0] - obs_length:
                target_frame = true_traj[obs_length + t:obs_length + t + 1]
            else:
                target_frame = None
            
            # 步骤4.2: 前向传播，更新LSTM状态
            # 输入：last_frame（上一步预测的位置）和 prev_grid（对应的社交数组）
            if target_frame is not None:
                # 如果提供了真实目标，可以用于更新LSTM状态（但预测仍基于last_frame）
                _, (h_states_tensor, c_states_tensor) = self.forward(last_frame, prev_grid, target_frame)
            else:
                # 没有真实目标，只进行前向传播
                (h_states_tensor, c_states_tensor) = self.forward(last_frame, prev_grid)
            
            # 将张量转换回列表，用于下一次迭代
            h_states = [h_states_tensor[i] for i in range(self.maxNumPeds)]
            c_states = [c_states_tensor[i] for i in range(self.maxNumPeds)]
            
            # 步骤4.3: 为每个行人生成预测
            newpos = torch.zeros(1, self.maxNumPeds, 5, device=device)
            for pedindex in range(self.maxNumPeds):
                # 跳过无效的行人（ID=0）
                if last_frame[0, pedindex, 0] == 0:
                    continue
                
                # 从LSTM隐藏状态生成输出
                output = self.output_w(h_states[pedindex]) + self.output_b
                
                # 提取分布参数：mux, muy（均值），sx, sy（标准差），corr（相关系数）
                mux, muy, sx, sy, corr = self.get_coef(output)
                
                # 采样下一个位置
                # 注意：这里使用的是预测的均值（mux, muy），而不是从分布中采样
                # 这会产生确定性的预测结果（每次运行结果相同）
                # 如果要从分布中采样，可以使用：next_x = mux + sx * torch.randn(...)
                next_x, next_y = self.sample_gaussian_2d(mux, muy, sx, sy, corr)
                
                # 保存预测结果：[pedID, next_x, next_y, 0, 0]
                # 注意：goal_x和goal_y设为0，因为预测时不需要goal信息
                newpos[0, pedindex, :] = torch.tensor([
                    last_frame[0, pedindex, 0],  # 保持行人ID
                    next_x.item(),  # 预测的x坐标
                    next_y.item(),  # 预测的y坐标
                    0,  # goal_x（预测时不需要）
                    0   # goal_y（预测时不需要）
                ], device=device)
            
            # 步骤4.4: 将预测结果添加到完整轨迹中
            ret = torch.cat([ret, newpos], dim=0)
            
            # ⚠️ 关键步骤：使用预测的位置作为下一步的输入（自回归）
            # 这意味着每一步的误差会累积到后续步骤
            last_frame = newpos
            
            # 步骤4.5: 更新社交数组
            # 使用新预测的位置重新计算社交数组，用于下一步预测
            # 因为其他行人的位置可能也发生了变化（如果也在预测）
            prev_grid_np = newpos[0].cpu().numpy()
            prev_grid = torch.FloatTensor(
                getSequenceGridMask(
                    prev_grid_np.reshape(1, self.maxNumPeds, 5)[:, :, :3],  # 只使用前3列：[pedID, x, y]
                    dimensions,
                    self.args.neighborhood_size,
                    self.grid_size
                )
            ).to(device)
        
        # ========== 步骤5: 返回完整轨迹 ==========
        # 返回：[obs_length + num, maxNumPeds, 5]
        # 包含观察部分（obs_length帧）+ 预测部分（num帧）
        return ret.cpu().numpy()

