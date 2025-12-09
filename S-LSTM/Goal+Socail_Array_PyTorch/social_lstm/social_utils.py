'''
Handles processing the input and target data in batches and sequences

Modified by : Simone Zamboni
Date : 2018-01-10

功能说明：
本文件实现了 SocialDataLoader 类，用于处理轨迹预测任务的数据加载和预处理。
主要功能包括：
1. 从CSV文件读取原始轨迹数据
2. 将数据组织成帧格式（每帧包含多个行人的位置信息）
3. 计算每个行人的目标位置（goal，即轨迹的最后一帧位置）
4. 生成训练和验证批次数据
5. 为每个样本添加goal信息和社交数组索引
'''

import os
import pickle
import numpy as np
import random

class SocialDataLoader():
    """
    社交LSTM数据加载器
    
    功能：
    - 从CSV文件加载原始轨迹数据
    - 将数据预处理成帧格式
    - 计算每个行人的目标位置（goal）
    - 生成训练和验证批次
    - 管理数据集的遍历指针
    """

    def __init__(self, batch_size=50, seq_length=5, maxNumPeds=70, datasets=[0, 1, 2, 3, 4], forcePreProcess=False, infer=False):
        """
        初始化数据加载器
        
        参数：
        - batch_size: 批次大小
        - seq_length: 序列长度（观察的帧数）
        - maxNumPeds: 每帧最大行人数
        - datasets: 使用的数据集索引列表
                   索引对应关系：
                   0 -> ucy/zara/zara01
                   1 -> ucy/zara/zara02
                   2 -> eth/univ
                   3 -> eth/hotel
                   4 -> ucy/univ
                   例如：[0, 1, 2, 3, 4] 表示使用全部5个数据集
        - forcePreProcess: 是否强制重新预处理数据
        - infer: 是否为推理模式
        """

        # Get the base directory (parent of social_lstm folder)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # List of data directories where raw data resides (rispetto all'originale e' stato cambiato)
        self.data_dirs = [
            os.path.join(base_dir, 'data', 'ucy', 'zara', 'zara01'),
            os.path.join(base_dir, 'data', 'ucy', 'zara', 'zara02'),
            os.path.join(base_dir, 'data', 'eth', 'univ'),
            os.path.join(base_dir, 'data', 'eth', 'hotel'),
            os.path.join(base_dir, 'data', 'ucy', 'univ')
        ]

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.infer = infer

        self.numDatasets = len(self.data_dirs)

        self.data_dir = os.path.join(base_dir, 'data')

        self.maxNumPeds = maxNumPeds

        self.batch_size = batch_size
        self.seq_length = seq_length

        self.val_fraction = 0.2
        self.takeOneInNFrames = 6
        data_file = os.path.join(self.data_dir, "social-trajectories.cpkl")

        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            self.frame_preprocess(self.used_data_dirs, data_file)

        self.load_preprocessed(data_file)

        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    def frame_preprocess(self, data_dirs, data_file):
        """
        数据预处理函数
        
        功能：
        1. 从CSV文件读取原始轨迹数据（格式：frameId, pedId, y, x）
        2. 将数据组织成帧格式：每帧包含该帧所有行人的[pedID, x, y]
        3. 分离训练集和验证集（验证集占20%）
        4. 计算每个行人的goal（目标位置）：遍历所有帧，保存每个行人最后出现的位置
        5. 将处理后的数据保存为pickle文件
        
        输出数据结构：
        - all_frame_data: 训练帧数据 [num_frames, maxNumPeds, 3]
        - valid_frame_data: 验证帧数据
        - goal: 每个行人的目标位置 [dataset_idx][ped_id] = [goal_x, goal_y]
        """

        all_frame_data = []

        valid_frame_data = []

        frameList_data = []

        numPeds_data = []

        dataset_index = 0

        frames = []  # list where alla the frames are stored in the format of all_frame_data
        all_peds = []  # array with the dimension of (numDirectory,b) with b the sum of each time all the pedestian appera
        dataset_validation_index = []

        # For each dataset
        for directory in data_dirs:

            file_path = os.path.join(directory, 'pixel_pos_interpolate.csv')

            data = np.genfromtxt(file_path, delimiter=',')

            frameList = np.unique(data[0, :]).tolist()

            # Number of frames
            numFrames = int(len(frameList)/self.takeOneInNFrames)*self.takeOneInNFrames

            # 数据划分逻辑：
            # - 如果 infer=True（推理模式）：不划分验证集，所有数据都用于测试
            # - 如果 infer=False（训练模式）：划分20%作为验证集，80%作为训练集
            if self.infer:
                valid_numFrames = 0  # 推理模式：不划分验证集
            else:
                # 训练模式：划分20%作为验证集
                valid_numFrames = int((numFrames * self.val_fraction)/self.takeOneInNFrames)*self.takeOneInNFrames

            dataset_validation_index.append(valid_numFrames)

            frameList_data.append(frameList)

            numPeds_data.append([])
            all_peds.append([])

            all_frame_data.append(np.zeros( (int((numFrames - valid_numFrames)/self.takeOneInNFrames), self.maxNumPeds, 3) ) )

            valid_frame_data.append(np.zeros(  (int(valid_numFrames/self.takeOneInNFrames), self.maxNumPeds, 3) ) )

            frames.append(np.zeros((numFrames, self.maxNumPeds, 3)))

            ind = 0
            while ind < numFrames:
                frame = frameList[ind]
                pedsInFrame = data[:, data[0, :] == frame]

                pedsList = pedsInFrame[1, :].tolist()

                numPeds_data[dataset_index].append(len(pedsList))

                pedsWithPos = []

                for ped in pedsList:
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    pedsWithPos.append([ped, current_x, current_y])
                    all_peds[dataset_index].append((ped))

                # 数据分配：
                # - 如果 infer=True：所有帧都放入 all_frame_data（用于测试）
                # - 如果 infer=False：
                #   * 前 valid_numFrames 帧 → valid_frame_data（验证集）
                #   * 后面的帧 → all_frame_data（训练集）
                if (ind >= valid_numFrames) or (self.infer):
                    # 训练集或测试集（推理模式下所有数据都是测试集）
                    all_frame_data[dataset_index][int((ind - valid_numFrames)/self.takeOneInNFrames), 0:len(pedsList), :] = np.array(pedsWithPos)
                else:
                    # 验证集（只在训练模式下使用）
                    valid_frame_data[dataset_index][int(ind/self.takeOneInNFrames), 0:len(pedsList), :] = np.array(pedsWithPos)

                frames[dataset_index][ind, 0:len(pedsList), :] = np.array(pedsWithPos)
                ind += self.takeOneInNFrames

            dataset_index += 1

        # ========== 计算每个行人的goal（目标位置） ==========
        # goal是每个行人在其完整轨迹中的最后一帧位置，用于提供方向信息

        unique_all_peds = []  # 存储每个数据集中所有唯一的行人ID

        # 获取每个数据集中的唯一行人ID
        dir = 0
        while dir < len(data_dirs):
            unique_all_peds.append(np.unique(all_peds[dir]))
            dir += 1

        goal = []  # goal数组：存储每个行人的目标位置 [dataset_idx][ped_id] = [goal_x, goal_y]

        # 初始化goal数组为全0
        dir = 0
        while dir < len(data_dirs):
            goal.append([])
            ped = 0
            #sembra che il valore len(unique_all_peds[dir]) non ritorni il numero di pedoni esatto in un video
            # e se non ci aggiungessimo una valore abbastanza alto darebbe errore.
            # Si e' quindi deciso di aggiungere una valore molto alto arbiratio per evitare errori, questa e' un
            # punto del codice che potrebbe assolutamente essere migliorato.
            while ped <= len(unique_all_peds[dir]) + 1000:
                goal[dir].append([0, 0])
                ped += 1
            dir += 1

        # 遍历所有帧，更新每个行人的最后已知位置
        # 由于是顺序遍历，最后保存的就是该行人在最后一帧的位置
        dir = 0
        while dir < len(frames):
            frame = 0
            while frame < len(frames[dir]):
                ped_n = 0
                # 遍历当前帧中的所有行人
                while ped_n < len(frames[dir][frame]):
                    ped_id = int(frames[dir][frame][ped_n][0])  # 获取当前行人的ID
                    goal[dir][ped_id][0] = frames[dir][frame][ped_n][1]  # 更新goal的x坐标
                    goal[dir][ped_id][1] = frames[dir][frame][ped_n][2]  # 更新goal的y坐标
                    ped_n += 1
                frame += 1
            dir += 1

        # 此时goal数组中存储的是每个行人在其轨迹最后一帧的位置

        # spiegazione di cosa viene salvato:
        # frameList_data[i] = tutti i numeri di frame dell'i-esimo dataset (se il dataset ha 700 frame ci sara un array di 700 elementi che vanno da 1 a 700
        # numpeds_data[i][j] = quandi pedoni ci sono nell'j-esimo frame dell'i-esimo dataset
        # all_frame_data[i][j]: della i-esima directory all j-esimo frame la lista di tutti i pedoni nell'ordine: [id,x,y], la lunghezza della lista e' maxNumPeds, e contiene i frame dopo l'ultimo valid_frame_data
        # valid frame data: uguale a all_frame_data come struttura solo che ha solo i frame di validzione
        # goal[i][j] = le coordinate x e y dell'obbiettivo pedone con id j del video i

        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_frame_data, goal), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        """
        加载预处理后的数据
        
        从pickle文件中加载：
        - data: 训练帧数据
        - frameList: 帧列表
        - numPedsList: 每帧的行人数量
        - valid_data: 验证帧数据
        - goals: 每个行人的目标位置
        """

        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        self.goals = self.raw_data[4] #prendo anche il goal dal file salvato
        counter = 0
        valid_counter = 0

        for dataset in range(len(self.data)):
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            print('Training data from dataset', dataset, ':', len(all_frame_data))
            print('Validation data from dataset', dataset, ':', len(valid_frame_data))
            # 计算可能的序列起点数量
            # 
            # 理论上的序列起点数量：
            # - 如果有1000帧，seq_length=20
            # - 理论上可以有 1000-20 = 980 个序列起点（滑动窗口）
            # - 因为最后一个序列起点是979，需要帧979-999（共21帧：20个输入+1个目标）
            #
            # 为什么除以 (seq_length+2) 而不是 (seq_length+1)？
            # 1. 保守估计：留一些余量，避免边界问题
            # 2. 实际使用：由于使用随机步长（randomUpdate=True），不会遍历所有可能的起点
            # 3. 简化计算：这样计算出的批次数更保守，确保不会超出数据范围
            #
            # 例如：1000帧，seq_length=20
            # - 理论序列数：1000-20 = 980
            # - 保守估计：1000/(20+2) = 45
            # - 实际使用：由于随机步长，可能只使用其中的一部分
            counter += int(len(all_frame_data) / (self.seq_length+2))
            valid_counter += int(len(valid_frame_data) / (self.seq_length+2))

        # 计算批次数量
        # 步骤1: 计算基础批次数
        # counter: 所有数据集中可能的序列起点总数
        # batch_size: 每个批次包含的序列数量
        base_num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)
        
        # 步骤2: 乘以2增加数据多样性
        # 原因：由于使用随机步长（randomUpdate=True），指针会随机移动
        # 这样可以覆盖更多不同的序列组合，提高模型的泛化能力
        # 例如：如果基础批次数是11，乘以2后变成22
        self.num_batches = base_num_batches * 2
        
        # 示例计算：
        # 假设有4个数据集，每个数据集有1000帧，seq_length=20，batch_size=16
        # counter = 4 * (1000 / (20+2)) = 4 * 45 = 180 个可能的序列起点
        # base_num_batches = 180 / 16 = 11
        # num_batches = 11 * 2 = 22 个批次/epoch

    def next_batch(self, randomUpdate=True):
        """
        获取下一个训练批次
        
        功能：
        1. 从当前数据集指针位置提取一个序列（seq_length帧）
        2. 为每个行人添加goal信息（goal_x, goal_y）
        3. 构建输入数据x_batch和目标数据y_batch
        
        训练流程说明：
        - 假设数据集有1000帧，seq_length=20
        - 理论上可以有约980个不同的序列（滑动窗口：0-19, 1-20, 2-21, ...）
        - 但实际训练时：
          * 如果 randomUpdate=True（默认）：指针随机移动1到seq_length步
            例如：从帧0提取后，可能跳到帧5、帧15等，不会遍历所有序列
            这样做的目的是增加数据随机性，提高模型泛化能力
          * 如果 randomUpdate=False：指针固定移动seq_length步
            例如：从帧0提取后，跳到帧20，再跳到帧40，会跳过中间序列
        
        返回：
        - x_batch: 输入数据列表，每个元素形状为 [seq_length, maxNumPeds, 5]
                  5个维度：[pedID, x, y, goal_x, goal_y]
        - y_batch: 目标数据列表，形状同x_batch，包含下一帧的位置
        - d: 数据集索引列表
        """
        x_batch = []
        y_batch = []
        d = []
        i = 0

        while i < self.batch_size:
            frame_data = self.data[self.dataset_pointer]

            idx = self.frame_pointer

            # 检查是否有足够的帧来提取一个完整的序列
            # 需要：seq_length个输入帧 + 1个目标帧 = seq_length+1 帧
            # 条件：idx + seq_length < frame_data.shape[0]
            # 例如：idx=979, seq_length=20, 需要帧979-999（共21帧）
            #      如果frame_data有1000帧（索引0-999），则979+20=999 < 1000，可以提取
            if idx + self.seq_length < frame_data.shape[0]:
                # 提取 seq_length+1 帧：用于构建输入和目标
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                # 输入序列：帧 idx 到 idx+seq_length-1（共seq_length帧）
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                # 目标序列：帧 idx+1 到 idx+seq_length（共seq_length帧，是输入的下一帧）
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]

                #list of the ID of all the pedestrian in the current batch
                pedID_list = np.unique(seq_frame_data[:, :, 0])

                # Number of unique peds the current batch
                numUniquePeds = pedID_list.shape[0]

                # 初始化输入和目标数据数组，从3维扩展到5维（添加goal_x和goal_y）
                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 5))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 5))

                # 遍历序列中的每一帧
                for seq in range(self.seq_length):
                    # 当前帧和下一帧的数据
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    # 遍历当前批次中的所有唯一行人
                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped]  # 获取行人ID

                        # 跳过无效的行人ID（0表示空位置）
                        if pedID == 0:
                            continue
                        else:
                            tped = []  # 目标数据（下一帧）
                            sped = []  # 输入数据（当前帧）

                            # 获取当前行人在当前帧的位置
                            temp_sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]

                            # 如果该行人在当前帧存在
                            if len(temp_sped) > 0:
                                # 添加行人的基本信息：[pedID, x, y]
                                iter = 0
                                while iter < len(temp_sped[0]):
                                    sped.append(temp_sped[0][iter])
                                    iter += 1

                                # 添加该行人的goal坐标：[goal_x, goal_y]
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            # 获取当前行人在下一帧的位置（作为目标）
                            temp_tped = tseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            # 如果该行人在下一帧也存在
                            if len(temp_tped) > 0:
                                iter = 0
                                # 添加下一帧的位置信息：[pedID, x, y]
                                while iter < len(temp_tped[0]):
                                    tped.append(temp_tped[0][iter])
                                    iter += 1
                                # 添加goal坐标
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            # 将数据写入sourceData和targetData
                            if len(sped) > 2:
                                sourceData[seq, ped, :] = sped  # [pedID, x, y, goal_x, goal_y]
                            if len(tped) > 2:
                                targetData[seq, ped, :] = tped  # [pedID, x, y, goal_x, goal_y]

                x_batch.append(sourceData)
                y_batch.append(targetData)

                # 移动帧指针，准备提取下一个序列
                if randomUpdate:
                    # 随机移动1到seq_length步，增加数据的随机性
                    # 这样不会遍历所有可能的序列，但可以提高模型的泛化能力
                    # 例如：seq_length=20，可能移动1-20步中的任意值
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    # 固定移动seq_length步，会跳过一些序列
                    # 例如：seq_length=20，从帧0提取，下次从帧20提取，跳过了帧1-19的序列
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1
            else:
                # 当前数据集已遍历完，切换到下一个数据集
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, d

    def next_valid_batch(self, randomUpdate=True):
        """
        获取下一个验证批次
        
        功能与next_batch相同，但使用验证数据集（valid_data）
        用于模型验证阶段
        """
        x_batch = []
        y_batch = []
        d = []
        i = 0
        while i < self.batch_size:
            frame_data = self.valid_data[self.valid_dataset_pointer]
            idx = self.valid_frame_pointer

            if idx + self.seq_length < frame_data.shape[0]:
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]

                # list of the ID of all the pedestrian in the current batch
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                # Number of unique peds the current batch
                numUniquePeds = pedID_list.shape[0]

                # sia sourceData che targetData sono stati ampliati da 3 a 5
                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 5))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 5))

                # per ogni frame della sequenza
                for seq in range(self.seq_length):
                    # frame attuale (ssqe_frame_data) e successivo (tseq_frame_data)
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    # per tutti i pedoni nel frame
                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped] #prendere il pedID

                        # se il pedone non esiste andare avanti al prossimo ciclo
                        if pedID == 0:
                            continue
                        else:
                            tped = [] #target data per questo pedone
                            sped = [] #sequence data per questo pedone

                            # array che contiene la posizione del pedone nel frame
                            temp_sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]

                            # se quel pedone e' presente nel frame, cioe' se ha una posizione(ed e' quindi salvata in temp_sped) allora si va avanti
                            if(len(temp_sped) > 0):
                                # aggiungere ai dati di input del pedone la posizione del pedone
                                iter = 0
                                while iter < len(temp_sped[0]):
                                    sped.append(temp_sped[0][iter])
                                    iter += 1

                                # e aggiungere i dati di input del pedone le coordinate del goal del pedone
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            # array che contiene i dati della posizione futura del pedone
                            temp_tped = (tseq_frame_data[tseq_frame_data[:, 0] == pedID, :])

                            # se quel pedone ha dati target, cioe' se ha una posizione anche nel frame successivo (questa quindi sara' salvata in temp_sped)
                            if(len(temp_tped) > 0) :
                                # aggiungere ai dati target di quel pedone la sua posizione futura
                                iter = 0
                                while iter < len(temp_tped[0]):
                                    tped.append(temp_tped[0][iter])
                                    iter += 1

                                # e aggiungere i dati target di quel pedone anche le coordinate del goal
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            # se sono state inserite delle informazioni in sped e tped allora vengono aggiunti a sourceData e targetData
                            if len(sped) > 2:
                                sourceData[seq, ped, : ] = sped
                            if len(tped) > 2:
                                targetData[seq, ped, :] = tped

                x_batch.append(sourceData)
                y_batch.append(targetData)

                # Advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1
            else:
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        """
        推进数据集指针
        
        当当前数据集的所有帧都处理完后，切换到下一个数据集
        如果所有数据集都处理完，则重置到第一个数据集
        """
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    def reset_batch_pointer(self, valid=False):
        """
        重置所有指针
        
        将数据集指针和帧指针重置到初始位置
        用于开始新的训练/验证周期
        """
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0


if __name__ == '__main__':
    """
    测试脚本：运行此文件时打印数据加载器的效果
    """
    print("=" * 80)
    print("SocialDataLoader 测试")
    print("=" * 80)
    
    # 创建数据加载器实例（使用较小的参数以便快速测试）
    print("\n[1] 初始化数据加载器...")
    data_loader = SocialDataLoader(
        batch_size=4,
        seq_length=8,
        maxNumPeds=70,
        datasets=[0, 1, 2, 3, 4],
        forcePreProcess=False,  # 如果数据已预处理，设为False
        infer=False
    )
    
    print("✓ 数据加载器初始化完成")
    
    # 打印数据集基本信息
    print("\n[2] 数据集基本信息:")
    print(f"  - 训练批次数量: {data_loader.num_batches}")
    print(f"  - 验证批次数量: {data_loader.valid_num_batches}")
    print(f"  - 批次大小: {data_loader.batch_size}")
    print(f"  - 序列长度: {data_loader.seq_length}")
    print(f"  - 最大行人数: {data_loader.maxNumPeds}")
    
    # 打印每个数据集的信息
    print("\n[3] 各数据集详细信息:")
    for i, dataset in enumerate(data_loader.data):
        print(f"  数据集 {i}:")
        print(f"    - 训练帧数: {len(dataset)}")
        if i < len(data_loader.valid_data):
            print(f"    - 验证帧数: {len(data_loader.valid_data[i])}")
        if i < len(data_loader.goals):
            # 统计有goal的行人数量
            non_zero_goals = sum(1 for goal in data_loader.goals[i] if goal[0] != 0 or goal[1] != 0)
            print(f"    - 有goal的行人数: {non_zero_goals}")
    
    # 获取一个训练批次
    print("\n[4] 获取训练批次数据...")
    data_loader.reset_batch_pointer(valid=False)
    x_batch, y_batch, d = data_loader.next_batch(randomUpdate=False)
    
    print(f"✓ 成功获取批次，包含 {len(x_batch)} 个样本")
    print(f"  - 输入数据形状: {x_batch[0].shape} (seq_length, maxNumPeds, 5)")
    print(f"  - 目标数据形状: {y_batch[0].shape} (seq_length, maxNumPeds, 5)")
    print(f"  - 数据集索引: {d}")
    
    # 打印第一个样本的详细信息
    print("\n[5] 第一个样本的详细信息:")
    sample_x = x_batch[0]
    sample_y = y_batch[0]
    
    # 统计有效行人（非零ID）
    valid_peds_in_first_frame = np.sum(sample_x[0, :, 0] > 0)
    valid_peds_in_last_frame = np.sum(sample_x[-1, :, 0] > 0)
    
    print(f"  - 第一帧有效行人数: {valid_peds_in_first_frame}")
    print(f"  - 最后一帧有效行人数: {valid_peds_in_last_frame}")
    
    # 显示第一个有效行人的数据示例
    first_valid_ped_idx = None
    for i in range(data_loader.maxNumPeds):
        if sample_x[0, i, 0] > 0:
            first_valid_ped_idx = i
            break
    
    if first_valid_ped_idx is not None:
        print(f"\n  - 第一个有效行人 (索引 {first_valid_ped_idx}) 的数据:")
        print(f"    第一帧输入: [pedID={sample_x[0, first_valid_ped_idx, 0]:.0f}, "
              f"x={sample_x[0, first_valid_ped_idx, 1]:.2f}, "
              f"y={sample_x[0, first_valid_ped_idx, 2]:.2f}, "
              f"goal_x={sample_x[0, first_valid_ped_idx, 3]:.2f}, "
              f"goal_y={sample_x[0, first_valid_ped_idx, 4]:.2f}]")
        print(f"    第一帧目标: [pedID={sample_y[0, first_valid_ped_idx, 0]:.0f}, "
              f"x={sample_y[0, first_valid_ped_idx, 1]:.2f}, "
              f"y={sample_y[0, first_valid_ped_idx, 2]:.2f}, "
              f"goal_x={sample_y[0, first_valid_ped_idx, 3]:.2f}, "
              f"goal_y={sample_y[0, first_valid_ped_idx, 4]:.2f}]")
        
        # 显示该行人在序列中的轨迹
        print(f"\n  - 该行人在序列中的轨迹 (前3帧):")
        for seq in range(min(3, data_loader.seq_length)):
            if sample_x[seq, first_valid_ped_idx, 0] > 0:
                print(f"    帧 {seq}: x={sample_x[seq, first_valid_ped_idx, 1]:.2f}, "
                      f"y={sample_x[seq, first_valid_ped_idx, 2]:.2f}")
    
    # 打印goal信息示例
    print("\n[6] Goal信息示例:")
    for dataset_idx in range(min(3, len(data_loader.goals))):
        non_zero_goals = [(i, goal) for i, goal in enumerate(data_loader.goals[dataset_idx]) 
                         if goal[0] != 0 or goal[1] != 0]
        if non_zero_goals:
            ped_id, goal = non_zero_goals[0]
            print(f"  数据集 {dataset_idx}, 行人ID {ped_id}: goal=({goal[0]:.2f}, {goal[1]:.2f})")
    
    # 数据统计
    print("\n[7] 数据统计:")
    all_x_coords = []
    all_y_coords = []
    for batch_idx in range(len(x_batch)):
        for seq in range(data_loader.seq_length):
            for ped in range(data_loader.maxNumPeds):
                if x_batch[batch_idx][seq, ped, 0] > 0:
                    all_x_coords.append(x_batch[batch_idx][seq, ped, 1])
                    all_y_coords.append(x_batch[batch_idx][seq, ped, 2])
    
    if all_x_coords:
        print(f"  - 坐标范围:")
        print(f"    x: [{min(all_x_coords):.2f}, {max(all_x_coords):.2f}]")
        print(f"    y: [{min(all_y_coords):.2f}, {max(all_y_coords):.2f}]")
        print(f"  - 平均坐标: x={np.mean(all_x_coords):.2f}, y={np.mean(all_y_coords):.2f}")
    
    print("\n" + "=" * 80)
    print("测试完成！数据加载器工作正常。")
    print("=" * 80)
