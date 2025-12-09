'''
PyTorch采样/测试脚本 - Social LSTM with Goal and Social Array

功能说明：
==========
本脚本用于测试训练好的模型，评估模型在测试集上的预测性能。

主要功能：
1. 加载训练好的模型权重
2. 在测试数据集上进行轨迹预测
3. 计算预测轨迹与真实轨迹的平均误差
4. 保存预测结果供后续分析和可视化

使用场景：
- 模型评估：测试模型在测试集上的表现
- 性能分析：计算平均预测误差
- 结果保存：保存预测结果用于可视化或进一步分析
'''

import numpy as np
import torch
import os
import pickle
import argparse

from social_utils import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask


def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    """
    计算预测轨迹与真实轨迹之间的平均欧氏距离误差
    
    功能：
    - 对于预测阶段（observed_length之后）的每一帧
    - 计算该帧所有行人的预测位置与真实位置的欧氏距离
    - 返回所有帧的平均误差
    
    参数：
    - predicted_traj: 预测的完整轨迹 [总帧数, maxNumPeds, 5]
    - true_traj: 真实的完整轨迹 [总帧数, maxNumPeds, 5]
    - observed_length: 观察长度（前obs_length帧是输入，之后是预测）
    - maxNumPeds: 最大行人数
    
    返回：
    - 平均误差（像素单位）
    """
    error = np.zeros(len(true_traj) - observed_length)
    for i in range(observed_length, len(true_traj)):
        pred_pos = predicted_traj[i, :]
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                continue
            elif pred_pos[j, 0] == 0:
                continue
            else:
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        if counter != 0:
            error[i - observed_length] = timestep_error / counter

    return np.mean(error)


def main():
    """
    主函数：模型测试流程
    
    流程：
    1. 解析命令行参数
    2. 加载训练好的模型
    3. 加载测试数据集
    4. 对每个测试轨迹进行预测
    5. 计算平均误差
    6. 保存结果
    """
    # 设置随机种子，确保结果可复现
    np.random.seed(1)
    torch.manual_seed(1)

    parser = argparse.ArgumentParser(description='测试训练好的Social LSTM模型')
    parser.add_argument('--obs_length', type=int, default=18,
                        help='观察长度：用于预测的历史帧数（输入）')
    parser.add_argument('--pred_length', type=int, default=2,
                        help='预测长度：要预测的未来帧数（输出）\n'
                             '注意：预测帧数越小，误差通常越小\n'
                             '原因：预测是自回归的，每一步的误差会累积到后续步骤')
    parser.add_argument('--test_dataset', type=int, default=0,
                        help='要测试的数据集索引（0-4）')
    parser.add_argument('--epoch', type=int, default=1,
                        help='要加载的模型epoch编号')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备（cuda或cpu）')
    sample_args = parser.parse_args()

    # ========== 步骤1: 加载模型配置和权重 ==========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(script_dir, 'save', str(sample_args.test_dataset))

    # 加载训练时保存的配置参数
    with open(os.path.join(save_directory, 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # 创建模型（推理模式）
    device = torch.device(sample_args.device)
    model = SocialModel(saved_args, infer=True).to(device)  # infer=True表示推理模式
    
    # 加载训练好的模型权重
    checkpoint_path = os.path.join(save_directory, 'social_model_epoch_{}.pth'.format(sample_args.epoch))
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('✓ 成功加载模型，epoch: {}'.format(sample_args.epoch))
    else:
        print('✗ 错误：找不到模型文件 {}'.format(checkpoint_path))
        return

    # ========== 步骤2: 加载测试数据集 ==========
    # 重要说明：测试数据的划分方式
    # ============================================
    # 测试数据是在数据集级别划分的，不是在单个数据集内部划分的
    # 
    # 训练时的划分：
    # - 有5个数据集：索引0,1,2,3,4
    # - 使用 leaveDataset 参数留出一个数据集作为测试集
    #   例如：leaveDataset=0 → 使用数据集1,2,3,4训练，数据集0作为测试集
    # - 在每个训练数据集内部，还会划分训练集和验证集（验证集占20%）
    #
    # 测试时的使用：
    # - test_dataset 参数指定要测试的数据集（就是训练时被留出的那个）
    # - infer=True 时，不会在数据集内部划分验证集，所有数据都用于测试
    # - 例如：test_dataset=0 表示测试数据集0（训练时被留出的数据集）
    #
    # 所以测试数据是完全独立的，没有参与训练过程
    # ============================================
    dataset = [sample_args.test_dataset]  # 只加载测试数据集
    # batch_size=1: 每次处理一个轨迹
    # seq_length=obs_length+pred_length: 需要观察帧+预测帧的总长度
    # infer=True: 推理模式，不分离训练/验证集
    data_loader = SocialDataLoader(
        1,  # batch_size=1，每次处理一个轨迹
        sample_args.pred_length + sample_args.obs_length,  # 总序列长度
        saved_args.maxNumPeds, 
        dataset, 
        forcePreProcess=False,  # 不强制重新预处理
        infer=True  # 推理模式
    )
    data_loader.reset_batch_pointer()

    # ========== 步骤3: 对测试集进行预测和评估 ==========
    results = []  # 存储所有预测结果
    total_error = 0  # 累计误差

    model.eval()  # 设置为评估模式（关闭dropout等）
    with torch.no_grad():  # 不计算梯度，节省内存和计算
        # 遍历测试集中的所有轨迹
        for b in range(data_loader.num_batches):
            # 获取一个测试轨迹
            # randomUpdate=False: 不使用随机步长，按顺序提取
            x, y, d = data_loader.next_batch(randomUpdate=False)
            x_batch, y_batch, d_batch = x[0], y[0], d[0]

            # 根据数据集确定图像尺寸
            if d_batch == 0 and dataset[0] == 0:
                dimensions = [640, 480]  # 数据集0的尺寸
            else:
                dimensions = [720, 576]  # 其他数据集的尺寸

            # 计算社交数组
            grid_batch = getSequenceGridMask(x_batch, dimensions, saved_args.neighborhood_size, saved_args.grid_size)

            # 提取观察部分（用于预测的输入）
            obs_traj = x_batch[:sample_args.obs_length]  # 前obs_length帧作为输入
            obs_grid = grid_batch[:sample_args.obs_length]  # 对应的社交数组

            print("********************** 预测新轨迹", b, "******************************")
            # 使用模型预测未来轨迹
            # obs_traj: 观察到的轨迹（输入）
            # obs_grid: 观察到的社交数组（输入）
            # dimensions: 图像尺寸
            # x_batch: 完整轨迹（用于计算误差）
            # sample_args.pred_length: 要预测的帧数
            complete_traj = model.sample(obs_traj, obs_grid, dimensions, x_batch, sample_args.pred_length)

            # 计算该轨迹的平均误差
            error = get_mean_error(complete_traj, x[0], sample_args.obs_length, saved_args.maxNumPeds)
            total_error += error

            print("已处理轨迹: {}/{}，当前误差: {:.4f}".format(b+1, data_loader.num_batches, error))
            # 保存结果：(真实轨迹, 预测轨迹, 观察长度)
            results.append((x[0], complete_traj, sample_args.obs_length))

    # ========== 步骤4: 输出和保存结果 ==========
    mean_error = total_error / data_loader.num_batches
    print("=" * 60)
    print("模型平均预测误差: {:.4f} 像素".format(mean_error))
    print("=" * 60)

    # 保存预测结果到pickle文件，供后续分析和可视化使用
    print("保存预测结果...")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    results_path = os.path.join(save_directory, 'social_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print("结果已保存到: {}".format(results_path))


if __name__ == '__main__':
    main()

