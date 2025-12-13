'''
PyTorch training script for Social LSTM with Goal and Social Array
'''

import torch
import torch.optim as optim
import argparse
import os
import time
import pickle
import numpy as np

from social_model import SocialModel
from social_utils import SocialDataLoader
from grid import getSequenceGridMask


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    parser.add_argument('--maxNumPeds', type=int, default=70,
                        help='Maximum Number of Pedestrians')
    parser.add_argument('--leaveDataset', type=int, default=0,
                        help='The dataset index to be left out in training')
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    # 设备选择：按照 cuda -> cpu -> mps 的顺序查找硬件支持
    if torch.cuda.is_available():
        default_device = 'cuda'
    else:
        # CPU总是可用，优先使用CPU（MPS需要手动通过 --device mps 指定）
        default_device = 'cpu'
    parser.add_argument('--device', type=str, default=default_device,
                        help='Device to use (cuda, cpu, or mps). Default priority: cuda > cpu > mps')
    args = parser.parse_args()
    train(args)


def train(args):
    datasets = list(range(5))
    datasets.remove(args.leaveDataset)

    # Create the SocialDataLoader object
    data_loader = SocialDataLoader(args.batch_size, args.seq_length, args.maxNumPeds, 
                                   datasets, forcePreProcess=True, infer=False)

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Log directory
    log_directory = os.path.join(script_dir, 'log', str(args.leaveDataset))
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = os.path.join(script_dir, 'save', str(args.leaveDataset))
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(os.path.join(save_directory, 'social_config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Create model
    device = torch.device(args.device)
    model = SocialModel(args).to(device)
    
    # Optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=args.decay_rate)

    print('Training begin')
    best_val_loss = 100
    best_epoch = 0

    # ========== 训练循环 ==========
    # 训练一轮（一个epoch）的流程：
    # 1. 重置数据指针到起始位置
    # 2. 遍历所有批次（num_batches）
    # 3. 每个批次包含 batch_size 个样本
    # 4. 每个样本从数据集中提取 seq_length 帧作为输入
    # 
    # 注意：由于使用随机步长（randomUpdate=True），训练一轮并不会遍历所有可能的序列
    # 例如：如果有1000帧，seq_length=20，理论上可以有约980个不同的序列（滑动窗口）
    # 但实际训练时，指针会随机移动1到seq_length步，所以会跳过一些序列
    # 这样做的目的是增加数据的随机性，提高模型的泛化能力
    for e in range(args.num_epochs):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate * (args.decay_rate ** e)
        
        # 重置训练数据指针，开始新的一轮训练
        data_loader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # 遍历所有训练批次
        # num_batches 是在 load_preprocessed 中计算的：
        # num_batches = (总帧数 / (seq_length+2)) / batch_size * 2
        for b in range(data_loader.num_batches):
            start = time.time()
            # 获取一个批次的数据
            # next_batch() 默认使用 randomUpdate=True，指针会随机移动1到seq_length步
            # 这意味着不是所有可能的序列都会被使用，但增加了数据的随机性
            x, y, d = data_loader.next_batch()
            loss_batch = 0

            for batch in range(data_loader.batch_size):
                x_batch = torch.FloatTensor(x[batch]).to(device)
                y_batch = torch.FloatTensor(y[batch]).to(device)
                d_batch = d[batch]

                if d_batch == 0 and datasets[0] == 0:
                    dataset_data = [640, 480]
                else:
                    dataset_data = [720, 576]

                grid_batch = getSequenceGridMask(x_batch.cpu().numpy(), dataset_data, 
                                                 args.neighborhood_size, args.grid_size)
                grid_batch = torch.FloatTensor(grid_batch).to(device)

                # Forward pass
                optimizer.zero_grad()
                loss, _ = model(x_batch, grid_batch, y_batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()

                loss_batch += loss.item()

            end = time.time()
            loss_batch = loss_batch / data_loader.batch_size
            loss_epoch += loss_batch
            
            # 打印训练进度
            # 批次编号说明：
            # - e * data_loader.num_batches + b: 当前批次的全局编号（从0开始）
            #   例如：epoch 29, batch 21 → 29 * 22 + 21 = 659
            # - args.num_epochs * data_loader.num_batches: 总批次数
            #   例如：30 epochs * 22 batches/epoch = 660 个批次
            # - e: 当前epoch编号（从0开始，所以29表示第30轮）
            # - loss_batch: 当前批次的平均损失
            print(
                "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                .format(
                    e * data_loader.num_batches + b,  # 全局批次编号
                    args.num_epochs * data_loader.num_batches,  # 总批次数
                    e,  # 当前epoch
                    loss_batch, end - start))

        loss_epoch /= data_loader.num_batches
        log_file_curve.write(str(e)+','+str(loss_epoch)+',')
        print('*****************')

        # Validation
        data_loader.reset_batch_pointer(valid=True)
        loss_epoch = 0

        model.eval()
        with torch.no_grad():
            for b in range(data_loader.valid_num_batches):
                x, y, d = data_loader.next_valid_batch()
                loss_batch = 0

                for batch in range(data_loader.batch_size):
                    x_batch = torch.FloatTensor(x[batch]).to(device)
                    y_batch = torch.FloatTensor(y[batch]).to(device)
                    d_batch = d[batch]

                    if d_batch == 0 and datasets[0] == 0:
                        dataset_data = [640, 480]
                    else:
                        dataset_data = [720, 576]

                    grid_batch = getSequenceGridMask(x_batch.cpu().numpy(), dataset_data,
                                                     args.neighborhood_size, args.grid_size)
                    grid_batch = torch.FloatTensor(grid_batch).to(device)

                    loss, _ = model(x_batch, grid_batch, y_batch)
                    loss_batch += loss.item()

                loss_batch = loss_batch / data_loader.batch_size
                loss_epoch += loss_batch

        model.train()
        loss_epoch /= data_loader.valid_num_batches

        # Update best validation loss
        if loss_epoch < best_val_loss:
            best_val_loss = loss_epoch
            best_epoch = e

        print('(epoch {}), valid_loss = {:.3f}'.format(e, loss_epoch))
        print('Best epoch', best_epoch, 'Best validation loss', best_val_loss)
        log_file_curve.write(str(loss_epoch)+'\n')
        print('*****************')

        # Save model after each epoch
        print('Saving model')
        checkpoint_path = os.path.join(save_directory, 'social_model_epoch_{}.pth'.format(e))
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_epoch,
        }, checkpoint_path)
        print("model saved to {}".format(checkpoint_path))

    print('Best epoch', best_epoch, 'Best validation loss', best_val_loss)
    log_file.write(str(best_epoch)+','+str(best_val_loss))

    log_file.close()
    log_file_curve.close()


if __name__ == '__main__':
    main()

