from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSE,maskedNLLTest
from torch.utils.data import DataLoader
import time
import math
import os
import logging
from datetime import datetime
import sys

def main():
    ## 创建保存模型的目录（在日志配置之前）
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    ## 配置日志记录（必须在所有 logger 使用之前）
    log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("开始训练 CS-LSTM 模型")
    logger.info("="*80)
    logger.info(f"模型保存目录: trained_models/")
    logger.info(f"训练日志文件: {log_filename}")

    ## Network Arguments
    args = {}
    # 自动检测设备：优先使用 MPS (Apple Silicon GPU)，然后是 CUDA，最后是 CPU
    if torch.backends.mps.is_available():
        args['use_cuda'] = False
        args['use_mps'] = True
        device = torch.device("mps")
        logger.info("使用 MPS (Apple Silicon GPU) 训练")
    elif torch.cuda.is_available():
        args['use_cuda'] = True
        args['use_mps'] = False
        device = torch.device("cuda")
        logger.info("使用GPU训练")
        logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        args['use_cuda'] = False
        args['use_mps'] = False
        device = torch.device("cpu")
        logger.info("CUDA和MPS不可用，使用CPU训练（速度较慢）")
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    args['out_length'] = 25
    args['grid_size'] = (13,3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = True
    args['train_flag'] = True



    # Initialize network
    net = highwayNet(args)
    net = net.to(device)


    ## Initialize optimizer
    pretrainEpochs = 5
    trainEpochs = 3
    optimizer = torch.optim.Adam(net.parameters())
    batch_size = 128
    crossEnt = torch.nn.BCELoss()


    ## Initialize data loaders
    # 在 macOS 上，num_workers=0 更稳定，避免 multiprocessing 问题
    # 如果需要多进程，可以设置为 2-4，但必须配合 if __name__ == '__main__' 使用
    num_workers = 0 if sys.platform == 'darwin' else 4
    trSet = ngsimDataset('data/TrainSet_traj.csv')
    valSet = ngsimDataset('data/ValSet_traj.csv')
    trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=num_workers,collate_fn=trSet.collate_fn)
    valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=num_workers,collate_fn=valSet.collate_fn)


    ## Variables holding train and validation loss values:
    train_loss = []
    val_loss = []
    prev_val_loss = math.inf

    ## 跟踪最佳验证损失
    best_val_loss = math.inf

    logger.info(f"训练配置: pretrainEpochs={pretrainEpochs}, trainEpochs={trainEpochs}, batch_size={batch_size}")
    logger.info(f"网络参数: encoder_size={args['encoder_size']}, decoder_size={args['decoder_size']}")
    logger.info(f"总训练轮数: {pretrainEpochs+trainEpochs}")
    logger.info(f"数据加载器 num_workers: {num_workers}")
    logger.info("-"*80)

    for epoch_num in range(pretrainEpochs+trainEpochs):
        if epoch_num == 0:
            logger.info('Pre-training with MSE loss')
        elif epoch_num == pretrainEpochs:
            logger.info('Training with NLL loss')


        ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
        net.train_flag = True

        # Variables to track training performance:
        avg_tr_loss = 0
        avg_tr_time = 0
        avg_lat_acc = 0
        avg_lon_acc = 0


        for i, data in enumerate(trDataloader):

            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            # 将数据移动到指定设备
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut = fut.to(device)
            op_mask = op_mask.to(device)

            # Forward pass
            if args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                # Pre-train with MSE loss to speed up training
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    # Train with NLL loss
                    l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                    avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                    avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)

            # Backprop and update weights
            optimizer.zero_grad()
            l.backward()
            a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            # Track average train loss and average train time:
            batch_time = time.time()-st_time
            avg_tr_loss += l.item()
            avg_tr_time += batch_time

            if i%100 == 99:
                eta = avg_tr_time/100*(len(trSet)/batch_size-i)
                log_msg = f"Epoch no: {epoch_num+1} | Epoch progress(%): {i/(len(trSet)/batch_size)*100:.2f} | Avg train loss: {avg_tr_loss/100:.4f} | Acc: {avg_lat_acc:.4f} {avg_lon_acc:.4f} | Validation loss prev epoch {prev_val_loss:.4f} | ETA(s): {int(eta)}"
                logger.info(log_msg)
                train_loss.append(avg_tr_loss/100)
                avg_tr_loss = 0
                avg_lat_acc = 0
                avg_lon_acc = 0
                avg_tr_time = 0
        # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



        ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
        net.train_flag = False

        logger.info(f"Epoch {epoch_num+1} complete. Calculating validation loss...")
        avg_val_loss = 0
        avg_val_lat_acc = 0
        avg_val_lon_acc = 0
        val_batch_count = 0
        total_points = 0

        for i, data  in enumerate(valDataloader):
            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            # 将数据移动到指定设备
            hist = hist.to(device)
            nbrs = nbrs.to(device)
            mask = mask.to(device)
            lat_enc = lat_enc.to(device)
            lon_enc = lon_enc.to(device)
            fut = fut.to(device)
            op_mask = op_mask.to(device)

            # Forward pass
            if args['use_maneuvers']:
                if epoch_num < pretrainEpochs:
                    # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                    net.train_flag = True
                    fut_pred, _ , _ = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    # During training with NLL loss, validate with NLL over multi-modal distribution
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
                    avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                    avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)

            avg_val_loss += l.item()
            val_batch_count += 1

        # Print validation loss and update display variables
        current_val_loss = avg_val_loss/val_batch_count
        logger.info(f"Validation loss: {current_val_loss:.4f} | Val Acc: {avg_val_lat_acc/val_batch_count*100:.4f} {avg_val_lon_acc/val_batch_count*100:.4f}")
        val_loss.append(current_val_loss)
        prev_val_loss = current_val_loss

        # 保存每个 epoch 的 checkpoint
        checkpoint_path = f'trained_models/checkpoint_epoch_{epoch_num+1}.pth'
        torch.save({
            'epoch': epoch_num + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': current_val_loss,
            'train_loss': train_loss,
            'val_loss_history': val_loss,
        }, checkpoint_path)
        logger.info(f"已保存 checkpoint: {checkpoint_path}")

        # 保存最佳模型（验证损失最低的）
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_path = 'trained_models/cslstm_m_best.pth'
            torch.save({
                'epoch': epoch_num + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss_history': val_loss,
            }, best_model_path)
            logger.info(f"✓ 新的最佳模型已保存 (val_loss: {best_val_loss:.4f}): {best_model_path}")

        #__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    # 训练结束后保存最终模型
    final_model_path = 'trained_models/cslstm_m_final.pth'
    torch.save({
        'epoch': pretrainEpochs + trainEpochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': prev_val_loss,
        'train_loss': train_loss,
        'val_loss_history': val_loss,
    }, final_model_path)
    logger.info("="*80)
    logger.info("训练完成！")
    logger.info(f"最终模型已保存: {final_model_path}")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"训练日志已保存: {log_filename}")
    logger.info("="*80)


if __name__ == '__main__':
    main()



