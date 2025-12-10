from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import os



## Network Arguments
args = {}
# 自动检测CUDA是否可用，如果不可用则使用CPU
args['use_cuda'] = torch.cuda.is_available()
if args['use_cuda']:
    print("使用GPU评估")
else:
    print("CUDA不可用，使用CPU评估（速度较慢）")
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
args['train_flag'] = False


# Evaluation metric:
metric = 'rmse'  #or rmse


# Initialize network
net = highwayNet(args)
# 加载模型权重（支持 .pth 和 .tar 格式以兼容旧模型）
model_path = 'trained_models/cslstm_m_best.pth'  # 优先使用最佳模型
if not os.path.exists(model_path):
    model_path = 'trained_models/cslstm_m_final.pth'  # 如果没有最佳模型，使用最终模型
if not os.path.exists(model_path):
    model_path = 'trained_models/cslstm_m.tar'  # 兼容旧格式
checkpoint = torch.load(model_path, map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载模型 checkpoint (epoch: {checkpoint.get('epoch', 'N/A')}, val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
else:
    net.load_state_dict(checkpoint)
    print("已加载模型权重")
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('data/TestSet_traj.csv')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

if args['use_cuda']:
    lossVals = torch.zeros(25).cuda()
    counts = torch.zeros(25).cuda()
else:
    lossVals = torch.zeros(25)
    counts = torch.zeros(25)


for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

    # Initialize Variables
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()

    if metric == 'nll':
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
    else:
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man*3 + lat_man
                fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedMSETest(fut_pred, fut, op_mask)


    lossVals +=l.detach()
    counts += c.detach()

if metric == 'nll':
    avg_loss_per_timestep = lossVals / counts
    print("\n" + "="*80)
    print("评估结果：每个时间步的平均负对数似然（NLL）损失")
    print("="*80)
    print(f"时间步范围: 1-25 (每个时间步 = 0.1秒, 总共2.5秒)")
    print(f"单位: 负对数似然 (值越小表示预测越好)")
    print("\n各时间步的NLL损失:")
    for i, loss_val in enumerate(avg_loss_per_timestep, 1):
        time_sec = i * 0.1
        print(f"  时间步 {i:2d} (t={time_sec:4.1f}s): {loss_val.item():7.4f}")
    print("\n" + "-"*80)
    print(f"平均NLL损失 (所有时间步): {avg_loss_per_timestep.mean().item():.4f}")
    print(f"最终时间步 (t=2.5s) NLL损失: {avg_loss_per_timestep[-1].item():.4f}")
    print("="*80)
    print("\n完整tensor输出:")
    print(avg_loss_per_timestep)
else:
    rmse_per_timestep = torch.pow(lossVals / counts, 0.5) * 0.3048  # Calculate RMSE and convert from feet to meters
    print("\n" + "="*80)
    print("评估结果：每个时间步的均方根误差（RMSE）")
    print("="*80)
    print(f"时间步范围: 1-25 (每个时间步 = 0.1秒, 总共2.5秒)")
    print(f"单位: 米 (值越小表示预测越好)")
    print("\n各时间步的RMSE:")
    for i, rmse_val in enumerate(rmse_per_timestep, 1):
        time_sec = i * 0.1
        print(f"  时间步 {i:2d} (t={time_sec:4.1f}s): {rmse_val.item():7.4f} 米")
    print("\n" + "-"*80)
    print(f"平均RMSE (所有时间步): {rmse_per_timestep.mean().item():.4f} 米")
    print(f"最终时间步 (t=2.5s) RMSE: {rmse_per_timestep[-1].item():.4f} 米")
    print("="*80)
    print("\n完整tensor输出:")
    print(rmse_per_timestep)


