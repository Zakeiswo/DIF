import sys
import os
import time
import logging
import glob
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime
import importlib
import yaml
import argparse
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from utils.config import parse_args, load_config, merge_args_with_config
import utils.utils as utils
from models.loss import structure_loss
from metric.metric import *

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = args.config
    config = load_config(config_path)
    config = merge_args_with_config(args, config)
    
    # 设置保存路径
    if 'time' in config['save']['path']:
        save_path = config['save']['path']
    else:
        save_path = config['save']['path'] + '-time:{}'.format(time.strftime("%m%d-%H%M%S"))
    
    # 如果不是测试阶段，创建实验目录并设置日志
    if not config['testflage']:
        # 创建实验目录，保存脚本
        utils.create_exp_dir_extend_2(save_path, 
                                     scripts_to_save=glob.glob('*.py'), 
                                     scripts_to_savemodel=glob.glob('./models/*.py'),
                                     scripts_to_savemodel2=glob.glob('./utils/*.py'))
        
        # TensorBoard相关代码已移除
        
        # 设置日志
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info(save_path)
    else:
        print("----------------------------------------!!!!!!注意这是测试阶段!!!!!!----------------------------------------")
    
    # 动态导入模型
    module_path = config['model']['path']
    class_name = config['model']['name']
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # 初始化模型
    model = model_class()
    model.cuda()
    
    # 设置优化器和参数
    params = model.parameters()
    # 确保学习率是浮点数
    print(f"配置文件中的学习率值: {config['training']['lr']}, 类型: {type(config['training']['lr'])}")
    lr = float(config['training']['lr'])
    print(f"转换为浮点数后的学习率值: {lr}")
    optimizer = torch.optim.Adam(params, lr)
    print(f"优化器初始化后的学习率: {optimizer.param_groups[0]['lr']}")
    
    # 学习率调度器 - 使用StepLR
    step_size = config['training']['decay_epoch']
    gamma = config['training']['decay_rate']
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    print(f"学习率调度器初始化后的学习率: {optimizer.param_groups[0]['lr']}")
    
    # 创建梯度缩放器用于混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 加载数据
    from data.dataloader import get_train_loader, get_test_loader
    
    train_loader, train_samples = get_train_loader(
        config['dataset']['train']['image_root'],
        config['dataset']['train']['gt_root'],
        config['dataset']['train']['depth_root'],
        batchsize=config['training']['batchsize'],
        trainsize=config['training']['trainsize'],
        num_workers=config['training']['numworkers']
    )
    
    # 使用第一个测试数据集进行训练过程中的验证
    if 'datasets' in config['dataset']['test']:
        # 新的配置结构：多个测试数据集
        first_dataset = config['dataset']['test']['datasets'][0]
        test_loader, test_samples = get_test_loader(
            first_dataset['image_root'],
            first_dataset['gt_root'],
            first_dataset['depth_root'],
            batchsize=1,
            trainsize=config['training']['trainsize'],
            num_workers=config['training']['numworkers']
        )
        print(f"使用数据集 '{first_dataset['name']}' 进行训练过程中的验证")
    else:
        # 旧的配置结构：单个测试数据集
        test_loader, test_samples = get_test_loader(
            config['dataset']['test']['image_root'],
            config['dataset']['test']['gt_root'],
            config['dataset']['test']['depth_root'],
            batchsize=1,
            trainsize=config['training']['trainsize'],
            num_workers=config['training']['numworkers']
        )
    
    print("CUDA设备:", torch.cuda.current_device())
    
    # 开始训练过程
    print("开始训练!")
    
    # 初始化最佳指标记录
    best_F_measure_rgb = 0
    cor_mae_rgb = 0
    cor_sm_rgb = 0
    cor_ep_rgb = 0
    
    try:
        for epoch in range(1, config['training']['epoch']):
            # 训练
            train_loss = train(
                train_loader, 
                model, 
                optimizer, 
                scaler,
                lr_scheduler,
                epoch, 
                config, 
                train_samples
            )
            
            # 更新学习率 - 在每个epoch结束后调用一次
            lr_scheduler.step()
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            if not config['testflage']:
                logging.info(f"Epoch {epoch} 结束后的学习率: {current_lr}")
            else:
                print(f"Epoch {epoch} 结束后的学习率: {current_lr}")
            
            # 训练结束后立即保存模型
            if not config['testflage']:
                # 保存最新模型
                utils.save(model, os.path.join(save_path, 'latest_epoch.pth'))
                logging.info(f"模型已保存到 {os.path.join(save_path, 'latest_epoch.pth')}，轮次: {epoch}")
                
                # 保存检查点
                if config['training']['save_ck']:
                    save_ck_path = os.path.join(save_path, 'checkpoint/')
                    if not os.path.exists(save_ck_path):
                        os.makedirs(save_ck_path)
                    torch.save(model.state_dict(), save_ck_path + '%d' % epoch + '.pth')
                    logging.info(f"检查点已保存到 {save_ck_path + '%d' % epoch + '.pth'}")
            
            # 测试并计算指标
            if not config['testflage']:
                logging.info("计算F-measure和mae")
            else:
                print("计算F-measure和mae")
            
            F_measure_rgb, mae_rgb, S_measure_rgb = test(
                model, 
                test_loader,
                test_samples,
                epoch, 
                config
            )
            
            # 记录当前epoch的指标
            if not config['testflage']:
                logging.info("Epoch:{}, F-measure-:{:.4f}, MAE:{:.4f}, S-measure:{:.4f} ".format(
                    epoch, F_measure_rgb, mae_rgb, S_measure_rgb))
                # TensorBoard记录已移除
            else:
                print("Epoch:{}, F-measure-:{:.4f}, MAE:{:.4f}, S-measure:{:.4f} ".format(
                    epoch, F_measure_rgb, mae_rgb, S_measure_rgb))
            
            # 如果当前F-measure更好，则更新最佳记录并保存模型
            if F_measure_rgb > best_F_measure_rgb:
                best_F_measure_rgb = F_measure_rgb
                cor_mae_rgb = mae_rgb
                cor_sm_rgb = S_measure_rgb
                cor_ep_rgb = epoch
                
                if not config['testflage']:
                    logging.info("最佳F-measure:{:.4f}, 对应MAE:{:.4f}, 对应S-measure:{:.4f}, 对应轮次:{} ".format(
                        best_F_measure_rgb, cor_mae_rgb, cor_sm_rgb, cor_ep_rgb))
                else:
                    print("最佳F-measure:{:.4f}, 对应MAE:{:.4f}, 对应S-measure:{:.4f}, 对应轮次:{} ".format(
                        best_F_measure_rgb, cor_mae_rgb, cor_sm_rgb, cor_ep_rgb))
                
                if not config['testflage']:
                    utils.save(model, os.path.join(save_path, 'best_epoch.pth'))
            
            # 保存最佳模型（基于测试结果）
            if not config['testflage'] and F_measure_rgb > best_F_measure_rgb:
                utils.save(model, os.path.join(save_path, 'best_epoch.pth'))
                logging.info(f"最佳模型已保存到 {os.path.join(save_path, 'best_epoch.pth')}，轮次: {epoch}")
        
        # 训练结束，输出最佳结果
        if not config['testflage']:
            logging.info("最佳RGB: F-measure:{:.4f}, 对应MAE:{:.4f}, 对应S-measure:{:.4f}, 对应轮次:{} ".format(
                best_F_measure_rgb, cor_mae_rgb, cor_sm_rgb, cor_ep_rgb))
        else:
            print("最佳RGB: F-measure:{:.4f}, 对应MAE:{:.4f}, 对应S-measure:{:.4f}, 对应轮次:{} ".format(
                best_F_measure_rgb, cor_mae_rgb, cor_sm_rgb, cor_ep_rgb))
    
    except KeyboardInterrupt:
        # 处理键盘中断，输出当前最佳结果
        if not config['testflage']:
            logging.info(
                "\n最佳RGB: F-measure:{:.4f}, 对应MAE:{:.4f}, 对应S-measure:{:.4f}, 对应轮次:{} ".format(
                    best_F_measure_rgb, cor_mae_rgb, cor_sm_rgb, cor_ep_rgb))
        else:
            print(
                "最佳RGB: F-measure:{:.4f}, 对应MAE:{:.4f}, 对应S-measure:{:.4f}, 对应轮次:{} ".format(
                    best_F_measure_rgb, cor_mae_rgb, cor_sm_rgb, cor_ep_rgb))

def train(train_loader, model, optimizer, scaler, lr_scheduler, epoch, config, train_samples):
    """
    训练模型的函数
    
    参数:
        train_loader: 训练数据加载器
        model: 要训练的模型
        optimizer: 优化器
        scaler: 梯度缩放器
        lr_scheduler: 学习率调度器
        epoch: 当前训练轮次
        config: 配置字典
        train_samples: 训练样本数
        
    返回:
        sum_loss: 总损失
    """
    model.train()
    sum_loss = 0.0
    torch.backends.cudnn.benchmark = True
    
    # 打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} 开始时的学习率: {current_lr}")
    
    # 使用tqdm显示训练进度
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, unit='images', ncols=75)
    loop.set_description('{} Epoch [{:03d}/{:03d}]'.format(
        datetime.now().replace(microsecond=0), epoch, config['training']['epoch']))
    
    for i, pack in loop:
        optimizer.zero_grad()
        images, depth, gts, _ = pack
        
        # 将数据转换为Variable并移至GPU
        images = Variable(images)
        depth = Variable(depth)
        gts = Variable(gts)
        
        images = images.cuda(non_blocking=True)
        depth = depth.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)
        
        # 处理深度图输入
        n, c, h, w = depth.size()
        depth = depth.view(n, h, w, 1).repeat(1, 1, 1, 3)
        depth = depth.transpose(3, 1)
        depth = depth.transpose(3, 2)
        
        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            predic_rgb = model(images.cuda(), depth.cuda())
            rgb_loss = structure_loss(predic_rgb, gts)
        
        # 累计损失
        sum_loss += rgb_loss
        
        # 反向传播
        scaler.scale(rgb_loss).backward()
        
        # 梯度裁剪
        utils.clip_gradient(optimizer, config['training']['clip'])
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        
        # 更新进度条
        loop.set_postfix_str('Loss: {:.4f}'.format(rgb_loss.data))
    
    # 记录训练损失
    if not config['testflage']:
        logging.info('总损失:{:.4f}'.format(sum_loss.data))
    else:
        print('总损失:{:.4f}'.format(sum_loss.data))
    
    return sum_loss

def test(model, test_loader, test_samples, epoch, config):
    """
    测试函数
    """
    model.eval()
    sum_loss = 0.0
    
    # 初始化评估指标计算器
    cal_fm = CalFM(num=test_samples)
    cal_mae = CalMAE(num=test_samples)
    cal_sm = CalSM(num=test_samples)
    
    for step, packs in enumerate(test_loader):
        input, depth, target, _ = packs
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        depth = depth.cuda(non_blocking=True)
        
        # 处理深度图输入
        n, c, h, w = depth.size()
        depth = depth.view(n, h, w, 1).repeat(1, 1, 1, 3)
        depth = depth.transpose(3, 1)
        depth = depth.transpose(3, 2)
        
        # 不计算梯度进行前向传播
        with torch.no_grad():
            output_rgb = model(input.cuda(), depth.cuda())
            sum_loss += structure_loss(output_rgb.detach(), target)
            output_rgb = torch.squeeze(output_rgb, 1)
        
        # 后处理预测结果
        predict_rgb = output_rgb.sigmoid().cpu().detach().numpy()
        target = target.data.cpu().numpy()
        
        # 确保 target 保持二维形状
        if len(target.shape) == 3 and target.shape[1] == 1:
            # 如果 target 是 (batch_size, 1, H, W)，移除通道维度
            target = np.squeeze(target, axis=1)
        elif len(target.shape) == 4 and target.shape[1] == 1:
            # 如果 target 是 (batch_size, 1, H, W)，移除通道维度
            target = np.squeeze(target, axis=1)
        
        for i in range(target.shape[0]):
            max_pred_array = predict_rgb[i].max()
            min_pred_array = predict_rgb[i].min()
            
            # 归一化预测结果
            if max_pred_array == min_pred_array:
                predict_rgb[i] = predict_rgb[i] / 255
            else:
                predict_rgb[i] = (predict_rgb[i] - min_pred_array) / (max_pred_array - min_pred_array)
            
            # 确保预测结果和目标掩码具有相同的形状
            pred_shape = predict_rgb[i].shape
            target_shape = target[i].shape
            
            if pred_shape != target_shape:
                print(f"形状不匹配: pred {pred_shape}, target {target_shape}")
                # 调整大小以匹配
                min_h = min(pred_shape[0], target_shape[0])
                min_w = min(pred_shape[1], target_shape[1])
                
                # 裁剪预测结果和目标掩码
                pred_resized = predict_rgb[i][:min_h, :min_w]
                target_resized = target[i][:min_h, :min_w]
                
                # 更新评估指标
                cal_fm.update(pred_resized, target_resized)
                cal_mae.update(pred_resized, target_resized)
                cal_sm.update(pred_resized, target_resized)
            else:
                # 更新评估指标
                cal_fm.update(predict_rgb[i], target[i])
                cal_mae.update(predict_rgb[i], target[i])
                cal_sm.update(predict_rgb[i], target[i])
    
    # 计算最终指标
    _, maxf, mmf, _, _ = cal_fm.show()
    mae = cal_mae.show()
    sm = cal_sm.show()
    
    return mmf, mae, sm

if __name__ == '__main__':
    main()
