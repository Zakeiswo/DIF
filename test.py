import sys
import os
import time
import logging
import glob
import torch
import argparse
import importlib
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from utils.config import load_config, merge_args_with_config
import utils.utils as utils
from models.loss import structure_loss
from metric.metric import *
from data.dataloader import get_test_loader

def main():
    """
    主函数 - 加载模型并进行测试
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = args.config
    config = load_config(config_path)
    config = merge_args_with_config(args, config)
    
    # 设置日志
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    
    # 动态导入模型
    module_path = config['model']['path']
    class_name = config['model']['name']
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # 初始化模型
    model = model_class()
    model.cuda()
    
    # 加载模型权重
    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        print(f"加载模型检查点: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print(f"错误: 检查点文件 {checkpoint_path} 不存在")
        return
    
    # 设置为评估模式
    model.eval()
    
    print(f"CUDA设备: {torch.cuda.current_device()}")
    
    # 检查配置文件格式
    if 'datasets' in config['dataset']['test']:
        # 新格式：支持多个数据集
        datasets = config['dataset']['test']['datasets']
    else:
        # 旧格式：单一数据集，为了兼容性创建列表
        datasets = [{
            'name': config['dataset']['test'].get('name', 'Unknown'),
            'image_root': config['dataset']['test']['image_root'],
            'gt_root': config['dataset']['test']['gt_root'],
            'depth_root': config['dataset']['test']['depth_root']
        }]
    
    # 存储所有数据集的结果
    all_results = []
    
    # 遍历每个数据集进行测试
    for dataset_config in datasets:
        dataset_name = dataset_config['name']
        print(f"\n测试数据集: {dataset_name}")
        print("加载测试数据")
        
        # 加载测试数据
        test_loader, test_samples = get_test_loader(
            dataset_config['image_root'],
            dataset_config['gt_root'],
            dataset_config['depth_root'],
            batchsize=1,
            trainsize=config['training']['trainsize'],
            num_workers=config['training']['numworkers']
        )
        
        print(f"测试样本数: {test_samples}")
        
        # 运行测试
        mmf, mae, sm, em, wfm = test(model, test_loader, test_samples, config)
        
        # 打印测试结果
        print(f"数据集 {dataset_name} 测试结果:")
        print(f"F-measure: {mmf:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"S-measure: {sm:.4f}")
        print(f"E-measure: {em:.4f}")
        print(f"WFM: {wfm:.4f}")
        
        # 保存结果
        all_results.append({
            'dataset': dataset_name,
            'mmf': mmf,
            'mae': mae,
            'sm': sm,
            'em': em,
            'wfm': wfm,
            'samples': test_samples
        })
    
    # 如果有多个数据集，以表格形式打印所有结果
    if len(all_results) > 1:
        total_samples = sum(r['samples'] for r in all_results)
        
        # 打印表格头
        print("\n" + "=" * 80)
        print("{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "数据集", "F-measure", "MAE", "S-measure", "E-measure", "WFM", "样本数"))
        print("-" * 80)
        
        # 打印每个数据集的结果
        for result in all_results:
            print("{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(
                result['dataset'],
                result['mmf'],
                result['mae'],
                result['sm'],
                result['em'],
                result['wfm'],
                result['samples']
            ))
        
        print("=" * 80)
        print(f"\n总样本数: {total_samples}")
        
        # 返回所有数据集的结果，而不是平均值
        return all_results
    
def test(model, test_loader, test_samples, config):
    """
    测试函数
    """
    model.eval()
    sum_loss = 0.0
    
    # 初始化评估指标计算器
    cal_fm = CalFM(num=test_samples)
    cal_mae = CalMAE(num=test_samples)
    cal_sm = CalSM(num=test_samples)
    cal_em = CalEM(num=test_samples)
    cal_wfm = CalWFM(num=test_samples)
    
    for step, packs in enumerate(tqdm(test_loader, desc="测试进度", ncols=75)):
        input, depth, target, name = packs
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
                print(f"形状不匹配: pred {pred_shape}, target {target_shape}, 图像: {name[i]}")
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
                cal_em.update(pred_resized, target_resized)
                cal_wfm.update(pred_resized, target_resized)
            else:
                # 更新评估指标
                cal_fm.update(predict_rgb[i], target[i])
                cal_mae.update(predict_rgb[i], target[i])
                cal_sm.update(predict_rgb[i], target[i])
                cal_em.update(predict_rgb[i], target[i])
                cal_wfm.update(predict_rgb[i], target[i])
    
    # 计算最终指标
    _, maxf, mmf, _, _ = cal_fm.show()
    mae = cal_mae.show()
    sm = cal_sm.show()
    em = cal_em.show()
    wfm = cal_wfm.show()
    
    return mmf, mae, sm, em, wfm

if __name__ == '__main__':
    main()
