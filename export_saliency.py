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
from PIL import Image

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
    主函数 - 加载模型并导出显著图
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output-dir', type=str, default='./results', help='输出目录')

    parser.add_argument('--evaluate', action='store_true', help='是否在导出的同时评估性能')
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
    
    # 遍历每个数据集进行测试和导出
    for dataset_config in datasets:
        dataset_name = dataset_config['name']
        print(f"\n处理数据集: {dataset_name}")
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
        
        # 创建输出目录
        output_dataset_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(output_dataset_dir, exist_ok=True)
        
        # 运行测试和导出
        if args.evaluate:
            mmf, mae, sm, em, wfm = export_and_evaluate(model, test_loader, test_samples, config, output_dataset_dir)
            
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
        else:
            export_saliency_maps(model, test_loader, output_dataset_dir)
            print(f"显著图已保存到: {output_dataset_dir}")
    
    # 如果有多个数据集且进行了评估，计算平均结果
    if args.evaluate and len(all_results) > 1:
        total_samples = sum(r['samples'] for r in all_results)
        avg_mmf = sum(r['mmf'] * r['samples'] for r in all_results) / total_samples
        avg_mae = sum(r['mae'] * r['samples'] for r in all_results) / total_samples
        avg_sm = sum(r['sm'] * r['samples'] for r in all_results) / total_samples
        avg_em = sum(r['em'] * r['samples'] for r in all_results) / total_samples
        avg_wfm = sum(r['wfm'] * r['samples'] for r in all_results) / total_samples
        
        print("\n所有数据集平均结果:")
        print(f"平均 F-measure: {avg_mmf:.4f}")
        print(f"平均 MAE: {avg_mae:.4f}")
        print(f"平均 S-measure: {avg_sm:.4f}")
        print(f"平均 E-measure: {avg_em:.4f}")
        print(f"平均 WFM: {avg_wfm:.4f}")
        print(f"总样本数: {total_samples}")

def export_saliency_maps(model, test_loader, output_dir):
    """
    导出显著图
    """
    model.eval()
    
    for step, packs in enumerate(tqdm(test_loader, desc="导出进度", ncols=75)):
        input, depth, target, name = packs
        input = input.cuda(non_blocking=True)
        depth = depth.cuda(non_blocking=True)
        
        # 处理深度图输入
        n, c, h, w = depth.size()
        depth = depth.view(n, h, w, 1).repeat(1, 1, 1, 3)
        depth = depth.transpose(3, 1)
        depth = depth.transpose(3, 2)
        
        # 不计算梯度进行前向传播
        with torch.no_grad():
            output_rgb = model(input.cuda(), depth.cuda())
            output_rgb = torch.squeeze(output_rgb, 1)
        
        # 后处理预测结果
        predict_rgb = output_rgb.sigmoid().cpu().detach().numpy()
        
        for i in range(predict_rgb.shape[0]):
            # 归一化预测结果到 0-1 范围
            max_pred = predict_rgb[i].max()
            min_pred = predict_rgb[i].min()
            
            if max_pred == min_pred:
                normalized_pred = predict_rgb[i] / 255
            else:
                normalized_pred = (predict_rgb[i] - min_pred) / (max_pred - min_pred)
            
            # 转换为 0-255 范围的 uint8 类型
            saliency_map = (normalized_pred * 255).astype(np.uint8)
            
            # 创建 PIL 图像
            img = Image.fromarray(saliency_map)
            
            # 获取不带扩展名的文件名
            filename = os.path.splitext(os.path.basename(name[i]))[0]
            
            # 保存图像
            save_path = os.path.join(output_dir, f"{filename}.png")
            img.save(save_path)

def export_and_evaluate(model, test_loader, test_samples, config, output_dir):
    """
    导出显著图并评估性能
    """
    model.eval()
    sum_loss = 0.0
    
    # 初始化评估指标计算器
    cal_fm = CalFM(num=test_samples)
    cal_mae = CalMAE(num=test_samples)
    cal_sm = CalSM(num=test_samples)
    cal_em = CalEM(num=test_samples)
    cal_wfm = CalWFM(num=test_samples)
    
    for step, packs in enumerate(tqdm(test_loader, desc="导出并评估", ncols=75)):
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
            
            # 转换为 0-255 范围的 uint8 类型并保存
            saliency_map = (predict_rgb[i] * 255).astype(np.uint8)
            img = Image.fromarray(saliency_map)
            
            # 获取不带扩展名的文件名
            filename = os.path.splitext(os.path.basename(name[i]))[0]
            
            # 保存图像
            save_path = os.path.join(output_dir, f"{filename}.png")
            img.save(save_path)
            
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
