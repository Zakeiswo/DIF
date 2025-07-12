import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import glob

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from utils.config import load_config, merge_args_with_config
from metric.metric import *

def main():
    """
    主函数 - 读取已保存的显著图并计算评估指标
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--pred-dir', type=str, required=True, help='预测显著图目录')
    parser.add_argument('--dataset-name', type=str, default=None, help='数据集名称，如果为None则使用pred-dir的目录名')
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = args.config
    config = load_config(config_path)
    config = merge_args_with_config(args, config)
    
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
    
    # 确定数据集名称
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.normpath(args.pred_dir))
    
    # 查找匹配的数据集配置
    dataset_config = None
    for ds in datasets:
        if ds['name'] == dataset_name:
            dataset_config = ds
            break
    
    if dataset_config is None:
        print(f"错误: 在配置文件中找不到名为 '{dataset_name}' 的数据集")
        return
    
    # 获取真实标签目录
    gt_root = dataset_config['gt_root']
    
    # 评估显著图
    evaluate_saliency_maps(args.pred_dir, gt_root, dataset_name)

def evaluate_saliency_maps(pred_dir, gt_dir, dataset_name):
    """
    评估显著图性能
    
    参数:
        pred_dir: 预测显著图目录
        gt_dir: 真实标签目录
        dataset_name: 数据集名称
    """
    # 获取所有预测显著图
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.png')))
    
    if not pred_files:
        print(f"错误: 在 {pred_dir} 中找不到预测显著图")
        return
    
    print(f"找到 {len(pred_files)} 个预测显著图")
    
    # 初始化评估指标计算器
    test_samples = len(pred_files)
    cal_fm = CalFM(num=test_samples)
    cal_mae = CalMAE(num=test_samples)
    cal_sm = CalSM(num=test_samples)
    cal_em = CalEM(num=test_samples)
    cal_wfm = CalWFM(num=test_samples)
    
    # 处理每个预测显著图
    for i, pred_file in enumerate(tqdm(pred_files, desc="评估进度", ncols=75)):
        # 获取文件名（不带扩展名）
        filename = os.path.splitext(os.path.basename(pred_file))[0]
        
        # 查找对应的真实标签文件
        gt_file = find_gt_file(gt_dir, filename)
        
        if gt_file is None:
            print(f"警告: 找不到 {filename} 的真实标签，跳过")
            continue
        
        # 读取预测显著图和真实标签
        pred = np.array(Image.open(pred_file).convert('L')) / 255.0
        gt = np.array(Image.open(gt_file).convert('L')) / 255.0
        
        # 确保预测结果和目标掩码具有相同的形状
        pred_shape = pred.shape
        gt_shape = gt.shape
        
        if pred_shape != gt_shape:
            print(f"形状不匹配: pred {pred_shape}, gt {gt_shape}, 图像: {filename}")
            # 调整大小以匹配
            min_h = min(pred_shape[0], gt_shape[0])
            min_w = min(pred_shape[1], gt_shape[1])
            
            # 裁剪预测结果和目标掩码
            pred_resized = pred[:min_h, :min_w]
            gt_resized = gt[:min_h, :min_w]
            
            # 更新评估指标
            cal_fm.update(pred_resized, gt_resized)
            cal_mae.update(pred_resized, gt_resized)
            cal_sm.update(pred_resized, gt_resized)
            cal_em.update(pred_resized, gt_resized)
            cal_wfm.update(pred_resized, gt_resized)
        else:
            # 更新评估指标
            cal_fm.update(pred, gt)
            cal_mae.update(pred, gt)
            cal_sm.update(pred, gt)
            cal_em.update(pred, gt)
            cal_wfm.update(pred, gt)
    
    # 计算最终指标
    _, maxf, mmf, _, _ = cal_fm.show()
    mae = cal_mae.show()
    sm = cal_sm.show()
    em = cal_em.show()
    wfm = cal_wfm.show()
    
    # 打印测试结果
    print(f"\n数据集 {dataset_name} 测试结果:")
    print(f"F-measure: {mmf:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"S-measure: {sm:.4f}")
    print(f"E-measure: {em:.4f}")
    print(f"WFM: {wfm:.4f}")
    print(f"有效样本数: {cal_fm.idx}")
    
    return mmf, mae, sm, em, wfm

def find_gt_file(gt_dir, filename):
    """
    在真实标签目录中查找对应的真实标签文件
    
    参数:
        gt_dir: 真实标签目录
        filename: 文件名（不带扩展名）
    
    返回:
        找到的真实标签文件路径，如果找不到则返回None
    """
    # 尝试不同的扩展名
    extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    for ext in extensions:
        gt_file = os.path.join(gt_dir, filename + ext)
        if os.path.exists(gt_file):
            return gt_file
    
    # 如果找不到完全匹配的文件名，尝试查找包含该文件名的文件
    for ext in extensions:
        pattern = os.path.join(gt_dir, f"*{filename}*{ext}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

if __name__ == '__main__':
    main()
