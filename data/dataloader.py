import os
import sys
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入原始数据加载器
try:
    from dataset.data_RGBD import get_loader as original_get_loader
    from dataset.data_RGBD import SalObjDataset, test_dataset
except ImportError:
    # 如果导入失败，我们需要自己实现数据加载器
    from data.dataset import SalObjDataset, test_dataset

def get_train_loader(image_root, gt_root, depth_root, batchsize, trainsize, num_workers):
    """
    获取训练数据加载器
    
    参数:
        image_root: 图像根目录
        gt_root: 真实标签根目录
        depth_root: 深度图根目录
        batchsize: 批次大小
        trainsize: 训练图像大小
        num_workers: 工作线程数
    
    返回:
        train_loader: 训练数据加载器
        train_samples: 训练样本数
    """
    try:
        # 尝试使用原始的get_loader函数
        return original_get_loader(image_root, depth_root, gt_root, batchsize, num_workers, trainsize)
    except:
        # 如果失败，使用我们自己的实现
        dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
        train_samples = len(dataset)
        train_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
        return train_loader, train_samples

def get_test_loader(image_root, gt_root, depth_root, batchsize, trainsize, num_workers):
    """
    获取测试数据加载器
    
    参数:
        image_root: 图像根目录
        gt_root: 真实标签根目录
        depth_root: 深度图根目录
        batchsize: 批次大小
        trainsize: 训练图像大小
        num_workers: 工作线程数
    
    返回:
        test_loader: 测试数据加载器
        test_samples: 测试样本数
    """
    try:
        # 尝试使用原始的get_loader函数
        return original_get_loader(image_root, depth_root, gt_root, batchsize, num_workers, trainsize, shuffle=False, iftrain=False)
    except Exception as e:
        # 如果失败，使用我们自己的实现
        dataset = test_dataset(image_root, gt_root, trainsize)
        test_samples = len(dataset)
        test_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers)
        return test_loader, test_samples
