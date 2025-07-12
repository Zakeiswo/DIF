import yaml
import os
import argparse

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件路径')
    parser.add_argument('--testflage', type=bool, default=False, help='如果是测试阶段，不保存模型')
    return parser.parse_args()

def load_config(config_path):
    """
    加载YAML配置文件
    
    参数:
        config_path: 配置文件路径
    
    返回:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_args_with_config(args, config):
    """
    将命令行参数与配置文件合并
    
    参数:
        args: 命令行参数
        config: 配置字典
    
    返回:
        合并后的配置字典
    """
    # 将命令行参数添加到配置中
    # 检查是否存在 testflage 参数，如果不存在则使用默认值 False
    if hasattr(args, 'testflage'):
        config['testflage'] = args.testflage
    else:
        # 如果没有定义 testflage 参数，默认为 False
        config['testflage'] = False
    
    return config
