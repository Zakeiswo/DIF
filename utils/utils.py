import os
import torch
import shutil
import numpy as np
import glob

def clip_gradient(optimizer, grad_clip):
    """
    梯度裁剪
    
    参数:
        optimizer: 优化器
        grad_clip: 梯度裁剪阈值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def create_exp_dir(path, scripts_to_save=None):
    """
    创建实验目录
    
    参数:
        path: 目录路径
        scripts_to_save: 要保存的脚本列表
    """
    if not os.path.exists(path):
        os.makedirs(path)
    print('实验目录 : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def create_exp_dir_extend_2(path, scripts_to_save=None, scripts_to_savemodel=None, scripts_to_savemodel2=None):
    """
    创建扩展的实验目录，可以保存三个文件夹
    
    参数:
        path: 目录路径
        scripts_to_save: 要保存的脚本列表
        scripts_to_savemodel: 要保存的模型脚本列表
        scripts_to_savemodel2: 要保存的其他模型脚本列表
    """
    if not os.path.exists(path):
        os.makedirs(path)
    print('实验目录 : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    
    if scripts_to_savemodel is not None:
        if not os.path.exists(os.path.join(path, 'model')):
            os.makedirs(os.path.join(path, 'model'))
        for script in scripts_to_savemodel:
            dst_file = os.path.join(path, 'model', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    
    if scripts_to_savemodel2 is not None:
        if not os.path.exists(os.path.join(path, 'model2')):
            os.makedirs(os.path.join(path, 'model2'))
        for script in scripts_to_savemodel2:
            dst_file = os.path.join(path, 'model2', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save(model, save_path):
    """
    保存模型
    
    参数:
        model: 要保存的模型
        save_path: 保存路径
    """
    torch.save(model.state_dict(), save_path)
