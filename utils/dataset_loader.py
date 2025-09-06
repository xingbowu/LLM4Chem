"""
数据集加载工具，支持本地路径和远程数据集
"""
import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from typing import Optional, List, Union


def is_local_dataset_path(path):
    """检查是否为本地数据集路径"""
    if isinstance(path, str):
        # 检查是否为绝对路径或相对路径
        if (os.path.isabs(path) or 
            path.startswith('./') or 
            path.startswith('../')):
            return True
        
        # 检查路径是否存在
        if os.path.exists(path):
            return True
            
        # 检查是否为常见的本地文件格式
        local_extensions = ['.json', '.jsonl', '.csv', '.tsv', '.txt', '.parquet']
        if any(path.endswith(ext) for ext in local_extensions):
            return True
            
    return False


def validate_local_dataset_path(dataset_path):
    """验证本地数据集路径是否有效"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"本地数据集路径不存在: {dataset_path}")
    
    path = Path(dataset_path)
    
    # 如果是目录，检查是否包含数据文件
    if path.is_dir():
        data_files = list(path.glob('*.json')) + list(path.glob('*.jsonl')) + \
                    list(path.glob('*.csv')) + list(path.glob('*.parquet'))
        if not data_files:
            raise FileNotFoundError(f"目录中未找到支持的数据文件: {dataset_path}")
        print(f"📁 找到 {len(data_files)} 个数据文件")
    
    # 如果是文件，检查文件格式
    elif path.is_file():
        supported_extensions = ['.json', '.jsonl', '.csv', '.tsv', '.txt', '.parquet']
        if not any(str(path).endswith(ext) for ext in supported_extensions):
            print(f"警告: 文件格式可能不被支持: {path.suffix}")
    
    return True


def load_local_dataset(dataset_path, split=None, tasks=None, **kwargs):
    """
    从本地路径加载数据集
    """
    validate_local_dataset_path(dataset_path)
    
    print(f"📁 从本地路径加载数据集: {os.path.abspath(dataset_path)}")
    
    path = Path(dataset_path)
    
    try:
        # 根据文件类型选择加载方式
        if path.is_dir():
            # 目录：自动检测文件类型
            json_files = list(path.glob('*.json')) + list(path.glob('*.jsonl'))
            if json_files:
                dataset = load_dataset('json', data_files=str(json_files[0]), **kwargs)
            else:
                csv_files = list(path.glob('*.csv'))
                if csv_files:
                    dataset = load_dataset('csv', data_files=str(csv_files[0]), **kwargs)
                else:
                    parquet_files = list(path.glob('*.parquet'))
                    if parquet_files:
                        dataset = load_dataset('parquet', data_files=str(parquet_files[0]), **kwargs)
                    else:
                        raise ValueError(f"目录中未找到支持的数据格式: {dataset_path}")
        
        elif str(path).endswith(('.json', '.jsonl')):
            # JSON/JSONL文件
            dataset = load_dataset('json', data_files=str(path), **kwargs)
        
        elif str(path).endswith(('.csv', '.tsv')):
            # CSV/TSV文件
            dataset = load_dataset('csv', data_files=str(path), **kwargs)
        
        elif str(path).endswith('.parquet'):
            # Parquet文件
            dataset = load_dataset('parquet', data_files=str(path), **kwargs)
        
        elif str(path).endswith('.txt'):
            # 文本文件
            dataset = load_dataset('text', data_files=str(path), **kwargs)
        
        else:
            # 尝试作为目录加载
            dataset = load_dataset(str(path), **kwargs)
        
        # 处理split参数
        if split and hasattr(dataset, split):
            dataset = dataset[split]
        elif split and 'train' in dataset:
            dataset = dataset['train']
        elif hasattr(dataset, 'train'):
            dataset = dataset.train
        
        # 处理tasks过滤
        if tasks is not None and hasattr(dataset, 'filter'):
            print(f"🔍 过滤任务: {tasks}")
            def filter_by_tasks(example):
                return example.get('task') in tasks if 'task' in example else True
            dataset = dataset.filter(filter_by_tasks)
        
        return dataset
        
    except Exception as e:
        print(f"本地数据集加载失败: {e}")
        raise e


def load_remote_dataset(dataset_path, split=None, tasks=None, **kwargs):
    """
    从远程加载数据集
    """
    print(f"🌐 从远程加载数据集: {dataset_path}")
    
    # 处理tasks参数
    if tasks is not None:
        kwargs['tasks'] = tasks
    
    return load_dataset(dataset_path, split=split, **kwargs)


def smart_load_dataset(dataset_path, split=None, tasks=None, **kwargs):
    """
    智能加载数据集，支持本地路径和远程数据集
    """
    if is_local_dataset_path(dataset_path):
        try:
            return load_local_dataset(dataset_path, split=split, tasks=tasks, **kwargs)
        except Exception as e:
            print(f"本地数据集加载失败，尝试远程加载: {e}")
            # 如果本地加载失败，尝试远程加载
            return load_remote_dataset(dataset_path, split=split, tasks=tasks, **kwargs)
    else:
        return load_remote_dataset(dataset_path, split=split, tasks=tasks, **kwargs)


def get_dataset_info(dataset_path):
    """
    获取数据集信息
    """
    if is_local_dataset_path(dataset_path):
        abs_path = os.path.abspath(dataset_path)
        size = 0
        file_count = 0
        
        if os.path.exists(abs_path):
            if os.path.isdir(abs_path):
                for root, dirs, files in os.walk(abs_path):
                    for file in files:
                        filepath = os.path.join(root, file)
                        try:
                            size += os.path.getsize(filepath)
                            file_count += 1
                        except (OSError, IOError):
                            pass
            else:
                try:
                    size = os.path.getsize(abs_path)
                    file_count = 1
                except (OSError, IOError):
                    pass
        
        return {
            "type": "local",
            "path": abs_path,
            "exists": os.path.exists(abs_path),
            "size": size,
            "file_count": file_count
        }
    else:
        return {
            "type": "remote",
            "path": dataset_path,
            "exists": True  # 假设远程数据集存在
        }


def format_dataset_size(size_bytes):
    """
    格式化数据集大小显示
    """
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def print_dataset_info(dataset_path, tasks=None):
    """
    打印数据集信息
    """
    info = get_dataset_info(dataset_path)
    
    if info["type"] == "local":
        size_str = format_dataset_size(info["size"])
        print(f"📁 使用本地数据集: {info['path']}")
        print(f"   数据集大小: {size_str}")
        if info["file_count"] > 1:
            print(f"   文件数量: {info['file_count']}")
    else:
        print(f"🌐 使用远程数据集: {info['path']}")
    
    if tasks:
        print(f"🔍 过滤任务: {tasks}")


# 为了向后兼容，提供一个替换load_dataset的函数
def load_dataset_smart(*args, **kwargs):
    """
    智能版本的load_dataset，支持本地路径
    可以直接替换 datasets.load_dataset 使用
    """
    if len(args) > 0:
        dataset_path = args[0]
        split = kwargs.get('split', None)
        tasks = kwargs.get('tasks', None)
        
        # 移除自定义参数，避免传递给原始load_dataset
        custom_kwargs = kwargs.copy()
        if 'tasks' in custom_kwargs:
            del custom_kwargs['tasks']
        
        return smart_load_dataset(dataset_path, split=split, tasks=tasks, **custom_kwargs)
    else:
        # 如果没有提供数据集路径，直接调用原始函数
        return load_dataset(*args, **kwargs)
