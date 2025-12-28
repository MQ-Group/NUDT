import shutil
import random
from pathlib import Path

from utils.sse import sse_print

def sample_dataset(source_dir, target_dir, train_count, val_count, seed=None):
    """
    从源数据集中随机选取指定数量的样本创建小数据集
    
    Args:
        source_dir: 源数据集目录
        target_dir: 目标数据集目录
        train_count: 训练集选取数量
        val_count: 验证集选取数量
        seed: 随机种子，确保可重复性
    """
    if seed is not None:
        random.seed(seed)
    
    # 设置路径
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 定义子目录结构
    splits = ['train', 'val']
    
    # 创建目标目录结构
    for split in splits:
        (target_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 统计数据
    selected_counts = {'train': train_count, 'val': val_count}
    actual_counts = {'train': 0, 'val': 0}
    
    # print(f"从数据集 {source_dir} 中随机选取样本...")
    # print(f"训练集: 选取 {train_count} 个样本")
    # print(f"验证集: 选取 {val_count} 个样本")
    
    for split in splits:
        # 源目录路径
        src_img_dir = source_path / 'images' / split
        src_label_dir = source_path / 'labels' / split
        
        # 目标目录路径
        dst_img_dir = target_path / 'images' / split
        dst_label_dir = target_path / 'labels' / split
        
        # 获取所有图片文件
        image_files = list(src_img_dir.glob('*.png'))
        
        if not image_files:
            # print(f"警告: {src_img_dir} 中没有找到图片文件")
            
            event = "dataset_sample_validated"
            data = {
                "status": "failure",
                "message": f"{src_img_dir}中没有找到图片文件"
            }
            sse_print(event, data)
            continue
        
        # 如果请求数量大于实际数量，则使用全部
        n_select = min(selected_counts[split], len(image_files))
        
        if n_select < selected_counts[split]:
            # print(f"警告: {split} 集只有 {len(image_files)} 个样本，将选取全部")
            event = "dataset_sample_validated"
            data = {
                "status": "warning",
                "message": f"警告: {split}集只有{len(image_files)}个样本，将选取全部."
            }
            sse_print(event, data)
        
        # 随机选取样本
        selected_images = random.sample(image_files, n_select)
        
        # print(f"\n{split} 集:")
        # print(f"  总样本数: {len(image_files)}")
        # print(f"  选取数量: {n_select}")
        event = "dataset_sample_validated"
        if split == 'train':
            data = {
                "message": "正在执行数据集抽取...",
                "progress": 0,
                "log": f"[0%] 从{split}集{len(image_files)}张样本中抽取{n_select}张样本."
            }
        else:
            data = {
                "message": "正在执行数据集抽取...",
                "progress": 50,
                "log": f"[50%] 从{split}集{len(image_files)}张样本中抽取{n_select}张样本."
            }
        sse_print(event, data)
        
        # 复制选中的文件
        for img_path in selected_images:
            # 对应的标签文件路径
            label_path = src_label_dir / f"{img_path.stem}.txt"
            
            # 复制图片文件
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            
            # 复制标签文件（如果存在）
            if label_path.exists():
                shutil.copy2(label_path, dst_label_dir / label_path.name)
            else:
                # print(f"警告: 标签文件 {label_path} 不存在")
                event = "dataset_sample_validated"
                data = {
                    "status": "warning",
                    "message": f"标签文件{label_path}不存在"
                }
                sse_print(event, data)
        
        actual_counts[split] = n_select
    
    # 打印统计信息
    # print("\n" + "="*50)
    # print("数据集创建完成!")
    # print(f"目标路径: {target_path}")
    # print(f"训练集: {actual_counts['train']} 个样本")
    # print(f"验证集: {actual_counts['val']} 个样本")
    # print("="*50)
    
   