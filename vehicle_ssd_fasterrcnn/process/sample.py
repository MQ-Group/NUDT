import os
import shutil
import random

from sse.sse import sse_print

def sample(src_dir, dst_dir, train_sample_num=None, val_sample_num=None, random_seed=42):
    """
    从VOC2007数据集源目录随机抽取文件到目标目录，保持相同组织结构。
    
    参数:
    src_dir: 源目录路径，例如"VOC2007"
    dst_dir: 目标目录路径，例如"VOC2007_sampled"
    train_sample_num: 从train.txt中随机抽取的文件数量，如果为None则抽取全部
    val_sample_num: 从val.txt中随机抽取的文件数量，如果为None则抽取全部
    random_seed: 随机种子，用于可重复的随机抽样
    """
    # 设置随机种子以确保结果可重复
    random.seed(random_seed)
    
    # 定义源子目录路径
    annotations_src = os.path.join(src_dir, "Annotations")
    jpegimages_src = os.path.join(src_dir, "JPEGImages")
    imagesets_src = os.path.join(src_dir, "ImageSets", "Main")
    
    # 定义目标子目录路径
    annotations_dst = os.path.join(dst_dir, "Annotations")
    jpegimages_dst = os.path.join(dst_dir, "JPEGImages")
    imagesets_dst = os.path.join(dst_dir, "ImageSets", "Main")
    
    # 创建目标目录（如果不存在）
    os.makedirs(annotations_dst, exist_ok=True)
    os.makedirs(jpegimages_dst, exist_ok=True)
    os.makedirs(imagesets_dst, exist_ok=True)
    
    # 要处理的txt文件和对应的抽样数量
    txt_files_config = [
        ("train.txt", train_sample_num),
        ("val.txt", val_sample_num)
    ]
    
    for txt_file, sample_num in txt_files_config:
        txt_path = os.path.join(imagesets_src, txt_file)
        
        # 读取文件名列表（假设每行一个文件名，不带扩展名）
        with open(txt_path, 'r') as f:
            all_file_names = [line.strip() for line in f.readlines() if line.strip()]  # 去除空行
        

        # 随机抽样
        if sample_num <= 0:
            # print(f"警告: {txt_file} 的抽样数量 {sample_num} 无效，跳过")
            event = "dataset_sample"
            data = {
                "status": "warning",
                "message": f"警告: {txt_file} 抽样数量 {sample_num} 无效，跳过"
            }
            sse_print(event, data)
            continue
            
        if sample_num > len(all_file_names):
            # print(f"警告: {txt_file} 的抽样数量 {sample_num} 超过总文件数 {len(all_file_names)}，使用全部文件")
            event = "dataset_sample"
            data = {
                "status": "warning",
                "message": f"警告: {txt_file} 的抽样数量 {sample_num} 超过总文件数 {len(all_file_names)}，使用全部文件"
            }
            sse_print(event, data)
            selected_files = all_file_names
        else:
            # 随机抽取指定数量的文件
            selected_files = random.sample(all_file_names, sample_num)
            # print(f"{txt_file}: 从 {len(all_file_names)} 个文件中随机抽取了 {len(selected_files)} 个文件")
            event = "dataset_sample"
            if txt_file == 'train.txt':
                data = {
                    "message": "正在执行数据集抽取...",
                    "progress": 0,
                    "log": f"[0%] 从train集{len(all_file_names)}张样本中抽取{sample_num}张样本."
                }
            else:
                data = {
                    "message": "正在执行数据集抽取...",
                    "progress": 50,
                    "log": f"[50%] 从val集{len(all_file_names)}张样本中抽取{sample_num}张样本."
                }
            sse_print(event, data)
        
        # 复制对应的.xml和.jpg文件
        copied_count = 0
        for name in selected_files:
            src_xml = os.path.join(annotations_src, name + ".xml")
            dst_xml = os.path.join(annotations_dst, name + ".xml")
            src_jpg = os.path.join(jpegimages_src, name + ".jpg")
            dst_jpg = os.path.join(jpegimages_dst, name + ".jpg")
            
            # 复制XML文件
            if os.path.exists(src_xml):
                shutil.copy2(src_xml, dst_xml)  # 复制并保留元数据
                xml_copied = True
            else:
                # print(f"警告: 文件 {src_xml} 不存在，跳过")
                event = "dataset_sample"
                data = {
                    "status": "warning",
                    "message": f"警告: 文件 {src_xml} 不存在，跳过"
                }
                sse_print(event, data)
                xml_copied = False
            
            # 复制JPG文件
            if os.path.exists(src_jpg):
                shutil.copy2(src_jpg, dst_jpg)
                jpg_copied = True
            else:
                # print(f"警告: 文件 {src_jpg} 不存在，跳过")
                event = "dataset_sample"
                data = {
                    "status": "warning",
                    "message": f"警告: 文件 {src_jpg} 不存在，跳过"
                }
                sse_print(event, data)
                jpg_copied = False
            
            # 如果两个文件都成功复制，计数增加
            if xml_copied and jpg_copied:
                copied_count += 1
        
        # 在目标目录中生成新的txt文件，内容为抽取的文件名列表
        new_txt_path = os.path.join(imagesets_dst, txt_file)
        with open(new_txt_path, 'w') as f:
            for name in selected_files:
                f.write(name + "\n")
        
        # print(f"已处理 {txt_file}，成功复制了 {copied_count} 个文件对（.xml和.jpg）\n")

if __name__ == "__main__":
    # 示例用法
    source_directory = "VOC2007"  # 替换为您的VOC2007源目录路径
    target_directory = "VOC2007_sampled"  # 替换为您希望保存抽取文件的目标目录路径
    
    # 参数设置：从train.txt中随机抽取50个文件，从val.txt中随机抽取20个文件
    train_sample_count = 50
    val_sample_count = 20
    
    # 调用函数执行随机抽取
    extract_voc_files_randomly(
        src_dir=source_directory,
        dst_dir=target_directory,
        train_sample_num=train_sample_count,
        val_sample_num=val_sample_count,
        random_seed=42  # 设置随机种子以确保结果可重复
    )
    
    print("随机文件抽取完成。目标目录结构已创建，文件已复制，新的txt文件已生成。")
