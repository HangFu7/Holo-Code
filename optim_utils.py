import torch
from datasets import load_dataset  # 用于加载 HuggingFace 的数据集
from typing import Any, Mapping
import json
import numpy as np
import os
from statistics import mean, stdev  # 用于计算平均值和标准差


# ---------------------------------------------------------
# 1. 基础工具函数
# ---------------------------------------------------------
def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    # 打开并读取 JSON 文件，返回字典格式的数据
    with open(filename) as fp:
        return json.load(fp)


# ---------------------------------------------------------
# 2. 数据集获取函数 (核心)
# ---------------------------------------------------------
def get_dataset(args):
    """
    根据传入参数 args.dataset_path 决定加载哪个数据集，
    并返回数据集对象以及对应的 Prompt 键名。
    """
    # 情况 A: 如果路径中包含 'laion' (如 LAION-400M/5B)
    if 'laion' in args.dataset_path:
        # 加载数据集的 'train' 分支
        dataset = load_dataset(args.dataset)['train']
        # LAION 数据集中存放文本提示词的键名通常是 'TEXT'
        prompt_key = 'TEXT'
        
    # 情况 B: 如果路径中包含 'coco' (MS-COCO 数据集)
    elif 'coco' in args.dataset_path:
        # 这里代码写死了一个本地路径，读取 COCO 的元数据
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            # COCO 的结构通常包含 'annotations'
            dataset = dataset['annotations']
            # COCO 中存放文本描述的键名是 'caption'
            prompt_key = 'caption'
            
    # 情况 C: 默认情况 (通常用于 Stable Diffusion Prompts 数据集)
    # 这对应你在 run_gaussian_shading.py 中看到的默认参数 'Gustavosta/Stable-Diffusion-Prompts'
    else:
        # 加载 HuggingFace 数据集的 'test' 分支
        dataset = load_dataset(args.dataset_path)['test']
        # 该数据集中存放文本提示词的键名是 'Prompt'
        prompt_key = 'Prompt'
    
    # 返回数据集对象和对应的键名，主程序会用 dataset[i][prompt_key] 来获取 Prompt
    return dataset, prompt_key


# ---------------------------------------------------------
# 3. 结果保存函数 (核心)
# ---------------------------------------------------------
def save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores):
    """
    保存实验指标到文本文件。
    根据当前进行的攻击类型（如 JPEG、Crop 等），自动决定写入哪个文件名。
    """
    # 定义一个字典，映射 args 中的参数名到输出文件名
    # 例如：如果 args.jpeg_ratio 有值，说明正在做 JPEG 攻击，结果存入 Jpeg.txt
    names = {
        'jpeg_ratio': "Jpeg.txt",
        'random_crop_ratio': "RandomCrop.txt",
        'random_drop_ratio': "RandomDrop.txt",
        'gaussian_blur_r': "GauBlur.txt",
        'gaussian_std': "GauNoise.txt",
        'median_blur_k': "MedBlur.txt",
        'resize_ratio': "Resize.txt",
        'sp_prob': "SPNoise.txt",
        'brightness_factor': "Color_Jitter.txt"
    }
    
    # 默认文件名：Identity.txt (表示无攻击，即原图测试)
    filename = "Identity.txt"
    
    # 遍历字典，检查 args 中哪个攻击参数不为 None
    # 一旦发现某个参数被设置了（例如 jpeg_ratio=25），就将文件名改为对应的 Jpeg.txt
    for option, name in names.items():
        if getattr(args, option) is not None:
            filename = name

    # 1. 先构建字符串 (Construct the string first)
    # 如果计算了 CLIP Score (reference_model 不为空)
    if args.reference_model is not None:
        
        # 格式：TPR检测率 | TPR溯源率 | 平均Bit准确率 | Bit准确率标准差 | 平均CLIP分数 | CLIP分数标准差
        log_str = 'tpr_detection:' + str(tpr_detection / args.num) + '       ' + \
                  'tpr_traceability:' + str(tpr_traceability / args.num) + '       ' + \
                  'mean_acc:' + str(mean(acc)) + '       ' + 'std_acc:' + str(stdev(acc)) + '       ' + \
                  'mean_clip_score:' + str(mean(clip_scores)) + '       ' + 'std_clip_score:' + str(stdev(clip_scores)) + '       ' + \
                  '\n'
        
    # 如果没有计算 CLIP Score
    else:
        log_str = 'tpr_detection:' + str(tpr_detection / args.num) + '       ' + \
                  'tpr_traceability:' + str(tpr_traceability / args.num) + '       ' + \
                  'mean_acc:' + str(mean(acc)) + '       ' + 'std_acc:' + str(stdev(acc)) + '       ' + \
                  '\n'

    # 2. 打印到屏幕 (Print to console)
    print("\n" + "="*50)
    print("Experiment Results:")
    print(log_str.strip()) # 去掉末尾换行符打印更美观
    print("="*50 + "\n")

    # 3. 写入文件 (Write to file)
    # 使用 "a" 模式打开文件，这样新的实验结果会追加到文件末尾，不会覆盖旧结果
    with open(os.path.join(args.output_path, filename), "a") as file:
        file.write(log_str)





