import argparse
import json
import torch
import os
import random
import numpy as np
from tqdm import tqdm
# 保持引用一致，确保环境兼容
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *
from pytorch_fid.fid_score import *

# [关键] 手动定义随机种子函数，确保和 gaussian_shading_fid 行为一致且不报错
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    print(f"🚀 Starting CLEAN (No Watermark) Generation...")

    # 1. 初始化模型 (逻辑严格对齐)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 尝试加载 Scheduler，增加容错 (和你遇到的 OSError 有关)
    try:
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    except OSError:
        print("⚠️ Warning: Could not load scheduler config from subfolder. Trying default config...")
        scheduler = DPMSolverMultistepScheduler.from_config(args.model_path)

    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # 2. 加载数据 (逻辑严格对齐)
    print(f"Loading prompts from {args.prompt_file}...")
    with open(args.prompt_file) as f:
        dataset = json.load(f)
        image_list = dataset['images']
        annotation_list = dataset['annotations']
        
        # 长度截断检查
        real_num = min(len(image_list), len(annotation_list))
        if args.num > real_num:
            print(f"Warning: args.num ({args.num}) is larger than dataset size. Truncating to {real_num}.")
            args.num = real_num

    # 3. 设置输出目录
    base_dir = os.path.join('./fid_outputs/coco', args.run_name)
    w_dir = os.path.join(base_dir, 'w_gen')
    os.makedirs(w_dir, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    print(f"Generating {args.num} images to: {w_dir}")

    # 4. 生成循环
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        
        # A. 获取数据
        current_prompt = annotation_list[i]['caption']
        file_name = image_list[i]['file_name']
        
        # B. 路径处理
        save_name = file_name.replace('.jpg', '.png')
        save_path = os.path.join(w_dir, save_name)

        # C. 核心差异：生成纯净高斯噪声 (Standard Gaussian Noise)
        # 严格控制种子
        set_random_seed(seed)
        
        # 计算 Latent 形状 (Batch=1, Channels=4, H/8, W/8)
        latent_shape = (1, 4, args.image_length // 8, args.image_length // 8)
        
        # 直接使用 torch.randn 生成标准正态分布噪声 (Clean)
        init_latents = torch.randn(
            latent_shape,
            device=device,
            dtype=torch.float16
        )
        
        # D. 推理
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents, # 传入纯净噪声
        )
        image_w = outputs.images[0]
        image_w.save(save_path)

    # 5. 计算 FID
    print(f"\n>>> Calculating FID for CLEAN images...")
    print(f"    GT: {args.gt_folder}")
    print(f"    Gen: {w_dir}")

    if not os.path.exists(args.gt_folder):
        print("Error: GT folder not found.")
        return

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    
    try:
        fid_value = calculate_fid_given_paths(
            [args.gt_folder, w_dir],
            50,
            device,
            2048,
            num_workers
        )
        print(f'\n{"="*40}')
        print(f'✨ RESULT: CLEAN | FID: {fid_value}')
        print(f'{"="*40}\n')
        
        with open(os.path.join(args.output_path, 'official_fid_results.txt'), "a") as file:
            file.write(f'Algo: Clean | Run: {args.run_name} | FID: {fid_value}\n')
            
    except Exception as e:
        print(f"❌ Error calculating FID: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Official Clean FID Calculation')
    
    parser.add_argument('--run_name', default="Official_Clean")
    parser.add_argument('--num', default=5000, type=int)
    
    # 路径参数
    parser.add_argument('--prompt_file', default='./fid_outputs/coco/meta_data.json')
    parser.add_argument('--gt_folder', default='./fid_outputs/coco/ground_truth')
    parser.add_argument('--output_path', default='./output/')
    # 默认值改为本地相对路径
    parser.add_argument('--model_path', default='./stable-diffusion-2-1-base') 
    
    # 生成参数
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    args = parser.parse_args()
    main(args)