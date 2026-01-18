import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from torchvision import transforms
from datasets import load_dataset 

# 导入你提供的 image_distortion
from image_utils import image_distortion

# 导入修改后的 Holo_Shading
from watermark import Holo_Shading 

# ==========================================
# 0. 配置攻击参数列表
# ==========================================
CROP_RATIOS = [0.7, 0.6, 0.5, 0.4, 0.3]
JPEG_QUALS  = [50, 40, 30, 20, 10]

# ==========================================
# 1. 内置 Dataset 加载函数
# ==========================================
def get_dataset(args):
    print(f"Loading dataset: {args.dataset_path}...")
    try:
        dataset = load_dataset(args.dataset_path)['train']
        prompt_key = 'text'
        return dataset, prompt_key
    except Exception as e:
        print(f"Warning: Failed to load dataset '{args.dataset_path}': {e}")
        print("Using dummy prompts for testing instead.")
        dummy_data = [{'text': 'a beautiful landscape with a mountain and a lake'} for _ in range(args.num)]
        return dummy_data, 'text'

# ==========================================
# 2. 辅助类：模拟 Args
# ==========================================
class AttackArgs:
    def __init__(self):
        self.jpeg_ratio = None
        self.random_crop_ratio = None
        self.random_drop_ratio = None
        self.gaussian_blur_r = None
        self.median_blur_k = None
        self.resize_ratio = None
        self.gaussian_std = None
        self.sp_prob = None
        self.brightness_factor = None
        self.contrast_factor = None
        self.translation_shift = None
        self.perspective_scale = None
        self.crop_scale_ratio = None
        self.composite_crop_jpeg = None
        self.vae_attack = False
        self.vae_quality = 3

# ==========================================
# 3. 辅助函数：图像预处理
# ==========================================
def transform_img_tensor(image, target_size=512):
    tform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    image = tform(image)
    return 2.0 * image - 1.0

# ==========================================
# 4. 核心消融逻辑
# ==========================================
def run_ablation(args):
    # --- 初始化 Pipeline ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Model: {args.model_path}...")
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = None

    # --- 准备 Inversion 用的 Embedding ---
    null_prompt = ''
    null_embeddings = pipe.get_text_embedding(null_prompt)

    # --- 加载数据集 ---
    dataset, prompt_key = get_dataset(args)

    # --- 定义对比模式 (只跑两个消融方案) ---
    modes = ['no_holo', 'no_ecc']
    
    # 结果容器结构: mode -> type -> param -> [acc_list]
    results = {}
    for m in modes:
        results[m] = {
            'crop': {r: [] for r in CROP_RATIOS},
            'jpeg': {q: [] for q in JPEG_QUALS}
        }

    print(f"\n{'='*60}")
    print(f"🔬 Running Ablation (No Holo vs No ECC)")
    print(f"   Samples: {args.num}")
    print(f"   Crop Ratios: {CROP_RATIOS}")
    print(f"   JPEG Qualities: {JPEG_QUALS}")
    print(f"{'='*60}\n")

    # --- 循环测试 ---
    for mode in modes:
        print(f"👉 Testing Variant: [{mode.upper()}]")
        
        # 初始化 Holo_Shading
        watermark = Holo_Shading(
            args.channel_copy, 
            args.hw_copy, 
            args.fpr, 
            args.user_number, 
            mode=mode 
        )

        for i in tqdm(range(args.num), desc=f"  Evaluating {mode}"):
            seed = args.gen_seed + i
            idx = i % len(dataset)
            current_prompt = dataset[idx][prompt_key]
            
            # 1. 生成 (Generation) - 每张图只生成一次，然后应用所有攻击
            init_latents_w = watermark.create_watermark_and_return_w()
            
            with torch.no_grad():
                outputs = pipe(
                    current_prompt,
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance_scale, 
                    num_inference_steps=args.num_inference_steps, 
                    height=args.image_length, 
                    width=args.image_length, 
                    latents=init_latents_w, 
                )
            image_w = outputs.images[0] # PIL Image

            # 2. 循环测试所有 Crop 参数
            for ratio in CROP_RATIOS:
                args_crop = AttackArgs()
                args_crop.random_crop_ratio = ratio
                img_crop = image_distortion(image_w.copy(), seed, args_crop)
                
                acc = invert_and_eval(pipe, watermark, img_crop, null_embeddings, args.num_inversion_steps, device)
                results[mode]['crop'][ratio].append(acc)

            # 3. 循环测试所有 JPEG 参数
            for qual in JPEG_QUALS:
                args_jpeg = AttackArgs()
                args_jpeg.jpeg_ratio = qual
                img_jpeg = image_distortion(image_w.copy(), seed, args_jpeg)
                
                acc = invert_and_eval(pipe, watermark, img_jpeg, null_embeddings, args.num_inversion_steps, device)
                results[mode]['jpeg'][qual].append(acc)

    # --- 打印结果 ---
    print_detailed_results(results, modes)

def invert_and_eval(pipe, watermark, image, text_embeddings, steps, device):
    """ 执行 Inversion 并提取水印 """
    img_tensor = transform_img_tensor(image).unsqueeze(0).to(text_embeddings.dtype).to(device)
    
    # Encode -> Inversion
    image_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings, 
        guidance_scale=1, 
        num_inference_steps=steps, 
    )
    
    # Decode
    res = watermark.eval_watermark(reversed_latents)
    if isinstance(res, tuple):
        return res[0] # 只取 Bit Accuracy
    return res

def print_detailed_results(results, modes):
    # 映射名称方便阅读
    name_map = {
        'no_holo': 'No Holo (Variant A)',
        'no_ecc':  'No ECC  (Variant B)'
    }

    print("\n" + "="*60)
    print("                ABLATION STUDY RESULTS")
    print("="*60)

    # --- Print JPEG Table ---
    print("\n[1] JPEG Compression Robustness (Bit Acc)")
    header = f"{'Metric':<10} | " + " | ".join([f"Q={q:<3}" for q in JPEG_QUALS])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for mode in modes:
        row_str = f"{name_map[mode]:<10} | "
        accs = []
        for q in JPEG_QUALS:
            avg_acc = np.mean(results[mode]['jpeg'][q])
            accs.append(f"{avg_acc:.4f}")
        print(row_str + " | ".join(accs))
    
    # --- Print Crop Table ---
    print("\n[2] Random Crop Robustness (Bit Acc)")
    header = f"{'Metric':<10} | " + " | ".join([f"R={r:<3}" for r in CROP_RATIOS])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for mode in modes:
        row_str = f"{name_map[mode]:<10} | "
        accs = []
        for r in CROP_RATIOS:
            avg_acc = np.mean(results[mode]['crop'][r])
            accs.append(f"{avg_acc:.4f}")
        print(row_str + " | ".join(accs))
        
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num', default=50, type=int, help="Sample count per variant")
    parser.add_argument('--model_path', default='./stable-diffusion-2-1-base')
    parser.add_argument('--dataset_path', default='./Stable-Diffusion-Prompts')
    parser.add_argument('--output_path', default='./output_ablation')
    
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=50, type=int)
    parser.add_argument('--gen_seed', default=2024, type=int)
    
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=6, type=int, help="Block size for ablation")
    parser.add_argument('--fpr', default=1e-6, type=float)
    parser.add_argument('--user_number', default=1000000, type=int)
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    run_ablation(args)