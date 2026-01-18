import argparse
import copy
import os 
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
import math
from PIL import Image
from diffusers import AutoencoderKL
from image_utils import vae_attack

def main(args):
    # 1. 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 初始化调度器
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    
    # 3. 初始化 Pipeline
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # 4. 加载 CLIP (可选)
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # 5. 加载数据集
    dataset, prompt_key = get_dataset(args)

    # 6. 算法选择
    if args.algo == 'holo': 
        print(f"\n{'='*40}")
        print(f"🚀 Running Algorithm: SAGE Framework (Holo-Code)")
        print(f"{'='*40}\n")
        watermark = Holo_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    elif args.algo == 'gs':
        print(f"\n{'='*40}")
        print(f"📉 Running Algorithm: Gaussian Shading (Baseline)")
        print(f"{'='*40}\n")
        if args.chacha:
             watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
        else:
             watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
                
    # [新增] GS with Sync
    elif args.algo == 'gssync':
        print(f"\n{'='*40}")
        print(f"📈 Running Algorithm: GS + Blind Sync (Strong Baseline)")
        print(f"{'='*40}\n")
        # 这里默认用 Chacha 版本，因为那更接近 Holo 的随机化
        watermark = Gaussian_Shading_Sync(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    
    elif args.algo == 'prc':
        print(f"\n{'='*40}")
        print(f"🧩 Running Algorithm: PRC Watermark (Baseline)")
        print(f"{'='*40}\n")
        watermark = PRC_Watermark(args)
    
    else:
        raise ValueError(f"Unknown algorithm type: {args.algo}")
        

    # 7. 创建目录
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "prompt"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "image"), exist_ok=True)
    
    # 8. 准备 Embeddings
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    acc = []
    clip_scores = []
    
    # 加载 VAE 用于攻击
    vae_model = None
    if args.vae_attack:
        print(">>> Loading VAE model for Attack...")
        vae_model = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae").to(device)

    # ===================== 9. 开始循环 =====================
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        # --- A. 生成 (Watermarking) ---
        set_random_seed(seed)
        
        # [回退] 不再传递参数，使用类内部的默认值 (默认是 1011)
        init_latents_w = watermark.create_watermark_and_return_w()
        
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale, 
            num_inference_steps=args.num_inference_steps, 
            height=args.image_length, 
            width=args.image_length, 
            latents=init_latents_w, 
        )
        image_w = outputs.images[0]
        
        # 保存水印图
        image_w.save(os.path.join(args.output_path, "image", f"img_{i}_watermarked.png"))

        # --- B. 攻击 (Distortion) ---
        image_w_distortion = image_distortion(image_w, seed, args, vae_model=vae_model)
        
        # 判定攻击保存逻辑
        gs_attack_params = [
            args.jpeg_ratio,
            args.random_crop_ratio,
            args.random_drop_ratio,    
            args.gaussian_blur_r,
            args.median_blur_k,
            args.resize_ratio,         
            args.gaussian_std,
            args.sp_prob,
            args.brightness_factor,
            args.contrast_factor,
            args.translation_shift,
            args.perspective_scale,
            args.crop_scale_ratio,
            args.composite_crop_jpeg,
            1 if args.vae_attack else None
        ]
        
        is_attacked = False
        if any(x is not None for x in gs_attack_params): 
            is_attacked = True
    #    if args.translation_shift > 0: is_attacked = True
    #    if args.perspective_scale > 0: is_attacked = True

        if is_attacked:
            attack_filename = os.path.join(args.output_path, "image", f"img_{i}_attacked.png")
            try:
                image_w_distortion.save(attack_filename)
            except Exception as e:
                print(f"Warning: Failed to save attacked image: {e}")

        
        # --- C. 提取 (Inversion) ---
        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings, 
            guidance_scale=1, 
            num_inference_steps=args.num_inversion_steps, 
        )

        # --- D. 评估 (Evaluation) ---
        # [兼容处理] 这里的 eval_watermark 可能会返回 (acc, id) 或者 acc
        # 我们只取 acc，忽略 ID
        eval_result = watermark.eval_watermark(reversed_latents_w)
        
        if isinstance(eval_result, tuple):
            acc_metric = eval_result[0] # 如果是元组，取第一个元素(准确率)
        else:
            acc_metric = eval_result    # 如果是数值，直接用

        acc.append(acc_metric)
        
        # 简单的进度打印，不再打印ID
        # tqdm.write(f"Step {i}: Bit Acc: {acc_metric:.4f}")

        if args.reference_model is not None:
            score = measure_similarity([image_w], current_prompt, ref_model,
                                       ref_clip_preprocess, ref_tokenizer, device)
            clip_score = score[0].item()
        else:
            clip_score = 0
        clip_scores.append(clip_score)


    # 10. 统计最终指标
    tpr_detection, tpr_traceability = watermark.get_tpr()
    save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    
    # 基础参数
    parser.add_argument('--num', default=1000, type=int) 
    parser.add_argument('--image_length', default=512, type=int) 
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base') 

    # 生成参数
    parser.add_argument('--guidance_scale', default=7.5, type=float) 
    parser.add_argument('--num_inference_steps', default=50, type=int) 
    parser.add_argument('--gen_seed', default=0, type=int) 
    parser.add_argument('--num_inversion_steps', default=None, type=int) 

    # 水印参数
    parser.add_argument('--channel_copy', default=1, type=int) 
    parser.add_argument('--hw_copy', default=8, type=int)       
    parser.add_argument('--user_number', default=1000000, type=int) 
    parser.add_argument('--fpr', default=0.000001, type=float) 
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher') 

    # 路径
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--reference_model', default=None) 
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts') 

    # 攻击参数
    parser.add_argument('--jpeg_ratio', default=None, type=int) 
    parser.add_argument('--random_crop_ratio', default=None, type=float) 
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--translation_shift', type=int, default=None) # 改为 None
    parser.add_argument('--perspective_scale', type=float, default=None) # 改为 None
    parser.add_argument('--crop_scale_ratio', type=float, default=None, 
                    help='Ratio of image side length to keep before scaling back. E.g., 0.5 means 2x Zoom.')
    parser.add_argument('--composite_crop_jpeg', nargs=2, type=float, default=None,
                    help='[crop_ratio, jpeg_quality]')
    
    # 算法选择
    parser.add_argument('--algo', default='holo', choices=['gs', 'holo', 'prc', 'gssync'], 
                        help='Algorithm type: gs, holo, or prc')
    
    parser.add_argument('--run_name', default='test_run', type=str, help='Name of this experiment run')
    parser.add_argument('--contrast_factor', type=float, default=None)
    parser.add_argument('--vae_attack', action='store_true', help='Enable VAE Compression attack')
    parser.add_argument('--vae_quality', type=int, default=3, 
                    help='Quality for VAE Compression (1-8). Lower is stronger attack. Default 3.')
    

    args = parser.parse_args()
    
    # 路径拼接
    args.output_path = os.path.join(args.output_path, args.run_name)
    os.makedirs(args.output_path, exist_ok=True)

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
