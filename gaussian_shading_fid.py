import argparse
import copy
from tqdm import tqdm
import json
import torch
import os
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *
from io_utils import *
from image_utils import *
from pytorch_fid.fid_score import *
from watermark import *

def main(args):
    # 1. 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # 2. 加载本地 JSON 数据 (meta_data.json)
    print(f"Loading prompts from {args.prompt_file}...")
    with open(args.prompt_file) as f:
        dataset = json.load(f)
        image_list = dataset['images']
        annotation_list = dataset['annotations']
        
        # [新增] 简单的长度检查，防止 num 设置过大导致越界
        if args.num > len(image_list) or args.num > len(annotation_list):
            print(f"Warning: args.num ({args.num}) is larger than dataset size. Truncating to {min(len(image_list), len(annotation_list))}.")
            args.num = min(len(image_list), len(annotation_list))

    # 3. 初始化算法
    print(f"Initializing Algorithm: {args.algo}")
    
    # [关键修改] 强制覆盖 Holo 的 hw_copy，防止命令行传错
    if args.algo == 'holo':
        if args.hw_copy != 6:
            print(f"⚠️ Notice: Forcing hw_copy to 6 for Holo-Code (User input was {args.hw_copy})")
            args.hw_copy = 6
        watermark = Holo_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
        
    elif args.algo == 'gs':
        if args.chacha:
            watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
        else:
            watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
            
    elif args.algo == 'prc':
        watermark = PRC_Watermark(args)
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    # 4. 设置输出目录
    base_dir = os.path.join('./fid_outputs/coco', args.run_name)
    w_dir = os.path.join(base_dir, 'w_gen')
    os.makedirs(w_dir, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    print(f"Saving generated images to: {w_dir}")

    # 5. 生成循环
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        
        # A. 获取数据
        current_prompt = annotation_list[i]['caption']
        file_name = image_list[i]['file_name'] 
        
        # B. 准备保存路径 (后缀改为 .png)
        save_name = file_name.replace('.jpg', '.png')
        save_path = os.path.join(w_dir, save_name)
        
        # 如果文件已存在，跳过? (可选，这里先不跳过，保证每次都是新的)
        # if os.path.exists(save_path): continue

        # C. 生成水印图
        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()
        
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]
        image_w.save(save_path)

    # 6. 计算 FID
    print(f"\n>>> Calculating FID between:")
    print(f"    1. Ground Truth: {args.gt_folder}")
    print(f"    2. Generated:    {w_dir}")

    if not os.path.exists(args.gt_folder):
        print(f"Error: GT folder not found at {args.gt_folder}")
        return

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    
    # 调用 pytorch-fid
    try:
        fid_value = calculate_fid_given_paths(
            [args.gt_folder, w_dir],
            50,       # batch size
            device,
            2048,     # dims
            num_workers
        )
        print(f'\n{"="*40}')
        print(f'🚀 RESULT: {args.algo} | FID: {fid_value}')
        print(f'{"="*40}\n')

        # 保存结果到 log
        with open(os.path.join(args.output_path, 'official_fid_results.txt'), "a") as file:
            file.write(f'Algo: {args.algo} | Run: {args.run_name} | FID: {fid_value}\n')
            
    except Exception as e:
        print(f"❌ Error calculating FID: {e}")
        print("Images are saved, you can try calculating FID manually.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Official FID Calculation')
    
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--algo', default='gs', choices=['gs', 'holo', 'prc'])
    parser.add_argument('--num', default=5000, type=int)
    
    # 关键路径
    parser.add_argument('--prompt_file', default='./fid_outputs/coco/meta_data.json')
    parser.add_argument('--gt_folder', default='./fid_outputs/coco/ground_truth')
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    
    # 生成参数
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    
    # 水印参数
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int) # 注意：Holo代码里会强转为6
    parser.add_argument('--chacha', action='store_true')
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--user_number', default=1000000, type=int)
    
    # PRC 参数
    parser.add_argument('--msg_type', type=str, default='binary')
    parser.add_argument('--nbits', type=int, default=32)
    parser.add_argument('--scaling_i', type=float, default=1.0)
    parser.add_argument('--scaling_w', type=float, default=1.0)
    parser.add_argument('--secret_key', type=str, default='utf-8')

    args = parser.parse_args()
    main(args)