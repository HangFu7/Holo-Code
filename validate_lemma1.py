import os
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from tqdm import tqdm

def compute_ecr_l2(diff_energy_map, box_top, box_left, box_h, box_w):
    """使用 L2 能量图计算 ECR"""
    total_energy = torch.sum(diff_energy_map).item()
    bbox_energy = torch.sum(
        diff_energy_map[box_top:box_top+box_h, box_left:box_left+box_w]
    ).item()
    
    if total_energy == 0: return 1.0
    return bbox_energy / total_energy

def validate_lemma1_statistical(model_path="./stable-diffusion-2-1-base", n_trials=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> Loading model from {model_path} to {device}...")
    
    # 【修复 1】固定 NumPy 种子，保证随机 Patch 位置绝对可复现
    np.random.seed(42)
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_path, scheduler=scheduler, torch_dtype=torch.float16
    ).to(device)
    pipe.safety_checker = None
    
    prompt = "A beautiful landscape photography, highly detailed, 8k"
    
    # 【强烈建议 4】使用 no_grad 避免无意义的显存占用和梯度计算波动
    with torch.no_grad():
        text_embeddings = pipe.get_text_embedding(prompt)
    
    to_tensor = transforms.ToTensor()
    forward_ecrs = []
    inverse_ecrs = []
    
    print(f"\n>>> Running {n_trials} reproducible trials for Lemma 1 Validation...")
    
    for i in tqdm(range(n_trials)):
        # 【修复 2】修正 randint 的上界 (exclusive)，确保能取到 48
        l_top = np.random.randint(0, 64 - 16 + 1)
        l_left = np.random.randint(0, 64 - 16 + 1)
        
        i_top = l_top * 8
        i_left = l_left * 8
        
        gen_z = torch.Generator(device=device).manual_seed(100 + i)
        z_clean = torch.randn((1, 4, 64, 64), generator=gen_z, device=device, dtype=torch.float16)
        
        eps = 0.08
        z_mask = z_clean.clone()
        z_mask[:, :, l_top:l_top+16, l_left:l_left+16] += eps * torch.randn(
            (1, 4, 16, 16), generator=gen_z, device=device, dtype=torch.float16
        )
        
        gen_pipe1 = torch.Generator(device=device).manual_seed(200 + i)
        gen_pipe2 = torch.Generator(device=device).manual_seed(200 + i)
        
        # 将所有的网络前向推断包裹在 no_grad 中
        with torch.no_grad():
            img_clean = pipe(prompt, num_inference_steps=20, latents=z_clean, generator=gen_pipe1).images[0]
            img_mask = pipe(prompt, num_inference_steps=20, latents=z_mask, generator=gen_pipe2).images[0]
            
            tensor_clean = to_tensor(img_clean)
            tensor_mask = to_tensor(img_mask)
            diff_x_energy = torch.sum((tensor_clean - tensor_mask)**2, dim=0)
            
            ecr_f = compute_ecr_l2(diff_x_energy, i_top, i_left, 128, 128)
            forward_ecrs.append(ecr_f)
            
            tensor_attack = tensor_clean.clone()
            tensor_attack[:, i_top:i_top+128, i_left:i_left+128] = 0.0
            
            input_clean = tensor_clean.unsqueeze(0).to(device, dtype=torch.float16)
            input_attack = tensor_attack.unsqueeze(0).to(device, dtype=torch.float16)
            
            z0_clean = pipe.get_image_latents(input_clean, sample=False)
            z0_attack = pipe.get_image_latents(input_attack, sample=False)
            
            z_prime_clean = pipe.forward_diffusion(
                latents=z0_clean, text_embeddings=text_embeddings, guidance_scale=1, num_inference_steps=20
            )
            z_prime_attack = pipe.forward_diffusion(
                latents=z0_attack, text_embeddings=text_embeddings, guidance_scale=1, num_inference_steps=20
            )
            
            diff_z_energy = torch.sum((z_prime_clean - z_prime_attack)**2, dim=1).squeeze(0).cpu().float()
            
            ecr_i = compute_ecr_l2(diff_z_energy, l_top, l_left, 16, 16)
            inverse_ecrs.append(ecr_i)

    # --- 统计汇总 ---
    # 【强烈建议 3】计算并输出面积基准线 (Baseline Area Fraction)
    baseline_forward = (128 * 128) / (512 * 512)
    baseline_inverse = (16 * 16) / (64 * 64)
    
    print("\n" + "="*50)
    print("📊 Statistical Results (over 10 randomized trials)")
    print("="*50)
    print(f"Baseline Area Fraction   : {baseline_forward:.2%} (Theoretical expectation for random diffusion)")
    print("-" * 50)
    print(f"Forward ECR (z_T -> x)   : {np.mean(forward_ecrs):.2%} ± {np.std(forward_ecrs):.2%}")
    print(f"Inverse ECR (x -> z'_T)  : {np.mean(inverse_ecrs):.2%} ± {np.std(inverse_ecrs):.2%}")

    # 绘图部分保持不变 (仅提取最后一次 trial 用于可视化)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(diff_x_energy.numpy(), cmap='hot', interpolation='nearest')
    axes[0].set_title(f'Forward Energy Map ($z_T \\to x$)\nSample ECR: {ecr_f:.2%} (Baseline: {baseline_forward:.2%})')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(diff_z_energy.numpy(), cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Inverse Energy Map ($x \\to z\'_T$)\nSample ECR: {ecr_i:.2%} (Baseline: {baseline_inverse:.2%})')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('lemma1_statistical_validation.pdf', dpi=300)
    print("\n✅ Validation complete! Sample heatmap saved to 'lemma1_statistical_validation.pdf'.")

if __name__ == "__main__":
    validate_lemma1_statistical()