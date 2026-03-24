import os 
# 降低多线程调度开销，使测试更稳定
os.environ["OMP_NUM_THREADS"] = "1" 

import argparse
from tqdm import tqdm
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt

# --- 导入你的工程模块 ---
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

# --- 导入 IoT 仿真器 ---
from iot_bit_simulator import IoTBitstreamSimulator

def get_watermark_algo(args):
    """工厂函数：根据参数返回对应的新实例化水印对象"""
    if args.algo == 'holo': 
        return Holo_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    elif args.algo == 'gs':
        if args.chacha:
            return Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
        else:
            return Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    elif args.algo == 'gssync':
        return Gaussian_Shading_Sync(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    elif args.algo == 'prc':
        return PRC_Watermark(args)
    else:
        return Holo_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    
    print(f"\n{'='*50}")
    print(f"📡 Running: Robustness 3.1 - IoT Network Simulation")
    print(f"{'='*50}\n")

    print(">>> Loading SD Pipeline...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path, scheduler=scheduler, torch_dtype=dtype
    ).to(device)
    pipe.safety_checker = None

    print(">>> Initializing IoT Bitstream Simulator...")
    # [修正 1] 初始化时设定 base_seed，后续不再更改
    simulator = IoTBitstreamSimulator(quality=90, seed_base=42)

    dataset, prompt_key = get_dataset(args)
    
    img_out_dir = os.path.join(args.output_path, "visual_samples")
    os.makedirs(img_out_dir, exist_ok=True)
    
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt).to(device=device, dtype=dtype)

    ber_list =[0.0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    trials = args.num 
    
    # =====================================================================
    # PHASE 1: 预生成数据集并缓存完整状态 (Image + Watermark Object)
    # =====================================================================
    print(f"\n>>> [Phase 1] Pre-generating {trials} images and caching states...")
    dataset_cache =[]
    
    # 安全保护：避免 trials 超过 dataset 长度
    actual_trials = min(trials, len(dataset))
    
    for i in tqdm(range(actual_trials), desc="Caching"):
        seed = i + args.gen_seed
        set_random_seed(seed)
        current_prompt = dataset[i][prompt_key]
        
        wm_instance = get_watermark_algo(args)
        init_latents_w = wm_instance.create_watermark_and_return_w()
        
        if isinstance(init_latents_w, np.ndarray):
            init_latents_w = torch.from_numpy(init_latents_w)
        init_latents_w = init_latents_w.to(device=device, dtype=dtype)

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
        img_w = outputs.images[0]
        
        dataset_cache.append({
            'image': img_w,
            'wm_obj': wm_instance
        })

    # =====================================================================
    # PHASE 2: 开始各 BER 的攻击与提取
    # =====================================================================
    print("\n>>> [Phase 2] Evaluating IoT transmission robustness...")
    results =[]

    for ber in ber_list:
        # [修正 1] 正确调用信道 RNG 刷新，保证公平
        simulator.reset_rng_for_ber(ber)
        
        decode_success = 0
        acc_list =[]
        
        # 外部统计 TPR 计数器
        tp_det_total = 0
        tp_tra_total = 0
        
        # [修正 3] 标记当前 BER 是否已经保存了可视化样本
        saved_img_for_this_ber = False

        for i in tqdm(range(actual_trials), desc=f"BER={ber:.1e}"):
            img_w = dataset_cache[i]['image']
            
            # [修正 4] 取消 deepcopy，直接引用
            wm_eval = dataset_cache[i]['wm_obj']

            # IoT 传输仿真
            ok, recovered_pil = simulator.simulate_transmission(img_w, ber=ber)
            
            if not ok:
                continue # Outage (丢包导致解码彻底失败)
                
            decode_success += 1
            
            # [修正 3] 只保存第一张成功解码的图像
            if not saved_img_for_this_ber:
                filename = f"ber_{ber:.1e}.png"
                recovered_pil.save(os.path.join(img_out_dir, filename))
                saved_img_for_this_ber = True

            # 提取
            image_w_distortion = transform_img(recovered_pil).unsqueeze(0).to(device=device, dtype=dtype)
            with torch.no_grad():
                image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
                reversed_latents_w = pipe.forward_diffusion(
                    latents=image_latents_w,
                    text_embeddings=text_embeddings, 
                    guidance_scale=1, 
                    num_inference_steps=args.num_inversion_steps, 
                )

            # [修正 2] 增量统计 TPR (完美避免内部累加污染)
            det_before = getattr(wm_eval, 'tp_onebit_count', 0)
            tra_before = getattr(wm_eval, 'tp_bits_count', 0)

            eval_result = wm_eval.eval_watermark(reversed_latents_w)
            acc_metric = eval_result[0] if isinstance(eval_result, tuple) else eval_result
            acc_list.append(acc_metric)
            
            det_after = getattr(wm_eval, 'tp_onebit_count', 0)
            tra_after = getattr(wm_eval, 'tp_bits_count', 0)

            # 只要本次评估后计数值增加了，就说明当前图提取成功
            if det_after > det_before:
                tp_det_total += 1
            if tra_after > tra_before:
                tp_tra_total += 1
            
        # --- 计算统计指标 ---
        success_rate = decode_success / float(actual_trials)
        acc_cond = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.5
        # 综合准确率：考虑 outage 的影响
        acc_eff = (success_rate * acc_cond) + ((1.0 - success_rate) * 0.5)

        # 端到端的 TPR：直接除以总实验次数 (Outage 视为失败)
        tpr_detection = tp_det_total / float(actual_trials)
        tpr_traceability = tp_tra_total / float(actual_trials)

        results.append({
            "ber": ber,
            "decode_success_rate": success_rate,
            "acc_cond": acc_cond, 
            "acc_eff": acc_eff,   
            "tpr_detection": tpr_detection,       
            "tpr_traceability": tpr_traceability, 
            "num_success": int(decode_success),
            "trials": int(actual_trials),
        })
        
        print(f"    -->[BER={ber:.1e}] Outage: {1-success_rate:.1%}, Acc(Eff): {acc_eff:.4f}, Det-TPR: {tpr_detection:.4f}, Trace-TPR: {tpr_traceability:.4f}")

    # =====================================================================
    # PHASE 3: Save and Plot Results
    # =====================================================================
    print("\n>>> Saving IoT Simulation Results...")
    csv_path = os.path.join(args.output_path, "iot_robustness.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # [修正 5] 画一张极其精美且专业的双Y轴图
    bers = [r["ber"] for r in results]
    succ = [r["decode_success_rate"] for r in results]
    acc_eff = [r["acc_eff"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    # 针对 0.0 做 symlog 处理，避免 log(0) 报错
    ax1.set_xscale("symlog", linthresh=1e-6)
    
    # 绘制 Decode Success Rate (左轴)
    line1 = ax1.plot(bers, succ, marker="s", color='#d62728', linewidth=2, label="Decode Success Rate")
    ax1.set_xlabel("Residual Bit Error Rate (BER)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Decode Success Rate", color='#d62728', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, which="major", ls="--", alpha=0.5)

    # 绘制 Effective Bit Accuracy (右轴)
    ax2 = ax1.twinx()
    line2 = ax2.plot(bers, acc_eff, marker="o", color='#1f77b4', linewidth=2.5, label="Effective Bit Accuracy")
    ax2.set_ylabel("Watermark Bit Accuracy", color='#1f77b4', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim([0.45, 1.05])

    # 合并图例
    lines = line1 + line2
    labels =[l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', frameon=True, edgecolor='black')

    fig.tight_layout()
    plt.savefig(os.path.join(args.output_path, "iot_robustness_ber.pdf"), dpi=300)
    plt.close()
    
    print(f"✅ All Done! Results and plots saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IoT Network Simulation')
    parser.add_argument('--num', default=50, type=int, help='Trials per BER') 
    parser.add_argument('--image_length', default=512, type=int) 
    parser.add_argument('--model_path', default='./stable-diffusion-2-1-base') 
    parser.add_argument('--guidance_scale', default=7.5, type=float) 
    parser.add_argument('--num_inference_steps', default=50, type=int) 
    parser.add_argument('--gen_seed', default=0, type=int) 
    parser.add_argument('--num_inversion_steps', default=50, type=int) 
    parser.add_argument('--channel_copy', default=1, type=int) 
    parser.add_argument('--hw_copy', default=6, type=int)       
    parser.add_argument('--user_number', default=1000000, type=int) 
    parser.add_argument('--fpr', default=0.000001, type=float) 
    parser.add_argument('--output_path', default='./output/iot_simulation')
    parser.add_argument('--dataset_path', default='./Stable-Diffusion-Prompts') 
    
    parser.add_argument('--algo', default='holo', choices=['gs', 'holo', 'prc', 'gssync'], help='Algorithm type')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    
    args = parser.parse_args()
    main(args)