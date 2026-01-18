import time
import torch
import numpy as np
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline

def main():
    # 1. 配置
    model_path = "./stable-diffusion-2-1-base"
    device = "cuda"
    num_inference_steps = 50
    test_rounds = 10  # 测试次数，取平均值
    
    print(f"正在加载模型: {model_path} ...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.safety_checker = None
    
    # 准备一个 Dummy Prompt
    prompt = "a photo of an astronaut riding a horse on mars"
    
    # 2. 预热 (Warm-up)
    # GPU 第一次运行需要编译内核，时间会很久，必须排除
    print("正在预热 GPU (Warm-up)...")
    _ = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    torch.cuda.synchronize() # 等待 GPU以此完成
    
    # 3. 正式测速
    print(f"开始测试 ({test_rounds} 次循环)...")
    times = []
    
    for i in range(test_rounds):
        # 必须使用 torch.cuda.synchronize() 确保计时准确
        torch.cuda.synchronize()
        start = time.time()
        
        # 执行生成
        _ = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed = (end - start) * 1000 # 转换为毫秒
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f} ms")
        
    # 4. 计算结果
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print("\n" + "="*40)
    print(f"Stable Diffusion Generation Time (50 steps)")
    print("="*40)
    print(f"Average Time: {avg_time:.2f} ms")
    print(f"Std Dev     : {std_time:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    main()