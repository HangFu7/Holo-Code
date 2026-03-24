import os
import time
import tracemalloc
import torch
from PIL import Image
from torchvision import transforms

# 强制禁用 GPU，模拟无显卡的边缘法证网关
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermark import Holo_Shading

print("🚀 启动纯 CPU 边缘网关环境...")
device = 'cpu'
MODEL_PATH = "./stable-diffusion-2-1-base"
TEST_IMAGE = "./output/comp_holo_Crop_0.5/image/img_0_attacked.png" # 随便找一张攻击过的图

scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')
pipe = InversableStableDiffusionPipeline.from_pretrained(MODEL_PATH, scheduler=scheduler, torch_dtype=torch.float32)
pipe.safety_checker = None
pipe = pipe.to(device)

watermark = Holo_Shading(1, 6, 0.000001, 1000000)

print("⏳ 开始性能与能耗剖析 (Profiling)...")
image_pil = Image.open(TEST_IMAGE).convert("RGB")
tform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor()])
image_tensor = (2.0 * tform(image_pil) - 1.0).unsqueeze(0).to(torch.float32).to(device)

# 开启内存追踪
tracemalloc.start()
start_time = time.time()

# 1. CPU 反演
image_latents_w = pipe.get_image_latents(image_tensor, sample=False)
tester_prompt = ''
text_embeddings = pipe.get_text_embedding(tester_prompt).to(device)

reversed_latents_w = pipe.forward_diffusion(
    latents=image_latents_w,
    text_embeddings=text_embeddings,
    guidance_scale=1,
    num_inference_steps=50,
)

# 2. CPU 提取与盲同步
_ = watermark.eval_watermark(reversed_latents_w)

total_time = time.time() - start_time
current, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

# 估算能耗: Intel Xeon Gold 6330 TDP = 205 W.  Energy (Joules) = Power (W) * Time (s)
TDP_WATTS = 205
energy_joules = TDP_WATTS * total_time

print("="*50)
print(f"📊 边缘网关法证提取性能剖析 (Edge Profiling)")
print(f"   CPU 提取总耗时 : {total_time:.2f} 秒")
print(f"   峰值内存占用   : {peak_mem / 1024 / 1024:.2f} MB")
print(f"   预估单次能耗   : {energy_joules:.2f} 焦耳 (Joules)")
print("="*50)