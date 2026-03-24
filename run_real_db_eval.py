import os
import time
import torch
import numpy as np
import faiss
from PIL import Image
from torchvision import transforms

# 导入你自己的流水线与工具
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from watermark import Holo_Shading  

# ==========================================
# 1. 实验参数与真实路径配置
# ==========================================
NUM_USERS = 1_000_000   
ACTUAL_PAYLOAD_BITS = 260    

# FAISS 强制要求 8 的倍数，向上补齐到 264 bits (即 33 Bytes)
FAISS_PAYLOAD_BITS = ((ACTUAL_PAYLOAD_BITS + 7) // 8) * 8  
PAYLOAD_BYTES = FAISS_PAYLOAD_BITS // 8                    

MODEL_PATH = "./stable-diffusion-2-1-base"  

# 指向你刚才跑出来的真实受损图片
TEST_IMAGE_PATH = "./output/comp_holo_Crop_0.5/image/img_0_attacked.png"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TRUE_USER_ID = 42000  

# ==========================================
# 2. 初始化真实生成模型与还原水印真值
# ==========================================
print("\n🚀 [1/4] 加载 Stable Diffusion 管道与 Holo-Code，提取真实水印特征...")
scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_PATH, subfolder='scheduler')
pipe = InversableStableDiffusionPipeline.from_pretrained(MODEL_PATH, scheduler=scheduler, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe = pipe.to(device)

# 【核心逻辑】：还原生成 img_0 时的随机种子和参数状态！
# (如果你生成 img_0 时用的不是 seed=0，请务必改成你当时用的 seed)
set_random_seed(0) 

# 按位置传参，避免 kwargs 报错
watermark = Holo_Shading(1, 6, 0.000001, NUM_USERS)

# 强制跑一遍生成逻辑，让 watermark 肚子里产生这张图绝对真实的 Ground Truth
_ = watermark.create_watermark_and_return_w()

# 抓取真实的 bits (注意：如果你的类里变量不叫 gt_message，请自行修改)
real_bits = watermark.gt_message.flatten().astype(np.uint8)

# 补齐真实 bits 到 264 位
if len(real_bits) < FAISS_PAYLOAD_BITS:
    pad_length = FAISS_PAYLOAD_BITS - len(real_bits)
    real_bits = np.pad(real_bits, (0, pad_length), 'constant', constant_values=0)

real_bytes = np.packbits(real_bits)

# ==========================================
# 3. 构建百万级工业数据库，并植入“真凶”数据
# ==========================================
print(f"\n🚀 [2/4] 正在构建 {NUM_USERS} 万用户的 FAISS 二进制数据库...")
# 随机生成 100 万个用户的纯随机 payload
db_vectors = np.random.randint(0, 256, size=(NUM_USERS, PAYLOAD_BYTES), dtype=np.uint8)

# 🚨 偷天换日：将刚刚提取的真实特征，覆盖到第 42000 个位置！
db_vectors[TRUE_USER_ID] = real_bytes

# 初始化并构建 FAISS 二进制检索引擎
index = faiss.IndexBinaryFlat(FAISS_PAYLOAD_BITS)
index.add(db_vectors)
print(f"   ✅ 建库完成！当前库容: {index.ntotal} 条记录。真凶数据已潜入 ID: {TRUE_USER_ID}")

# ==========================================
# 4. 执行真实的图片反演与特征提取 (Real Inversion)
# ==========================================
print(f"\n🚀 [3/4] 读取受损图片，执行物理反演提取...")
print(f"   -> 加载图片: {TEST_IMAGE_PATH}")

image_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
tform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
])
image_tensor = (2.0 * tform(image_pil) - 1.0).unsqueeze(0).to(torch.float16).to(device)

image_latents_w = pipe.get_image_latents(image_tensor, sample=False)
tester_prompt = ''
text_embeddings = pipe.get_text_embedding(tester_prompt).to(device, dtype=torch.float16)

start_inv_time = time.time()
reversed_latents_w = pipe.forward_diffusion(
    latents=image_latents_w,
    text_embeddings=text_embeddings,
    guidance_scale=1,
    num_inference_steps=50,
)
inv_time = time.time() - start_inv_time
print(f"   ✅ 反演完成，耗时: {inv_time:.2f} 秒")

# 提取残破的 bit 序列
eval_result = watermark.eval_watermark(reversed_latents_w)
# 根据你代码的返回值结构动态解包
if isinstance(eval_result, tuple):
    acc, extracted_bits = eval_result
else:
    # 如果只返回了准确率，你需要去 watermark.py 里修改，让它把提取到的序列也 return 出来
    raise ValueError("eval_watermark 必须返回提取出的 bits 序列用于溯源检索！")

# 格式化提取出的 bits
extracted_bits = extracted_bits.flatten().astype(np.uint8)

# 对提取出来的序列进行补齐 (补齐到 264 位)
if len(extracted_bits) < FAISS_PAYLOAD_BITS:
    pad_length = FAISS_PAYLOAD_BITS - len(extracted_bits)
    extracted_bits = np.pad(extracted_bits, (0, pad_length), 'constant', constant_values=0)

extracted_bytes = np.packbits(extracted_bits)

# ==========================================
# 5. 真实的百万级 FAISS 盲同步撞库
# ==========================================
print(f"\n🚀 [4/4] 将真实提取的比特流送入 FAISS 数据库进行 100万用户比对...")

start_db_time = time.time()

query_vectors = np.expand_dims(extracted_bytes, axis=0)
distances, user_ids = index.search(query_vectors, 1)

best_user_id = user_ids[0, 0]
min_distance = distances[0, 0]
matched_acc = 1.0 - (min_distance / FAISS_PAYLOAD_BITS)

db_time = (time.time() - start_db_time) * 1000

print("\n" + "="*60)
print(f"🏆 真实系统端到端测试结果汇总 🏆")
print("="*60)
print(f"   真实的作恶用户 ID : {TRUE_USER_ID}")
print(f"   数据库锁定的 ID   : {best_user_id}")
print(f"   溯源是否成功      : {'✅ YES' if TRUE_USER_ID == best_user_id else '❌ NO'}")
print(f"   数据库提取准确率  : {matched_acc*100:.2f}%")
print(f"   FAISS 检索耗时    : {db_time:.2f} ms")
print("="*60)