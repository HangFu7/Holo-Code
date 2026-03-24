import torch
import os
os.environ['OMP_NUM_THREADS'] = '1'

# 从你的 watermark.py 导入 Holo_Shading 类
from watermark import Holo_Shading

print(">>> Initializing Holo_Shading...")
# 按你之前的设定初始化 (1通道, 6次复制, fpr=1e-6, user=1M)
watermark = Holo_Shading(1, 6, 0.000001, 1000000)

print(">>> Generating dummy latent tensor...")
# 生成一个假的 4x64x64 的特征图 (加个 batch size 1 更严谨: 1x4x64x64)
dummy_w = torch.randn(1, 4, 64, 64)

# 启动测速！
watermark.benchmark_sync(dummy_w)