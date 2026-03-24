import cv2
import numpy as np
import torch
from PIL import Image

class IoTBitstreamSimulator:
    def __init__(self, quality=90, seed_base=42):
        self.quality = quality
        self.seed_base = int(seed_base)
        self.rng = np.random.default_rng(self.seed_base)

    def reset_rng_for_ber(self, ber):
        # 加上 round，彻底杜绝浮点数造成的 seed 碰撞隐患
        seed = self.seed_base + int(round(ber * 1e9))
        self.rng = np.random.default_rng(seed)

    def simulate_transmission(self, img_input, ber):
        if isinstance(img_input, torch.Tensor):
            img_tensor = img_input.detach().cpu().float()
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            if img_np.min() < 0:
                img_np = (img_np + 1.0) / 2.0
            img_np = np.clip(img_np, 0.0, 1.0)
            img_uint8 = (img_np * 255.0 + 0.5).astype(np.uint8)

        elif isinstance(img_input, Image.Image):
            img_uint8 = np.array(img_input.convert("RGB"))
        else:
            raise ValueError("Unsupported input type. Use PyTorch Tensor or PIL Image.")

        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
        if not ok:
            return False, None

        enc = enc.reshape(-1).astype(np.uint8)
        bitstream = np.unpackbits(enc)

        if ber > 0:
            flip = (self.rng.random(bitstream.size) < ber).astype(np.uint8)
            bitstream = np.bitwise_xor(bitstream, flip)

        corrupted_bytes = np.packbits(bitstream).reshape(-1).astype(np.uint8)

        recovered_bgr = cv2.imdecode(corrupted_bytes, cv2.IMREAD_COLOR)
        if recovered_bgr is None:
            return False, None

        recovered_rgb = cv2.cvtColor(recovered_bgr, cv2.COLOR_BGR2RGB)
        return True, Image.fromarray(recovered_rgb)