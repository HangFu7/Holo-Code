import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import random
import torchvision.transforms.functional as TF
import os
import io

# ============================================================
# 1. 依赖库检查与模型缓存
# ============================================================

try:
    from compressai.zoo import bmshj2018_factorized
    COMPRESSAI_AVAILABLE = True
except ImportError:
    COMPRESSAI_AVAILABLE = False

# 全局缓存，避免重复加载
_COMPRESSAI_MODEL_CACHE = {}

# ============================================================
# 2. 基础辅助函数
# ============================================================

def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def transform_img(image, target_size=512):
    tform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    image = tform(image)
    return 2.0 * image - 1.0

def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x

def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)

# ============================================================
# 3. 核心攻击函数 (VAE, Composite, ROI Zoom)
# ============================================================

def vae_attack(image, quality=3, device='cuda'):
    """
    [修正] 函数名改回 vae_attack 以兼容 run.py 的 import。
    使用 CompressAI 进行 VAE 压缩攻击。
    quality: 1 (强攻击/最糊) -> 8 (弱攻击/清晰)
    """
    if not COMPRESSAI_AVAILABLE:
        print("Error: compressai not installed. Skipping VAE attack.")
        return image

    # 范围限制 1-8
    quality = max(1, min(int(quality), 8))
    
    # 懒加载模型
    model_key = f"bmshj2018_{quality}"
    if model_key not in _COMPRESSAI_MODEL_CACHE:
        print(f"Loading CompressAI Model: bmshj2018-factorized (q={quality})...")
        net = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
        _COMPRESSAI_MODEL_CACHE[model_key] = net
    
    net = _COMPRESSAI_MODEL_CACHE[model_key]

    # PIL -> Tensor
    # CompressAI 需要输入为 Tensor (0-1)
    img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        out = net(img_tensor)
        x_hat = out['x_hat'].clamp(0, 1)

    # Tensor -> PIL
    rec_img = transforms.ToPILImage()(x_hat.squeeze().cpu())
    return rec_img


# [新增] 复合攻击函数：完全复用你提供的 Crop 和 JPEG 逻辑
# ============================================================
def composite_crop_jpeg(image, crop_ratio, jpeg_quality, seed=None):
    """
    顺序：1. Random Crop (Masking/Pad Black) -> 2. JPEG Compression
    """
    if seed is not None:
        set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    # --- Step 1: Random Crop (Masking) ---
    # 逻辑直接复制自你的单项攻击代码
    width, height = image.size
    img_np = np.array(image)
    
    new_w = int(width * crop_ratio)
    new_h = int(height * crop_ratio)
    
    # 边界检查
    if new_w >= width or new_h >= height:
        img_masked_pil = image
    else:
        start_x = np.random.randint(0, width - new_w + 1)
        start_y = np.random.randint(0, height - new_h + 1)
        
        padded = np.zeros_like(img_np)
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = img_np[start_y:start_y+new_h, start_x:start_x+new_w]
        img_masked_pil = Image.fromarray(padded)

    # --- Step 2: JPEG Compression ---
    # 逻辑直接复制自你的单项攻击代码 (使用 BytesIO)
    buffer = io.BytesIO()
    img_masked_pil.save(buffer, format="JPEG", quality=int(jpeg_quality))
    buffer.seek(0)
    img_final = Image.open(buffer).copy()
    buffer.close()
    
    return img_final

# ============================================================
# 4. 主攻击入口函数
# ============================================================

def image_distortion(img, seed, args, vae_model=None):
    set_random_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --------------------------------------------------------
    # Priority 1: 复合攻击 (Composite) - 互斥
    # --------------------------------------------------------
    if hasattr(args, 'composite_crop_jpeg') and args.composite_crop_jpeg is not None:
        try:
            # args.composite_crop_jpeg 应该是一个列表 [0.5, 30]
            vals = args.composite_crop_jpeg
            if isinstance(vals, list) and len(vals) == 2:
                c_ratio = float(vals[0])
                j_qual = int(vals[1])
                return composite_crop_jpeg(img, c_ratio, j_qual, seed)
        except Exception as e:
            print(f"Composite Attack Error: {e}")
            return img

    # --------------------------------------------------------
    # Priority 2: 独占型高级攻击 - 互斥
    # --------------------------------------------------------
    
    # VAE Attack (CompressAI)
    # 只要 args.vae_attack 为 True 且 args.vae_quality 被设置（默认3），就执行
    # 注意：这里不再使用传入的 vae_model (Stable Diffusion自带的)，而是用 CompressAI
    if hasattr(args, 'vae_attack') and args.vae_attack:
        quality = getattr(args, 'vae_quality', 3)
        try:
            # 直接调用上面改名后的 vae_attack
            return vae_attack(img, quality=quality, device=device)
        except Exception as e:
            print(f"VAE Attack Error: {e}")
            return img

    # Crop & Scale (ROI Zoom)
    if hasattr(args, 'crop_scale_ratio') and args.crop_scale_ratio is not None:
        return crop_and_scale_attack(img, args.crop_scale_ratio, seed)

    # --------------------------------------------------------
    # Priority 3: 标准单一攻击 (可串行)
    # --------------------------------------------------------

    # 1. JPEG
    if args.jpeg_ratio is not None:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=int(args.jpeg_ratio))
        buffer.seek(0)
        img = Image.open(buffer).copy()
        buffer.close()

    # 2. Random Crop (Masking)
    if args.random_crop_ratio is not None:
        width, height = img.size
        img_np = np.array(img)
        new_w = int(width * args.random_crop_ratio)
        new_h = int(height * args.random_crop_ratio)
        
        start_x = np.random.randint(0, width - new_w + 1)
        start_y = np.random.randint(0, height - new_h + 1)
        
        padded = np.zeros_like(img_np)
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = img_np[start_y:start_y+new_h, start_x:start_x+new_w]
        img = Image.fromarray(padded)

    # 3. Random Drop (Masking Inner)
    if args.random_drop_ratio is not None:
        width, height = img.size
        img_np = np.array(img)
        new_w = int(width * args.random_drop_ratio)
        new_h = int(height * args.random_drop_ratio)
        start_x = np.random.randint(0, width - new_w + 1)
        start_y = np.random.randint(0, height - new_h + 1)
        # 挖空中间
        img_np[start_y:start_y+new_h, start_x:start_x+new_w] = 0
        img = Image.fromarray(img_np)

    # 4. Resize
    if args.resize_ratio is not None:
        w, h = img.size
        new_size = int(w * args.resize_ratio)
        img = img.resize((new_size, new_size), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)

    # 5. Filters
    if args.gaussian_blur_r is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
    if args.median_blur_k is not None:
        img = img.filter(ImageFilter.MedianFilter(args.median_blur_k))

    # 6. Noises
    if args.gaussian_std is not None:
        img_np = np.array(img).astype(float)
        noise = np.random.normal(0, args.gaussian_std, img_np.shape) * 255.0
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

    if args.sp_prob is not None:
        img_np = np.array(img)
        h, w, c = img_np.shape
        mask = np.random.rand(h, w)
        salt_mask = mask < (args.sp_prob / 2)
        pepper_mask = mask > (1 - args.sp_prob / 2)
        for k in range(c):
            img_np[:,:,k][salt_mask] = 255
            img_np[:,:,k][pepper_mask] = 0
        img = Image.fromarray(img_np)

    # 7. Photometric
    if args.brightness_factor is not None and args.brightness_factor > 0:
        img = TF.adjust_brightness(img, args.brightness_factor)
    if args.contrast_factor is not None and args.contrast_factor > 0:
        img = TF.adjust_contrast(img, args.contrast_factor)
        
    # 8. Geometric
    if hasattr(args, 'translation_shift') and args.translation_shift is not None and args.translation_shift > 0:
        shift = args.translation_shift
        img = TF.affine(img, angle=0, translate=(shift, shift), scale=1.0, shear=0)

    if hasattr(args, 'perspective_scale') and args.perspective_scale is not None and args.perspective_scale > 0:
        perspective_aug = transforms.RandomPerspective(distortion_scale=args.perspective_scale, p=1.0, fill=0)
        img = perspective_aug(img)

    return img