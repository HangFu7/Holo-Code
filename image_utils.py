import io
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF

from ldpc_utils import SystematicLDPC

# ============================================================
# 1. 依赖库检查与模型缓存
# ============================================================

try:
    from compressai.zoo import bmshj2018_factorized
    COMPRESSAI_AVAILABLE = True
except ImportError:
    COMPRESSAI_AVAILABLE = False

_COMPRESSAI_MODEL_CACHE = {}
_LDPC_MODEL_CACHE = {}

# ============================================================
# 2. 基础辅助函数
# ============================================================

def set_random_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def pil_to_np_uint8(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def compute_psnr_pil(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = pil_to_np_uint8(img1).astype(np.float32)
    arr2 = pil_to_np_uint8(img2).astype(np.float32)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))

# ============================================================
# 3. 核心攻击函数
# ============================================================

def vae_attack(image, quality=3, device='cuda'):
    """
    使用 CompressAI 进行 VAE 压缩攻击。
    quality: 1 (强攻击/最糊) -> 8 (弱攻击/清晰)
    """
    if not COMPRESSAI_AVAILABLE:
        print("Warning: compressai not installed. Skipping VAE attack.")
        return image

    quality = max(1, min(int(quality), 8))
    model_key = f"bmshj2018_{quality}"

    if model_key not in _COMPRESSAI_MODEL_CACHE:
        print(f"Loading CompressAI Model: bmshj2018-factorized (q={quality})...")
        net = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
        _COMPRESSAI_MODEL_CACHE[model_key] = net

    net = _COMPRESSAI_MODEL_CACHE[model_key]

    img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(img_tensor)
        x_hat = out['x_hat'].clamp(0, 1)

    rec_img = transforms.ToPILImage()(x_hat.squeeze().cpu())
    return rec_img


def composite_crop_jpeg(image, crop_ratio, jpeg_quality, seed=None):
    """
    顺序：1. Random Crop (Masking/Pad Black) -> 2. JPEG Compression
    """
    if seed is not None:
        set_random_seed(seed)

    width, height = image.size
    img_np = np.array(image)

    new_w = int(width * crop_ratio)
    new_h = int(height * crop_ratio)

    if new_w >= width or new_h >= height or new_w <= 0 or new_h <= 0:
        img_masked_pil = image
    else:
        start_x = np.random.randint(0, width - new_w + 1)
        start_y = np.random.randint(0, height - new_h + 1)

        padded = np.zeros_like(img_np)
        padded[start_y:start_y + new_h, start_x:start_x + new_w] = \
            img_np[start_y:start_y + new_h, start_x:start_x + new_w]
        img_masked_pil = Image.fromarray(padded)

    buffer = io.BytesIO()
    img_masked_pil.save(buffer, format="JPEG", quality=int(jpeg_quality))
    buffer.seek(0)
    img_final = Image.open(buffer).convert("RGB").copy()
    buffer.close()

    return img_final


def crop_and_scale_attack(image, crop_ratio, seed=None):
    """
    中心区域保留/随机区域保留后再缩放回原尺寸，模拟 ROI zoom.
    """
    if seed is not None:
        set_random_seed(seed)

    width, height = image.size
    new_w = int(width * crop_ratio)
    new_h = int(height * crop_ratio)

    if new_w >= width or new_h >= height or new_w <= 0 or new_h <= 0:
        return image

    start_x = np.random.randint(0, width - new_w + 1)
    start_y = np.random.randint(0, height - new_h + 1)

    cropped = image.crop((start_x, start_y, start_x + new_w, start_y + new_h))
    resized = cropped.resize((width, height), Image.BILINEAR)
    return resized

# ============================================================
# 4. IoT bitstream transmission simulator + LDPC
# ============================================================

@dataclass
class IoTTransmissionStats:
    jpeg_quality: int
    packet_bytes: int
    severity: str
    channel_code: str
    total_packets: int
    erased_packets: int
    corrupted_packets: int
    effective_loss_rate: float
    decode_success: int
    psnr: float


def pil_to_jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    data = buffer.getvalue()
    buffer.close()
    return data


def jpeg_bytes_to_pil(data: bytes) -> Image.Image:
    buffer = io.BytesIO(data)
    img = Image.open(buffer).convert("RGB").copy()
    buffer.close()
    return img


def packetize_bytes(data: bytes, packet_bytes: int) -> List[bytes]:
    return [data[i:i + packet_bytes] for i in range(0, len(data), packet_bytes)]


def depacketize_bytes(packets: List[bytes]) -> bytes:
    return b"".join(packets)


def get_ge_params(severity: str) -> Tuple[float, float, float]:
    """
    返回：
    p_g2b: Good -> Bad
    p_b2g: Bad -> Good
    p_bad : 在 Bad 状态下发生损伤的概率
    """
    severity = severity.lower()
    if severity == "mild":
        return 0.02, 0.35, 0.35
    elif severity == "moderate":
        return 0.05, 0.25, 0.55
    elif severity == "severe":
        return 0.08, 0.18, 0.75
    else:
        raise ValueError(f"Unknown IoT severity: {severity}")


def get_ldpc_model(k=2048, m=1024, col_weight=3, max_iter=20, seed=0):
    key = (k, m, col_weight, max_iter, seed)
    if key not in _LDPC_MODEL_CACHE:
        _LDPC_MODEL_CACHE[key] = SystematicLDPC(
            k=k,
            m=m,
            col_weight=col_weight,
            max_iter=max_iter,
            seed=seed,
        )
    return _LDPC_MODEL_CACHE[key]


def channel_encode_bytes(
    data: bytes,
    scheme="none",
    ldpc_k=2048,
    ldpc_m=1024,
    ldpc_col_weight=3,
    ldpc_max_iter=20,
    seed=0,
):
    if scheme == "none":
        return data, {"scheme": "none"}
    elif scheme == "ldpc":
        model = get_ldpc_model(
            k=ldpc_k,
            m=ldpc_m,
            col_weight=ldpc_col_weight,
            max_iter=ldpc_max_iter,
            seed=seed,
        )
        coded_bytes, meta = model.encode_bytes(data)
        meta = dict(meta)
        meta["scheme"] = "ldpc"
        return coded_bytes, meta
    else:
        raise ValueError(f"Unknown channel coding scheme: {scheme}")

def ge_channel_on_llr_blocks(
    llr_blocks: np.ndarray,
    severity: str = "moderate",
    mode: str = "erasure",
    bit_flip_prob_bad: float = 0.02,
):
    """
    在 LDPC codeword 级别施加 burst 信道 (基于 LLR 软信息)
    llr_blocks: float64, 正常的 LLR (例如 0比特映射为+10，1比特映射为-10)
    """
    p_g2b, p_b2g, p_bad = get_ge_params(severity)

    rx_llr = llr_blocks.copy()
    state = "G"
    erased = 0
    corrupted = 0
    total = len(llr_blocks)

    for i in range(total):
        r = np.random.rand()
        if state == "G":
            if r < p_g2b:
                state = "B"
        else:
            if r < p_b2g:
                state = "G"

        if state == "B" and np.random.rand() < p_bad:
            if mode == "erasure":
                # LLR = 0.0 代表完全不确定（即擦除）
                rx_llr[i, :] = 0.0  
                erased += 1
            elif mode == "corruption":
                # 随机翻转某些位的符号 (误码)
                flip_mask = (np.random.rand(rx_llr.shape[1]) < bit_flip_prob_bad)
                rx_llr[i, flip_mask] *= -1.0
                corrupted += 1
            elif mode == "mixed":
                if np.random.rand() < 0.5:
                    rx_llr[i, :] = 0.0
                    erased += 1
                else:
                    flip_mask = (np.random.rand(rx_llr.shape[1]) < bit_flip_prob_bad)
                    rx_llr[i, flip_mask] *= -1.0
                    corrupted += 1
            else:
                raise ValueError(f"Unknown channel mode: {mode}")

    info = {
        "total_packets": total,
        "erased_packets": erased,
        "corrupted_packets": corrupted,
        "effective_loss_rate": (erased + corrupted) / max(1, total)
    }
    return rx_llr, info

def ge_channel_on_packets(
    packets: List[bytes],
    severity: str = "moderate",
    mode: str = "erasure",
) -> Tuple[List[bytes], Dict]:
    """
    mode:
      - 'erasure'   : 整包擦除为全零
      - 'corruption': 整包随机篡改部分字节
      - 'mixed'     : 两者混合
    """
    p_g2b, p_b2g, p_bad = get_ge_params(severity)

    out_packets = []
    state = "G"
    erased = 0
    corrupted = 0

    for pkt in packets:
        r = np.random.rand()
        if state == "G":
            if r < p_g2b:
                state = "B"
        else:
            if r < p_b2g:
                state = "G"

        pkt_mut = bytearray(pkt)

        if state == "B" and np.random.rand() < p_bad and len(pkt_mut) > 0:
            if mode == "erasure":
                pkt_mut = bytearray(len(pkt_mut))
                erased += 1

            elif mode == "corruption":
                n_flip = max(1, len(pkt_mut) // 16)
                n_flip = min(n_flip, len(pkt_mut))
                idxs = np.random.choice(len(pkt_mut), size=n_flip, replace=False)
                for idx in idxs:
                    pkt_mut[idx] ^= np.random.randint(1, 256)
                corrupted += 1

            elif mode == "mixed":
                if np.random.rand() < 0.5:
                    pkt_mut = bytearray(len(pkt_mut))
                    erased += 1
                else:
                    n_flip = max(1, len(pkt_mut) // 16)
                    n_flip = min(n_flip, len(pkt_mut))
                    idxs = np.random.choice(len(pkt_mut), size=n_flip, replace=False)
                    for idx in idxs:
                        pkt_mut[idx] ^= np.random.randint(1, 256)
                    corrupted += 1
            else:
                raise ValueError(f"Unknown channel mode: {mode}")

        out_packets.append(bytes(pkt_mut))

    info = {
        "total_packets": len(packets),
        "erased_packets": erased,
        "corrupted_packets": corrupted,
        "effective_loss_rate": (erased + corrupted) / max(1, len(packets))
    }
    return out_packets, info


def safe_decode_corrupted_jpeg(tx_bytes: bytes, fallback_img: Image.Image) -> Tuple[Image.Image, int]:
    """
    优先尝试直接解码 JPEG；
    如果失败，则回退为对原图做一次低质量 JPEG 重编码，
    以保证输出仍是合法图像，可供 inversion 使用。
    """
    try:
        rx_img = jpeg_bytes_to_pil(tx_bytes)
        print(f"[JPEG decode] success, bytes={len(tx_bytes)}")
        return rx_img, 1
    except Exception as e:
        print(f"[JPEG decode] failed, bytes={len(tx_bytes)}, err={repr(e)}")
        buffer = io.BytesIO()
        fallback_img.save(buffer, format="JPEG", quality=20)
        degraded = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB").copy()
        buffer.close()
        return degraded, 0


def iot_bitstream_attack(
    img: Image.Image,
    jpeg_quality: int = 50,
    packet_bytes: int = 1024,
    severity: str = "moderate",
    channel_mode: str = "mixed",
    channel_code: str = "none",
    ldpc_k: int = 2048,
    ldpc_m: int = 1024,
    ldpc_col_weight: int = 3,
    ldpc_max_iter: int = 20,
    seed: int = 0,
) -> Tuple[Image.Image, IoTTransmissionStats]:
    """
    Two modes:

    1) none:
       PIL -> JPEG bytes -> packetization -> GE packet channel -> depacketize
           -> JPEG decode/recover -> PIL

    2) ldpc:
       PIL -> JPEG bytes -> bits -> LDPC encode(bit-blocks)
           -> GE channel on LDPC bit-blocks
           -> LDPC decode(bit-blocks) -> bits -> JPEG bytes
           -> JPEG decode/recover -> PIL
    """
    # 1. Source coding
    src_bytes = pil_to_jpeg_bytes(img, quality=jpeg_quality)

    # ========================================================
    # Branch A: No channel coding
    # ========================================================
    if channel_code == "none":
        packets = packetize_bytes(src_bytes, packet_bytes=packet_bytes)

        rx_packets, info = ge_channel_on_packets(
            packets,
            severity=severity,
            mode=channel_mode
        )

        rx_src_bytes = depacketize_bytes(rx_packets)

        rx_img, decode_success = safe_decode_corrupted_jpeg(
            rx_src_bytes,
            fallback_img=img
        )

        # 1. 先计算第一次尝试解码出来的图的 PSNR
        psnr = compute_psnr_pil(img, rx_img)

        # ====================================================
        # 2. 【这里就是你要加的 PSNR 安全阀】
        # 如果 Pillow 强行解出了灰色烂图或花屏图（PSNR < 25），
        # 我们强行判定物理层崩溃，并启动低质量 Fallback。
        # ====================================================
        if psnr < 25.0:
            print(f"[Branch A Warning] Pillow parsed corrupted bits (PSNR={psnr:.2f}). Triggering Fallback.")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=20)
            rx_img = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB").copy()
            buffer.close()
            # 重新计算 Fallback 图的 PSNR
            psnr = compute_psnr_pil(img, rx_img)
            # 判定原码流解码失败
            decode_success = 0  
        # ====================================================

        # 3. 将最终的参数打包进 stats 统计对象中
        stats = IoTTransmissionStats(
            jpeg_quality=jpeg_quality,
            packet_bytes=packet_bytes,
            severity=severity,
            channel_code=channel_code,
            total_packets=info["total_packets"],
            erased_packets=info["erased_packets"],
            corrupted_packets=info["corrupted_packets"],
            effective_loss_rate=info["effective_loss_rate"],
            decode_success=decode_success,
            psnr=psnr,
        )
        return rx_img, stats

    # ========================================================
    # Branch B: LDPC channel coding (加入交织器与真实包级信道)
    # ========================================================
    elif channel_code == "ldpc":
        src_bits = np.unpackbits(np.frombuffer(src_bytes, dtype=np.uint8))

        model = get_ldpc_model(
            k=ldpc_k, m=ldpc_m, col_weight=ldpc_col_weight, max_iter=ldpc_max_iter, seed=seed
        )

        # 1. LDPC 编码
        coded_blocks, code_meta = model.encode_bit_blocks(src_bits)
        coded_bits_1d = coded_blocks.flatten()

        # 2. 【核心通信技术：交织器 (Interleaver)】
        # 打乱比特顺序，将连续的突发丢包打散成随机单比特错误，完美释放 LDPC 的威力！
        rng = np.random.RandomState(seed)
        interleaver_idx = rng.permutation(len(coded_bits_1d))
        deinterleaver_idx = np.argsort(interleaver_idx)

        interleaved_bits = coded_bits_1d[interleaver_idx]

        # 3. 映射为软判决 LLR
        tx_llr_1d = np.where(interleaved_bits == 0, 10.0, -10.0)

        # 4. 模拟真实基于 Packet 的突发信道 (1024 Bytes = 8192 bits/packet)
        packet_bits = packet_bytes * 8
        num_packets = int(np.ceil(len(tx_llr_1d) / packet_bits))

        rx_llr_1d = tx_llr_1d.copy()
        p_g2b, p_b2g, p_bad = get_ge_params(severity)
        state = "G"
        erased = corrupted = 0

        for i in range(num_packets):
            # 状态转移
            if state == "G":
                if np.random.rand() < p_g2b: state = "B"
            else:
                if np.random.rand() < p_b2g: state = "G"

            # 发生损伤
            if state == "B" and np.random.rand() < p_bad:
                start_idx = i * packet_bits
                end_idx = min((i + 1) * packet_bits, len(rx_llr_1d))

                if channel_mode == "erasure":
                    rx_llr_1d[start_idx:end_idx] = 0.0
                    erased += 1
                elif channel_mode == "corruption":
                    flip_mask = (np.random.rand(end_idx - start_idx) < 0.02)
                    rx_llr_1d[start_idx:end_idx][flip_mask] *= -1.0
                    corrupted += 1
                elif channel_mode == "mixed":
                    if np.random.rand() < 0.5:
                        rx_llr_1d[start_idx:end_idx] = 0.0
                        erased += 1
                    else:
                        flip_mask = (np.random.rand(end_idx - start_idx) < 0.02)
                        rx_llr_1d[start_idx:end_idx][flip_mask] *= -1.0
                        corrupted += 1

        info = {
            "total_packets": num_packets,
            "erased_packets": erased,
            "corrupted_packets": corrupted,
            "effective_loss_rate": (erased + corrupted) / max(1, num_packets)
        }

        # 5. 解交织 (De-interleave)
        rx_llr_deinterleaved = rx_llr_1d[deinterleaver_idx]

        # 6. 还原为 Blocks 并解码
        rx_llr_blocks = rx_llr_deinterleaved.reshape(-1, model.n)

        try:
            rx_bits = model.decode_llr_blocks(rx_llr_blocks, code_meta)
            rx_src_bytes = np.packbits(rx_bits).tobytes()
        except Exception as e:
            rx_src_bytes = b""

        # 7. 解码与 Fallback
        rx_img, decode_success = safe_decode_corrupted_jpeg(rx_src_bytes, fallback_img=img)
        psnr = compute_psnr_pil(img, rx_img)

        # 【追加安全阀】：如果 Pillow 强行解出了灰色烂图（PSNR < 25），说明物理层彻底崩溃
        # 我们强行判定解码失败，并启动低质量 Fallback，防止实验链完全崩坏
        if psnr < 25.0:
            print(f"[JPEG Warning] Pillow parsed corrupted bits (PSNR={psnr:.2f}). Triggering Fallback.")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=20)
            rx_img = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB").copy()
            buffer.close()
            psnr = compute_psnr_pil(img, rx_img)
            decode_success = 0  # 真实物理意义上，原码流已经报废

        stats = IoTTransmissionStats(
            jpeg_quality=jpeg_quality, packet_bytes=packet_bytes, severity=severity,
            channel_code=channel_code, total_packets=info["total_packets"],
            erased_packets=info["erased_packets"], corrupted_packets=info["corrupted_packets"],
            effective_loss_rate=info["effective_loss_rate"], decode_success=decode_success, psnr=psnr,
        )
        return rx_img, stats

# ============================================================
# 5. 主攻击入口函数
# ============================================================

def image_distortion(img, seed, args, vae_model=None):
    set_random_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --------------------------------------------------------
    # Priority 1: 复合攻击 (互斥)
    # --------------------------------------------------------
    if hasattr(args, 'composite_crop_jpeg') and args.composite_crop_jpeg is not None:
        try:
            vals = args.composite_crop_jpeg
            if isinstance(vals, list) and len(vals) == 2:
                c_ratio = float(vals[0])
                j_qual = int(vals[1])
                return composite_crop_jpeg(img, c_ratio, j_qual, seed)
        except Exception as e:
            print(f"Composite Attack Error: {e}")
            return img

    # --------------------------------------------------------
    # Priority 2: 独占型高级攻击 (互斥)
    # --------------------------------------------------------
    if hasattr(args, 'vae_attack') and args.vae_attack:
        quality = getattr(args, 'vae_quality', 3)
        try:
            return vae_attack(img, quality=quality, device=device)
        except Exception as e:
            print(f"VAE Attack Error: {e}")
            return img

    if hasattr(args, 'crop_scale_ratio') and args.crop_scale_ratio is not None:
        try:
            return crop_and_scale_attack(img, args.crop_scale_ratio, seed)
        except Exception as e:
            print(f"Crop-and-Scale Attack Error: {e}")
            return img

    # --------------------------------------------------------
    # Priority 3: IoT bitstream attack
    # --------------------------------------------------------
    if hasattr(args, 'iot_attack') and args.iot_attack:
        try:
            jpeg_quality = getattr(args, 'iot_jpeg_quality', 50)
            packet_bytes = getattr(args, 'iot_packet_bytes', 1024)
            severity = getattr(args, 'iot_severity', 'moderate')
            channel_mode = getattr(args, 'iot_channel_mode', 'mixed')
            channel_code = getattr(args, 'iot_channel_code', 'none')

            ldpc_k = getattr(args, 'iot_ldpc_k', 2048)
            ldpc_m = getattr(args, 'iot_ldpc_m', 1024)
            ldpc_col_weight = getattr(args, 'iot_ldpc_col_weight', 3)
            ldpc_max_iter = getattr(args, 'iot_ldpc_max_iter', 20)

            img, stats = iot_bitstream_attack(
                img,
                jpeg_quality=jpeg_quality,
                packet_bytes=packet_bytes,
                severity=severity,
                channel_mode=channel_mode,
                channel_code=channel_code,
                ldpc_k=ldpc_k,
                ldpc_m=ldpc_m,
                ldpc_col_weight=ldpc_col_weight,
                ldpc_max_iter=ldpc_max_iter,
                seed=seed,
            )

            args._last_iot_stats = stats

            print(
                f"[IoT] code={stats.channel_code}, q={stats.jpeg_quality}, pkt={stats.packet_bytes}, "
                f"sev={stats.severity}, pkts={stats.total_packets}, erased={stats.erased_packets}, "
                f"corr={stats.corrupted_packets}, loss={stats.effective_loss_rate:.3f}, "
                f"decode_success={stats.decode_success}, psnr={stats.psnr:.2f}"
            )

            #return img

        except Exception as e:
            print("=" * 80)
            print("[IoT Attack Error]")
            print(repr(e))
            print("=" * 80)
            args._last_iot_stats = None
            #return img

    # --------------------------------------------------------
    # Priority 4: 标准单一攻击 (可串行)
    # --------------------------------------------------------

    # 1. JPEG
    if args.jpeg_ratio is not None:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=int(args.jpeg_ratio))
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB").copy()
        buffer.close()

    # 2. Random Crop (Masking)
    if args.random_crop_ratio is not None:
        width, height = img.size
        img_np = np.array(img)
        new_w = int(width * args.random_crop_ratio)
        new_h = int(height * args.random_crop_ratio)

        new_w = max(1, min(new_w, width))
        new_h = max(1, min(new_h, height))

        start_x = np.random.randint(0, width - new_w + 1)
        start_y = np.random.randint(0, height - new_h + 1)

        padded = np.zeros_like(img_np)
        padded[start_y:start_y + new_h, start_x:start_x + new_w] = \
            img_np[start_y:start_y + new_h, start_x:start_x + new_w]
        img = Image.fromarray(padded)

    # 3. Random Drop (Masking Inner)
    if args.random_drop_ratio is not None:
        width, height = img.size
        img_np = np.array(img)
        new_w = int(width * args.random_drop_ratio)
        new_h = int(height * args.random_drop_ratio)

        new_w = max(1, min(new_w, width))
        new_h = max(1, min(new_h, height))

        start_x = np.random.randint(0, width - new_w + 1)
        start_y = np.random.randint(0, height - new_h + 1)

        img_np[start_y:start_y + new_h, start_x:start_x + new_w] = 0
        img = Image.fromarray(img_np)

    # 4. Resize
    if args.resize_ratio is not None:
        w, h = img.size
        new_size = max(1, int(w * args.resize_ratio))
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
            img_np[:, :, k][salt_mask] = 255
            img_np[:, :, k][pepper_mask] = 0
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
        perspective_aug = transforms.RandomPerspective(
            distortion_scale=args.perspective_scale,
            p=1.0,
            fill=0
        )
        img = perspective_aug(img)

    return img