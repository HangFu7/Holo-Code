import os
import pickle
import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import betainc
from functools import reduce
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

# =========================================================================
#  Optional PRC Modules (For Baseline Comparison)
# =========================================================================
try:
    from prc_core.prc import KeyGen, Encode, Detect, Decode
    import prc_core.pseudogaussians as prc_gaussians
    PRC_AVAILABLE = True
except ImportError:
    PRC_AVAILABLE = False
    print("⚠️ Warning: PRC modules not found. PRC_Watermark class will fail if used.")

# =========================================================================
#  Gaussian Shading (ChaCha20 Version) - Baseline 1
# =========================================================================
class Gaussian_Shading_chacha:
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength // (self.ch * self.hw * self.hw)

        # 投票阈值
        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        
        # 统计计数
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        
       # 阈值计算
        self.tau_onebit = None
        self.tau_bits = None
        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def stream_key_encrypt(self, sd):
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1, self.ch, self.hw, self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        w = self.truncSampling(m)
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor.cuda()

    def diffusion_inverse(self, watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        
        if correct >= self.tau_onebit: self.tp_onebit_count += 1
        if correct >= self.tau_bits: self.tp_bits_count += 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count


# =========================================================================
#  Gaussian Shading (Original XOR Version) - Baseline 2
# =========================================================================
class Gaussian_Shading:
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength // (self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = int(message[i])
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        self.key = torch.randint(0, 2, [1, 4, 64, 64]).cuda()
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1, self.ch, self.hw, self.hw)
        # XOR Randomization
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m)
        return w

    def diffusion_inverse(self, watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_m):
        reversed_m = (reversed_m > 0).int()
        reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        
        if correct >= self.tau_onebit: self.tp_onebit_count += 1
        if correct >= self.tau_bits: self.tp_bits_count += 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count


# =========================================================================
#  Auxiliary: Naive Coder for Ablation (Variant B: No ECC)
# =========================================================================
class NaiveCoder:
    """
    朴素重复编码器 (用于消融实验 Variant B: w/o ECC)
    不使用 Hamming 矩阵，而是简单的重复填充，用于证明 ECC 的编码增益。
    Input: 4 bits -> Output: 7 slots (利用重复填充)
    Mapping: [b0, b1, b2, b3] -> [b0, b1, b2, b3, b0, b1, b2]
    """
    def __init__(self, scale=1.0):
        self.scale = scale

    def encode(self, message_bits):
        # message_bits shape: (N, 4)
        # 重复前3位来填充7位空间: [b0,b1,b2,b3] + [b0,b1,b2]
        redundant = message_bits[:, :3] 
        c = np.concatenate([message_bits, redundant], axis=1) # Shape: (N, 7)
        return np.where(c == 0, -self.scale, self.scale)

    def decode_soft(self, noisy_blocks):
        # noisy_blocks shape: (N, 7)
        # 简单的平均解码 (Repetition Decoding)
        # b0 由 slot 0 和 slot 4 决定
        s0 = (noisy_blocks[:, 0] + noisy_blocks[:, 4]) / 2
        s1 = (noisy_blocks[:, 1] + noisy_blocks[:, 5]) / 2
        s2 = (noisy_blocks[:, 2] + noisy_blocks[:, 6]) / 2
        s3 = noisy_blocks[:, 3] # b3 只有1份，没有重复 (这也展示了没有ECC的脆弱性)
        
        scores = np.stack([s0, s1, s2, s3], axis=1)
        
        # Hard decision: >0 is 1, <0 is 0
        decoded_bits = (scores > 0).astype(int)
        
        # 模拟信心值 (用于兼容接口，消融实验主要看 Acc)
        confidence = np.abs(scores).mean(axis=1)
        
        return decoded_bits, np.mean(confidence)

# =========================================================================
#  Holo-Code (Modified for Ablation)
# =========================================================================

class LatticeCoder:
    """ 高性能 Hamming(7,4) 编解码器 (原版保持不变) """
    def __init__(self, scale=1.0):
        self.scale = scale
        self.G = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], dtype=int)
        
        self.msg_lookup = np.array([list(map(int, format(i, '04b'))) for i in range(16)], dtype=int)
        code_binary = np.dot(self.msg_lookup, self.G) % 2
        self.codewords = np.where(code_binary == 0, -scale, scale).astype(np.float32)

    def encode(self, message_bits):
        c = np.dot(message_bits, self.G) % 2
        return np.where(c == 0, -self.scale, self.scale)

    def decode_soft(self, noisy_blocks):
        scores = np.dot(noisy_blocks, self.codewords.T)
        best_indices = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1) 
        return self.msg_lookup[best_indices], np.mean(max_scores)


class Holo_Shading:
    # 1. 修改 __init__ 增加 mode 参数
    def __init__(self, ch_factor, hw_factor, fpr, user_number, 
                 secret_key=b'MySecretKey123456789012345678901', 
                 mode='full'):
        """
        mode: 
          'full': Holo-Code (Standard)
          'no_holo': Variant A (With ECC, No Scramble)
          'no_ecc': Variant B (With Scramble, No ECC/Hamming)
        """
        self.mode = mode  # <--- 保存模式
        self.ch = ch_factor
        self.hw = hw_factor
        
        # 1. Latent Params
        self.c, self.h, self.w = 4, 64, 64
        self.latentlength = self.c * self.h * self.w
        
        # 2. Key Generation
        seed = int.from_bytes(secret_key[:4], 'little') 
        self.rng = np.random.RandomState(seed)
        
        # A. Permutation (Holographic Scrambling)
        self.perm_indices = self.rng.permutation(self.latentlength)
        self.inv_perm_indices = np.argsort(self.perm_indices)
        
        # B. Whitening Mask
        self.whitening_mask = self.rng.randint(0, 2, self.latentlength)
        
        # 3. Capacity Setup
        self.pixels_per_patch = self.ch * self.hw * self.hw
        self.total_patches = self.latentlength // self.pixels_per_patch
        self.num_blocks = self.total_patches // 7
        self.used_patches = self.num_blocks * 7
        self.capacity_bits = self.num_blocks * 4
        
        # <--- 2. 根据 mode 选择编码器
        if self.mode == 'no_ecc':
            print(f"[{mode}] Initializing NaiveCoder (Repetition only)...")
            self.coder = NaiveCoder()
        else:
            print(f"[{mode}] Initializing LatticeCoder (Hamming 7,4)...")
            self.coder = LatticeCoder()
            
        self.marklength = self.capacity_bits
        
        print(f"[Holo-Code] Mode: {self.mode} | Capacity: {self.capacity_bits} bits")

        # 4. Threshold Calculation (保持不变)
        self.tau_onebit = None
        self.tau_bits = None
        for i in range(self.marklength):
            p_val = betainc(i+1, self.marklength-i, 0.5)
            fpr_onebit = p_val
            fpr_bits = p_val * user_number
            
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength
        
        if self.tau_onebit is None: self.tau_onebit = 1.0
        if self.tau_bits is None: self.tau_bits = 1.0

        self.gt_message = None 
        self.tp_onebit_count = 0
        self.tp_bits_count = 0

    def truncSampling_fast(self, message_bits_01):
        raw_noise = np.abs(np.random.randn(self.latentlength))
        z = np.where(message_bits_01 == 0, -raw_noise, raw_noise)
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        # 1. 随机生成 Payload
        message_bits = np.random.randint(0, 2, self.capacity_bits)
        self.gt_message = message_bits

        # 2. Encoding (coder 已经根据 mode 替换了)
        msg_blocks = self.gt_message.reshape(self.num_blocks, 4)
        encoded_bpsk = self.coder.encode(msg_blocks).flatten()
        encoded_vals = (encoded_bpsk > 0).astype(int)
        
        # 3. Expansion
        expanded_vals = np.repeat(encoded_vals, self.pixels_per_patch)
        m_ordered = np.zeros(self.latentlength, dtype=int)
        limit = min(len(expanded_vals), self.latentlength)
        m_ordered[:limit] = expanded_vals[:limit]
        
        # 4. Scrambling (根据 mode 决定是否跳过)
        if self.mode == 'no_holo':
            # Variant A: 直接使用 ordered mapping
            m_scrambled = m_ordered 
        else:
            # Full & Variant B: 使用 Permutation
            m_scrambled = m_ordered[self.perm_indices]
        
        # 5. Whitening
        m_final = m_scrambled ^ self.whitening_mask
        
        # 6. Sampling
        w = self.truncSampling_fast(m_final)
        return w

    def attempt_decode(self, w_flat_numpy):
        # 1. De-Whitening
        whitening_correction = 1 - 2 * self.whitening_mask 
        w_corrected = w_flat_numpy * whitening_correction
        
        # 2. Inverse Permutation (根据 mode 决定是否跳过)
        if self.mode == 'no_holo':
            w_ordered_soft = w_corrected
        else:
            w_ordered_soft = w_corrected[self.inv_perm_indices]
        
        # 3. Pooling
        w_valid = w_ordered_soft[:self.used_patches * self.pixels_per_patch]
        w_reshaped = w_valid.reshape(self.num_blocks * 7, self.pixels_per_patch)
        symbol_means = np.mean(w_reshaped, axis=1)
        
        # 4. Decoding (coder 已经根据 mode 替换了)
        decoded_bits, confidence = self.coder.decode_soft(symbol_means.reshape(self.num_blocks, 7))
        return decoded_bits.flatten(), confidence

    # ... eval_watermark 和 get_tpr 保持不变 ...
    def eval_watermark(self, reversed_w):
        # (原代码 eval_watermark 逻辑无需修改，它调用的是 attempt_decode)
        # ... (此处省略原代码) ...
        # 请直接保留原来的 eval_watermark 代码块
        # 统一转到 CPU 进行几何搜索，避免设备冲突
        target_tensor = reversed_w.cpu().float()
        
        best_acc = 0.0
        best_confidence = -float('inf')
        
        # === 几何感知盲同步 (Geometry-Aware Blind Synchronization) ===
        
        # 1. Scale Search Space (抗 Resize / Perspective)
        scale_grid = [1.0]
        scale_grid.extend(np.arange(0.9, 1.1, 0.02).tolist()) 
        scale_grid = sorted(list(set([round(s, 2) for s in scale_grid])), key=lambda x: abs(x-1.0))

        # 2. Shift Search Space (抗 Translation / Crop 对齐)
        shift_range = 1 
        
        original_h, original_w = 64, 64

        for scale in scale_grid:
            # --- Geometric Transform: Scale ---
            if scale != 1.0:
                # interpolate 需要 4D 输入 (N,C,H,W)
                w_rescaled = F.interpolate(
                    target_tensor, 
                    scale_factor=scale, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                w_rescaled = target_tensor

            # --- Geometric Alignment: Padding/Cropping ---
            curr_h, curr_w = w_rescaled.shape[2], w_rescaled.shape[3]
            canvas = torch.zeros(1, 4, original_h, original_w)
            
            # Center alignment logic
            dst_y = max(0, (original_h - curr_h) // 2)
            dst_x = max(0, (original_w - curr_w) // 2)
            src_y = max(0, (curr_h - original_h) // 2)
            src_x = max(0, (curr_w - original_w) // 2)
            
            h_len = min(original_h, curr_h)
            w_len = min(original_w, curr_w)
            
            try:
                canvas[:, :, dst_y:dst_y+h_len, dst_x:dst_x+w_len] = \
                    w_rescaled[:, :, src_y:src_y+h_len, src_x:src_x+w_len]
            except: continue

            # --- Geometric Transform: Shift ---
            for dy in range(-shift_range, shift_range + 1):
                for dx in range(-shift_range, shift_range + 1):
                    w_shifted = torch.roll(canvas, shifts=(dy, dx), dims=(2, 3))
                    w_flat = w_shifted.numpy().flatten()
                    
                    # --- Robust Decoding ---
                    decoded_bits, confidence = self.attempt_decode(w_flat)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        if self.gt_message is not None:
                            correct = np.sum(decoded_bits == self.gt_message)
                            best_acc = correct / len(self.gt_message)

        if best_acc >= self.tau_onebit: self.tp_onebit_count += 1
        if best_acc >= self.tau_bits: self.tp_bits_count += 1
        
        return best_acc

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count

# =========================================================================
#  PRC Watermark (Baseline 2)
# =========================================================================
class PRC_Watermark:
    def __init__(self, args):
        if not PRC_AVAILABLE:
            raise ImportError("PRC modules not installed.")
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n = 4 * 64 * 64
        self.fpr = args.fpr
        self.msg_len = 256  
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.key_dir = './keys'
        os.makedirs(self.key_dir, exist_ok=True)
        self.key_path = os.path.join(self.key_dir, f'prc_key_n{self.n}_msg{self.msg_len}_fpr{self.fpr}.pkl')
        
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                self.encoding_key, self.decoding_key = pickle.load(f)
        else:
            print(f"Generating new PRC keys...")
            self.encoding_key, self.decoding_key = KeyGen(
                self.n, message_length=self.msg_len, false_positive_rate=self.fpr
            )
            with open(self.key_path, 'wb') as f:
                pickle.dump((self.encoding_key, self.decoding_key), f)
        self.gt_message = None

    def create_watermark_and_return_w(self):
        self.gt_message = np.random.randint(0, 2, self.msg_len)
        prc_codeword = Encode(self.encoding_key, self.gt_message)
        init_latents = prc_gaussians.sample(prc_codeword)
        init_latents = init_latents.reshape(1, 4, 64, 64).to(self.device).to(torch.float16)
        return init_latents

    def eval_watermark(self, reversed_latents):
        latents = reversed_latents.detach().cpu().to(torch.float32).flatten()
        latents = (latents - latents.mean()) / (latents.std() + 1e-8)
        posteriors = torch.erf(latents / math.sqrt(2))
        
        try:
            is_detected = Detect(self.decoding_key, posteriors)
        except: is_detected = False
        if is_detected: self.tp_onebit_count += 1 
        
        bit_acc = 0.5 
        try:
            result = Decode(self.decoding_key, posteriors)
            if result is not None and result[0] is not None:
                recovered_msg, is_success = result
                if is_success: self.tp_bits_count += 1
                match_count = np.sum(recovered_msg == self.gt_message)
                bit_acc = match_count / self.msg_len
        except: pass
        return bit_acc

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count
    
    
    
# =========================================================================
#  Gaussian Shading with Blind Synchronization (Strong Baseline)
# =========================================================================
class Gaussian_Shading_Sync(Gaussian_Shading_chacha):
    """
    为了公平对比，给 GS 也加上同样的几何搜索机制。
    这能证明 Holo 的优势不仅仅来源于搜索，而是来源于全息拓扑。
    """
    def eval_watermark(self, reversed_w):
        # 统一转到 CPU
        target_tensor = reversed_w.cpu().float()
        
        best_acc = 0.0
        
        # 使用和 Holo 完全一样的搜索空间
        # Scale Search
        scale_grid = [1.0]
        # 如果你测试透视/缩放，这里也应该开启
        scale_grid.extend(np.arange(0.9, 1.1, 0.02).tolist()) 
        scale_grid = sorted(list(set([round(s, 2) for s in scale_grid])), key=lambda x: abs(x-1.0))

        # Shift Search
        shift_range = 1 
        
        original_h, original_w = 64, 64

        for scale in scale_grid:
            # --- Scale ---
            if scale != 1.0:
                w_rescaled = F.interpolate(
                    target_tensor, 
                    scale_factor=scale, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                w_rescaled = target_tensor

            # --- Alignment (Padding/Cropping) ---
            curr_h, curr_w = w_rescaled.shape[2], w_rescaled.shape[3]
            canvas = torch.zeros(1, 4, original_h, original_w)
            
            dst_y = max(0, (original_h - curr_h) // 2)
            dst_x = max(0, (original_w - curr_w) // 2)
            src_y = max(0, (curr_h - original_h) // 2)
            src_x = max(0, (curr_w - original_w) // 2)
            
            h_len = min(original_h, curr_h)
            w_len = min(original_w, curr_w)
            
            try:
                canvas[:, :, dst_y:dst_y+h_len, dst_x:dst_x+w_len] = \
                    w_rescaled[:, :, src_y:src_y+h_len, src_x:src_x+w_len]
            except: continue

            # --- Shift Loop ---
            for dy in range(-shift_range, shift_range + 1):
                for dx in range(-shift_range, shift_range + 1):
                    w_shifted = torch.roll(canvas, shifts=(dy, dx), dims=(2, 3))
                    
                    # --- GS 原始解码逻辑 ---
                    # 注意：GS 需要输入 int 类型
                    reversed_m = (w_shifted > 0).int()
                    reversed_sd = self.stream_key_decrypt(reversed_m.numpy().flatten())
                    reversed_watermark = self.diffusion_inverse(reversed_sd)
                    
                    # 计算准确率
                    correct = (reversed_watermark == self.watermark).float().mean().item()
                    
                    if correct > best_acc:
                        best_acc = correct

        # 更新统计 (只算最好的那次)
        if best_acc >= self.tau_onebit: self.tp_onebit_count += 1
        if best_acc >= self.tau_bits: self.tp_bits_count += 1
        
        return best_acc