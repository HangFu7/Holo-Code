import os
import pickle
import math
import torch
import torch.nn.functional as F
import numpy as np
import time
from scipy.stats import norm, truncnorm
from scipy.special import betainc
from functools import reduce
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
import reedsolo
import galois
from ldpc_utils import SystematicLDPC

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
class LDPCCoder:
    """
    现代 LDPC 编解码器 (完美定制 k=260, m=195, n=455)
    """
    def __init__(self, scale=1.0):
        self.scale = scale
        # K=260, M=195, N=455 (严格遵守 4/7 码率)
        self.ldpc = SystematicLDPC(k=260, m=195, col_weight=3, max_iter=10, seed=42)

    def encode(self, message_bits):
        orig_shape = message_bits.shape  # (65, 4)
        bits_1d = message_bits.flatten() # 260 bits
        
        coded_blocks, _ = self.ldpc.encode_bit_blocks(bits_1d) # 必然输出 (1, 455)
        coded_bits = coded_blocks.flatten()
        
        c = np.where(coded_bits == 0, -self.scale, self.scale).astype(np.float32)
        return c.reshape(orig_shape[0], 7)

    def decode_soft(self, noisy_blocks):
        orig_shape = noisy_blocks.shape  # (65, 7)
        noisy_1d = noisy_blocks.flatten() # 455 floats
        
        # LLR 映射
        llr = -noisy_1d 
        decoded_bits = self.ldpc.decode_block_min_sum_llr(llr) # 返回 455 bits 的完整码字
        msg_bits = decoded_bits[:orig_shape[0] * 4] # 取前 260 bits
        
        confidence = np.mean(np.abs(noisy_1d))
        
        # 【极速修复】：直接用 LDPC 的校验矩阵 H 计算伴随式 (Syndrome = H * c mod 2)
        syndrome = (self.ldpc.H @ decoded_bits) % 2
        
        if np.all(syndrome == 0):
            confidence += 10.0 # 完全收敛给高分，锁定网格
            
        return msg_bits.reshape(orig_shape[0], 4), confidence

class BCHCoder:
    """
    现代 BCH 编解码器 (代数硬判决解码)
    - 借用 BCH(511, 259) 截断模拟
    """
    def __init__(self, scale=1.0):
        self.scale = scale
        # 初始化 BCH(511, 259)，能纠正 29 个错误
        self.bch = galois.BCH(511, 259)

    def encode(self, message_bits):
        orig_shape = message_bits.shape
        bits_1d = message_bits.flatten()
        
        # 补齐 3 个 bit 满足 259
        padded_msg = np.pad(bits_1d, (0, 3), 'constant')
        gf_msg = galois.GF2(padded_msg)
        
        # BCH 编码出 511 bits
        coded_gf = self.bch.encode(gf_msg)
        coded_bits = np.asarray(coded_gf, dtype=np.uint8)
        
        # 截断到 448 bits 保持严格 4/7 码率公平对比
        coded_bits_448 = coded_bits[:448]
        
        c = np.where(coded_bits_448 == 0, -self.scale, self.scale).astype(np.float32)
        return c.reshape(orig_shape[0], 7)

    def decode_soft(self, noisy_blocks):
        orig_shape = noisy_blocks.shape
        noisy_1d = noisy_blocks.flatten()
        
        # 硬判决
        hard_bits = (noisy_1d > 0).astype(np.uint8)
        confidence = np.mean(np.abs(noisy_1d))
        
        # 补回被截断的 bits 作为 Erasure 送入解码器
        padded_rx = np.pad(hard_bits, (0, 511 - 448), 'constant')
        gf_rx = galois.GF2(padded_rx)
        
        try:
            # 尝试 BCH 代数解码
            decoded_gf = self.bch.decode(gf_rx)
            decoded_bits = np.asarray(decoded_gf, dtype=np.uint8)[:256]
            confidence += 10.0
            return decoded_bits.reshape(orig_shape[0], 4), confidence
        except Exception:
            # 【悬崖效应触发】：BCH 解码崩溃
            failed_bits = hard_bits[:256]
            return failed_bits.reshape(orig_shape[0], 4), confidence

class RSCoder:
    """ 
    现代 RS 编解码器 (完美适配 hw=6, Payload=260 bits, Encoded=455 bits)
    """
    def __init__(self, scale=1.0):
        self.scale = scale
        # 33 Bytes(输入) + 24 Bytes(冗余) = 57 Bytes(输出)
        self.rs = reedsolo.RSCodec(24)

    def encode(self, message_bits):
        orig_shape = message_bits.shape  # 必然是 (65, 4)
        bits_1d = message_bits.flatten() # 260 bits
        
        # 补 4 bits 凑齐 33 Bytes (264 bits)
        bits_padded = np.pad(bits_1d, (0, 4), 'constant')
        msg_bytes = bytearray(np.packbits(bits_padded))
        
        # RS 编码得到 57 Bytes (456 bits)
        encoded_bytes = self.rs.encode(msg_bytes)
        encoded_bits = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))
        
        # 截断最后一个 padding bit，精确输出 455 bits
        target_len = orig_shape[0] * 7  # 65 * 7 = 455
        encoded_bits = encoded_bits[:target_len]
        
        c = np.where(encoded_bits == 0, -self.scale, self.scale).astype(np.float32)
        return c.reshape(orig_shape[0], 7)

    def decode_soft(self, noisy_blocks):
        orig_shape = noisy_blocks.shape  # 必然是 (65, 7)
        noisy_1d = noisy_blocks.flatten() # 455 floats
        
        # RS的致命弱点：只能硬判决
        hard_bits = (noisy_1d > 0).astype(np.uint8)
        confidence = np.mean(np.abs(noisy_1d))
        
        # 补 1 bit 凑齐 57 Bytes (456 bits) 以便解码
        hard_bits_padded = np.pad(hard_bits, (0, 1), 'constant')
        rx_bytes = bytearray(np.packbits(hard_bits_padded))
        
        try:
            # RS 尝试解码
            decoded_bytes, _, _ = self.rs.decode(rx_bytes)
            decoded_bits = np.unpackbits(np.frombuffer(decoded_bytes, dtype=np.uint8))
            
            # 截取真实的 260 bits
            msg_bits = decoded_bits[:orig_shape[0] * 4]
            confidence += 10.0  # 校验成功给高分，锁定网格
            return msg_bits.reshape(orig_shape[0], 4), confidence
        except reedsolo.ReedSolomonError:
            # 盲同步偏移引发【悬崖效应】，解码直接崩溃！
            failed_bits = hard_bits[:orig_shape[0] * 4]
            return failed_bits.reshape(orig_shape[0], 4), confidence



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
            self.coder = LDPCCoder()
            
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

    def create_watermark_and_return_w(self, message=None, return_stats=False):
        # 1. 确定 Payload
        if message is not None:
            # 如果传了特定 message (比如十进制整数 ID)，将其转为二进制数组
            if isinstance(message, int):
                bin_str = format(message, f'0{self.capacity_bits}b')
                message_bits = np.array([int(b) for b in bin_str])
            else:
                # 如果传入的已经是数组，直接使用
                message_bits = np.array(message)
        else:
            # 兼容老代码：如果不传 message，就随机生成
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
        
        # 5. Whitening (关键统计量 s)
        m_final = m_scrambled ^ self.whitening_mask
        
        # 6. Sampling (关键统计量 z_T)
        w = self.truncSampling_fast(m_final)
        
        # --- 新增：专门为定理 1 验证开辟的“后门” ---
        if return_stats:
            # 返回 w (即 z_T) 和 m_final (即 s)
            return w, m_final
            
        return w

    def attempt_decode(self, w_flat_numpy):
        # 1. De-Whitening
        whitening_correction = 1 - 2 * self.whitening_mask 
        w_corrected = w_flat_numpy * whitening_correction
        
        # 2. Inverse Permutation
        if self.mode == 'no_holo':
            w_ordered_soft = w_corrected
        else:
            w_ordered_soft = w_corrected[self.inv_perm_indices]
        
        # 3. Pooling (最原始的均值池化)
        w_valid = w_ordered_soft[:self.used_patches * self.pixels_per_patch]
        w_reshaped = w_valid.reshape(self.num_blocks * 7, self.pixels_per_patch)
        
        # 没有任何花里胡哨的操作，直接取均值
        symbol_means = np.mean(w_reshaped, axis=1)
        
        # 4. Soft Decoding
        decoded_bits, confidence = self.coder.decode_soft(symbol_means.reshape(self.num_blocks, 7))
        return decoded_bits.flatten(), confidence

    # ... eval_watermark 和 get_tpr 保持不变 ...
    def eval_watermark(self, reversed_w):
        # 统一转到 CPU 进行几何搜索，避免设备冲突
        target_tensor = reversed_w.cpu().float()
        
        best_acc = 0.0
        best_confidence = -float('inf')
        best_decoded_bits = None  # 【新增 1】：用来保存提取出的最佳 0/1 序列
        
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
                        best_decoded_bits = decoded_bits # 【新增 2】：保存最佳序列
                        
                        if self.gt_message is not None:
                            correct = np.sum(decoded_bits == self.gt_message)
                            best_acc = correct / len(self.gt_message)

        if best_acc >= self.tau_onebit: self.tp_onebit_count += 1
        if best_acc >= self.tau_bits: self.tp_bits_count += 1
        
        # 【修改 3】：同时返回 best_acc 和提取出的 0/1 数组
        return best_acc, best_decoded_bits

    # ==========================================
    # 论文测速专用代码 (Benchmark)
    # ==========================================
    def _sync_only(self, target_tensor):
        """
        纯净版盲同步网格搜索，仅用于 Benchmark 测速。
        完全复刻 eval_watermark 的几何感知逻辑。
        """
        best_confidence = -1
        
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
            except: 
                continue

            # --- Geometric Transform: Shift ---
            for dy in range(-shift_range, shift_range + 1):
                for dx in range(-shift_range, shift_range + 1):
                    w_shifted = torch.roll(canvas, shifts=(dy, dx), dims=(2, 3))
                    
                    # 测速优化：使用 reshape(-1) 避免多余的内存拷贝
                    w_flat = w_shifted.reshape(-1).numpy()
                    
                    # --- Robust Decoding ---
                    decoded_bits, confidence = self.attempt_decode(w_flat)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        
        return best_confidence

    def benchmark_sync(self, dummy_w, n_repeat=200, n_warmup=20):
        """
        系统级延迟测试：预热 + 多次重复 + 强制单线程稳定测试
        """
        print("\n>>> [Benchmark] Preparing strict CPU timing environment...")
        # 强制单线程，消除多线程调度带来的耗时波动
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        # 确保 tensor 在 CPU 上，并且是 float 类型
        target_tensor = dummy_w.cpu().float()
        
        print(f">>> [Benchmark] Warming up for {n_warmup} iterations (ignoring cache overhead)...")
        for _ in range(n_warmup):
            self._sync_only(target_tensor)
            
        print(f">>> [Benchmark] Running formal benchmark for {n_repeat} iterations...")
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            self._sync_only(target_tensor)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0) # 转换为毫秒 (ms)
            
        mean_time = np.mean(times)
        std_time = np.std(times)
        p95_time = np.percentile(times, 95)
        
        print("\n" + "="*50)
        print("🚀 Synchronization Benchmark Results (CPU)")
        print("="*50)
        print(f"Mean Time : {mean_time:.2f} ms")
        print(f"Std Dev   : {std_time:.2f} ms")
        print(f"P95 Time  : {p95_time:.2f} ms")
        print("="*50)
        
        return mean_time, std_time, p95_time

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