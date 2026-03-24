import numpy as np

def bytes_to_bits(data: bytes) -> np.ndarray:
    if len(data) == 0:
        return np.zeros((0,), dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) == 0:
        return b""
    pad = (-len(bits)) % 8
    if pad > 0:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

class SystematicLDPC:
    """
    轻量系统化 LDPC (Numpy 高度优化版)
      H = [A | I_m]
      c = [u | p], p = A u mod 2
    """
    def __init__(self, k=1024, m=1024, col_weight=3, max_iter=20, seed=0):
        self.k = int(k)
        self.m = int(m)
        self.n = self.k + self.m
        self.col_weight = int(col_weight)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

        self.A = self._build_sparse_A()
        self.H = self._build_H()

    def _build_sparse_A(self):
        rng = np.random.RandomState(self.seed)
        A = np.zeros((self.m, self.k), dtype=np.uint8)
        row_counts = np.zeros(self.m, dtype=np.int32)

        for j in range(self.k):
            candidate_rows = np.argsort(row_counts)[: max(self.col_weight * 4, self.col_weight)]
            chosen = rng.choice(candidate_rows, size=self.col_weight, replace=False)
            A[chosen, j] = 1
            row_counts[chosen] += 1

        return A

    def _build_H(self):
        I = np.eye(self.m, dtype=np.uint8)
        return np.concatenate([self.A, I], axis=1)

    def encode_bit_blocks(self, msg_bits: np.ndarray):
        msg_bits = np.asarray(msg_bits, dtype=np.uint8)
        orig_len = len(msg_bits)

        pad_len = (-orig_len) % self.k
        if pad_len > 0:
            msg_bits = np.concatenate([msg_bits, np.zeros(pad_len, dtype=np.uint8)])

        blocks = msg_bits.reshape(-1, self.k)
        
        # 🚀 优化 1：完全向量化的全矩阵编码，抛弃了原来低效的 for 循环
        if len(blocks) > 0:
            # parity = (blocks @ A^T) % 2
            parity = (blocks @ self.A.T) % 2
            coded_blocks = np.concatenate([blocks, parity], axis=1).astype(np.uint8)
        else:
            coded_blocks = np.zeros((0, self.n), dtype=np.uint8)

        meta = {
            "orig_len": orig_len,
            "pad_len": pad_len,
            "n_blocks": len(blocks),
            "k": self.k,
            "m": self.m,
            "n": self.n,
            "col_weight": self.col_weight,
            "max_iter": self.max_iter,
            "seed": self.seed,
        }
        return coded_blocks, meta

    def decode_block_min_sum_llr(self, channel_llr: np.ndarray) -> np.ndarray:
        """
        🚀 优化 2：纯 Numpy 矩阵运算版 Min-Sum BP 解码
        消灭了上千万次的 Python 内部循环和字典查询，速度提升上千倍！
        """
        channel_llr = np.asarray(channel_llr, dtype=np.float64)
        mask = self.H  # 形状 (m, n) 的掩码矩阵
        
        # 初始化 变量节点 到 校验节点 的消息 (V2C)
        V2C = channel_llr[np.newaxis, :] * mask
        
        hard = (channel_llr < 0).astype(np.uint8)
        
        for _ in range(self.max_iter):
            # --- 校验节点更新 Check to Variable (C2V) ---
            # 1. 计算符号
            signs = np.sign(V2C)
            signs[signs == 0] = 1.0
            signs = np.where(mask == 1, signs, 1.0) # 非连接边不影响符号积
            
            sign_prod = np.prod(signs, axis=1, keepdims=True)
            C2V_sign = sign_prod * signs 
            
            # 2. 计算幅值 (Min-Sum)
            abs_V2C = np.where(mask == 1, np.abs(V2C), np.inf)
            
            # 找到每行的最小幅值和索引
            min1_idx = np.argmin(abs_V2C, axis=1)
            min1_val = abs_V2C[np.arange(self.m), min1_idx]
            
            # 找到每行的次小幅值 (把最小值替换为无穷大再求最小)
            abs_V2C_copy = abs_V2C.copy()
            abs_V2C_copy[np.arange(self.m), min1_idx] = np.inf
            min2_val = np.min(abs_V2C_copy, axis=1)
            
            # 构建幅值矩阵 (除最小值所在列之外全取最小值，最小值所在列取次小值)
            C2V_mag = np.tile(min1_val[:, np.newaxis], (1, self.n))
            C2V_mag[np.arange(self.m), min1_idx] = min2_val
            
            # 组合 C2V 消息并应用掩码过滤掉不存在的边
            C2V = C2V_sign * C2V_mag
            C2V = C2V * mask
            
            # --- 变量节点更新 Variable to Check (V2C) ---
            C2V_sum = np.sum(C2V, axis=0) # 聚合所有进入变量节点的消息
            
            # 计算后验 LLR
            posterior = channel_llr + C2V_sum
            
            # 硬判决与早停机制 (Early Stopping)
            hard = (posterior < 0).astype(np.uint8)
            if np.all((self.H @ hard) % 2 == 0):
                return hard # 全部校验通过，提前结束！
                
            # 更新下一轮的 V2C (减去从当前校验节点传来的消息)
            V2C = channel_llr[np.newaxis, :] + C2V_sum[np.newaxis, :] - C2V
            V2C = V2C * mask
            
        return hard

    def decode_llr_blocks(self, llr_blocks: np.ndarray, meta: dict) -> np.ndarray:
        if llr_blocks.size == 0:
            return np.zeros((0,), dtype=np.uint8)

        decoded_blocks =[]
        for llr in llr_blocks:
            dec = self.decode_block_min_sum_llr(llr)
            decoded_blocks.append(dec[:self.k]) # 截取系统信息位

        msg_bits = np.concatenate(decoded_blocks, axis=0) if decoded_blocks else np.zeros((0,), dtype=np.uint8)
        msg_bits = msg_bits[:meta["orig_len"]]
        return msg_bits

    def encode_bytes(self, data: bytes):
        bits = bytes_to_bits(data)
        coded_blocks, meta = self.encode_bit_blocks(bits)
        coded_bits = coded_blocks.reshape(-1)
        coded_bytes = bits_to_bytes(coded_bits)
        meta = dict(meta)
        meta["coded_bits_len"] = len(coded_bits)
        return coded_bytes, meta