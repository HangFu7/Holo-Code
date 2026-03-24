import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_run_lengths(mask):
    """计算连续 0 (被擦除位) 的游程长度。为简便忽略了边界游程。"""
    ones_indices = np.where(mask == 1)[0]
    if len(ones_indices) < 2:
        return np.array([])
    return np.diff(ones_indices) - 1

def lag_corr(mask, k):
    """计算滞后 k 阶的自相关系数，包含 NaN 防护"""
    if k >= len(mask): return 0.0
    c = np.corrcoef(mask[:-k], mask[k:])[0, 1]
    return float(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))

def validate_theorem2_paper_grade(n_trials=1000, target_rho=0.5):
    C, H, W = 4, 64, 64
    D = C * H * W
    
    print(f">>> Running {n_trials} trials for Theorem 2 Validation...")
    print(f">>> Latent Space: {C}x{H}x{W} (D={D}), Target Rho: {target_rho:.2f}\n")
    
    # 物理掩码设置
    mask_top = np.zeros((C, H, W), dtype=np.int8)
    mask_top[:, :H//2, :] = 1
    
    mask_center = np.zeros((C, H, W), dtype=np.int8)
    crop_h = int(np.round(H * np.sqrt(target_rho)))
    crop_w = int(np.round(W * np.sqrt(target_rho)))
    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2
    mask_center[:, start_h:start_h+crop_h, start_w:start_w+crop_w] = 1
    
    rho_top = mask_top.mean()
    rho_center = mask_center.mean()
    
    attacks = {
        'Top-Half Crop': (mask_top.flatten(), rho_top),
        'Center Crop': (mask_center.flatten(), rho_center)
    }
    
    lags = [1, 2, 4, 8, 16, 32, 64, 128]
    
    stats_records = {
        'Top-Half Crop': {'retention': [], 'autocorr': {k: [] for k in lags}, 'run_lengths': []},
        'Baseline (Top)': {'retention': [], 'autocorr': {k: [] for k in lags}, 'run_lengths': []},
        'Center Crop': {'retention': [], 'autocorr': {k: [] for k in lags}, 'run_lengths': []},
        'Baseline (Center)': {'retention': [], 'autocorr': {k: [] for k in lags}, 'run_lengths': []}
    }
    
    np.random.seed(42)
    
    # 【新增】微观逐位边缘概率追踪 (Per-index Marginal Probability)
    # 随机抽取 1024 个逻辑位进行长期追踪
    sample_idx = np.random.choice(D, size=1024, replace=False)
    freq_records = {
        'Top-Half Crop': np.zeros_like(sample_idx, dtype=np.float64),
        'Center Crop': np.zeros_like(sample_idx, dtype=np.float64)
    }
    
    for i in tqdm(range(n_trials)):
        pi = np.random.permutation(D)
        
        for name, (p_mask, actual_rho) in attacks.items():
            l_mask = p_mask[pi]
            
            # 【新增】累加被抽样逻辑比特的保留次数
            freq_records[name] += l_mask[sample_idx]
            
            stats_records[name]['retention'].append(l_mask.mean())
            for k in lags:
                stats_records[name]['autocorr'][k].append(lag_corr(l_mask, k))
            if i % 10 == 0:
                stats_records[name]['run_lengths'].extend(get_run_lengths(l_mask))
                
            # 【优化】更清晰的 Baseline 命名
            base_name = "Baseline (Top)" if name == "Top-Half Crop" else "Baseline (Center)"
            l_mask_random = np.random.choice([0, 1], size=D, p=[1-actual_rho, actual_rho])
            
            stats_records[base_name]['retention'].append(l_mask_random.mean())
            for k in lags:
                stats_records[base_name]['autocorr'][k].append(lag_corr(l_mask_random, k))
            if i % 10 == 0:
                stats_records[base_name]['run_lengths'].extend(get_run_lengths(l_mask_random))

    # --- 1. 打印宏观与相关性表格 ---
    print("\n" + "="*85)
    print(f"{'Attack / Baseline Type':<25} | {'Ret. Mean':<10} | Mean Autocorrelation at various Lags")
    lag_header = " | ".join([f"L-{k:<3}" for k in lags])
    print(f"{'':<25} | {'':<10} | {lag_header}")
    print("-" * 85)
    
    for attack in stats_records.keys():
        r_mean = np.mean(stats_records[attack]['retention'])
        lag_means = [np.mean(stats_records[attack]['autocorr'][k]) for k in lags]
        lag_str = " | ".join([f"{val:>7.4f}" for val in lag_means])
        print(f"{attack:<25} | {r_mean:<10.4f} | {lag_str}")
        
    # --- 2. 打印微观逐位边缘概率验证 ---
    print("\n" + "="*85)
    print("[Marginal Probability: Per-index Survival Rate over 1024 sampled bits]")
    for name in ['Top-Half Crop', 'Center Crop']:
        actual_rho = attacks[name][1]
        freq_records[name] /= n_trials # 计算经验保留率
        std_dev = freq_records[name].std()
        max_dev = np.max(np.abs(freq_records[name] - actual_rho))
        print(f"{name:<25} -> Target: {actual_rho:.4f} | Emp. Mean: {freq_records[name].mean():.4f} | Std: {std_dev:.4f} | Max Dev: {max_dev:.4f}")
        
    # --- 3. 打印游程长度几何分布偏差 ---
    print("\n" + "="*85)
    print("[Memoryless Property: Maximum CDF Deviation from Geometric Model]")
    
    for name in ['Top-Half Crop', 'Center Crop']:
        actual_rho = attacks[name][1]
        runs = np.array(stats_records[name]['run_lengths'])
        val, counts = np.unique(runs, return_counts=True)
        ecdf = np.cumsum(counts) / len(runs)
        
        theoretical_cdf = 1 - (1 - actual_rho)**(val + 1)
        max_deviation = np.max(np.abs(ecdf - theoretical_cdf))
        print(f"{name:<25} (rho={actual_rho:.4f}) -> Max CDF Deviation: {max_deviation:.4f}")

    # --- 画图部分保持不变 (略) ---
    plt.figure(figsize=(8, 5))
    y_top = [np.mean(stats_records['Top-Half Crop']['autocorr'][k]) for k in lags]
    y_base_top = [np.mean(stats_records['Baseline (Top)']['autocorr'][k]) for k in lags]
    plt.plot(lags, y_top, marker='o', label='Top-Half Crop', color='blue', linewidth=2)
    plt.plot(lags, y_base_top, linestyle='--', color='lightblue', label='Baseline (Top)', alpha=0.8)
    
    y_center = [np.mean(stats_records['Center Crop']['autocorr'][k]) for k in lags]
    y_base_center = [np.mean(stats_records['Baseline (Center)']['autocorr'][k]) for k in lags]
    plt.plot(lags, y_center, marker='s', label='Center Crop', color='red', linewidth=2)
    plt.plot(lags, y_base_center, linestyle='--', color='lightcoral', label='Baseline (Center)', alpha=0.8)
    
    plt.axhline(0, color='black', linewidth=1.5, linestyle=':')
    plt.xscale('log', base=2)
    plt.xticks(lags, [str(k) for k in lags])
    plt.xlabel('Lag ($k$)')
    plt.ylabel('Mean Autocorrelation')
    plt.title('Empirical Validation of Negligible Inter-bit Correlation')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig('theorem2_autocorrelation.pdf', dpi=300)
    print("\n✅ Validation complete! Plot saved as 'theorem2_autocorrelation.pdf'.")

if __name__ == "__main__":
    validate_theorem2_paper_grade()