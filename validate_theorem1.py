import os
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from watermark import Holo_Shading

def validate_theorem1(n_trials=1000, D=16384):
    watermark = Holo_Shading(1, 6, 0.000001, 1000000)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    p_hat_list = []
    z_means, z_vars, z_skews, z_kurts = [], [], [], []
    ks_p_values_raw, ks_p_values_std = [], []
    qq_samples = [] # 【采纳建议3】准确重命名

    print(f">>> Running {n_trials} reproducible trials for Theorem 1 empirical validation (D={D})...")

    for i in tqdm(range(n_trials)):
        random_m = np.random.randint(0, 1000000)
        z_T, s = watermark.create_watermark_and_return_w(message=random_m, return_stats=True)
        
        # --- Target 1a: s distribution ---
        s_np = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
        uniq = np.unique(s_np)
        if np.all(np.isin(uniq, [0, 1])):
            s_binary = s_np.astype(np.float64)
        else:
            s_binary = (s_np > 0).astype(np.float64)
        p_hat_list.append(float(s_binary.mean()))
        
        # --- Target 1b: z_T Gaussianity ---
        z_flat = z_T.detach().cpu().numpy().flatten().astype(np.float64) if isinstance(z_T, torch.Tensor) else z_T.flatten().astype(np.float64)
            
        z_mean = float(np.mean(z_flat))
        z_std = float(np.std(z_flat, ddof=1))
        
        z_means.append(z_mean)
        z_vars.append(float(np.var(z_flat, ddof=1)))
        
        # 【采纳建议2】增加高阶统计矩：偏度与峰度
        z_skews.append(float(stats.skew(z_flat)))
        z_kurts.append(float(stats.kurtosis(z_flat, fisher=True)))
        
        # 【采纳建议1】双重 KS 检验 (Raw & Standardized)
        sub = np.random.choice(z_flat, 500, replace=False)
        _, p_raw = stats.kstest(sub, 'norm') # Against strict N(0,1)
        
        sub_std = (sub - z_mean) / z_std
        _, p_std = stats.kstest(sub_std, 'norm') # Shape sanity check
        
        ks_p_values_raw.append(p_raw)
        ks_p_values_std.append(p_std)
        
        if i % 100 == 0:
            qq_samples.extend(np.random.choice(z_flat, 1000, replace=False))

    # --- Statistics Summary ---
    p_hat_arr = np.array(p_hat_list)
    print(f"\n[Target 1a - Empirical Symmetry of s]")
    print(f"Mean of p_hat: {np.mean(p_hat_arr):.6f} (Expected near 0.5)")
    print(f"Std of p_hat: {np.std(p_hat_arr):.6f}")
    print(f"Max Deviation: {np.max(np.abs(p_hat_arr - 0.5)):.6f}")

    print(f"\n[Target 1b - Empirical Gaussianity of z_T]")
    print(f"Global Mean: {np.mean(z_means):.6f} (Expected near 0.0)")
    print(f"Global Variance (ddof=1): {np.mean(z_vars):.6f} (Expected near 1.0)")
    print(f"Skewness: {np.mean(z_skews):.6f} (Expected near 0.0)")
    print(f"Kurtosis: {np.mean(z_kurts):.6f} (Expected near 0.0)")
    
    alpha = 0.05
    print(f"KS pass rate raw (N(0,1)): {(np.array(ks_p_values_raw) > alpha).mean():.2%}")
    print(f"KS pass rate standardized (shape): {(np.array(ks_p_values_std) > alpha).mean():.2%}")
    print("Note: KS is reported as a supporting sanity check using subsampling.")

    plot_theorem1_results(p_hat_arr, np.array(qq_samples))

def plot_theorem1_results(p_hat, z_samples):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(p_hat, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0.5, color='darkred', linestyle='dashed', linewidth=2)
    ax1.set_title('Empirical Distribution of $\hat{p}$ (Ones Ratio)')
    
    stats.probplot(z_samples, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Modulated Latents $z_T$')
    
    plt.tight_layout()
    plt.savefig('theorem1_empirical_validation.png', dpi=300)
    print("\n✅ Empirical validation plots saved as 'theorem1_empirical_validation.png'")

if __name__ == "__main__":
    validate_theorem1()