# ================= 配置区域 =================
# 1. 基础设置
NUM_IMAGES=100  # 建议 50 或 100 以获取稳定 Acc
MODEL_PATH="./stable-diffusion-2-1-base"
DATASET_PATH="./Stable-Diffusion-Prompts"

# 2. 攻击强度 (VAE Noise Std)
# 范围拉大到 2.0 以观察性能下降趋势
NOISE_LEVELS=("2.0" "1.5" "1.0" "0.5" "0.0")

# 3. 对比方案
# 注意：确保你的代码里 args.algo 支持这些名字
ALGOS=("gs" "prc" "holo")

# ================= 循环执行 =================

echo "开始 VAE Regeneration 鲁棒性测试..."

for ALGO in "${ALGOS[@]}"; do
    
    # 自动适配参数
    if [ "$ALGO" == "holo" ]; then
        HW_COPY=6
        CH_COPY=1
    else
        # GS, GS+Sync, PRC 都是 8x8 (256 bits)
        HW_COPY=8
        CH_COPY=1
    fi

    for NOISE in "${NOISE_LEVELS[@]}"; do
        
        RUN_NAME="Bench_VAE_${ALGO}_std${NOISE}"

        echo ""
        echo "--------------------------------------------------------"
        echo "▶ 正在运行: $ALGO | VAE Noise Std: $NOISE"
        echo "--------------------------------------------------------"

        # 运行 Python 脚本
        # grep 提取最后的结果行，方便你直接看数据
        python run_gaussian_shading.py \
            --run_name "$RUN_NAME" \
            --algo "$ALGO" \
            --num "$NUM_IMAGES" \
            --model_path "$MODEL_PATH" \
            --dataset_path "$DATASET_PATH" \
            --channel_copy "$CH_COPY" \
            --hw_copy "$HW_COPY" \
            --fpr 0.000001 \
            --vae_attack \
            --vae_noise_std "$NOISE" \
            | grep -E "tpr_detection|mean_acc" 

    done
done

echo ""
echo "✅ 所有 VAE 实验完成！"