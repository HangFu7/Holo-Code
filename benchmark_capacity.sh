# ================= 配置区域 =================
# 图片数量
NUM=50
# 路径
MODEL="./stable-diffusion-2-1-base"
DATA="./Stable-Diffusion-Prompts"

# 1. 容量设置 (对应的 hw_copy)
# 12 -> ~64 bits
# 6  -> ~260 bits (Standard)
# 3  -> ~1040 bits
HW_SETTINGS=("12" "6" "3")

# 2. Crop 攻击参数 (保留比例 rho)
# 0.1 (只剩10%, 极强) -> 0.6 (剩60%, 中等)
CROPS=("0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5")

# ================= 执行循环 =================

echo "开始容量权衡 (Capacity-Robustness Trade-off) 实验..."
echo "Holo-Code | Num: $NUM"

for HW in "${HW_SETTINGS[@]}"; do
    
    # 打印当前容量的标识
    if [ "$HW" == "12" ]; then CAP="64b"; fi
    if [ "$HW" == "6" ]; then CAP="260b"; fi
    if [ "$HW" == "3" ]; then CAP="1040b"; fi

    echo ""
    echo "=================================================="
    echo "测试容量: $CAP (hw_copy=$HW)"
    echo "=================================================="

    for RHO in "${CROPS[@]}"; do
        
        RUN_NAME="ABLATION_Cap${CAP}_Crop${RHO}"
        
        echo ">>> Running Crop Ratio: $RHO"
        
        # 运行并抓取 Acc
        python run_gaussian_shading.py \
            --run_name "$RUN_NAME" \
            --algo holo \
            --num "$NUM" \
            --model_path "$MODEL" \
            --dataset_path "$DATA" \
            --channel_copy 1 \
            --hw_copy "$HW" \
            --fpr 0.000001 \
            --random_crop_ratio "$RHO" \
            | grep "mean_acc"
            
    done
done

echo ""
echo "✅ 所有容量消融实验完成！"