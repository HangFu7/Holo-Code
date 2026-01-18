# ================= 配置区域 =================
# 图片数量: 50 (为了曲线平滑，建议跑50张)
NUM_IMAGES=100
MODEL_PATH="./stable-diffusion-2-1-base"
DATASET_PATH="./Stable-Diffusion-Prompts"

# 1. 固定几何攻击参数: Crop 0.4 (保留 40%)
# 这个参数是你刚才测出来效果最好的，保持不变
CROP_RATIO=0.4

# 2. 变化信号攻击参数: JPEG Quality
# 从 90 (轻微) 到 20 (极强)
JPEG_LEVELS=("40")

# 3. 对比方案
ALGOS=("gssync" "holo")

echo "开始复合攻击曲线测试: Fixed Crop $CROP_RATIO + Varying JPEG..."

for ALGO in "${ALGOS[@]}"; do
    
    if [ "$ALGO" == "holo" ]; then
        HW_COPY=6
    else
        HW_COPY=8
    fi

    for Q in "${JPEG_LEVELS[@]}"; do
        RUN_NAME="COMP_${ALGO}_C${CROP_RATIO}_J${Q}"
        
        echo ""
        echo ">>> Running $ALGO with Crop $CROP_RATIO + JPEG $Q"
        
        # 运行并提取关键指标
        # 结果会直接打印在屏幕上，请记录下来填入绘图脚本
        python run_gaussian_shading.py \
            --run_name "$RUN_NAME" \
            --algo "$ALGO" \
            --num "$NUM_IMAGES" \
            --model_path "$MODEL_PATH" \
            --dataset_path "$DATASET_PATH" \
            --channel_copy 1 \
            --hw_copy "$HW_COPY" \
            --fpr 0.000001 \
            --composite_crop_jpeg "$CROP_RATIO" "$Q" \
            | grep -E "tpr_detection|tpr_traceability|mean_acc"
    done
done

echo "✅ 实验完成，请记录数据画图。"