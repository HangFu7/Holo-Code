NUM_IMAGES=100
MODEL_PATH="./stable-diffusion-2-1-base"
DATASET_PATH="./Stable-Diffusion-Prompts"

# [修改] 使用 Quality 等级 (1=强攻击, 6=弱攻击)
QUALITIES=("5" "4" "3" "2" "1") 

ALGOS=("gs" "prc" "holo")

# ================= 循环执行 =================

echo "开始 CompressAI VAE 鲁棒性测试..."

for ALGO in "${ALGOS[@]}"; do
    
    if [ "$ALGO" == "holo" ]; then
        HW_COPY=6
        CH_COPY=1
    else
        HW_COPY=8
        CH_COPY=1
    fi

    for Q in "${QUALITIES[@]}"; do
        
        RUN_NAME="VAE_Q${Q}_${ALGO}"

        echo ""
        echo "--------------------------------------------------------"
        echo "▶ Algo: $ALGO | VAE Quality: $Q (Lower is Stronger)"
        echo "--------------------------------------------------------"

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
            --vae_quality "$Q" \
            | grep -E "tpr_detection|mean_acc" 

    done
done