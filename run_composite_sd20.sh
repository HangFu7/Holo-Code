#!/bin/bash

# ==========================================
# SD 2.0 模型泛化性测试：云边复合攻击批量脚本
# ==========================================

NUM=100
# 指向我们刚刚下载好的 SD 2.0 目录
MODEL_PATH="./stable-diffusion-2-base" 
DATASET_PATH="./Stable-Diffusion-Prompts"

# 固定的 IoT 传输恶劣条件 (触发降级)
IOT_CODE="none"
IOT_SEV="moderate"

ALGOS=("holo" "gssync" "prc")

# 6 种毁天灭地的二次攻击
ATTACK_NAMES=("Crop_0.5" "Drop_0.8" "JPEG_20" "Blur_6" "Noise_0.05" "Bright_4")
ATTACK_PARAMS=(
    "--random_crop_ratio 0.5"
    "--random_drop_ratio 0.8"
    "--jpeg_ratio 20"
    "--gaussian_blur_r 6"
    "--gaussian_std 0.05"
    "--brightness_factor 4"
)

LOG_DIR="composite_logs_sd20"
mkdir -p $LOG_DIR
SUMMARY_CSV="composite_summary_sd20.csv"

# 初始化 CSV 表头
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "Model,Algorithm,Image_Attack,TPR_Det,TPR_Tra,Mean_Acc" > $SUMMARY_CSV
fi

echo "=========================================================="
echo "🚀 开始执行 [SD 2.0 泛化测试] 双重复合攻击"
echo "=========================================================="

for algo in "${ALGOS[@]}"; do

    if [ "$algo" == "holo" ]; then HW_COPY=6; else HW_COPY=8; fi

    for i in "${!ATTACK_NAMES[@]}"; do
        attack_name="${ATTACK_NAMES[$i]}"
        attack_param="${ATTACK_PARAMS[$i]}"
        
        RUN_NAME="comp_sd20_${algo}_${attack_name}"
        TEMP_LOG="${LOG_DIR}/${RUN_NAME}.log"
        
        echo "⏳ [$(date +'%H:%M:%S')] 运行 -> 模型: SD2.0 | 算法: $algo | 攻击: $attack_name"

        python run_holo_code.py \
            --algo $algo \
            --run_name $RUN_NAME \
            --num $NUM \
            --model_path $MODEL_PATH \
            --dataset_path $DATASET_PATH \
            --channel_copy 1 \
            --hw_copy $HW_COPY \
            --fpr 0.000001 \
            --iot_attack \
            --iot_channel_code $IOT_CODE \
            --iot_severity $IOT_SEV \
            --iot_channel_mode mixed \
            $attack_param \
            > "$TEMP_LOG" 2>&1

        # 提取核心指标
        RES_LINE=$(grep "tpr_detection:" "$TEMP_LOG" | tail -n 1)
        TPR_DET=$(echo "$RES_LINE" | sed -n 's/.*tpr_detection:\([0-9.]*\).*/\1/p')
        TPR_TRA=$(echo "$RES_LINE" | sed -n 's/.*tpr_traceability:\([0-9.]*\).*/\1/p')
        MEAN_ACC=$(echo "$RES_LINE" | sed -n 's/.*mean_acc:\([0-9.]*\).*/\1/p')

        # 写入 CSV (最前面加了 SD2.0 的标识)
        echo "SD2.0,$algo,$attack_name,${TPR_DET:-N/A},${TPR_TRA:-N/A},${MEAN_ACC:-N/A}" >> $SUMMARY_CSV
        
        echo "✅ 完成 -> 数据已写入"
    done
done

echo "🎉 SD 2.0 测试完毕！请查看 $SUMMARY_CSV"