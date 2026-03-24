#!/bin/bash

# ==========================================
# 真实云边复合攻击批量测试脚本 (IoT传输降级 + 极限二次篡改)
# ==========================================

NUM=100
MODEL_PATH="./stable-diffusion-2-1-base"
DATASET_PATH="./Stable-Diffusion-Prompts"

# 固定 IoT 传输的恶劣条件 (无信道编码 + 中度丢包，必然触发 Fallback 降级)
IOT_CODE="none"
IOT_SEV="moderate"

# 算法对比
ALGOS=("holo" "gssync" "prc")

# 6 种毁天灭地的二次攻击 (赶尽杀绝级别)
ATTACK_NAMES=("Crop_0.5" "Drop_0.8" "JPEG_20" "Blur_6" "Noise_0.05" "Bright_4")
ATTACK_PARAMS=(
    "--random_crop_ratio 0.5"
    "--random_drop_ratio 0.8"
    "--jpeg_ratio 20"
    "--gaussian_blur_r 6"
    "--gaussian_std 0.05"
    "--brightness_factor 4"
)

LOG_DIR="composite_logs"
mkdir -p $LOG_DIR
SUMMARY_CSV="composite_summary.csv"

# 初始化纯净版 CSV 表头 (只看核心水印存活指标)
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "Algorithm,Image_Attack,TPR_Det,TPR_Tra,Mean_Acc" > $SUMMARY_CSV
fi

echo "=========================================================="
echo "🚀 开始执行 [IoT传输崩溃 + 极限图像篡改] 双重复合攻击"
echo "固定前置条件: IoT 传输 ($IOT_CODE, $IOT_SEV) -> 触发画质降级"
echo "=========================================================="

for algo in "${ALGOS[@]}"; do

    # 动态分配 hw_copy 参数，保证基线不报错
    if [ "$algo" == "holo" ]; then HW_COPY=6; else HW_COPY=8; fi

    for i in "${!ATTACK_NAMES[@]}"; do
        attack_name="${ATTACK_NAMES[$i]}"
        attack_param="${ATTACK_PARAMS[$i]}"
        
        RUN_NAME="comp_${algo}_${attack_name}"
        TEMP_LOG="${LOG_DIR}/${RUN_NAME}.log"
        
        echo "⏳ [$(date +'%H:%M:%S')] 运行 -> 算法: $algo | 叠加攻击: $attack_name"

        # 执行复合攻击
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

        # ====================================================
        # 📊 只提取最核心的水印存活指标
        # ====================================================
        RES_LINE=$(grep "tpr_detection:" "$TEMP_LOG" | tail -n 1)
        TPR_DET=$(echo "$RES_LINE" | sed -n 's/.*tpr_detection:\([0-9.]*\).*/\1/p')
        TPR_TRA=$(echo "$RES_LINE" | sed -n 's/.*tpr_traceability:\([0-9.]*\).*/\1/p')
        MEAN_ACC=$(echo "$RES_LINE" | sed -n 's/.*mean_acc:\([0-9.]*\).*/\1/p')

        # 容错处理
        TPR_DET=${TPR_DET:-"N/A"}
        TPR_TRA=${TPR_TRA:-"N/A"}
        MEAN_ACC=${MEAN_ACC:-"N/A"}

        # 写入 CSV
        echo "$algo,$attack_name,$TPR_DET,$TPR_TRA,$MEAN_ACC" >> $SUMMARY_CSV
        
        echo "✅ 完成 -> 数据已写入 CSV"
    done
done

echo "🎉 所有复合攻击测试完毕！请查看 $SUMMARY_CSV"