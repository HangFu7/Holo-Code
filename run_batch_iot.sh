#!/bin/bash

# ==========================================
# 批量 IoT bitstream transmission 测试脚本
# mild / moderate / severe 三档
# ==========================================

NUM=50
MODEL_PATH="./stable-diffusion-2-1-base"
DATASET_PATH="./Stable-Diffusion-Prompts"

ALGOS=("gs" "prc" "holo")
SEVERITIES=("mild" "moderate" "severe")

IOT_JPEG_QUALITY=50
IOT_PACKET_BYTES=1024
IOT_CHANNEL_MODE="mixed"

OUT_BASE="./output"

echo "=========================================================="
echo "🚀 开始批量 IoT bitstream transmission 测试"
echo "测试算法: ${ALGOS[*]}"
echo "信道强度: ${SEVERITIES[*]}"
echo "每组测试图片数: $NUM"
echo "JPEG Quality: $IOT_JPEG_QUALITY"
echo "Packet Bytes: $IOT_PACKET_BYTES"
echo "Channel Mode: $IOT_CHANNEL_MODE"
echo "输出根目录: $OUT_BASE"
echo "=========================================================="

for ALGO in "${ALGOS[@]}"; do

    if [ "$ALGO" == "holo" ]; then
        HW=6
    else
        HW=8
    fi

    for SEV in "${SEVERITIES[@]}"; do

        RUN_NAME="iot_${ALGO}_q${IOT_JPEG_QUALITY}_${SEV}"

        echo ""
        echo "----------------------------------------------------------"
        echo "▶ 正在运行算法: [$ALGO] | hw_copy: $HW | severity: $SEV"
        echo "▶ run_name: $RUN_NAME"
        echo "----------------------------------------------------------"

        python run_holo_code.py \
            --algo "$ALGO" \
            --num "$NUM" \
            --model_path "$MODEL_PATH" \
            --dataset_path "$DATASET_PATH" \
            --channel_copy 1 \
            --hw_copy "$HW" \
            --fpr 0.000001 \
            --output_path "$OUT_BASE" \
            --run_name "$RUN_NAME" \
            --iot_attack \
            --iot_jpeg_quality "$IOT_JPEG_QUALITY" \
            --iot_packet_bytes "$IOT_PACKET_BYTES" \
            --iot_severity "$SEV" \
            --iot_channel_mode "$IOT_CHANNEL_MODE"

        EXIT_CODE=$?

        if [ $EXIT_CODE -ne 0 ]; then
            echo "❌ 算法 [$ALGO] 在 severity=[$SEV] 运行失败，退出码: $EXIT_CODE"
        else
            echo "✅ 算法 [$ALGO] 在 severity=[$SEV] 运行完成！"
            echo "📁 结果目录: $OUT_BASE/$RUN_NAME"
        fi

    done
done

echo ""
echo "=========================================================="
echo "🎉 所有批量实验完成！"
echo "=========================================================="