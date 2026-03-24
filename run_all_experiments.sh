#!/bin/bash

# ==========================================
# AIGC 水印云边 IoT 传输端到端批量实验脚本
# (纯净版：终端实时显示进度，无冗余日志，全自动写CSV)
# ==========================================

# 1. 基础参数设置
NUM=100
MODEL_PATH="./stable-diffusion-2-1-base"
DATASET_PATH="./Stable-Diffusion-Prompts"
CHANNEL_COPY=1
FPR=0.000001
JPEG_QUALITY=50
PACKET_BYTES=1024
CHANNEL_MODE="mixed" 

# 2. 实验维度矩阵
ALGOS=("holo" "gs" "prc")
CODES=("ldpc" "none")
SEVERITIES=("mild" "moderate" "severe")

# 3. 创建输出 CSV 与临时缓存文件
SUMMARY_CSV="experiment_summary.csv"
TEMP_LOG=".temp_run_output.log" # 隐藏的临时文件，跑完自动覆盖，不占空间

# 初始化 CSV 表头 (如果文件不存在则写入)
# 【修复：if 和 [ 之间加了空格】
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "Algorithm,Channel_Code,Severity,HW_Copy,Decode_Success_Rate,Effective_Loss_Rate,PSNR,TPR_Detection,TPR_Traceability,Mean_Acc,Std_Acc" > $SUMMARY_CSV
fi

echo "=========================================================="
echo "🚀 开始批量执行端到端 IoT 传输水印实验"
echo "测试算法: ${ALGOS[*]}"
echo "信道编码: ${CODES[*]}"
echo "信道强度: ${SEVERITIES[*]}"
echo "测试数量: $NUM 张图片 / 组"
echo "结果汇总: $SUMMARY_CSV (实时追加)"
echo "=========================================================="

# 4. 三重循环嵌套，遍历所有组合
for algo in "${ALGOS[@]}"; do

    # 🔥 动态分配 hw_copy 参数
    # 【修复：if 和 [ 之间加了空格】
    if [ "$algo" == "holo" ]; then
        CURRENT_HW_COPY=6
    else
        CURRENT_HW_COPY=8
    fi

    for code in "${CODES[@]}"; do
        for severity in "${SEVERITIES[@]}"; do
            
            RUN_NAME="exp_${algo}_${code}_${severity}"
            
            echo ""
            echo "=========================================================="
            echo "⏳[$(date +'%H:%M:%S')] 正在运行 -> 算法: $algo | 编码: $code | 强度: $severity | HW_Copy: $CURRENT_HW_COPY"
            echo "=========================================================="

            # 运行 Python 脚本，使用 tee 将输出实时打印到终端，同时存入临时文件供抓取
            python run_holo_code.py \
                --algo $algo \
                --run_name $RUN_NAME \
                --num $NUM \
                --model_path $MODEL_PATH \
                --dataset_path $DATASET_PATH \
                --channel_copy $CHANNEL_COPY \
                --hw_copy $CURRENT_HW_COPY \
                --fpr $FPR \
                --iot_attack \
                --iot_channel_code $code \
                --iot_jpeg_quality $JPEG_QUALITY \
                --iot_packet_bytes $PACKET_BYTES \
                --iot_severity $severity \
                --iot_channel_mode $CHANNEL_MODE \
                --iot_ldpc_k 1024 \
                --iot_ldpc_m 1024 \
                --iot_ldpc_col_weight 3 \
                --iot_ldpc_max_iter 20 \
                2>&1 | tee $TEMP_LOG

            # ====================================================
            # 📊 从临时文件中自动提取指标，并追加到 CSV 表格中
            # ====================================================
            
            # 提取通信层指标
            DECODE_SUCC=$(grep "Mean Decode Success Rate" "$TEMP_LOG" | awk '{print $6}')
            EFF_LOSS=$(grep "Mean Effective Loss Rate" "$TEMP_LOG" | awk '{print $6}')
            PSNR=$(grep "Mean PSNR" "$TEMP_LOG" | awk '{print $4}')
            
            # 提取水印层指标
            RES_LINE=$(grep "tpr_detection:" "$TEMP_LOG" | tail -n 1)
            TPR_DET=$(echo "$RES_LINE" | sed -n 's/.*tpr_detection:\([0-9.]*\).*/\1/p')
            TPR_TRA=$(echo "$RES_LINE" | sed -n 's/.*tpr_traceability:\([0-9.]*\).*/\1/p')
            MEAN_ACC=$(echo "$RES_LINE" | sed -n 's/.*mean_acc:\([0-9.]*\).*/\1/p')
            STD_ACC=$(echo "$RES_LINE" | sed -n 's/.*std_acc:\([0-9.]*\).*/\1/p')

            # 容错处理：如果发生意外导致未提取到数据，则填 N/A
            DECODE_SUCC=${DECODE_SUCC:-"N/A"}
            EFF_LOSS=${EFF_LOSS:-"N/A"}
            PSNR=${PSNR:-"N/A"}
            TPR_DET=${TPR_DET:-"N/A"}
            TPR_TRA=${TPR_TRA:-"N/A"}
            MEAN_ACC=${MEAN_ACC:-"N/A"}
            STD_ACC=${STD_ACC:-"N/A"}

            # 写入 CSV
            echo "$algo,$code,$severity,$CURRENT_HW_COPY,$DECODE_SUCC,$EFF_LOSS,$PSNR,$TPR_DET,$TPR_TRA,$MEAN_ACC,$STD_ACC" >> $SUMMARY_CSV

            echo "✅[$(date +'%H:%M:%S')] 本组完成！结果已成功写入 $SUMMARY_CSV"

        done
    done
done

# 清理临时文件
rm -f $TEMP_LOG

echo ""
echo "🎉🎉🎉 所有 18 组实验已全部执行完毕！"
echo "📊 请直接查看当前目录下的表格文件: $SUMMARY_CSV"