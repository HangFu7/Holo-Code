#!/bin/bash

# ================= ⚔️ Holo vs GSsync 鲁棒性对决 =================
# 核心目标: 排除 Search 带来的增益，单纯对比编码结构 (Holo Lattice vs GS Random)
# 样本数: 100 (如果想跑论文最终版，建议改为 1000)
NUM=100

MODEL="./stable-diffusion-2-1-base"
DATA="./Stable-Diffusion-Prompts"
FPR=0.000001
PREFIX="COMPARE_SYNC"

# --- 关键参数差异 ---
# Holo: 使用更密集的 hw=6 (你的核心优势配置)
HOLO_HW=6
# GSsync: 使用标准的 hw=8 (GS 原始配置，但也加上了 Sync 搜索)
GS_HW=8
CHANNEL=1

echo "========================================================="
echo "🚀 STARTING HOLO vs GSSYNC BENCHMARK"
echo "   - Samples: $NUM"
echo "   - Holo HW: $HOLO_HW | GSsync HW: $GS_HW"
echo "========================================================="

# 定义双跑函数
run_compare() {
    local ATTACK_NAME=$1
    local PARAM_NAME=$2
    local VAL=$3
    
    # 1. Run Holo
    echo "   > [Holo] ${ATTACK_NAME} ${VAL}..."
    python run_gaussian_shading.py --run_name "${PREFIX}_Holo_${ATTACK_NAME}_${VAL}" \
        --algo holo --num $NUM --channel_copy $CHANNEL --hw_copy $HOLO_HW --fpr $FPR \
        --model_path $MODEL --dataset_path $DATA $PARAM_NAME $VAL

    # 2. Run GSsync (带几何搜索的 GS)
    echo "   > [GSsync] ${ATTACK_NAME} ${VAL}..."
    python run_gaussian_shading.py --run_name "${PREFIX}_GSsync_${ATTACK_NAME}_${VAL}" \
        --algo gssync --num $NUM --channel_copy $CHANNEL --hw_copy $GS_HW --fpr $FPR \
        --model_path $MODEL --dataset_path $DATA $PARAM_NAME $VAL
}

# =======================================================
# 1. Degradation (退化)
# =======================================================
echo "" && echo ">>> [1/4] Degradation Attacks..."

# JPEG
for VAL in 60 50 40 30 20 10; do
    run_compare "JPEG" "--jpeg_ratio" $VAL
done

# Gaussian Blur
for VAL in 2 4 6 8 10 12; do
    run_compare "Blur" "--gaussian_blur_r" $VAL
done

# Gaussian Noise
for VAL in 0.04 0.08 0.12 0.16 0.20; do
    run_compare "GNoise" "--gaussian_std" $VAL
done

# Median Filter
for VAL in 3 7 11 15 19; do
    run_compare "Median" "--median_blur_k" $VAL
done

# Resize
for VAL in 0.9 0.7 0.5 0.3 0.1; do
    run_compare "Resize" "--resize_ratio" $VAL
done

# S&P Noise
for VAL in 0.1 0.2 0.3 0.4 0.5; do
    run_compare "SPNoise" "--sp_prob" $VAL
done

# =======================================================
# 2. Geometric (几何 - 重点战场!)
# =======================================================
echo "" && echo ">>> [2/4] Geometric Attacks (Critical)..."

# Random Crop (GSsync 应该在这一项上会有所提升，但看 Holo 是否依然领先)
for VAL in 0.9 0.7 0.5 0.4 0.3 0.25 0.2 0.15 0.1; do
    run_compare "Crop" "--random_crop_ratio" $VAL
done

# Random Drop
for VAL in 0.6 0.7 0.8 0.9 0.95; do
    run_compare "Drop" "--random_drop_ratio" $VAL
done

# Translation Shift
for VAL in {1..16}; do
    run_compare "Shift" "--translation_shift" $VAL
done

# Perspective
for VAL in 0.04 0.06 0.08 0.10 0.12 0.16 0.20; do
    run_compare "Persp" "--perspective_scale" $VAL
done

# =======================================================
# 3. Photometric (光度)
# =======================================================
echo "" && echo ">>> [3/4] Photometric Attacks..."

# Brightness
for VAL in 2 4 6 8 10 12; do
    run_compare "Bright" "--brightness_factor" $VAL
done

# Contrast
for VAL in 2 3 4 5 6 7; do
    run_compare "Contrast" "--contrast_factor" $VAL
done

# =======================================================
# 4. VAE Attack
# =======================================================
echo "" && echo ">>> [4/4] VAE Regeneration..."

echo "   > [Holo] VAE..."
python run_gaussian_shading.py --run_name "${PREFIX}_Holo_VAE" \
    --algo holo --num $NUM --channel_copy $CHANNEL --hw_copy $HOLO_HW --fpr $FPR \
    --model_path $MODEL --dataset_path $DATA --vae_attack

echo "   > [GSsync] VAE..."
python run_gaussian_shading.py --run_name "${PREFIX}_GSsync_VAE" \
    --algo gssync --num $NUM --channel_copy $CHANNEL --hw_copy $GS_HW --fpr $FPR \
    --model_path $MODEL --dataset_path $DATA --vae_attack

echo ""
echo "🏆 HOLO vs GSSYNC COMPARISON FINISHED!"