#!/bin/bash

# =================================================================
# 🧪 Holo-Code Official Re-run (FID + CLIP)
# =================================================================
# 目标: 生成 5000 张图 -> 算 FID -> 算 CLIP
# 状态: 严格对齐 GS/PRC 的实验设置
# =================================================================

# --- 核心配置 ---
NUM=5000
MODEL="./stable-diffusion-2-1-base"
META_JSON="./fid_outputs/coco/meta_data.json"
GT_FOLDER="./fid_outputs/coco/ground_truth"
# 本地 CLIP 模型路径
CLIP_WEIGHTS="./clip-vit-g-14/open_clip_pytorch_model.bin"

# Holo 参数 (关键: hw_copy=6)
ALGO="holo"
RUN_NAME="Official_Holo"
HW_COPY=6
FPR=0.000001

echo "#############################################################"
echo "🚀 STARTING HOLO-CODE OFFICIAL BENCHMARK"
echo "   - Run Name: $RUN_NAME"
echo "   - Samples: $NUM"
echo "   - HW Copy: $HW_COPY (Critical for Robustness)"
echo "#############################################################"

# ================= STEP 1: 生成图片 & 计算 FID =================
echo ""
echo ">>> [Step 1/2] Generating Images & Calculating FID..."

# 注意：这里调用的是 gaussian_shading_fid.py (专门测 FID 的脚本)
python gaussian_shading_fid.py \
    --run_name "$RUN_NAME" \
    --algo "$ALGO" \
    --num $NUM \
    --fpr $FPR \
    --prompt_file "$META_JSON" \
    --gt_folder "$GT_FOLDER" \
    --model_path "$MODEL" \
    --channel_copy 1 \
    --hw_copy $HW_COPY

if [ $? -ne 0 ]; then
    echo "❌ Error in FID Generation. Stopping."
    exit 1
fi

# ================= STEP 2: 计算 CLIP Score (离线模式) =================
echo ""
echo ">>> [Step 2/2] Calculating CLIP Score..."

IMG_DIR="./fid_outputs/coco/${RUN_NAME}/w_gen"

# 检查图片目录是否存在
if [ ! -d "$IMG_DIR" ]; then
    echo "❌ Error: Image directory missing: $IMG_DIR"
    exit 1
fi

# 调用 calc_clip.py (离线计算脚本)
python calc_clip.py \
    --run_name "$RUN_NAME" \
    --image_folder "$IMG_DIR" \
    --json_path "$META_JSON" \
    --pretrained_path "$CLIP_WEIGHTS"

echo ""
echo "#############################################################"
echo "✅ HOLO EXPERIMENT COMPLETE!"
echo "   - FID: Check 'output/official_fid_results.txt'"
echo "   - CLIP: Check 'final_clip_results.txt'"
echo "#############################################################"