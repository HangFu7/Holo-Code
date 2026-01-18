# ================= 配置区域 =================
# 1. 基础设置
# 固定种子，确保每次生成的原图完全一致
SEED=42
# 只生成 1 张图
NUM=1

# [修正] 使用您的本地路径
MODEL_PATH="./stable-diffusion-2-1-base"
DATASET_PATH="./Stable-Diffusion-Prompts"

# 输出总目录
OUTPUT_ROOT="paper_figures_signal"

# 2. Holo 参数 (保持实验一致)
ALGO="holo"
CH_COPY=1
HW_COPY=6
FPR=0.000001

# ===========================================

# 创建收集图片的文件夹
mkdir -p "$OUTPUT_ROOT/final_gallery"

echo "开始生成 Figure 6.3.1 的素材图..."
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATASET_PATH"

# 定义执行函数
run_attack() {
    local task_name=$1
    local filename=$2
    local extra_args=$3

    echo ""
    echo ">>> Running: $task_name ($filename)"
    
    python run_gaussian_shading.py \
        --run_name "temp_fig_$task_name" \
        --algo "$ALGO" \
        --num "$NUM" \
        --gen_seed "$SEED" \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --channel_copy "$CH_COPY" \
        --hw_copy "$HW_COPY" \
        --fpr "$FPR" \
        $extra_args

    # 复制并重命名
    if [ "$task_name" == "Original" ]; then
        cp "output/temp_fig_$task_name/image/img_0_watermarked.png" "$OUTPUT_ROOT/final_gallery/$filename"
    else
        cp "output/temp_fig_$task_name/image/img_0_attacked.png" "$OUTPUT_ROOT/final_gallery/$filename"
    fi
}

# ================= 执行生成 (a)-(j) =================

# (a) Original (无攻击)
#run_attack "Original" "a_Original.png" ""

# (b) JPEG QF=20
#run_attack "JPEG_20" "b_JPEG.png" "--jpeg_ratio 20"

# (c) VAE-Regen 0.3
run_attack "VAE_0.3" "c_VAE.png" "--vae_attack --vae_noise_std 0.3"

# (d) Resize 0.3
#run_attack "Resize_0.3" "d_Resize.png" "--resize_ratio 0.3"

# (e) Gaussian Noise 0.08
#run_attack "GNoise_0.08" "e_GNoise.png" "--gaussian_std 0.08"

# (f) Salt & Pepper 0.3
#run_attack "SPNoise_0.3" "f_SPNoise.png" "--sp_prob 0.3"

# (g) Gaussian Blur r=8
#run_attack "GBlur_8" "g_GBlur.png" "--gaussian_blur_r 8"

# (h) Median Filter k=11
#run_attack "Median_11" "h_Median.png" "--median_blur_k 11"

# (i) Brightness 6
#run_attack "Bright_6" "i_Bright.png" "--brightness_factor 6"

# (j) Contrast 3
#run_attack "Contrast_3" "j_Contrast.png" "--contrast_factor 3"

echo ""
echo "✅ 所有图片已生成！"
echo "请查看文件夹: $OUTPUT_ROOT/final_gallery"