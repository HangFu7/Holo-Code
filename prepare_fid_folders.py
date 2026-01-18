import os
import shutil
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    # 你的实验输出目录，例如 ./output/QUALITY_CLIP_Holo
    parser.add_argument('--source_dir', type=str, required=True) 
    # 临时存放 clean 图片的目录
    parser.add_argument('--clean_dir', type=str, default='./temp_fid/clean')
    # 临时存放 watermarked 图片的目录
    parser.add_argument('--wm_dir', type=str, default='./temp_fid/wm')
    args = parser.parse_args()

    # 1. 创建目标文件夹
    os.makedirs(args.clean_dir, exist_ok=True)
    os.makedirs(args.wm_dir, exist_ok=True)
    
    source_img_dir = os.path.join(args.source_dir, "image")
    if not os.path.exists(source_img_dir):
        print(f"Error: Directory {source_img_dir} does not exist!")
        return

    files = os.listdir(source_img_dir)
    print(f"📂 Processing {len(files)} files from {source_img_dir}...")

    count_clean = 0
    count_wm = 0

    # 2. 遍历并复制
    for f in tqdm(files):
        src_path = os.path.join(source_img_dir, f)
        
        # 识别 Clean 图片
        if "clean" in f and f.endswith(".png"):
            shutil.copy(src_path, os.path.join(args.clean_dir, f))
            count_clean += 1
            
        # 识别 Watermarked 图片 (通常是不带 clean 且带 watermarked 或者是默认生成的图)
        # 根据你的命名逻辑调整，假设是 *_watermarked.png
        elif "watermarked" in f and f.endswith(".png"):
            shutil.copy(src_path, os.path.join(args.wm_dir, f))
            count_wm += 1

    print(f"\n✅ Done!")
    print(f"   Clean images: {count_clean} -> {args.clean_dir}")
    print(f"   Watermarked images: {count_wm} -> {args.wm_dir}")

if __name__ == "__main__":
    main()