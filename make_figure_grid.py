import os
import matplotlib.pyplot as plt
from PIL import Image

# === 配置区域 ===
# 选定的图片编号 (注意：必须用字符串格式，否则 Python 会报 SyntaxError)
SELECTED_IDS = [
    "000000030675", 
    "000000379453", 
    "000000039956", 
    "000000168330"
] 

# 文件夹路径
BASE_DIR = "fid_outputs/coco"

# 只需要文件夹列表，顺序对应：Clean -> GS -> PRC -> Holo
FOLDERS = [
    "Official_Clean/w_gen",
    "Official_GS/w_gen",
    "Official_PRC/w_gen",
    "Official_Holo/w_gen"
]

OUTPUT_FILENAME = "Figure_Quality_Comparison.png"
# =================

def main():
    # === 核心修改 1: 交换行列定义 ===
    # 现在行数 = 方法数量，列数 = 图片数量
    num_rows = len(FOLDERS)
    num_cols = len(SELECTED_IDS)
    
    # 创建画布
    # figsize 可以根据需要调整，这里设置大一点保证分辨率
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    
    print("正在生成纯净无标签拼图...")

    # === 核心修改 2: 交换循环层级 ===
    # 外层循环遍历“方法” (行 r)
    # 内层循环遍历“图片ID” (列 c)
    for r, folder in enumerate(FOLDERS):
        for c, img_id in enumerate(SELECTED_IDS):
            
            img_path = os.path.join(BASE_DIR, folder, f"{img_id}.png")
            
            # 处理 axes 索引
            if num_rows == 1: ax = axes[c]
            elif num_cols == 1: ax = axes[r]
            else: ax = axes[r, c]
            
            # 读取并显示图片
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                print(f"❌ 警告: 找不到图片 {img_path}")
                ax.imshow([[1]], cmap='gray') 
                ax.axis('off')
                continue
            
            # 移除所有干扰元素
            ax.axis('off')

    # 布局调整 (紧密排列)
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    
    # 保存图片
    print(f"保存结果到: {OUTPUT_FILENAME}")
    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight', pad_inches=0)
    print("✅ 完成！每一行是同一个方法生成的不同图片。")

if __name__ == "__main__":
    main()