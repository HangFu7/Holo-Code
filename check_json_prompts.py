import json
import os

# ================= 配置区域 =================
# 这里填你想要在论文里展示的物体/场景关键词
# 建议覆盖：纹理(Texture)、平滑背景(Flat)、复杂结构(Structure)
TARGET_KEYWORDS = {
    "Train (工业纹理)": ["train", "locomotive", "railroad", "steam engine"],
    "Airplane (蓝天/平滑)": ["airplane", "plane", "jet", "aircraft", "sky"],
    "Cat (毛发细节)": ["cat", "kitten", "feline"],
    "Pizza/Food (复杂色彩)": ["pizza", "food", "sandwich", "cake"],
    "Bedroom (室内结构)": ["bedroom", "living room", "kitchen", "furniture"],
    "Clock (精细线条)": ["clock", "watch", "tower"],
    "Person (人脸/姿态)": ["woman", "man", "person", "portrait"]
}

JSON_PATH = 'fid_outputs/coco/meta_data.json'
# ===========================================

def main():
    if not os.path.exists(JSON_PATH):
        print(f"❌ 找不到文件: {JSON_PATH}")
        return

    print(f"正在读取 {JSON_PATH} ...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 统一数据格式为列表
    prompts_list = []
    if isinstance(data, list):
        prompts_list = data
    elif isinstance(data, dict):
        if 'annotations' in data: prompts_list = data['annotations']
        elif 'captions' in data: prompts_list = data['captions']
        else: prompts_list = list(data.values())

    print(f"✅ 加载完成，共 {len(prompts_list)} 条提示词。\n")
    print("="*60)
    print("正在搜索符合论文展示要求的图片编号 (Index)...")
    print("="*60)

    # 2. 遍历搜索
    # 结果字典： { "Category": [ (index, prompt_text), ... ] }
    results = {cat: [] for cat in TARGET_KEYWORDS}

    for idx, item in enumerate(prompts_list):
        # 提取文本
        text = ""
        if isinstance(item, str): text = item
        elif isinstance(item, dict):
            # 尝试取值
            for k in ['caption', 'text', 'prompt', 'Prompt']:
                if k in item: text = item[k]; break
        
        text_lower = text.lower()

        # 匹配关键词
        for category, keywords in TARGET_KEYWORDS.items():
            for kw in keywords:
                # 简单匹配：单词在句子里，且句子不要太长太乱
                if kw in text_lower and len(text) < 200: 
                    results[category].append((idx, text))
                    break # 命中一个关键词就不再重复添加同一类别

    # 3. 打印结果
    for category, items in results.items():
        if not items:
            continue
            
        print(f"\n📂 类别: {category} (找到 {len(items)} 张)")
        print("-" * 60)
        
        # 为了不刷屏，每个类别只显示前 5 个最合适的（长度适中的）
        # 优先展示 Prompt 长度在 20-100 字符之间的，通常构图较好
        good_samples = [x for x in items if 30 < len(x[1]) < 100]
        display_items = good_samples[:5] if good_samples else items[:5]
        
        for idx, prompt in display_items:
            print(f"  [图片编号: {idx}] -> 文件名: {idx}.png")
            print(f"  Prompt: \"{prompt}\"")
            print("  . . .")
    
    print("\n" + "="*60)
    print("💡 使用说明:")
    print("1. 记下上面心仪的 [图片编号] (例如 123)")
    print("2. 进入文件夹: fid_outputs/coco/Official_Holo/ (或其他方案文件夹)")
    print("3. 找到对应的图片: 123.png")
    print("4. 对比 Clean/GS/Holo 同一编号的图片质量。")

if __name__ == "__main__":
    main()