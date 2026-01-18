import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

def main(args):
    print(f"🚀 Starting CLIP Evaluation for: {args.image_folder}")
    
    # 1. 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载本地 CLIP 模型
    # 你的模型路径: ./clip-vit-g-14/open_clip_pytorch_model.bin
    print(f"Loading CLIP model: {args.model_name}...")
    print(f"Weights path: {args.pretrained_path}")
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, 
            pretrained=args.pretrained_path, 
            device=device
        )
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        print("Tip: Ensure you renamed the local 'open_clip' folder to 'open_clip_legacy'!")
        return

    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    # 3. 加载 Prompts (meta_data.json)
    print(f"Loading prompts from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
        image_list = data['images']
        annotation_list = data['annotations']

    scores = []
    
    # 4. 遍历并计算
    print("Calculating scores...")
    for i in tqdm(range(len(image_list))):
        # A. 获取 Prompt 和 文件名
        prompt = annotation_list[i]['caption']
        file_name = image_list[i]['file_name']
        
        # B. 对齐文件名 (.jpg -> .png)
        # FID 实验生成的图片都是 .png 结尾
        file_name = file_name.replace('.jpg', '.png')
        image_path = os.path.join(args.image_folder, file_name)
        
        # C. 检查图片是否存在 (防止 FID 还没跑完报错)
        if not os.path.exists(image_path):
            continue

        # D. 计算分数
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            text = tokenizer([prompt]).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                # 归一化
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # 计算余弦相似度
                score = (image_features @ text_features.T).item()
                scores.append(score)
        except Exception as e:
            pass

    # 5. 统计与保存
    if len(scores) > 0:
        mean_score = sum(scores) / len(scores)
        print(f"\n{'='*40}")
        print(f"✅ Folder: {os.path.basename(args.image_folder)}")
        print(f"📊 Mean CLIP Score: {mean_score:.4f}")
        print(f"🖼️ Images Processed: {len(scores)}")
        print(f"{'='*40}\n")
        
        # 追加写入总结果文件
        with open("final_clip_results.txt", "a") as f:
            f.write(f"Experiment: {args.run_name} | Score: {mean_score:.4f}\n")
    else:
        print("❌ No images found. Check path or wait for FID generation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True, help="Label for the result txt")
    parser.add_argument("--json_path", type=str, default="./fid_outputs/coco/meta_data.json")
    
    # CLIP 配置
    parser.add_argument("--model_name", type=str, default="ViT-g-14")
    # 指向你刚才 ls 看到的 .bin 文件路径
    parser.add_argument("--pretrained_path", type=str, default="./clip-vit-g-14/open_clip_pytorch_model.bin")
    
    args = parser.parse_args()
    main(args)