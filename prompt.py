from datasets import load_dataset

# 1. 加载和 run.py 一样的数据集
# 注意：如果你运行时指定了 --dataset_path，请在这里修改
dataset_path = './Stable-Diffusion-Prompts' 

print(f"正在加载数据集: {dataset_path} ...")
dataset = load_dataset(dataset_path)['test']

# 2. 获取第 43 张图片的提示词
# 计算机从 0 开始计数，第 43 张图片的索引是 42
index = 43

try:
    prompt = dataset[index]['Prompt']
    print("\n" + "="*50)
    print(f"【第 {index+1} 张图片 (Index {index}) 的提示词】:")
    print("="*50)
    print(f"\n{prompt}\n")
    print("="*50)
except Exception as e:
    print(f"读取错误: {e}")
    # 有些数据集的列名可能是 'text' 而不是 'Prompt'
    print("尝试查看数据结构:", dataset[0])