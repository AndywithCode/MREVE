import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 模型配置
MODELS = [
    {"name": "Ground Truth", "file": "groundtruth_with_reward.jsonl", "color": "#1f77b4"},
    {"name": "GPT-5.4", "file": "gpt-5.4_output_with_reward.jsonl", "color": "#ff7f0e"},
    {"name": "DeepSeek V3.2", "file": "deepseek-v3.2_output_with_reward.jsonl", "color": "#2ca02c"},
    {"name": "Doubao Seed 2.0 Pro", "file": "doubao-seed-2.0-pro_output_with_reward.jsonl", "color": "#d62728"}
]

OUTPUT_DIR = "/home/wyx/KitPatch-63E8/Code/LLM_test"

def load_r_structure(file_path):
    """加载指定文件中的r_structure数据"""
    r_structures = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"加载{os.path.basename(file_path)}"):
            sample = json.loads(line)
            r_analysis = sample.get("r_analysis", sample.get("r_structure", 0.0))
            r_structures.append(r_analysis)
    return np.array(r_structures)

def main():
    # 加载所有模型的数据
    model_data = []
    for model in MODELS:
        file_path = os.path.join(OUTPUT_DIR, model["file"])
        r_struct = load_r_structure(file_path)
        model_data.append({
            "name": model["name"],
            "data": r_struct,
            "color": model["color"],
            "mean": r_struct.mean(),
            "std": r_struct.std()
        })
        print(f"✅ {model['name']} 加载完成，均值: {r_struct.mean():.4f}, 标准差: {r_struct.std():.4f}")

    # 生成多模型r_structure直方图对比
    plt.figure(figsize=(12, 7))
    for model in model_data:
        plt.hist(model["data"], bins=20, edgecolor='black', alpha=0.5,
                 color=model["color"], label=f"{model['name']} (mean={model['mean']:.4f})")

    plt.title('R_Structure Distribution Comparison Across Models', fontsize=16)
    plt.xlabel('R_Structure Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    hist_path = os.path.join(OUTPUT_DIR, "all_models_r_structure_histogram.png")
    plt.savefig(hist_path, dpi=300)
    print(f"\n📊 多模型R_Structure直方图已保存到: {hist_path}")

    # 生成多模型r_structure箱线图对比
    plt.figure(figsize=(12, 7))
    box_data = [model["data"] for model in model_data]
    labels = [f"{model['name']}\n(mean={model['mean']:.4f})" for model in model_data]
    colors = [model["color"] for model in model_data]

    box = plt.boxplot(box_data, patch_artist=True, labels=labels, medianprops={'color': 'red'})

    # 设置颜色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title('R_Structure Boxplot Comparison Across Models', fontsize=16)
    plt.ylabel('R_Structure Score', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加统计值标注
    for i, model in enumerate(model_data):
        data = model["data"]
        max_val = np.max(data)
        plt.text(i+1, max_val + 0.03, f"Max: {max_val:.2f}\nMin: {np.min(data):.2f}",
                ha='center', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()

    box_path = os.path.join(OUTPUT_DIR, "all_models_r_structure_boxplot.png")
    plt.savefig(box_path, dpi=300)
    print(f"📊 多模型R_Structure箱线图已保存到: {box_path}")

    print("\n🎉 所有对比图表生成完成！")

if __name__ == "__main__":
    main()