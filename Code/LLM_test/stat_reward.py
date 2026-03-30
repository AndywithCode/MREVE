import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="统计LLM输出的reward得分并生成图表")
    parser.add_argument("--input", "-i", required=True, help="输入jsonl文件路径")
    parser.add_argument("--output-dir", "-o", default="/home/wyx/KitPatch-63E8/Code/LLM_test", help="输出结果目录")
    parser.add_argument("--prefix", "-p", default="", help="输出文件前缀，用于区分不同模型结果")
    args = parser.parse_args()

    # 构造输出路径
    os.makedirs(args.output_dir, exist_ok=True)
    if args.prefix:
        prefix = f"{args.prefix}_"
    else:
        prefix = ""
    OUTPUT_PATH = os.path.join(args.output_dir, f"{prefix}reward_stat_result.txt")
    HISTOGRAM_PATH = os.path.join(args.output_dir, f"{prefix}reward_histogram.png")
    BOXPLOT_PATH = os.path.join(args.output_dir, f"{prefix}reward_boxplot.png")
    R_STRUCTURE_HISTOGRAM_PATH = os.path.join(args.output_dir, f"{prefix}r_structure_histogram.png")
    R_STRUCTURE_BOXPLOT_PATH = os.path.join(args.output_dir, f"{prefix}r_structure_boxplot.png")

    # 读取所有样本
    rewards = []
    r_formats = []
    r_structures = []
    r_repairs = []
    r_locations = []
    b_jsons = []

    with open(args.input, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取样本中"):
            sample = json.loads(line)
            rewards.append(sample.get("reward", 0.0))
            r_formats.append(sample.get("r_format", 0.0))
            # 兼容r_analysis和r_structure两个字段
            r_analysis = sample.get("r_analysis", sample.get("r_structure", 0.0))
            r_structures.append(r_analysis)
            r_repairs.append(sample.get("r_repair", 0.0))
            r_locations.append(sample.get("r_location", 0.0))
            b_jsons.append(1 if sample.get("b_json", False) else 0)

    # 转换为numpy数组
    rewards = np.array(rewards)
    r_formats = np.array(r_formats)
    r_structures = np.array(r_structures)
    r_repairs = np.array(r_repairs)
    r_locations = np.array(r_locations)
    b_jsons = np.array(b_jsons)

    # 计算均值和标准差
    def calc_stat(arr):
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": len(arr)
        }

    stats = {
        "total_samples": len(rewards),
        "reward": calc_stat(rewards),
        "r_format": calc_stat(r_formats),
        "r_structure": calc_stat(r_structures),
        "r_repair": calc_stat(r_repairs),
        "r_location": calc_stat(r_locations),
        "json_success_rate": float(b_jsons.mean())
    }

    # 生成结果文本
    result_text = "Reward统计结果:\n"
    result_text += "="*50 + "\n"
    result_text += f"总样本数: {stats['total_samples']}\n"
    result_text += f"JSON格式成功率: {stats['json_success_rate']:.2%}\n\n"

    for key in ["reward", "r_format", "r_structure", "r_repair", "r_location"]:
        stat = stats[key]
        result_text += f"{key}:\n"
        result_text += f"  均值: {stat['mean']:.4f}\n"
        result_text += f"  标准差: {stat['std']:.4f}\n"
        result_text += f"  最小值: {stat['min']:.4f}\n"
        result_text += f"  最大值: {stat['max']:.4f}\n"
        result_text += "\n"

    # 打印结果
    print(result_text)

    # 保存到文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(result_text)

    print(f"统计结果已保存到 {OUTPUT_PATH}")

    # 生成Reward直方图
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Reward Distribution Histogram', fontsize=14)
    plt.xlabel('Reward Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axvline(rewards.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {rewards.mean():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PATH, dpi=300)
    print(f"Reward直方图已保存到 {HISTOGRAM_PATH}")

    # 生成箱线图
    plt.figure(figsize=(12, 6))
    box_data = [rewards, r_formats, r_structures, r_repairs, r_locations]
    labels = ['Total Reward', 'Format', 'Structure', 'Repair', 'Location']
    box = plt.boxplot(box_data, patch_artist=True, labels=labels, medianprops={'color': 'red'})

    # 设置颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title('Reward Components Boxplot', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加统计值标注
    for i, (data, label) in enumerate(zip(box_data, labels)):
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        min_val = np.min(data)
        max_val = np.max(data)
        plt.text(i+1, max_val + 0.02, f'Max: {max_val:.2f}\nQ3: {q3:.2f}\nMedian: {median:.2f}\nQ1: {q1:.2f}\nMin: {min_val:.2f}',
                ha='center', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(BOXPLOT_PATH, dpi=300)
    print(f"Reward箱线图已保存到 {BOXPLOT_PATH}")

    # 生成r_structure单独直方图
    plt.figure(figsize=(10, 6))
    plt.hist(r_structures, bins=20, edgecolor='black', alpha=0.7, color='#2ca02c')
    plt.title('R_Structure Distribution Histogram', fontsize=14)
    plt.xlabel('R_Structure Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axvline(r_structures.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {r_structures.mean():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(R_STRUCTURE_HISTOGRAM_PATH, dpi=300)
    print(f"R_Structure直方图已保存到 {R_STRUCTURE_HISTOGRAM_PATH}")

    # 生成r_structure单独箱线图
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(r_structures, patch_artist=True, labels=['R_Structure'], medianprops={'color': 'red'})

    # 设置颜色
    box['boxes'][0].set_facecolor('#2ca02c')
    box['boxes'][0].set_alpha(0.7)

    plt.title('R_Structure Boxplot', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加统计值标注
    median = np.median(r_structures)
    q1 = np.percentile(r_structures, 25)
    q3 = np.percentile(r_structures, 75)
    min_val = np.min(r_structures)
    max_val = np.max(r_structures)
    plt.text(1, max_val + 0.02, f'Max: {max_val:.2f}\nQ3: {q3:.2f}\nMedian: {median:.2f}\nQ1: {q1:.2f}\nMin: {min_val:.2f}',
            ha='center', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(R_STRUCTURE_BOXPLOT_PATH, dpi=300)
    print(f"R_Structure箱线图已保存到 {R_STRUCTURE_BOXPLOT_PATH}")

if __name__ == "__main__":
    main()