import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置
INPUT_FILE = "/home/wyx/KitPatch-63E8/Code/LLM_test/test_after_sft3.jsonl"
OUTPUT_IMAGE = "generated_explanation_length_distribution.png"
BIN_WIDTH = 50  # 每段50个单词

def main():
    # 读取所有样本
    word_counts = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="统计单词数中"):
            sample = json.loads(line)
            text = sample.get("generated_explanation", "")
            words = text.split()
            word_counts.append(len(words))

    word_counts = np.array(word_counts)

    # 统计信息
    print("="*50)
    print(f"总样本数: {len(word_counts)}")
    print(f"最短长度: {word_counts.min()} 词")
    print(f"最长长度: {word_counts.max()} 词")
    print(f"平均长度: {word_counts.mean():.1f} 词")
    print(f"中位数: {np.median(word_counts):.1f} 词")
    print(f"标准差: {word_counts.std():.1f} 词")
    print("="*50)

    # 创建直方图
    plt.figure(figsize=(12, 7))

    # 计算bins，从0到最大值+BIN_WIDTH，间隔BIN_WIDTH
    max_count = word_counts.max()
    bins = np.arange(0, max_count + BIN_WIDTH, BIN_WIDTH)

    n, bins, patches = plt.hist(word_counts, bins=bins, edgecolor='black', alpha=0.7, color='#1f77b4')

    # 在每个柱子上显示数量
    for i in range(len(patches)):
        height = patches[i].get_height()
        if height > 0:
            plt.text(patches[i].get_x() + patches[i].get_width()/2, height + 1,
                    f"{int(height)}", ha='center', va='bottom')

    plt.title('Generated Explanation Length Distribution (words)', fontsize=16)
    plt.xlabel('Word Count (bin width = 50 words)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 设置x轴刻度
    plt.xticks(bins)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\n📊 分布直方图已保存到: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
