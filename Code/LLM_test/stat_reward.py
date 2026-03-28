import json
import numpy as np
from tqdm import tqdm

# 输入路径
INPUT_PATH = "/home/wyx/KitPatch-63E8/Code/LLM_test/llm_output_with_reward.jsonl"
OUTPUT_PATH = "/home/wyx/KitPatch-63E8/Code/LLM_test/reward_stat_result.txt"

def main():
    # 读取所有样本
    rewards = []
    r_formats = []
    r_analyses = []
    r_repairs = []
    r_locations = []
    b_jsons = []

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取样本中"):
            sample = json.loads(line)
            rewards.append(sample.get("reward", 0.0))
            r_formats.append(sample.get("r_format", 0.0))
            r_analyses.append(sample.get("r_analysis", 0.0))
            r_repairs.append(sample.get("r_repair", 0.0))
            r_locations.append(sample.get("r_location", 0.0))
            b_jsons.append(1 if sample.get("b_json", False) else 0)

    # 转换为numpy数组
    rewards = np.array(rewards)
    r_formats = np.array(r_formats)
    r_analyses = np.array(r_analyses)
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
        "r_analysis": calc_stat(r_analyses),
        "r_repair": calc_stat(r_repairs),
        "r_location": calc_stat(r_locations),
        "json_success_rate": float(b_jsons.mean())
    }

    # 生成结果文本
    result_text = "Reward统计结果:\n"
    result_text += "="*50 + "\n"
    result_text += f"总样本数: {stats['total_samples']}\n"
    result_text += f"JSON格式成功率: {stats['json_success_rate']:.2%}\n\n"

    for key in ["reward", "r_format", "r_analysis", "r_repair", "r_location"]:
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

if __name__ == "__main__":
    main()