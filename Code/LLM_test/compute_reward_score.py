import json
import sys
import os
import argparse
from tqdm import tqdm

# 添加reward模块所在的目录到sys.path
sys.path.append("/home/wyx/KitPatch-63E8/Code/patch_generation")
from reward import compute_reward

def main():
    parser = argparse.ArgumentParser(description="计算LLM输出的reward分项得分")
    parser.add_argument("--input", "-i", required=True, help="输入去重后的模型输出jsonl路径")
    parser.add_argument("--output", "-o", required=True, help="输出带reward的jsonl路径")
    args = parser.parse_args()

    # 输入输出路径
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output

    # 读取所有样本
    samples = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    # 计算每个样本的reward
    results = []
    for sample in tqdm(samples, desc="计算reward中"):
        llm_output = sample.get("llm_output", "")
        ref_analysis = sample.get("analysis", "")
        ground_patch = sample.get("ground_patch", "")
        buggy_lines = sample.get("buggy_location", [])

        if not llm_output:
            print("样本缺少llm_output字段，跳过")
            continue

        # 调用compute_reward
        reward, r_format, r_analysis, r_repair, r_location, b_json = compute_reward(
            text=llm_output,
            ref_analysis=ref_analysis,
            ground_patch=ground_patch,
            buggy_lines=buggy_lines
        )

        # 保存结果
        sample.update({
            "reward": float(reward),
            "r_format": float(r_format),
            "r_analysis": float(r_analysis),
            "r_repair": float(r_repair),
            "r_location": float(r_location),
            "b_json": b_json
        })
        results.append(sample)

    # 保存带reward的结果
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"计算完成，结果已保存到 {OUTPUT_PATH}，共处理 {len(results)} 个样本")

if __name__ == "__main__":
    main()