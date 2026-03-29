import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="对JSONL文件按照指定字段去重")
    parser.add_argument("--input", "-i", default="/home/wyx/KitPatch-63E8/Code/LLM_test/gpt-5.4_output.jsonl", help="输入JSONL文件路径")
    parser.add_argument("--output", "-o", default="/home/wyx/KitPatch-63E8/Code/LLM_test/gpt-5.4_output_deduplicated.jsonl", help="去重后输出JSONL文件路径")
    parser.add_argument("--deduplicate-field", "-f", default="generated_explanation", help="用于去重的字段名")
    args = parser.parse_args()

    # 读取所有样本并去重
    seen_values = set()
    unique_samples = []
    duplicate_count = 0

    with open(args.input, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)

        for line in tqdm(f, total=total_lines, desc="去重处理中"):
            sample = json.loads(line)
            # 检查去重字段是否存在
            if args.deduplicate_field not in sample:
                print(f"警告：样本缺少{args.deduplicate_field}字段，直接保留")
                unique_samples.append(sample)
                continue

            field_value = sample[args.deduplicate_field]
            if field_value not in seen_values:
                seen_values.add(field_value)
                unique_samples.append(sample)
            else:
                duplicate_count += 1

    # 保存去重后的结果
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in unique_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ 去重完成！")
    print(f"📊 原文件样本数: {total_lines}")
    print(f"🗑️  重复样本数: {duplicate_count}")
    print(f"✅ 去重后样本数: {len(unique_samples)}")
    print(f"💾 结果已保存到: {args.output}")

if __name__ == "__main__":
    main()