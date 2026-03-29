import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="调用LLM API进行推理并保存结果")
    parser.add_argument("--api-key", "-k", default="685308fe-2bc0-4e32-9483-20ea7354e845", help="LLM API密钥")
    parser.add_argument("--base-url", "-u", default="https://ark.cn-beijing.volces.com/api/coding/v3", help="API基础URL")
    parser.add_argument("--model", "-m", default="doubao-seed-2.0-pro", help="模型名称")
    parser.add_argument("--input", "-i", default="/home/wyx/KitPatch-63E8/Code/LLM_test/test_after_sft3.jsonl", help="输入测试数据jsonl路径")
    parser.add_argument("--output", "-o", default="/home/wyx/KitPatch-63E8/Code/LLM_test/llm_output.jsonl", help="输出结果jsonl路径")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="生成温度")
    parser.add_argument("--max-tokens", "-l", type=int, default=2048, help="最大生成长度")
    args = parser.parse_args()

    # 初始化客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    def call_llm(prompt):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"调用LLM出错: {e}")
            return None

    # 读取所有样本
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
    # 读取所有样本
    samples = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    # 处理每个样本
    results = []
    for sample in tqdm(samples, desc="处理样本中"):
        prompt = sample.get("prompt", "")
        if not prompt:
            print("样本缺少prompt字段，跳过")
            continue
        llm_output = call_llm(prompt)
        if llm_output:
            sample["llm_output"] = llm_output
            results.append(sample)

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"处理完成，结果已保存到 {args.output}，共处理 {len(results)} 个样本")

if __name__ == "__main__":
    main()