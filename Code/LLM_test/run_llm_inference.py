import json
import os
from openai import OpenAI
from tqdm import tqdm

# 配置模型信息
API_KEY = "685308fe-2bc0-4e32-9483-20ea7354e845"
BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
MODEL_NAME = "doubao-seed-2.0-pro"

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 输入输出路径
INPUT_PATH = "/home/wyx/KitPatch-63E8/Code/LLM_test/test_after_sft3.jsonl"
OUTPUT_PATH = "/home/wyx/KitPatch-63E8/Code/LLM_test/llm_output.jsonl"

def call_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用LLM出错: {e}")
        return None

def main():
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
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"处理完成，结果已保存到 {OUTPUT_PATH}，共处理 {len(results)} 个样本")

if __name__ == "__main__":
    main()