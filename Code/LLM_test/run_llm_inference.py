import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

# 模型API配置池 - 新增模型在这里添加配置
MODEL_CONFIGS = {
    "doubao-seed-2.0-pro": {
        "api_key": "685308fe-2bc0-4e32-9483-20ea7354e845",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "model": "doubao-seed-2.0-pro"
    },
    "doubao-seed-2.0-code": {
        "api_key": "685308fe-2bc0-4e32-9483-20ea7354e845",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "model": "doubao-seed-2.0-code"
    },
    "gpt-5.4": {
        "api_key": "sk-70e924fa69410befe4bad7d42a7fc18f8b07936d1a5edae43ee68bcb62985b4c",
        "base_url": "https://codex.sakurapy.de/v1/",
        "model": "gpt-5.4"
    }, 
    "deepseek-v3.2": {
        "api_key": "sk-7a1c889ae2b04cfd8db07d74c8201cb9",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    }
}

def main():
    parser = argparse.ArgumentParser(description="调用LLM API进行推理并保存结果")
    # 支持直接选择模型池中的预配置模型
    parser.add_argument("--use-model", "-c", choices=MODEL_CONFIGS.keys(), help="使用预配置的模型，直接从模型池中加载配置")
    # 自定义配置参数（优先级高于预配置）
    parser.add_argument("--api-key", "-k", help="LLM API密钥（自定义时必填）")
    parser.add_argument("--base-url", "-u", help="API基础URL（自定义时必填）")
    parser.add_argument("--model", "-m", help="模型名称（自定义时必填）")
    parser.add_argument("--input", "-i", default="/home/wyx/KitPatch-63E8/Code/LLM_test/test_after_sft3.jsonl", help="输入测试数据jsonl路径")
    parser.add_argument("--output", "-o", default="/home/wyx/KitPatch-63E8/Code/LLM_test/llm_output.jsonl", help="输出结果jsonl路径")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="生成温度")
    parser.add_argument("--max-tokens", "-l", type=int, default=2048, help="最大生成长度")
    args = parser.parse_args()

    # 加载配置
    api_key = args.api_key
    base_url = args.base_url
    model_name = args.model

    # 如果使用预配置模型，从池中加载
    if args.use_model:
        config = MODEL_CONFIGS[args.use_model]
        api_key = config["api_key"]
        base_url = config["base_url"]
        model_name = config["model"]
        # 允许自定义参数覆盖预配置
        if args.api_key:
            api_key = args.api_key
        if args.base_url:
            base_url = args.base_url
        if args.model:
            model_name = args.model

    # 检查必填参数
    if not api_key or not base_url or not model_name:
        print("❌ 错误：必须指定--use-model选择预配置模型，或者同时提供--api-key、--base-url和--model参数")
        exit(1)

    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    def call_llm(prompt):
        try:
            response = client.chat.completions.create(
                model=model_name,
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
        for line in f:
            samples.append(json.loads(line))

    # 断点续传：检查输出文件是否存在，读取已处理的样本
    processed_samples = []
    processed_prompts = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                processed_samples.append(sample)
                if "prompt" in sample:
                    processed_prompts.add(sample["prompt"])
        print(f"🔍 发现已有输出文件，已处理 {len(processed_samples)} 个样本，将从断点继续")

    # 过滤出未处理的样本
    unprocessed_samples = []
    for sample in samples:
        prompt = sample.get("prompt", "")
        if not prompt:
            print("样本缺少prompt字段，跳过")
            continue
        if prompt not in processed_prompts:
            unprocessed_samples.append(sample)

    print(f"📋 总样本数: {len(samples)}, 已处理: {len(processed_samples)}, 待处理: {len(unprocessed_samples)}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 处理每个样本，边处理边写入（避免中途丢失结果）
    new_processed_count = 0
    with open(args.output, 'a', encoding='utf-8') as f:
        for sample in tqdm(unprocessed_samples, desc="处理样本中"):
            prompt = sample["prompt"]
            llm_output = call_llm(prompt)
            if llm_output:
                sample["llm_output"] = llm_output
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f.flush()  # 立即写入磁盘，避免缓存丢失
                new_processed_count += 1

    total_processed = len(processed_samples) + new_processed_count
    print(f"✅ 处理完成，结果已保存到 {args.output}")
    print(f"📊 本次新处理: {new_processed_count} 个样本，累计处理: {total_processed} 个样本")
    if total_processed == len(samples):
        print("🎉 所有样本已全部处理完成！")

if __name__ == "__main__":
    main()