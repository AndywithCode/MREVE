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
    "kimi-k2.5": {
        "api_key": "685308fe-2bc0-4e32-9483-20ea7354e845",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "model": "kimi-k2.5"
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
    },
    "claude-sonnet-4.6": {
        "api_key": "sk-1pC2G9tUaAYrVzvomglnuOcrD5FDJBTdeEWk48o90vusFeN9",
        "base_url": "https://api.nih.cc/v1",
        "model": "anthropic/claude-sonnet-4.6"
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

    def call_llm(prompt, max_retries=15):
        import time
        # 每个请求前先等待3秒，严格控制请求速率
        time.sleep(3)

        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_msg = str(e)
                print(f"调用LLM出错 (第{attempt}/{max_retries}次): {e}")

                # 针对不同错误类型调整等待时间
                if "429" in error_msg or "Rate limit exceeded" in error_msg:
                    # 速率限制错误，等待更长时间
                    wait_time = 60 * attempt
                    print(f"检测到速率限制，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                elif "500" in error_msg or "Cursor API" in error_msg:
                    # 服务端错误，等待更长时间
                    wait_time = 45 * attempt
                    print(f"检测到服务端错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                elif attempt < max_retries:
                    # 其他错误，按指数退避
                    wait_time = 10 * attempt
                    time.sleep(wait_time)
        print("已达最大重试次数，跳过该样本")
        return None

    # 读取所有样本
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    # 断点续传：按 generated_explanation 字段逐项对比，找出缺失样本
    processed_keys = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    key = sample.get("generated_explanation", "")
                    if key:
                        processed_keys.add(key)
                except json.JSONDecodeError:
                    continue
        print(f"🔍 发现已有输出文件，已处理 {len(processed_keys)} 个样本（按generated_explanation匹配），将补全缺失样本")

    # 按 generated_explanation 找出缺失样本
    unprocessed_samples = []
    for sample in samples:
        key = sample.get("generated_explanation", "")
        prompt = sample.get("prompt", "")
        if not prompt:
            print("样本缺少prompt字段，跳过")
            continue
        if not key or key not in processed_keys:
            unprocessed_samples.append(sample)

    print(f"📋 总样本数: {len(samples)}, 已处理: {len(processed_keys)}, 待处理: {len(unprocessed_samples)}")

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:  # 只有当输出路径包含目录时才创建目录
        os.makedirs(output_dir, exist_ok=True)

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

    total_processed = len(processed_keys) + new_processed_count
    print(f"✅ 处理完成，结果已保存到 {args.output}")
    print(f"📊 本次新处理: {new_processed_count} 个样本，累计处理: {total_processed} 个样本")
    if total_processed == len(samples):
        print("🎉 所有样本已全部处理完成！")

if __name__ == "__main__":
    main()