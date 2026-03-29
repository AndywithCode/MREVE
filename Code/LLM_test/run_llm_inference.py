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
    "gpt-5.4": {
        "api_key": "sk-88cc79b3fe547f95f19990115bc9e3d388c0320c0c95edababdf706fb6653dcc",
        "base_url": "https://www.ananapi.com/",
        "model": "gpt-5.4"
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