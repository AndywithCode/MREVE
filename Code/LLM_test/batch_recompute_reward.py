import os
import subprocess
import json

# 配置
MODELS = [
    "groundtruth",
    "gpt-5.4",
    "doubao-seed-2.0-pro",
    "doubao-seed-2.0-code",
    "deepseek-v3.2",
    "claude-sonnet-4.6",
    "kimi-k2.5"
]

RESULT_DIR = "/home/wyx/KitPatch-63E8/Code/LLM_test/result"
os.makedirs(RESULT_DIR, exist_ok=True)

def run_command(cmd, description):
    """运行命令并返回是否成功"""
    print(f"\n{'='*80}")
    print(f"🚀 开始执行: {description}")
    print(f"命令: {cmd}")
    print('='*80)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)

        if result.returncode == 0:
            print(f"\n✅ {description} 执行成功")
            return True
        else:
            print(f"\n❌ {description} 执行失败，返回码: {result.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ {description} 执行异常: {str(e)}")
        return False

def main():
    print("🔄 开始批量重新计算所有模型的reward得分")
    print(f"📁 结果输出目录: {RESULT_DIR}")
    print(f"🤖 待处理模型: {', '.join(MODELS)}")

    # 1. 首先处理groundtruth
    print("\n" + "="*100)
    print("📌 第1步: 重新计算groundtruth reward")
    success = run_command("python compute_groundtruth_reward.py", "计算groundtruth reward")
    if not success:
        print("❌ groundtruth计算失败，终止流程")
        return

    success = run_command(f"python stat_reward.py --input {RESULT_DIR}/groundtruth_with_reward.jsonl --output-dir {RESULT_DIR} --prefix groundtruth", "生成groundtruth统计图表")
    if not success:
        print("❌ groundtruth统计生成失败")

    # 2. 处理其他模型
    for i, model in enumerate(MODELS[1:], 2):
        print("\n" + "="*100)
        print(f"📌 第{i}步: 处理模型 {model}")

        # 检查是否已经有去重后的文件
        dedup_file = f"{model}_output_deduplicated.jsonl"
        if not os.path.exists(dedup_file):
            # 先去重
            input_file = f"{model}_output.jsonl"
            if not os.path.exists(input_file):
                print(f"❌ 找不到模型 {model} 的输入文件 {input_file}，跳过")
                continue

            success = run_command(f"python deduplicate_jsonl.py --input {input_file} --output {dedup_file}", f"{model} 输出文件去重")
            if not success:
                print(f"❌ {model} 去重失败，跳过")
                continue

        # 计算reward
        output_file = f"{RESULT_DIR}/{model}_output_with_reward.jsonl"
        success = run_command(f"python compute_reward_score.py --input {dedup_file} --output {output_file}", f"计算 {model} 的reward得分")
        if not success:
            print(f"❌ {model} reward计算失败，跳过")
            continue

        # 生成统计图表
        success = run_command(f"python stat_reward.py --input {output_file} --output-dir {RESULT_DIR} --prefix {model}", f"生成 {model} 的统计结果和图表")
        if not success:
            print(f"❌ {model} 统计图表生成失败")

    # 3. 生成多模型对比图表
    print("\n" + "="*100)
    print("📌 最后一步: 生成所有模型的r_structure对比图表")
    success = run_command("python compare_r_structure.py", "生成多模型对比图表")
    if success:
        print("\n🎉 所有模型处理完成！")
    else:
        print("\n⚠️  对比图表生成失败")

if __name__ == "__main__":
    main()
