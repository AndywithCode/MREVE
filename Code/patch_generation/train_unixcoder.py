import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from ppomodel import UniXcoderPPO, generate_rollout, ppo_update
from reward import compute_reward
import json
from tqdm import tqdm
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "/media/models/unixcoder/unixcoder-base-nine"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
m = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = UniXcoderPPO(m).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

with open(f"/home/wyx/KitPatch-63E8/Datasets/prompt/generate_prompt.txt") as f:
    generate_prompt = f.read()

def train_step(sample):
    prompt = generate_prompt.format(language=sample["language"], 
                                    contextCode=sample["context_code"], 
                                    cveDescription=sample["cveDescription"], 
                                    cweName=sample["cwe_type"], 
                                    cweDescription=sample["cwe_description"], 
                                    commitMessage=sample["commitMessage"])

    rollout = generate_rollout(DEVICE, model, tokenizer, prompt)

    reward, _, _, _ , _= compute_reward(
        rollout["text"],
        sample["ground_patch"],
        # sample["buggy_lines"],
        sample["buggy_code"],
    )

    loss = ppo_update(
        model,
        tokenizer,
        optimizer,
        prompt,
        rollout,
        reward,
    )

    return {
        "reward": reward,
        "loss": loss,
    }

def train(train_dataset, epochs=3, save_dir="/home/wyx/KitPatch-63E8/Results/explain_result/bigvul/checkpoints", log_interval=10):
    model.train()
    step = 0
    all_logs = []

    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        
        for sample in tqdm(train_dataset):
            stats = train_step(sample)
            step += 1

            if step % log_interval == 0:
                print(
                    f"[step {step}] "
                    f"loss={stats['loss']:.4f} "
                    f"reward={stats['reward']:.3f}"
                )

            all_logs.append({
                "step": step,
                "loss": stats["loss"],
                "reward": stats["reward"]
            })

        # 每个 epoch 存一次模型
        torch.save(
            model.state_dict(),
            f"{save_dir}/unixcoder_ppo_epoch{epoch+1}.pt"
        )

    # 保存训练日志
    with open(f"{save_dir}/train_log.json", "w") as f:
        json.dump(all_logs, f, indent=2)

def build_train_dataset(args):
    with open(f"/home/wyx/KitPatch-63E8/Datasets/train_data/{args.dataset}/sample_data.json", "r") as f:
        sample_data = json.load(f)
    train_dataset = []
    for file, meta in tqdm(sample_data.items()):
        sample = {
            "language": meta["language"],
            "context_code": meta["context_code"],
            "cveDescription": meta["cveDescription"],
            "cwe_type": meta["cwe_type"],
            "cwe_description": meta["cwe_description"],
            "commitMessage": meta["commit_message"],
            "buggy_lines": meta["location"],
            "ground_patch": meta["diff"],
            "buggy_code": meta["vul"]
        }
        train_dataset.append(sample)
    return train_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bigvul",
                        help="Dataset to use")
    parser.add_argument("--epoch", type=int, default=3)
    args = parser.parse_args()

    train_dataset = build_train_dataset(args)
    print(f"Loaded {len(train_dataset)} training samples.")

    train(
        train_dataset=train_dataset,
        epochs=args.epoch,
        save_dir="./checkpoints",
        log_interval=20
    )