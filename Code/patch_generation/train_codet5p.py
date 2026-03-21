import os
import csv
import json
import argparse
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from reward import compute_reward
from peft import LoraConfig, get_peft_model
import torch.distributed as dist

# ================= Utils =================
def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")

# ================= Dataset =================

class VulExplainDataset(Dataset):
    def __init__(self, data, tokenizer, prompt_tpl, max_len=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_tpl = prompt_tpl
        self.max_len = max_len

    def build_prompt(self, item):
        prompt = self.prompt_tpl.format(
            language=item["language"],
            contextCode=item["context_code"],
            cveDescription=item["cveDescription"],
            cweName=item["cwe_type"],
            cweDescription=item["cwe_description"],
            commitMessage=item["commitMessage"], 
            vul=item["buggy_code"]
        )
        return prompt.strip()

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self.build_prompt(item)

        enc = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt": prompt,
            "ground_patch": item["ground_patch"],
            "location": torch.tensor(item["location"], dtype=torch.long),
            "analysis": item["output"],
        }

    def __len__(self):
        return len(self.data)


class VulExplainSFTDataset(Dataset):
    def __init__(self, data, tokenizer, prompt_tpl, max_len=1024, tgt_max_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_tpl = prompt_tpl
        self.max_len = max_len
        self.tgt_max_len = tgt_max_len

    def build_prompt(self, item):
        return self.prompt_tpl.format(
            language=item["language"],
            contextCode=item["context_code"],
            cveDescription=item["cveDescription"],
            cweName=item["cwe_type"],
            cweDescription=item["cwe_description"],
            commitMessage=item["commitMessage"], 
            vul=item["buggy_code"]
        ).strip()

    def build_target(self, item):
        target = item["target"]
        return target.strip()

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self.build_prompt(item)
        target = self.build_target(item)

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
        )

        dec = self.tokenizer(
            target,
            truncation=True,
            max_length=self.tgt_max_len,
        )

        labels = dec["input_ids"]
        labels = [x if x != self.tokenizer.pad_token_id else -100 for x in labels]

        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels)
        }

    def __len__(self):
        return len(self.data)

# ================= Data Loader =================

def build_dataset(args, isTrain=True):
    path = "sample_data_v1.json" if isTrain else "test_data_v1.json"
    with open(f"/home/wyx/KitPatch-63E8/Datasets/train_data/{args.dataset}/{path}") as f:
        sample_data = json.load(f)

    dataset = []
    for _, meta in tqdm(sample_data.items()):

        target_dict = {
            "analysis": meta["output"], 
            "localization": meta["location"], 
            "repair": meta["diff"]
        }

        sample = {
            "language": meta["language"],
            "context_code": meta["context_code"],
            "cveDescription": meta["cveDescription"],
            "cwe_type": meta["cwe_type"],
            "cwe_description": meta["cwe_description"],
            "commitMessage": meta["commit_message"],
            "ground_patch": meta["diff"],
            "buggy_code": meta["vul"], 
            "target": json.dumps(target_dict, ensure_ascii=False)
        }
        dataset.append(sample)

    return dataset


# ================= Evaluation =================

@torch.no_grad()
def evaluate(args, model, tokenizer, dataloader, save_path, split_name="tarin"):
    model.eval()
    results = []
    bad_words_ids = tokenizer(
        ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"],
        add_special_tokens=False
    ).input_ids

    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            bad_words_ids=bad_words_ids
        )

        for i in range(outputs.size(0)):
            text = tokenizer.decode(outputs[i], skip_special_tokens=True)

            reward, r_format, r_structure, r_repair, r_location, b_json = compute_reward(
                text=text,
                ref_analysis=batch["analysis"][i],
                ground_patch=batch["ground_patch"][i],
                buggy_lines=batch["location"][i].tolist()
            )

            results.append({
                "b_json": b_json,
                "reward": float(reward),
                "r_format": float(r_format),
                "r_structure": float(r_structure),
                "r_repair": float(r_repair),
                "r_location": float(r_location), 
                "generated_explanation": text,
                "ground_patch": batch["ground_patch"][i],
                "buggy_location": batch["location"][i].tolist(),
                "analysis": batch["analysis"][i],
                "prompt": batch["prompt"][i],
            })

    with open(save_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model.train()
    return results


# ================= Training =================

def run_sft(args, device):

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_data = build_dataset(args)
    with open("/home/wyx/KitPatch-63E8/Datasets/prompt/explain_prompt.txt") as f:
        generate_prompt = f.read()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        # device_map={"": device.index}, 
        trust_remote_code=True
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id

    model.tie_weights()

    
    # ===== LoRA =====
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        # target_modules=["q", "v"]  # CodeT5+ T5结构通用 for 770m
        target_modules=["q_proj", "v_proj"] # for 2b
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # ================

    dataset = VulExplainSFTDataset(raw_data, tokenizer, generate_prompt)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )
    
    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    sft_save_dir = os.path.join(args.sft_save_dir, os.path.basename(os.path.normpath(args.base_model_path))+f"_{now}")
    os.makedirs(sft_save_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=sft_save_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=args.sft_epoch,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    print("🚀 Start SFT Training...")
    trainer.train()
    trainer.save_model(sft_save_dir)
    print("🚀 SFT Finished.")

    # ===== dump after sft =====
    print("📦 Dump after SFT...")

    test_raw = build_dataset(args, False)

    gen_train = VulExplainDataset(raw_data, tokenizer, generate_prompt)
    gen_test = VulExplainDataset(test_raw, tokenizer, generate_prompt)

    train_loader = DataLoader(gen_train, batch_size=1, shuffle=False)
    test_loader = DataLoader(gen_test, batch_size=1, shuffle=False)

    dump_dir = os.path.join(sft_save_dir, "final_generations")
    os.makedirs(dump_dir, exist_ok=True)

    evaluate(args, model, tokenizer, train_loader, 
             os.path.join(dump_dir, "train_after_sft.jsonl"), "train")

    evaluate(args, model, tokenizer, test_loader, 
             os.path.join(dump_dir, "test_after_sft.jsonl"), "test")

    return sft_save_dir


def train_ppo(args, device):

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    #     args.sft_save_dir,
    #     torch_dtype=torch.bfloat16,
    #     # device_map="auto", 
    #     device_map={"": device.index},
    #     trust_remote_code=True
    # )

    # ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    #     args.sft_save_dir,
    #     torch_dtype=torch.bfloat16,
    #     # device_map="auto", 
    #     device_map={"": device.index},
    #     trust_remote_code=True
    # )

    # ===== load SFT LoRA model with value head =====
    policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        args.sft_save_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        # device_map={"": device.index},
        trust_remote_code=True
    ).cuda()

    policy_model.pretrained_model.gradient_checkpointing_enable()
    policy_model.pretrained_model.config.use_cache = False

    # ref model
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        args.sft_save_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        # device_map={"": device.index},
        # device_map={"": "cpu"},
        trust_remote_code=True
    ).cuda()
    # ===============================================
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    with open("/home/wyx/KitPatch-63E8/Datasets/prompt/explain_prompt.txt") as f:
        generate_prompt = f.read()

    raw_data = build_dataset(args)
    test_data = build_dataset(args, False)
    train_dataset = VulExplainDataset(raw_data, tokenizer, generate_prompt)
    test_dataset = VulExplainDataset(test_data, tokenizer, generate_prompt)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ppo_config = PPOConfig(
        model_name="codet5p-770m",
        learning_rate=1e-5,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        ppo_epochs=args.ppo_epoch,
        max_grad_norm=1.0,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.5,
        target_kl=0.1,
        log_with=None
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer
    )

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_path = f"/home/wyx/KitPatch-63E8/Results/logs/codet5p/log_{now}"
    os.makedirs(log_path, exist_ok=True)
    log_file = open(os.path.join(log_path, "train_log.csv"), "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["epoch", "step", "mean_scores", "policykl", "loss_total", "loss_policy", "loss_value", "entropy"])

    print("🚀 Start PPO Training...")

    bad_words_ids = tokenizer(
        ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"],
        add_special_tokens=False
    ).input_ids
    for epoch in range(args.ppo_epoch):
        for step, batch in enumerate(tqdm(train_dataloader)):

            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            # ====== generate ======
            with torch.no_grad():
                outputs = policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    bad_words_ids=bad_words_ids
                )

            responses = []
            rewards = []

            for i in range(outputs.size(0)):
                text = tokenizer.decode(outputs[i], skip_special_tokens=True)

                r, _, _, _, _, _ = compute_reward(
                    text=text,
                    ref_analysis=batch["analysis"][i],
                    ground_patch=batch["ground_patch"][i],
                    buggy_lines=batch["location"][i].tolist()
                )

                responses.append(outputs[i])
                rewards.append(r)

            rewards = torch.tensor(rewards).cuda()

            # ====== PPO Step ======
            query_tensors = [q for q in input_ids]
            response_tensors = responses
            score_list = [rewards]
            stats = ppo_trainer.step(
                queries=query_tensors,
                responses=response_tensors,
                scores=score_list
            )

            logger.writerow([
                epoch, step, 
                float(stats["ppo/mean_scores"]), 
                float(stats["ppo/policy/policykl"]), 
                float(stats["ppo/loss/total"]), 
                float(stats["ppo/loss/policy"]), 
                float(stats["ppo/loss/value"]), 
                float(stats["ppo/policy/entropy"]), 
            ])

            if step % 20 == 0:
                print(f"Epoch {epoch} Step {step} Reward {rewards.mean().item():.3f}")

        # ====== evaluation each epoch =====
        print(f"Epoch {epoch} evaluation...")
        evaluate(args, policy_model, tokenizer, test_dataloader, os.path.join(log_path, f"test_epoch_{epoch}.jsonl"))
        print(f"Epoch {epoch} finished")

    log_file.close()
    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = os.path.join(args.save_dir, f"codet5p_ppo_{now}")
    os.makedirs(save_dir, exist_ok=True)
    ppo_trainer.save_pretrained(save_dir)

    # ===== final dump =====
    print("📦 Dump final after PPO...")

    final_dir = os.path.join(args.save_dir, "final_generations")
    os.makedirs(final_dir, exist_ok=True)

    evaluate(args, policy_model, tokenizer, train_dataloader, 
             os.path.join(final_dir, "train_after_ppo.jsonl"), "train")


# ================= Main =================
# torchrun --nproc_per_node=2 train_codet5p.py
if __name__ == "__main__":
    # codet5p max_in=2048, max_out=2048
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bigvul")
    parser.add_argument("--ppo_epoch", type=int, default=3)
    parser.add_argument("--sft_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="/media/wyx/saved_models/codet5p_ppo", help="Saved model after ppo")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--base_model_path", type=str, default="/media/models/Salesforce/codet5p-770m", help="Original model path")
    parser.add_argument("--sft_save_dir", type=str, default="/media/wyx/saved_models/codet5p_sft/codet5p-770m_2026_01_16_16_16", help="Saved model after sft")
    args = parser.parse_args()

    # device = setup_ddp()
    device = {}

    sft_dir = run_sft(args, device)
    args.sft_save_dir = sft_dir

    train_ppo(args, device)
