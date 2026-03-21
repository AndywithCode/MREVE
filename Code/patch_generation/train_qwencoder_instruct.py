import os
import json
import csv
import argparse
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling, 
    BitsAndBytesConfig, 
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)
from reward import compute_reward

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================= Dataset =================

class VulExplainDataset(Dataset):
    def __init__(self, data, tokenizer, prompt_tpl, max_len=3072):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_tpl = prompt_tpl
        self.max_len = max_len

    def build_prompt(self, item):
        user_prompt = self.prompt_tpl.format(
            language=item["language"],
            contextCode=item["context_code"],
            cveDescription=item["cveDescription"],
            cweName=item["cwe_type"],
            cweDescription=item["cwe_description"],
            commitMessage=item["commitMessage"], 
            vul=item["buggy_code"]
        ).strip()
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

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
    def __init__(self, data, tokenizer, prompt_tpl, max_len=3072):
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
        ).strip()
        return prompt

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self.build_prompt(item)
        target_dict = {
            "analysis": item["output"], 
            "localization": item["location"], 
            "repair": item["ground_patch"]
        }
        target_json = json.dumps(target_dict, ensure_ascii=False)
        target = target_json.strip() + "\n<|im_end|>"

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )

        input_ids = torch.tensor(enc["input_ids"])
        labels = input_ids.clone()

        # ===== begin mask prompt: 只对Explanation计算loss =====
        prompt_enc = self.tokenizer(
            prompt + "\n",
            truncation=True,
            max_length=self.max_len,
        )
        prompt_len = len(prompt_enc["input_ids"])

        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:prompt_len] = -100   # 不计算loss
        # ===== end mask prompt: 只对Explanation计算loss =====

        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": labels
        }

    def __len__(self):
        return len(self.data)


# ================= Data =================

def build_dataset(args, isTrain=True):
    path = "sample_data_v1.json" if isTrain else "test_data_v1.json"
    with open(f"/home/wyx/KitPatch-63E8/Datasets/train_data/{args.dataset}/{path}") as f:
        sample_data = json.load(f)

    dataset = []
    for _, meta in sample_data.items():
        dataset.append({
            "language": meta["language"],
            "context_code": meta["context_code"],
            "cveDescription": meta["cveDescription"],
            "cwe_type": meta["cwe_type"],
            "cwe_description": meta["cwe_description"],
            "commitMessage": meta["commit_message"],
            "ground_patch": meta["diff"],
            "buggy_code": meta["vul"],
            "location": meta["location"],
            "output": meta["output"],
        })
    return dataset


# ================= Evaluation =================
@torch.no_grad()
def evaluate(args, model, tokenizer, dataloader, save_path):
    model.eval()
    results = []

    bad_words = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    for batch in tqdm(dataloader, desc="Evaluating"):

        input_ids = batch["input_ids"].cuda()

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].cuda(),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")],
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids
        )

        for i in range(outputs.size(0)):
            # text = tokenizer.decode(outputs[i][input_ids.shape[1]:], skip_special_tokens=True).strip()
            text = tokenizer.decode(outputs[i][input_ids.shape[1]:], skip_special_tokens=False)
            stop_tokens = [
                "<|im_end|>",
                "<|endoftext|>",
                "Human:",
                "\nHuman",
                "json"
            ]
            for s in stop_tokens:
                if s in text:
                    text = text.split(s)[0]
            text = text.strip()

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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model.train()
    return results


@torch.no_grad()
def dump_generation_to_json(args, model, tokenizer, dataloader, save_path, split_name="train"):
    model.eval()
    results = []

    bad_words = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    for batch in tqdm(dataloader, desc=f"Dumping {split_name} generations"):

        input_ids = batch["input_ids"].cuda()

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].cuda(),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")],
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids
        )

        for i in range(outputs.size(0)):
            # gen_text = tokenizer.decode(outputs[i][input_ids.shape[1]:], skip_special_tokens=True).strip()
            gen_text = tokenizer.decode(outputs[i][input_ids.shape[1]:], skip_special_tokens=False)
            stop_tokens = [
                "<|im_end|>",
                "<|endoftext|>",
                "Human:",
                "\nHuman",
                "json",
            ]
            for s in stop_tokens:
                if s in gen_text:
                    gen_text = gen_text.split(s)[0]
            gen_text = gen_text.strip()

            reward, r_format, r_structure, r_repair, r_location, b_json = compute_reward(
                text=gen_text,
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
                "generated_explanation": gen_text,
                "ground_patch": batch["ground_patch"][i],
                "buggy_location": batch["location"][i].tolist(),
                "analysis": batch["analysis"][i],
                "prompt": batch["prompt"][i],
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model.train()


# ================= 按阶段选择模型并dump =================
def dump_by_stage(args, stage="base", ppo_model=None):
    """
    stage: one of ["base", "sft", "ppo"]
    ppo_model: required if stage == "ppo"
    """

    assert stage in ["base", "sft", "ppo"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ===== load prompt =====
    with open("/home/wyx/KitPatch-63E8/Datasets/prompt/explain_prompt.txt") as f:
        generate_prompt = f.read()

    # ===== build dataset =====
    train_raw = build_dataset(args, isTrain=True)
    test_raw = build_dataset(args, isTrain=False)

    train_dataset = VulExplainDataset(train_raw, tokenizer, generate_prompt)
    test_dataset = VulExplainDataset(test_raw, tokenizer, generate_prompt)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ===== load model by stage =====
    if stage == "base":
        print("📦 Using BASE model for dump...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_4bit=True
        )

    elif stage == "sft":
        print("📦 Using SFT model (LoRA) for dump...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_4bit=True
        )
        model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_cfg)
        model.load_adapter(args.sft_save_dir, adapter_name="default")

    elif stage == "ppo":
        print("📦 Using PPO policy model for dump...")
        assert ppo_model is not None, "ppo_model must be provided when stage='ppo'"
        model = ppo_model

    # ===== dump paths =====
    dump_root = os.path.join(args.save_dir, "stage_generations", stage)
    os.makedirs(dump_root, exist_ok=True)

    dump_generation_to_json(
        args, model, tokenizer, train_loader,
        os.path.join(dump_root, "train.jsonl"),
        split_name=f"train_{stage}"
    )

    dump_generation_to_json(
        args, model, tokenizer, test_loader,
        os.path.join(dump_root, "test.jsonl"),
        split_name=f"test_{stage}"
    )

    print(f"✅ Dump finished for stage = {stage}")


# ================= SFT =================

def run_sft(args):

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    with open("/home/wyx/KitPatch-63E8/Datasets/prompt/explain_prompt.txt") as f:
        generate_prompt = f.read()

    raw = build_dataset(args)

    train_dataset = VulExplainSFTDataset(raw, tokenizer, generate_prompt)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_4bit=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    sft_save_dir = os.path.join(args.sft_save_dir, f"qwencoder_instruct_sft_{now}")
    os.makedirs(sft_save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=sft_save_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=args.sft_epoch,
        logging_steps=20,
        save_steps=500,
        bf16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("🚀 Start SFT...")
    trainer.train()
    trainer.save_model(sft_save_dir)

    # ===== Dump generations after SFT =====
    print("📦 Dumping train/test generations after SFT...")

    with open("/home/wyx/KitPatch-63E8/Datasets/prompt/explain_prompt.txt") as f:
        generate_prompt = f.read()

    train_raw = build_dataset(args, isTrain=True)
    test_raw = build_dataset(args, isTrain=False)

    gen_train_dataset = VulExplainDataset(train_raw, tokenizer, generate_prompt)
    gen_test_dataset = VulExplainDataset(test_raw, tokenizer, generate_prompt)

    train_loader = DataLoader(gen_train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(gen_test_dataset, batch_size=1, shuffle=False)

    dump_dir = os.path.join(sft_save_dir, "final_generations")
    os.makedirs(dump_dir, exist_ok=True)

    dump_generation_to_json(
        args, model, tokenizer, train_loader,
        os.path.join(dump_dir, "train_after_sft.jsonl"),
        split_name="train"
    )

    dump_generation_to_json(
        args, model, tokenizer, test_loader,
        os.path.join(dump_dir, "test_after_sft.jsonl"),
        split_name="test"
    )

    print("✅ SFT generation dump finished.")

    return sft_save_dir


# ================= PPO =================

def train_ppo(args):

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # load_in_4bit=True, 
        quantization_config=bnb_config,
    )
    policy_model.config.use_cache = False
    policy_model.pretrained_model.config.use_cache = False
    policy_model.pretrained_model = prepare_model_for_kbit_training(
        policy_model.pretrained_model
    )
    policy_model.pretrained_model.gradient_checkpointing_enable()
    policy_model.pretrained_model = get_peft_model(policy_model.pretrained_model, lora_cfg)
    policy_model.pretrained_model.load_adapter(args.sft_save_dir, adapter_name="default")
    policy_model.train()

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # load_in_4bit=True, 
        quantization_config=bnb_config, 
    )
    ref_model.config.use_cache = False
    ref_model.pretrained_model.config.use_cache = False
    ref_model.pretrained_model = prepare_model_for_kbit_training(
        ref_model.pretrained_model
    )
    ref_model.pretrained_model = get_peft_model(ref_model.pretrained_model, lora_cfg)
    ref_model.pretrained_model.load_adapter(args.sft_save_dir, adapter_name="default")
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    with open("/home/wyx/KitPatch-63E8/Datasets/prompt/explain_prompt.txt") as f:
        generate_prompt = f.read()

    train_data = build_dataset(args)
    train_dataset = VulExplainDataset(train_data, tokenizer, generate_prompt)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_data = build_dataset(args, isTrain=False)
    test_dataset = VulExplainDataset(test_data, tokenizer, generate_prompt)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ppo_config = PPOConfig(
        learning_rate=2e-6,
        batch_size=1,
        mini_batch_size=1,
        ppo_epochs=args.ppo_epoch,
        target_kl=0.08,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_path = f"/home/wyx/KitPatch-63E8/Results/logs/qwencoder_instruct/log_{now}"
    os.makedirs(log_path, exist_ok=True)
    log_file = open(os.path.join(log_path, "train_log.csv"), "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["epoch", "step", "mean_scores", "policykl", "loss_total", "loss_policy", "loss_value", "entropy"])

    print("🚀 Start PPO Training...")

    log_dir = os.path.join(args.save_dir, "eval_logs")
    os.makedirs(log_dir, exist_ok=True)

    bad_words = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    for epoch in range(args.ppo_epoch):
        for step, batch in enumerate(tqdm(train_loader)):

            input_ids = batch["input_ids"].cuda()

            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

            with torch.no_grad():
                response = policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=batch["attention_mask"].cuda(),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8, 
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")],
                    pad_token_id=tokenizer.eos_token_id,
                    bad_words_ids=bad_words_ids
                )

            responses = []
            rewards = []
            for i in range(response.size(0)):
                resp_ids = response[i][input_ids.shape[1]:]
                resp_ids = truncate_response(resp_ids, tokenizer)

                # text = tokenizer.decode(response[i][input_ids.shape[1]:], skip_special_tokens=True).strip()
                # text = tokenizer.decode(response[i][input_ids.shape[1]:], skip_special_tokens=False)
                text = tokenizer.decode(resp_ids, skip_special_tokens=False)
                stop_tokens = [
                    "<|im_end|>",
                    "<|endoftext|>",
                    "Human:",
                    "\nHuman",
                    "json"
                ]
                for s in stop_tokens:
                    if s in text:
                        text = text.split(s)[0]
                text = text.strip()

                r, _, _, _, _, _ = compute_reward(
                    text=text,
                    ref_analysis=batch["analysis"][i],
                    ground_patch=batch["ground_patch"][i],
                    buggy_lines=batch["location"][i].tolist()
                )
                # responses.append(response[i][input_ids.shape[1]:])
                responses.append(resp_ids)
                rewards.append(r)

            # rewards = torch.tensor(rewards).cuda()

            query_tensors = [q for q in input_ids]
            response_tensors = responses
            stats = ppo_trainer.step(
                queries=query_tensors,
                responses=response_tensors,
                # scores=[rewards],
                scores=rewards.detach().cpu().tolist()
            )
            # stats = ppo_trainer.step(
            #     queries=[input_ids[0]],
            #     responses=[response[0]],
            #     scores=[rewards[0]],
            # )

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
        evaluate(args, policy_model, tokenizer, test_loader, os.path.join(log_path, f"test_epoch_{epoch}.jsonl"))
        print(f"Epoch {epoch} finished")

    log_file.close()

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = os.path.join(args.save_dir, f"qwencoder_instruct_ppo_{now}")
    os.makedirs(save_dir, exist_ok=True)
    ppo_trainer.save_pretrained(save_dir)

    # ===== Final dump after PPO training =====
    print("📦 Dumping final train/test generations after PPO...")

    final_dump_dir = os.path.join(save_dir, "final_generations")
    os.makedirs(final_dump_dir, exist_ok=True)

    dump_generation_to_json(
        args, policy_model, tokenizer, train_loader,
        os.path.join(final_dump_dir, "train_after_ppo.jsonl"),
        split_name="train"
    )

    print("✅ Final generation dump finished.")

def truncate_response(resp_ids, tokenizer):

    stop_ids = {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>")
    }

    for idx, tok in enumerate(resp_ids):
        if tok.item() in stop_ids:
            return resp_ids[:idx+1]

    return resp_ids


# ================= Main =================

if __name__ == "__main__":
    # codeqwen1.5 max_in+max_out=65536
    # qwen2.5-coder max_in+max_out=32k
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bigvul")
    parser.add_argument("--ppo_epoch", type=int, default=3)
    parser.add_argument("--sft_epoch", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=768)

    parser.add_argument("--base_model_path", type=str,
                        default="/media/models/Qwen2.5-Coder-1.5B-Instruct", help="Original model path")

    parser.add_argument("--sft_save_dir", type=str,
                        default="/media/wyx/saved_models/qwencoder_instruct_sft", help="Saved model after sft")

    parser.add_argument("--save_dir", type=str,
                        default="/media/wyx/saved_models/qwencoder_instruct_ppo", help="Saved model after ppo")

    args = parser.parse_args()

    sft_dir = run_sft(args)
    args.sft_save_dir = sft_dir

    # dump_by_stage(args, stage="base")
    # dump_by_stage(args, stage="sft")
    # dump_by_stage(args, stage="ppo", ppo_model=policy_model)

    train_ppo(args)
