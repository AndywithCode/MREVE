import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= Config =================

MODEL_PATH = "/media/models/Qwen2.5-Coder-1.5B-Instruct"
TRAINED_PATH = "/media/wyx/saved_models/qwencoder_instruct_sft/qwencoder_instruct_sft_2026_03_13_16_48"

TEST_PATH = "/home/wyx/KitPatch-63E8/Datasets/train_data/bigvul/test_data_v1.json"
KB_PATH   = "/home/wyx/KitPatch-63E8/Datasets/train_data/bigvul/sample_data_v1.json"

PROMPT_BASE_PATH = "/home/wyx/KitPatch-63E8/Datasets/prompt/explain_prompt.txt"
PROMPT_RAG_PATH  = "/home/wyx/KitPatch-63E8/Datasets/prompt/rag_explain_prompt.txt"

SAVE_PATH = "/home/wyx/KitPatch-63E8/Results/rag_inference/rag_results.jsonl"

THRESHOLD = 0.65
MAX_NEW_TOKENS = 768


# ================= Load Model =================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_4bit=True
    )

    model = PeftModel.from_pretrained(model, TRAINED_PATH)
    model.eval()

    return model, tokenizer


# ================= Prompt =================

def build_base_prompt(item, tpl):
    return tpl.format(
        language=item["language"],
        contextCode=item["context_code"],
        cveDescription=item["cveDescription"],
        cweName=item["cwe_type"],
        cweDescription=item["cwe_description"],
        commitMessage=item["commit_message"],
        vul=item["vul"]
    ).strip()


def build_rag_prompt(item, tpl):
    return tpl.format(
        language=item["language"],
        contextCode=item["context_code"],
        cveDescription=item["cveDescription"],
        cweName=item["cwe_type"],
        cweDescription=item["cwe_description"],
        commitMessage=item["commit_message"],
        vul=item["vul"],
        similar_vul=item["similar_vul"],
        similar_explain=item["similar_explain"],
        similar_diff=item["similar_diff"]
    ).strip()


# ================= Generation =================

@torch.no_grad()
def generate(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=3072
    ).to(model.device)

    output = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>")
        ],
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(
        output[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=False
    )

    # 截断
    for s in ["<|im_end|>", "<|endoftext|>", "Human:", "\nHuman", "json"]:
        if s in text:
            text = text.split(s)[0]

    return text.strip()


# ================= RAG Pipeline =================

def run_rag():
    model, tokenizer = load_model()

    with open(TEST_PATH) as f:
        test_data = json.load(f)

    with open(KB_PATH) as f:
        kb_data = json.load(f)

    with open(PROMPT_BASE_PATH) as f:
        base_tpl = f.read()

    with open(PROMPT_RAG_PATH) as f:
        rag_tpl = f.read()

    results = []

    for key, item in tqdm(test_data.items()):
        # ===== Step 2: 获取score =====
        score = item["retrieved"][0].get("score", 0.0)

        use_rag = score >= THRESHOLD

        # ===== Step 3/4/5: 分支 =====
        if use_rag:

            prompt = build_rag_prompt(item, rag_tpl)
            mode = "RAG"

        else:
            prompt = build_base_prompt(item, base_tpl)
            mode = "BASE"

        # ===== Step 4/5: 推理 =====
        output = generate(model, tokenizer, prompt)

        # ===== Step 6: 保存 =====
        results.append({
            "id": key,
            "mode": mode,
            "score": score,
            "prompt": prompt,
            "output": output
        })

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Done. Saved to {SAVE_PATH}")


# ================= Main =================

if __name__ == "__main__":
    run_rag()