import torch
import torch.nn as nn
import torch.nn.functional as F
from repair_kitpatch import run_validation


# for ppo, we need both policy head and value head
class UniXcoderPPO(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        hidden = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        values = self.value_head(outputs.hidden_states[-1]).squeeze(-1)
        return logits, values


# 生成 + logprob + value
def generate_rollout(DEVICE, model, tokenizer, prompt, max_new_tokens=256):
    model.eval()
    prompt = "<decoder-only>\n" + prompt + "\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        gen = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            # do_sample=False,
            # num_beams=1,
            # temperature=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = gen.sequences
    scores = torch.stack(gen.scores, dim=1)

    log_probs = torch.log_softmax(scores, dim=-1)
    chosen_log_probs = log_probs.gather(
        -1, gen_ids[:, -scores.size(1):].unsqueeze(-1)
    ).squeeze(-1).sum(dim=1)

    _, values = model(
        gen_ids,
        torch.ones_like(gen_ids),
    )

    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # 只 decode 新生成部分
    # gen_len = gen_ids.shape[1] - inputs["input_ids"].shape[1]
    # generated_ids = gen_ids[0, -gen_len:]

    # text = tokenizer.decode(
    #     generated_ids,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=True
    # )

    text = text.replace("<decoder-only>", "").strip()

    return {
        "text": text,
        "log_probs": chosen_log_probs.detach(),
        "values": values[:, -1].detach(),
        "input_ids": gen_ids,
    }

# advantage
def compute_advantage(reward, value, gamma=0.99, lam=0.95):
    advantage = reward - value
    return advantage.detach()

def ppo_loss(new_log_probs, old_log_probs, advantage, value, reward, clip_eps=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)

    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantage, clipped * advantage)

    value_loss = F.mse_loss(value, torch.tensor([reward]).to(value.device))

    return policy_loss + 0.5 * value_loss

def ppo_update(model, tokenizer, optimizer, prompt, rollout, reward):
    model.train()

    advantage = compute_advantage(reward, rollout["values"])

    logits, values = model(
        rollout["input_ids"],
        torch.ones_like(rollout["input_ids"]),
    )

    log_probs = F.log_softmax(logits, dim=-1)
    new_log_probs = log_probs.gather(
        -1, rollout["input_ids"].unsqueeze(-1)
    ).squeeze(-1).sum(dim=1)

    loss = ppo_loss(
        new_log_probs,
        rollout["log_probs"],
        advantage,
        values[:, -1],
        reward,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
