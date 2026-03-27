import re
import os
import json
from difflib import SequenceMatcher
# from bert_score import score as bert_score
from bert_score import BERTScorer

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/media/models"
# os.environ["TRANSFORMERS_CACHE"] = "/media/models"

MODEL_TYPE = "distilroberta-base"
MODEL_PATH = "/media/models/distilroberta-base"

bert_scorer = None

def get_bert_scorer():
    global bert_scorer

    if bert_scorer is None:
        bert_scorer = BERTScorer(
            model_type=MODEL_TYPE,
            lang="en",
            device="cuda:1",
            rescale_with_baseline=True
        )

    return bert_scorer

def _trim_to_valid_json(text):
    """
    从第一个 { 开始扫描，找到 balance=0 的位置
    截断为一个完整 JSON object
    """
    start = text.find("{")
    if start == -1:
        return None

    balance = 0
    end_pos = None

    for i in range(start, len(text)):
        c = text[i]

        if c == "{":
            balance += 1
        elif c == "}":
            balance -= 1

            if balance == 0:
                end_pos = i
                break

    if end_pos is None:
        return text[start:]

    return text[start:end_pos + 1]


def _fix_brackets(text):
    """
    修复 {} 和 [] 数量不一致问题
    """
    open_b = text.count("{")
    close_b = text.count("}")

    if open_b > close_b:
        text += "}" * (open_b - close_b)

    open_s = text.count("[")
    close_s = text.count("]")

    if open_s > close_s:
        text += "]" * (open_s - close_s)

    return text


def _clean_json_text(text):
    """
    清理 LLM 常见 JSON 噪声
    """
    # 删除注释
    text = re.sub(r"//.*", "", text)

    # 删除 trailing comma
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text


def _extract_json(text):
    """
    稳定 JSON 提取
    """
    json_text = _trim_to_valid_json(text)

    if json_text is None:
        return None

    json_text = _clean_json_text(json_text)
    json_text = _fix_brackets(json_text)

    try:
        return json.loads(json_text)
    except:
        return None

def compute_reward(text, ref_analysis, ground_patch, buggy_lines, have_loc=False):
    data = _extract_json(text)

    # ========= JSON 成功 ==============
    if data is not None:
        r_format = format_reward(data)

        pred_analysis = data.get("analysis", "")
        pred_patch = data.get("repair", "")
        pred_loc = data.get("localization", [])

        r_analysis = analysis_reward(pred_analysis, ref_analysis, ground_patch)
        r_repair = repair_reward(pred_patch, ground_patch)
        r_location = localization_reward(pred_loc, buggy_lines)
        
    else:
    # ========= JSON 失败 fallback ==============
        # analysis fallback
        r_analysis = analysis_reward(text, ref_analysis, ground_patch)

        # patch fallback: 提取 diff 行
        diff_lines = []
        for line in text.split("\n"):
            if line.strip().startswith("+") or line.strip().startswith("-"):
                diff_lines.append(line)

        pred_patch = "\n".join(diff_lines)
        r_repair = repair_reward(pred_patch, ground_patch)

        # location fallback: 提取数字
        nums = re.findall(r"\d+", text)
        pred_loc = [int(n) for n in nums[:5]]
        r_location = localization_reward(pred_loc, buggy_lines)

        # format penalty
        r_format = -0.5

    r_len = length_penalty(text)

    if not have_loc:
        r_location = 1.0

    reward = (
        0.5 * r_analysis +
        0.3 * r_repair +
        0.1 * r_location +
        0.1 * r_format +
        r_len
    )

    # fallback reward 上限
    reward = max(min(reward, 1.0), -1.0)

    return reward, r_format, r_analysis, r_repair, r_location, True if data is not None else False

# 防止模型乱输出
def format_reward(data):
    if not isinstance(data, dict):
        return 0.0
    
    required = ["analysis", "localization", "repair"]
    for k in required:
        if k not in data:
            return 0.0
    
    if not isinstance(data["localization"], list):
        return 0.0
    
    return 1.0

def analysis_reward(pred_analysis, ref_analysis, ground_patch):
    '''
    - 结构完整
    - BERTScore 语义相似度
    - 与 patch / 漏洞变量一致
    '''
    if not pred_analysis:
        return 0.0
    
    score = get_bert_scorer()

    # 1️⃣ 语义相似度
    try:
        _, _, F1 = score.score(
            [pred_analysis], 
            [ref_analysis], 
        )
        semantic = F1.item()
    except:
        semantic = 0.0

    # 2️⃣ 结构完整度
    structure = explanation_structure_reward(pred_analysis)

    # 3️⃣ patch 变量一致性
    token_align = token_overlap(pred_analysis, ground_patch)

    return 0.5 * semantic + 0.3 * structure + 0.2 * token_align

# 结构要素reward
def explanation_structure_reward(cot: str) -> float:
    """
    Reward explanation completeness and structure.
    Range: [0, 1]
    """
    if cot is None:
        return 0.0
    if not isinstance(cot, list):
        cot = str(cot)
    if len(cot.strip()) == 0:
        return 0.0

    cot_lower = cot.lower()
    score = 0.0

    # 1. Root cause
    if any(k in cot_lower for k in [
        "because", "due to", "caused by", "root cause", "reason"
    ]):
        score += 0.25

    # 2. Vulnerable location
    if any(k in cot_lower for k in [
        "line", "statement", "function", "variable", "this code"
    ]):
        score += 0.25

    # 3. Security impact
    if any(k in cot_lower for k in [
        "vulnerab", "overflow", "null", "crash", "memory", "attack"
    ]):
        score += 0.25

    # 4. Fix rationale
    if any(k in cot_lower for k in [
        "fix", "patch", "prevent", "avoid", "ensure", "check"
    ]):
        score += 0.25

    return min(score, 1.0)

def token_overlap(text1, text2):
    t1 = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text1))
    t2 = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text2))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t2)

def repair_reward(pred_patch, ground_patch):
    if not pred_patch:
        return 0.0

    # 1️⃣ exact match
    em = exact_match(pred_patch, ground_patch)

    # 2️⃣ token F1
    f1 = repair_f1(pred_patch, ground_patch)

    # 3️⃣ edit similarity
    edit = edit_similarity(pred_patch, ground_patch)

    return 0.5 * f1 + 0.3 * edit + 0.2 * em

def exact_match(output, ground_patch):
    return int(output.strip() == ground_patch.strip())

def repair_f1(output, ground_patch):
    output_tokens = set(output.strip().split())
    ground_tokens = set(ground_patch.strip().split())
    if not output_tokens or not ground_tokens:
        return 0.0
    precision = len(output_tokens & ground_tokens) / len(output_tokens)
    recall = len(output_tokens & ground_tokens) / len(ground_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def edit_similarity(output, ground_patch):
    return SequenceMatcher(None, output, ground_patch).ratio()

def localization_reward(pred_lines, gold_lines):
    if not isinstance(pred_lines, list):
        return 0.0
    if not gold_lines:
        return 0.0

    pred = set(pred_lines)
    gold = set(gold_lines)

    if not pred:
        return 0.0

    precision = len(pred & gold) / len(pred)
    recall = len(pred & gold) / len(gold)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def length_penalty(text, min_len=1000, max_len=3500):
    n = len(text.split())

    if n < min_len:
        return -0.3
    elif n > max_len:
        return -0.001 * (n - max_len)
    else:
        return 0.0