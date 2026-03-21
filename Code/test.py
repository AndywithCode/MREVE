import pickle
import json
import re
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

'''
graph_data: {
    'features': [[double*4098], [double*4098], [...]*3540],
    'adj': [[int*5375], [int*5375], [(1 or 2 or 3)*5375], 3540]
}
'''
# with open('/home/wyx/KitPatch-63E8/Datasets/graph_data/bigvul_graph_data.pkl', 'rb') as file:
#     graph_data = pickle.load(file)

'''
graph_embedding: {
    'CVE-YYYY-N': [double*128],
    ... len: 1965
}
'''
# with open('/home/wyx/KitPatch-63E8/Datasets/graph_embedding/bigvul_graph_embeddings.pkl', 'rb') as file:
#     graph_embedding = pickle.load(file)

'''
retrieved_results_bigvul_cvefixes: {
    '000': {
        'target_vul_code': 'vul_code',
        'target_patch_code': 'fix_code',
        'target_ids': file path, 
        'ctxs': [
            {
                'id': file path,
                'vul_code': 'vul_code',
                'patch_code': 'fix_code',
                'score': double
            },
            ... len: 100
        ]   
    },
    ... len: 986
}
'''
# with open('/home/wyx/KitPatch-63E8/Datasets/kitpatch_data/DPR-main/retrieved_results_bigvul_cvefixes.json', 'r') as file:
#     retrieved_results_bigvul_cvefixes = json.load(file)

'''
bigvul_cvefixes_data: {
    'project': {
        '0': 'samba', '1': 'openssl', '2': 'savannah', ...
    },
    'vul_type': {
        '0': 'CWE-264', '1': 'CWE-310', '2': 'CWE-119', ...
    },
    'patch_filepath': {
        '0': '/path/to/patch/file1', '1': '/path/to/patch/file2', ...
    },
    'vul_filepath': {
        '0': '/path/to/vul/file1', '1': '/path/to/vul/file2', ...
    },
    'function_name': {
        '0': 'function_name1', '1': 'function_name2', ...
    },
    'vul_func_code': {
        '0': 'vulnerable function code1', '1': 'vulnerable function code2', ...
    }
    ...
}
'''
# with open('/home/wyx/KitPatch-63E8/Datasets/kitpatch_data/vulnerability_repair/bigvul_cvefixes_data.json', 'r') as file:
#     bigvul_cvefixes_data = json.load(file)

'''
cvefixes_data: {
    same as bigvul_cvefixes_data
}
'''
# with open('/home/wyx/KitPatch-63E8/Datasets/kitpatch_data/vulnerability_repair/cvefixes_data.json', 'r') as file:
#     cvefixes_data = json.load(file)

# fi = '0--CWE-264--CVE-2015-8467--samldb.c--samldb_check_user_account_control_acl.c'
# fi_ = "--".join(fi.split("--")[2:-1]).split("_slice_")[0]
# print(fi_)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME = "/media/models/unixcoder/unixcoder-base-nine"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, is_decoder=True)
# # model = AutoModel.from_pretrained(MODEL_NAME)
# model.to(DEVICE)
# context = '''
# <decoder-only>// You are an automated vulnerability repair tool.\n\n// Generate a step-by-step reasoning process to analyze and fix the vulnerability in the given C/C++ function, following these three steps:\nStep 1: Vulnerability Localization\nIdentify the specific lines or code regions containing the vulnerability.\nStep 2: Root Cause Analysis\nAnalyze the root cause of the vulnerability and describe the conditions under which it may be exploited.\nStep 3: Repair and Refactoring\nPropose and implement a fix that eliminates the vulnerability, preserves original functionality, and follows secure coding best practices.\nOutput: Reasoning path and the code diff between the vulnerable and fixed functions.\n\n//Vul-fix Context\n\n//Context Code 1:\nvoid fz_init_cached_color_converter(fz_context *ctx, fz_color_converter *cc, fz_colorspace *is, fz_colorspace *ds, fz_colorspace *ss, const fz_color_params *params)\n{\n\tint n = ss->n;\n\tfz_cached_color_converter *cached = fz_malloc_struct(ctx, fz_cached_color_converter);\n\n\tcc->opaque = cached;\n\tcc->convert = fz_cached_color_convert;\n\tcc->ds = ds ? ds : fz_device_gray(ctx);\n\tcc->ss = ss;\n\tcc->is = is;\n\n\tfz_try(ctx)\n\t{\n\t\tfz_find_color_converter(ctx, &cached->base, is, cc->ds, ss, params);\n\t\tcached->hash = fz_new_hash_table(ctx, 256, n * sizeof(float), -1, fz_free);\n\t}\n\tfz_catch(ctx)\n\t{\n                fz_drop_color_converter(ctx, &cached->base);\n                fz_drop_hash_table(ctx, cached->hash);\n                fz_free(ctx, cached);\n                fz_rethrow(ctx);\n        }\n }\n\n\n// CVE Description\nIn Artifex MuPDF 1.12.0 and earlier, multiple use of uninitialized value bugs in the PDF parser could allow an attacker to cause a denial of service (crash) or influence program flow via a crafted file.\n\n// CWE Name and Description\nImproper Input Validation: The product receives input or data, but it does not validate or incorrectly validates that the input has the properties that are required to process the data safely and correctly.\n\n// Commit Message\n\n\nLet's think step by step.
# '''
# tokens_ids = tokenizer(context, truncation=True, max_length=512)
# source_ids = tokens_ids.to(DEVICE)
# prediction_ids = model.generate(**source_ids, max_length=128)
# predictions = tokenizer.decode(prediction_ids[0])
# print(context)
# print("===================================")
# print(predictions)

# import os
# import torch
# import torch.distributed as dist

# def is_rank0():
#     return not dist.is_initialized() or dist.get_rank() == 0


# def setup_ddp():
#     if not dist.is_initialized():
#         dist.init_process_group("nccl")
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#     return torch.device(f"cuda:{local_rank}")

# if is_rank0():
#     print("🚀 Start SFT Training")
# device = setup_ddp()
# print(device)

# from bert_score import BERTScorer
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = "/media/models"
# os.environ["TRANSFORMERS_CACHE"] = "/media/models"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
model_path = "/media/models/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# print(tokenizer.special_tokens_map)
# model = AutoModel.from_pretrained(model_path)
# print("Model and tokenizer loaded successfully.")
# bert_scorer = BERTScorer(
#     model_type="distilroberta-base",
#     lang="en",
#     device="cuda:1",
#     rescale_with_baseline=True
# )
# print(bert_scorer.score(["The cat is on the mat."], ["The cat is on the mat."]))


text = '''{\"analysis\": \"**Chain of Thought Analysis:**\\n**Step 1: Vulnerability Localization**\\nThe vulnerability is located in the `check_mtab` function where it calls `check_newline()` twice with the same arguments (`progname`, `devname`) and (`progname`, `dir`). This pattern suggests potential issues related to improper newline checking.\\n**Step 2: Root Cause Analysis**\\nThe root cause is insufficient validation of input strings before processing them. Specifically:\\n1. The function checks for newline characters in both `progname` and `devname`.\\n2. If either contains a newline character, it returns `EX_USAGE` without further validation.\\n3. This could lead to:\\n   - Incorrect handling of paths containing newline characters\\n   - Potential buffer overflows due to invalid input\\n   - Denial of Service through corrupted mtab files\\n   - Security vulnerabilities like CVE-2010-0547\\n**Step 3: Repair and Refactoring**\\nThe fix should:\\n1. Validate all inputs before performing any checks\\n2. Maintain the original functionality for valid inputs\\n3. Add proper error handling for invalid inputs\\n4. Follow secure coding practices by validating all parameters\\n**Fixed Function:**\\n**Key Changes Made:**\\n1. Added parameter validation for `progname` and `dir` before calling `check_newline`\\n2. Maintained the original functionality for valid inputs\\n3. Added error handling for invalid inputs\\n4. Kept the same return value logic for successful cases\\nThis fixed version ensures that:\\n- All inputs are validated before processing\\n- Invalid inputs are handled gracefully\\n- The original functionality remains intact\\n- Proper error codes are returned when necessary\", \"localization\": [], \"repair\": \"+ \\tif (!is_valid_path(progname)) {\\n+ \\t\\treturn EX_USAGE;\\n+ \\t}\\n+ \\tif (!is_valid_path(dir)) {\\n+ \\t\\treturn EX_USAGE;\\n+ \\t}\"}\n```'''
text = '''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou are a security vulnerability analysis and patch generation assistant.\n\nYour task is to analyze the given vulnerable C/C++ function using the provided security context and generate a structured JSON output.\n\nYou MUST follow the reasoning steps internally, but ONLY output the final JSON object.\n\nDo NOT output explanations outside the JSON.\nDo NOT output markdown.\nDo NOT output additional text before or after the JSON.\n\n=== TASKS ===\n\n1) Vulnerability Localization\n- Identify the exact vulnerable lines inside the given function.\n- Return line numbers relative to the provided vulnerable function. \n\n2) Root Cause Analysis\n- Explain step-by-step:\n  - Where the vulnerability occurs\n  - Why it happens (missing checks, incorrect assumptions, unsafe operations, etc.)\n  - Under what conditions it can be triggered \n\n3) Repair\n- Generate a minimal and correct unified diff patch. \n- Use standard unified diff format: \n    - old line\n    + new line\n\n=== STRICT OUTPUT FORMAT === \n\n{\n  \"analysis\": \"full step-by-step vulnerability reasoning\",\n  \"localization\": [int, ...],\n  \"repair\": \"unified diff patch\"\n} \n\n=== CONTEXT ===\n\n[Vulnerable Function]\nint __ref online_pages(unsigned long pfn, unsigned long nr_pages)\n{\n\tunsigned long onlined_pages = 0;\n\tstruct zone *zone;\n\tint need_zonelists_rebuild = 0;\n\tint nid;\n\tint ret;\n\tstruct memory_notify arg;\n\n\tlock_memory_hotplug();\n\targ.start_pfn = pfn;\n\targ.nr_pages = nr_pages;\n\targ.status_change_nid = -1;\n\n\tnid = page_to_nid(pfn_to_page(pfn));\n\tif (node_present_pages(nid) == 0)\n\t\targ.status_change_nid = nid;\n\n\tret = memory_notify(MEM_GOING_ONLINE, &arg);\n\tret = notifier_to_errno(ret);\n\tif (ret) {\n\t\tmemory_notify(MEM_CANCEL_ONLINE, &arg);\n\t\tunlock_memory_hotplug();\n\t\treturn ret;\n\t}\n\t/*\n\t * This doesn't need a lock to do pfn_to_page().\n\t * The section can't be removed here because of the\n\t * memory_block->state_mutex.\n\t */\n\tzone = page_zone(pfn_to_page(pfn));\n\t/*\n\t * If this zone is not populated, then it is not in zonelist.\n\t * This means the page allocator ignores this zone.\n\t * So, zonelist must be updated after online.\n\t */\n\tmutex_lock(&zonelists_mutex);\n\tif (!populated_zone(zone))\n\t\tneed_zonelists_rebuild = 1;\n\n\tret = walk_system_ram_range(pfn, nr_pages, &onlined_pages,\n\t\tonline_pages_range);\n\tif (ret) {\n\t\tmutex_unlock(&zonelists_mutex);\n\t\tprintk(KERN_DEBUG \"online_pages [mem %#010llx-%#010llx] failed\\n\",\n\t\t       (unsigned long long) pfn << PAGE_SHIFT,\n\t\t       (((unsigned long long) pfn + nr_pages)\n\t\t\t    << PAGE_SHIFT) - 1);\n\t\tmemory_notify(MEM_CANCEL_ONLINE, &arg);\n\t\tunlock_memory_hotplug();\n\t\treturn ret;\n\t}\n \n \tzone->present_pages += onlined_pages;\n \tzone->zone_pgdat->node_present_pages += onlined_pages;\n\tif (need_zonelists_rebuild)\n\t\tbuild_all_zonelists(NULL, zone);\n\telse\n\t\tzone_pcp_update(zone);\n \n \tmutex_unlock(&zonelists_mutex);\n \n \tinit_per_zone_wmark_min();\n \n\tif (onlined_pages) {\n \t\tkswapd_run(zone_to_nid(zone));\n\t\tnode_set_state(zone_to_nid(zone), N_HIGH_MEMORY);\n\t}\n \n \tvm_total_pages = nr_free_pagecache_pages();\n \n\twriteback_set_ratelimit();\n\n\tif (onlined_pages)\n\t\tmemory_notify(MEM_ONLINE, &arg);\n\tunlock_memory_hotplug();\n\n\treturn 0;\n}\n\n\n[Context Code]\n\n//Context Code 1:\nint __ref online_pages(unsigned long pfn, unsigned long nr_pages)\n{\n\tunsigned long onlined_pages = 0;\n\tstruct zone *zone;\n\tint need_zonelists_rebuild = 0;\n\tint nid;\n\tint ret;\n\tstruct memory_notify arg;\n\n\tlock_memory_hotplug();\n\targ.start_pfn = pfn;\n\targ.nr_pages = nr_pages;\n\targ.status_change_nid = -1;\n\n\tnid = page_to_nid(pfn_to_page(pfn));\n\tif (node_present_pages(nid) == 0)\n\t\targ.status_change_nid = nid;\n\n\tret = memory_notify(MEM_GOING_ONLINE, &arg);\n\tret = notifier_to_errno(ret);\n\tif (ret) {\n\t\tmemory_notify(MEM_CANCEL_ONLINE, &arg);\n\t\tunlock_memory_hotplug();\n\t\treturn ret;\n\t}\n\t/*\n\t * This doesn't need a lock to do pfn_to_page().\n\t * The section can't be removed here because of the\n\t * memory_block->state_mutex.\n\t */\n\tzone = page_zone(pfn_to_page(pfn));\n\t/*\n\t * If this zone is not populated, then it is not in zonelist.\n\t * This means the page allocator ignores this zone.\n\t * So, zonelist must be updated after online.\n\t */\n\tmutex_lock(&zonelists_mutex);\n\tif (!populated_zone(zone))\n\t\tneed_zonelists_rebuild = 1;\n\n\tret = walk_system_ram_range(pfn, nr_pages, &onlined_pages,\n\t\tonline_pages_range);\n\tif (ret) {\n\t\tmutex_unlock(&zonelists_mutex);\n\t\tprintk(KERN_DEBUG \"online_pages [mem %#010llx-%#010llx] failed\\n\",\n\t\t       (unsigned long long) pfn << PAGE_SHIFT,\n\t\t       (((unsigned long long) pfn + nr_pages)\n\t\t\t    << PAGE_SHIFT) - 1);\n\t\tmemory_notify(MEM_CANCEL_ONLINE, &arg);\n\t\tunlock_memory_hotplug();\n\t\treturn ret;\n\t}\n \n \tzone->present_pages += onlined_pages;\n \tzone->zone_pgdat->node_present_pages += onlined_pages;\n\tif (need_zonelists_rebuild)\n\t\tbuild_all_zonelists(NULL, zone);\n\telse\n\t\tzone_pcp_update(zone);\n \n \tmutex_unlock(&zonelists_mutex);\n \n \tinit_per_zone_wmark_min();\n \n\tif (onlined_pages) {\n \t\tkswapd_run(zone_to_nid(zone));\n\t\tnode_set_state(zone_to_nid(zone), N_HIGH_MEMORY);\n\t}\n \n \tvm_total_pages = nr_free_pagecache_pages();\n \n\twriteback_set_ratelimit();\n\n\tif (onlined_pages)\n\t\tmemory_notify(MEM_ONLINE, &arg);\n\tunlock_memory_hotplug();\n\n\treturn 0;\n}\n\n\n[CVE Description]\nThe online_pages function in mm/memory_hotplug.c in the Linux kernel before 3.6 allows local users to cause a denial of service (NULL pointer dereference and system crash) or possibly have unspecified other impact in opportunistic circumstances by using memory that was hot-added by an administrator.\n\n[CWE]\n: \n\n[Commit Message]\nWhen online_pages() is called to add new memory to an empty zone, it rebuilds all zone lists by calling build_all_zonelists().  But there's a bug which prevents the new zone to be added to other nodes' zone lists.  online_pages() { \tbuild_all_zonelists() \t..... \tnode_set_state(zone_to_nid(zone), N_HIGH_MEMORY) }  Here the node of the zone is put into N_HIGH_MEMORY state after calling build_all_zonelists(), but build_all_zonelists() only adds zones from nodes in N_HIGH_MEMORY state to the fallback zone lists. build_all_zonelists()      ->__build_all_zonelists() \t->build_zonelists() \t    ->find_next_best_node() \t\t->for_each_node_state(n, N_HIGH_MEMORY)  So memory in the new zone will never be used by other nodes, and it may cause strange behavor when system is under memory pressure.  So put node into N_HIGH_MEMORY state before calling build_all_zonelists().  Signed-off-by: Jianguo Wu <wujianguo@huawei.com> Signed-off-by: Jiang Liu <liuj97@gmail.com> Cc: Mel Gorman <mgorman@suse.de> Cc: Michal Hocko <mhocko@suse.cz> Cc: Minchan Kim <minchan@kernel.org> Cc: Rusty Russell <rusty@rustcorp.com.au> Cc: Yinghai Lu <yinghai@kernel.org> Cc: Tony Luck <tony.luck@intel.com> Cc: KAMEZAWA Hiroyuki <kamezawa.hiroyu@jp.fujitsu.com> Cc: KOSAKI Motohiro <kosaki.motohiro@jp.fujitsu.com> Cc: David Rientjes <rientjes@google.com> Cc: Keping Chen <chenkeping@huawei.com> Signed-off-by: Andrew Morton <akpm@linux-foundation.org> Signed-off-by: Linus Torvalds <torvalds@linux-foundation.org>\n\n=== OUTPUT JSON ===<|im_end|>\n<|im_start|>assistant\n'''

print(len(text.strip()))
print(len(tokenizer.encode(text)))

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

data = _extract_json(text)
print(data)