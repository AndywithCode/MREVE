from __future__ import absolute_import, division, print_function
import os
import time
import json
import torch
import random
import argparse
import tiktoken
import warnings
import re
import string
import regex
import pickle
import numpy as np
import pandas as pd
import faiss
import torch.nn.functional as F
from model import Model
from openai import OpenAI
from sklearn.cluster import KMeans
from py2neo import Graph
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from data_preprocess import clean_parse_bigvul

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# encoding_name = 'cl100k_base'
# encoding = tiktoken.get_encoding(encoding_name)


# client = OpenAI(api_key="xx", base_url="https://api.deepseek.com")

# graph = Graph("bolt://xx.xx.xx.xx:7687", auth=("neo4j", "xx")) 



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    normal_prediction = normalize_answer(prediction)
    normal_groundtruth = normalize_answer(ground_truth)
    if normal_groundtruth == normal_prediction:
        return True
    return False

def ems(prediction, ground_truth):
    return exact_match_score(prediction, ground_truth)

def remove_here_sentences_multiline(text):
    pattern = r'Here.*?:'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def remove_empty_lines(text):
    return '\n'.join([line for line in text.splitlines() if line.strip() != ''])


def run_validation(output, ground_patch_func):
    pred_patch = output
    cot = ''
    try:
        pred_patch = pred_patch.split("```")[1].strip("c\n")   
    except:
        pass
    
    pred_patch = remove_c_comments(pred_patch)
    ground_patch_func = remove_c_comments(ground_patch_func)

    prediction_result = pred_patch.split('\n')
    new_prediction = []
    for res in prediction_result:
        res = res.strip()
        if res.startswith('-') and not res.startswith('---') and not res.endswith('.c') and not res.endswith('.h>') and "#include" not in res:
            res = res.strip('-').strip()
            if res.startswith('//') or res.endswith('*/') or len(res)==0:
                continue
            elif '//' in res:
                res = res.split('//')[0]
            res = '-' + ' ' + res
            new_prediction.append(res)
    for res in prediction_result:
        res = res.strip()
        if res.startswith('+') and not res.startswith('+++') and not res.endswith('.c') and not res.endswith('.h>') and "#include" not in res:
            res = res.strip('+').strip()
            if res.startswith('//') or res.endswith('*/') or len(res)==0:
                continue
            elif '//' in res:
                res = res.split('//')[0]
            res = '+' + ' ' + res
            new_prediction.append(res)
    new_prediction = ' '.join(new_prediction)
    new_prediction = ' '.join(new_prediction.split())
    ground_patch_func = ' '.join(ground_patch_func.split())
    new_prediction = clean_tokens(new_prediction)
    ground_patch_func = clean_tokens(ground_patch_func)
    valid = ems(new_prediction, ground_patch_func)

    try:
        cot = output.split("```")[0] + output.split("```")[2]
        cot = remove_here_sentences_multiline(cot)
        cot = remove_empty_lines(cot)
    except:
        pass

    return valid, cot, new_prediction, ground_patch_func


def request_engine(messages):
    ret = None
    while ret is None:
        try:
            ret = client.chat.completions.create(model="deepseek-chat", messages=messages, stream=False)  
        except Exception as e:
            print(e)
            return None
    return ret


def extract_all_context_files(type, args):
    context_slice_path = f"/home/wyx/KitPatch-63E8/Datasets/{args.dataset}/vul-fix_context/"  
    context_files = {}
    for fi in os.listdir(context_slice_path):
        if args.dataset == "bigvul":
            fi_ = "--".join(fi.split("--")[2:]).split("_slice_")[0]
            # fi_: CVE-2015-8467--samldb.c--samldb_check_user_account_control_acl.c
            # fi: /home/wyx/KitPatch-63E8/Datasets/kitpatch_data/vulnerability_repair/bigvul_cvefixes/patch/0--CWE-264--CVE-2015-8467--samldb.c--samldb_check_user_account_control_acl.c
            if not fi_.startswith('CVE-'):
                continue
        else:
            fi_ = "--".join(fi.split("--")[1:]).split("_slice_")[0]
        
        if fi_ not in context_files:
            context_files[fi_] = [fi]
        else:
            context_files[fi_].append(fi)

    return context_files, context_slice_path

def extract_context_code(filename, context_files, context_slice_path):
    context_file = context_files[filename]
    context_code = ''
    for index, f in enumerate(context_file):
        with open(os.path.join(context_slice_path, f)) as a:
            content = a.read()
        context_code = context_code + f'\n//Context Code {str(index+1)}:\n' + content

    return context_code

def get_embedding(args, model, tokenizer, think_dataset):
    with open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_deepseek.json", 'r') as f:
        cot = json.load(f)
    for file, repair_result in cot.items():
        buggy = think_dataset[file]["vul"]
        tokenized_code = tokenizer.encode_plus(buggy, max_length=400, return_tensors="pt")
        outputs = model(**tokenized_code)    
        cot[file]["embedding"] = outputs[0][0, 0, :].detach().numpy().tolist()
    with open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'w') as f:
        json.dump(cot, f, indent=4)

def encodeCOTEmbedding(args, think_dataset):
    model = BGEM3FlagModel('/media/models/bgem3/bge-m3', use_fp16=True)
    with open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_without_step-cot.json", 'r') as f:
        cot = json.load(f)
    for file, repair_result in cot.items():
        buggy = think_dataset[file]["vul"]
        # outputs = model.encode(buggy, batch_size=12, max_length=8192)['dense_vecs']
        outputs = model.encode(buggy, max_length=8192)['dense_vecs']
        cot[file]["embedding"] = outputs.tolist()
    with open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'w') as f:
        json.dump(cot, f, indent=4)

def loadCOTEmbedding(args):
    with open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r') as f:
        cot_embedding_dataset = json.load(f)
    return cot_embedding_dataset

def encodeInferenceEmbedding(args, inference_dataset):
    model = BGEM3FlagModel('/media/models/bgem3/bge-m3', use_fp16=True)
    for file, repair_result in inference_dataset.items():
        buggy = inference_dataset[file]["vul"]
        outputs = model.encode(buggy, max_length=8192)['dense_vecs']
        inference_dataset[file]["embedding"] = outputs.tolist()
    with open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_embedding.json", 'w') as f:
        json.dump(inference_dataset, f, indent=4)

def loadInferenceEmbedding(args):
    with open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_embedding.json", 'r') as f:
        inference_embedding_dataset = json.load(f)
    return inference_embedding_dataset

def get_clusters(args, think_dataset):
    cot = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r'))
    cot_think = {}
    for file in think_dataset.keys():
        if file in cot:
            cot_think[file] = cot[file]
    embeddings = np.asarray([repair_result["embedding"] for _, repair_result in cot_think.items()]) 
    kmeans = KMeans(n_clusters=args.n_example, n_init=10, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels
    

def get_example_from_clusters(args, model, tokenizer, buggy, labels, file):
    with open(f"/home/wyx/KitPatch-63E8/Datasets/graph_embedding/{args.dataset}_graph_embeddings.pkl", "rb") as f:
        graph_data = pickle.load(f)
    
    cot = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r'))
    embeddings = []
    for _, repair_result in cot.items():
        try:
            emb_ = repair_result["embedding"] + graph_data['-'.join(_.split('-')[:3])]
        except:
            emb_ = repair_result["embedding"] + [float(0)]*128
        embeddings.append(emb_)
    embeddings = np.asarray(embeddings)
    
    points = [file for file, repair_result in cot.items()]

    selected_points = []
    
    tokenized_code = tokenizer.encode_plus(buggy, max_length=400, return_tensors="pt")
    outputs = model(**tokenized_code)    
    buggy_embedding = outputs[0][0, 0, :].detach().numpy().tolist()
    try:
        buggy_embedding = buggy_embedding + graph_data['-'.join(file.split('-')[:3])]
    except:
        buggy_embedding = buggy_embedding + [float(0)]*128

    for i in range(args.n_example):
        cluster_points = np.where(labels == i)[0]
        cluster_embeddings = [embeddings[point] for point in cluster_points]
        similarities = cosine_similarity([buggy_embedding], cluster_embeddings)
        most_similar_index = np.argmax(similarities)
        selected_point = cluster_points[most_similar_index]
        selected_points.append(points[selected_point])
    return selected_points

def get_example_by_embedding(args, model, tokenizer, buggy_embedding, labels, file):
    
    cot = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r'))

    embeddings = []
    for _, repair_result in cot.items():
        emb_ = repair_result["embedding"]
        embeddings.append(emb_)
    embeddings = np.asarray(embeddings)

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(np.array([buggy_embedding]), args.n_example)
    
    points = list(cot.keys())

    selected_points = []
    for i in range(args.n_example):
        selected_point = indices[0][i]
        selected_points.append(points[selected_point])
    return selected_points

def retrieve_topk(query_embedding, context_tensor, topk=1):
    """
    返回 top-k 相似样本索引及其相似度
    """
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)

    # 计算 cosine similarity
    similarities = F.cosine_similarity(query_tensor, context_tensor, dim=1)

    # 取 top-k
    topk_values, topk_indices = torch.topk(similarities, topk)

    return topk_indices.tolist(), topk_values.tolist()

def get_examples_by_embedding(args):
    '''
    faiss检索
    '''
    cot = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r'))
    querys = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_embedding.json", 'r'))
    query_embeddings = []

    for _, repair_result in querys.items():
        emb_ = repair_result["embedding"]
        query_embeddings.append(emb_)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    faiss.normalize_L2(query_embeddings)
    query_embeddings = np.ascontiguousarray(query_embeddings)

    embeddings = []
    for _, repair_result in cot.items():
        emb_ = repair_result["embedding"]
        embeddings.append(emb_)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    embeddings = np.ascontiguousarray(embeddings)

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(query_embeddings, args.n_example)
    
    points = list(cot.keys())
    query_points = list(querys.keys())

    selected_points = {}
    for q_idx in range(len(query_embeddings)):
        q_points = []
        for distance, idx in zip(distances[q_idx], indices[q_idx]):
            key = points[idx]
            q_points.append({
                "key": key, 
                "score": float(distance)
            })
        selected_points[query_points[q_idx]] = q_points
    with open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_retrieved.json", 'w') as f:
        json.dump(selected_points, f, indent=4)

def get_examples_by_embedding_v1(args):
    '''
    retrieve_topk检索
    '''
    cot = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r'))
    querys = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_embedding.json", 'r'))

    # ========================
    # 构造 query embedding
    # ========================
    query_embeddings = []
    for _, repair_result in querys.items():
        query_embeddings.append(repair_result["embedding"])

    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

    # ========================
    # 构造 context embedding
    # ========================
    embeddings = []
    for _, repair_result in cot.items():
        embeddings.append(repair_result["embedding"])

    embeddings = np.asarray(embeddings, dtype=np.float32)
    context_tensor = torch.tensor(embeddings, dtype=torch.float32)

    points = list(cot.keys())
    query_points = list(querys.keys())

    selected_points = {}

    # ========================
    # 逐 query 检索
    # ========================
    for q_idx, query_emb in enumerate(query_embeddings):

        topk_indices, topk_scores = retrieve_topk(
            query_emb,
            context_tensor,
            args.n_example
        )

        q_points = []
        for idx, score in zip(topk_indices, topk_scores):
            key = points[idx]
            q_points.append({
                "key": key,
                "score": float(score)
            })

        selected_points[query_points[q_idx]] = q_points

    # ========================
    # 保存结果
    # ========================
    with open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_retrieved_v1.json", 'w') as f:
        json.dump(selected_points, f, indent=4)

def get_examples(args):
    
    cot = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r'))

    # sample数据
    querys = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r'))
    think_dataset = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/think_dataset.json", 'r'))
    output = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/firp_kitpatch.json", 'r'))
    # test数据
    # querys = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_embedding.json", 'r'))
    # think_dataset = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_dataset.json", 'r'))
    # output = json.load(open(f"/home/wyx/KitPatch-63E8/Results/repair_result/bigvul_kitpatch.json", 'r'))

    knowledge_dataset = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/think_dataset.json", 'r'))
    # knowledge_dataset = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/total_dataset.json", 'r'))
    df_commit = pd.read_csv(f"/home/wyx/KitPatch-63E8/Code/VulKG_construction/import/{args.dataset}/Commit_message.csv")
    df_cve = pd.read_csv(f"/home/wyx/KitPatch-63E8/Code/VulKG_construction/import/{args.dataset}/CVE_knowledge.csv")
    df_cwe = pd.read_csv(f"/home/wyx/KitPatch-63E8/Code/VulKG_construction/import/{args.dataset}/CWE_knowledge.csv")
    knowledge_output = json.load(open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/firp_kitpatch.json", 'r'))
    query_embeddings = []

    for file, repair_result in querys.items():
        emb_ = repair_result["embedding"]
        query_embeddings.append(emb_)
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    # faiss.normalize_L2(query_embeddings)
    # query_embeddings = np.ascontiguousarray(query_embeddings)

    embeddings = []
    for _, repair_result in cot.items():
        emb_ = repair_result["embedding"]
        embeddings.append(emb_)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    # faiss.normalize_L2(embeddings)
    # embeddings = np.ascontiguousarray(embeddings)

    # res = faiss.StandardGpuResources()
    # index = faiss.GpuIndexFlatIP(res, embeddings.shape[1])
    # index.add(embeddings)
    # distances, indices = index.search(query_embeddings, args.n_example)

    context_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    points = list(cot.keys())
    query_points = list(querys.keys())
    # context_files为同一文件内的内容
    context_files = {}
    context_slice_path = "/home/wyx/KitPatch-63E8/Datasets/kitpatch_data/vulnerability_repair/bigvul_cvefixes/vul"
    for fi in os.listdir(context_slice_path):
        fi_ = "--".join(fi.split("--")[2:-1]).split("_slice_")[0]
        if not fi_.startswith('CVE-'):
                continue
        if fi_ not in context_files:
            context_files[fi_] = [fi]
        else:
            context_files[fi_].append(fi)
    # for fi in total_dataset.keys():
    #     fi_ = "--".join(fi.split("--")[:-1]).split("_slice_")[0]
    #     if fi_ not in context_files:

    selected_points = {}
    for q_idx in range(len(query_embeddings)):
        fi = query_points[q_idx]
        if not fi.startswith('CVE-'):
            continue
        q_points = []
        selected_point = {}
        context_code = ''
        flag = False

        topk_indices, topk_scores = retrieve_topk(
            query_embeddings[q_idx],
            context_tensor,
            args.n_example
        )
        # for distance, idx in zip(distances[q_idx], indices[q_idx]):
        for idx, distance in zip(topk_indices, topk_scores):
            key = points[idx]
            q_points.append({
                "key": key, 
                "score": float(distance)
            })
            if distance >= args.sim_threshold and not flag and key!=fi and (key in knowledge_output):
                selected_point["similar_vul"] = knowledge_dataset[key]["vul"]
                selected_point["similar_diff"] = knowledge_dataset[key]["diff"]
                selected_point["similar_explain"] = knowledge_output[key]["output"]
                flag = True

        selected_point["retrieved"] = q_points
        if not flag:
            selected_point["similar_vul"] = ""
            selected_point["similar_diff"] = ""
            selected_point["similar_explain"] = ""
        try:
            selected_point["language"] = think_dataset[fi]["language"]
        except:
            selected_point["language"] = "C/C++"
        CVE_ID = '-'.join(fi.split('-')[:3])
        match_cve = df_cve[df_cve['cveID'] == CVE_ID]
        CWE_ID = match_cve['cweID'].values[0] if not match_cve.empty else ''
        selected_point["cveDescription"] = match_cve['cveDescription'].values[0] if not match_cve.empty else ''
        if match_cve.empty:
            match_cve_commit = df_commit[df_commit['cveID'] == CVE_ID]
            if match_cve_commit.empty:
                selected_point["commit_message"] = ''
            else:
                selected_point["commit_message"] = match_cve_commit['commitMessage'].values[0]
        else:
            if pd.isna(match_cve['commitMsg'].values[0]):
                match_cve_commit = df_commit[df_commit['cveID'] == CVE_ID]
                if match_cve_commit.empty:
                    selected_point["commit_message"] = ''
                else:
                    selected_point["commit_message"] = match_cve_commit['commitMessage'].values[0]
            else:
                selected_point["commit_message"] = match_cve['commitMsg'].values[0]
        if pd.isna(selected_point["commit_message"]):
            selected_point["commit_message"] = ''
        if CWE_ID != '':
            match_cwe = df_cwe[df_cwe['cweID'] == CWE_ID]
            if not match_cwe.empty:
                selected_point["cwe_type"] = match_cwe['cweName'].values[0]
                selected_point["cwe_description"] = match_cwe['cweDescription'].values[0]
            else:
                selected_point["cwe_type"] = ''
                selected_point["cwe_description"] = ''
        else:
            selected_point["cwe_type"] = ''
            selected_point["cwe_description"] = ''
        
        # context_code这里可以改成提取vul-fix_context的内容
        # context_code也可以直接用retrieved的第一个
        filename = "--".join(fi.split("--")[:-1]).split("_slice_")[0]
        if filename in context_files:
            context_file = context_files[filename]
            for index, f in enumerate(context_file):
                with open(os.path.join(context_slice_path, f)) as a:
                    content = a.read()
                context_code = context_code + f'\n//Context Code {str(index+1)}:\n' + content
        
        selected_point["context_code"] = context_code
        selected_point["location"] = think_dataset[fi]["location"]
        selected_point["vul"] = think_dataset[fi]["vul"]
        selected_point["diff"] = think_dataset[fi]["diff"]
        selected_point["fix"] = think_dataset[fi]["fix"]
        # sample数据
        selected_point["output"] = output[fi]["output"]
        # test数据
        # flag = False
        # for o in output[fi]["result"]:
        #     validation,_, _, _  = run_validation(o["pred_patch"], output[fi]["ground_patch"])
        #     if validation and not flag:
        #         selected_point["output"] = o["output"]
        #         flag = True
        #         break
        # if not flag:
        #     selected_point["output"] = think_dataset[fi]["diff"]
        
        selected_points[fi] = selected_point
    
    # sample数据
    with open(f"/home/wyx/KitPatch-63E8/Datasets/train_data/{args.dataset}/sample_data_v1.json", 'w') as f:
        json.dump(selected_points, f, indent=4)
    # test数据
    # with open(f"/home/wyx/KitPatch-63E8/Datasets/train_data/{args.dataset}/test_data_v1.json", 'w') as f:
    #     json.dump(selected_points, f, indent=4)

def merge_json_files():
    think_dataset = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/bigvul/think_dataset.json", 'r'))
    inference_dataset = json.load(open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/bigvul/inference_dataset.json", 'r'))
    merge_json_dataset = {**think_dataset, **inference_dataset}
    with open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/bigvul/total_dataset.json", 'w') as f:
        json.dump(merge_json_dataset, f, indent=4)

def add_bug_comments(code_string, buggy_line_numbers):
    lines = code_string.split("\n")
    for line_number in buggy_line_numbers:
        if 1 <= line_number <= len(lines):
            lines[line_number-1] += "//Vulnerable line"  
            
    modified_code_string = "\n".join(lines)
    return modified_code_string


def extract_kg_context(filename):
    CVE_ID = '-'.join(filename.split('-')[:3])
    query = """MATCH (v:Vulnerability) WHERE v.cveID = '%s' RETURN v""" % (CVE_ID)
    try:
        vul_node_attributes = graph.run(query).data()[0]['v']
        cveDescription = vul_node_attributes['cveDescription']
    except:
        vul_node_attributes, cveDescription = '', ''

    query = """MATCH (v:Vulnerability)-[r:EXAMPLE_OF]->(m) WHERE v.cveID = '%s' RETURN m""" % (CVE_ID)
    try:
        cwe_type = graph.run(query).data()[0]['m']['cweName']
        cwe_description = graph.run(query).data()[0]['m']['cweDescription']
    except:
        cwe_type, cwe_description = '', ''
    
    query = """MATCH (v:Vulnerability)-[r:HAS_COMMIT]->(m) WHERE v.cveID = '%s' RETURN m""" % (CVE_ID)
    commitMessage = graph.run(query).data()[0]['m']['commitMessage']
    if commitMessage is None:
        commitMessage = ""
    
    return cveDescription, commitMessage, cwe_type, cwe_description


def remove_c_comments(code):
    """
    移除C/C++代码中的注释，同时保留字符串内容
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    cleaned_code = re.sub(pattern, replacer, code)
    
    cleaned_code = '\n'.join(
        line for line in cleaned_code.split('\n')
        if line.strip() != ''
    )

    return cleaned_code


def generate_chain_of_thought(args, think_dataset):
    with open(f"/home/wyx/KitPatch-63E8/Datasets/prompt/generate_prompt.txt") as f:
        generate_prompt = f.read()
    print(generate_prompt)
    try:
        with open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot.json", 'r') as f:
            cot = json.load(f)
    except:
        cot = {}

    context_slice_path = f"/home/wyx/KitPatch-63E8/Datasets/vul-fix_context/{args.dataset}/"
    context_files = {}
    for fi in os.listdir(context_slice_path):
        fi_ = "--".join(fi.split("--")[1:]).split("_slice_")[0]
        if fi_ not in context_files:
            context_files[fi_] = [fi]
        else:
            context_files[fi_].append(fi)

    i = 0
    for file, bug in think_dataset.items():
        print(i)
        i += 1
        if file in cot.keys(): continue
        print("Repairing vulnerability {} ... ".format(file.split(".")[0]))
        cveDescription, commitMessage, cwe_type, cwe_description = extract_kg_context(file)

        buggy_func = bug["vul"]
        ground_patch_func = bug["diff"]
        buggy_lines = bug["location"]
        try:
            language = bug["language"]
        except:
            language = "C/C++"
        modified_func = add_bug_comments(buggy_func, buggy_lines)
        if file in context_files:
            context_file = context_files[file]
            context_code = ''
            for index, f in enumerate(context_file):
                with open(os.path.join(context_slice_path, f)) as a:
                    content = a.read()
                context_code = context_code + f'\n//Context Code {str(index+1)}:\n' + content
        else:
            context_code = modified_func

        
        prompt = generate_prompt.format(language=language, contextCode=context_code, cveDescription=cveDescription, cweName=cwe_type, cweDescription=cwe_description, commitMessage=commitMessage)  

        # 取样次数
        for _ in range(args.sample):
            print(file + '---' + str(_))
            messages = [
                        {"role": "system", "content": "You are an Automatic Vulnerability Repair Tool"},
                        {"role": "user", "content": prompt}
                    ]
    
            output = request_engine(messages).choices[0].message.content
            valid, pred_cot, _, _ = run_validation(output, ground_patch_func)
            if valid:
                break
        cot[file] = {"output": pred_cot, "valid": valid}
        with open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot.json", 'w') as f:
            json.dump(cot, f, indent=4)
    print('a')


def chain_of_thought_repair(args, Select_model, Select_tokenizer, think_dataset, inference_files, output_filename="SynEq"):
    with open(f"/home/wyx/KitPatch-63E8/Datasets/prompt/repair_prompt.txt") as f:
        repair_prompt = f.read()
    with open(f"/home/wyx/KitPatch-63E8/Results/cot/{args.dataset}/cot_embedding.json", 'r') as f:
        cot = json.load(f)

    try:
        with open(f"/home/wyx/KitPatch-63E8/Results/repair_result/{args.dataset}_{output_filename}.json", 'r') as f:
            repair = json.load(f)
    except:
        repair = {}

    labels = get_clusters(args, think_dataset)
    
    inference_files_new = {}
    for key, value in inference_files.items():
        if key not in repair.keys():
            inference_files_new[key] = value

    test_context_files, test_context_slice_path = extract_all_context_files("test", args)
    example_context_files, example_context_slice_path = extract_all_context_files("cot", args)

    for file, bug in tqdm(inference_files_new.items()):
        print("Repairing vulnerability {} ... ".format(file.split(".")[0]))
        # examples = get_example_from_clusters(args, Select_model, Select_tokenizer, bug['vul'], labels, file)
        examples = get_example_by_embedding(args, Select_model, Select_tokenizer, bug['embedding'], labels, file)

        if file in repair.keys():
            repair_results = repair[file]
        else:
            repair_results = []
        
        try:
            vul_code = extract_context_code(file, test_context_files, test_context_slice_path)  
        except:
            vul_code = add_bug_comments(bug["vul"], bug["location"])

        try:
            language = bug['language']
        except:
            language = 'c/c++'
        for i in range(len(examples)):
            try:
                example_vul = extract_context_code(examples[i], example_context_files, example_context_slice_path)
            except:
                example_vul = add_bug_comments(think_dataset[examples[i]]["vul"], think_dataset[examples[i]]["location"])
            instruct = "\n\n//Here is the diff:\n\n```%s\n```" % think_dataset[examples[i]]["diff"]
            prompt = repair_prompt.format(language=language, example_vul=example_vul, example_cot=cot[examples[i]]["output"]+instruct, vul=vul_code)  
            ground_patch_func = bug["diff"]

            messages = [
                        {"role": "system", "content": "You are an Automatic Vulnerability Repair Tool."},
                        {"role": "user", "content": prompt}
                    ]
            repair_result = []
            try:
                output = request_engine(messages).choices[0].message.content
            except:
                continue
            valid, message, pred_patch, ground_patch = run_validation(output, ground_patch_func)
            # print(pred_patch)
            repair_result.append({"output": output, "valid": valid, "pred_patch": pred_patch, "ground_patch": ground_patch})
            if valid: 
                break

            repair_results.append(repair_result)
            repair[file] = repair_results
            with open(f"/home/wyx/KitPatch-63E8/Results/repair_result/{args.dataset}_{output_filename}.json", 'w') as f:
                json.dump(repair, f, indent=4)
            
            if valid:
                break


def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path) 
    model = Model(model, config, tokenizer, args)
    model.to(device)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  

    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    model_dir = os.path.join(args.model_dir, '{}'.format(checkpoint_prefix))  
    model = torch.load(model_dir)
    
    return model, tokenizer    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bigvul",
                        help="Dataset to use")
    parser.add_argument("--n_example", type=int, default=5)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--chance", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # model
    # parser.add_argument("--model_dir", default="saved_models", type=str)
    parser.add_argument("--model_dir", default="/media/wyx/models/saved_models", type=str)
    parser.add_argument("--model_name_or_path", default="/media/models/unixcoder/unixcoder-base-nine", type=str, 
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--sim_threshold", default=0, type=float,
                        help="similarity threshold for retrieving examples.")
    
    args = parser.parse_args()

    out_folder = f'/home/wyx/KitPatch-63E8/Results/cot1/{args.dataset}'
    os.makedirs(out_folder, exist_ok=True)

    # d4j_dataset = clean_parse_bigvul(args.dataset, "/home/wyx/KitPatch-63E8/Datasets/")
    set_seed(args.seed)

    # total_files = list(d4j_dataset.keys())
    # num_test = int(len(total_files) * 0.2)
    # test_files = random.choices(total_files, k=num_test)
    # think_dataset = {key: value for key, value in d4j_dataset.items() if key not in test_files}   
    # inference_dataset = {key: value for key, value in d4j_dataset.items() if key in test_files}

    # generate_chain_of_thought(args, think_dataset)
    
    # Select_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # Select_model = AutoModel.from_pretrained(args.model_dir)
    # get_embedding(args, Select_model, Select_tokenizer, think_dataset)
    
    # ==========begin: generate & save COT embeddings=====================
    # with open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/think_dataset.json", 'r') as f:
    #     think_dataset = json.load(f)
    # encodeCOTEmbedding(args, think_dataset)
    # ===========end: generate & save COT embeddings=====================
    # ==========begin: generate & save inference embeddings=====================
    # with open(f"/home/wyx/KitPatch-63E8/Datasets/source_code/{args.dataset}/inference_dataset.json", 'r') as f:
    #     inference_dataset = json.load(f)
    # encodeInferenceEmbedding(args, inference_dataset)
    # ===========end: generate & save inference embeddings=====================

    # ==========begin: RAG for inference_dataset=====================
    # get_examples_by_embedding(args)
    # get_examples_by_embedding_v1(args)
    # ==========end: RAG for inference_dataset=====================
    # merge_json_files()
    # ==========begin: get train_data from think_dataset=====================
    get_examples(args)
    # ==========end: get train_data from think_dataset=====================

    # inference_embedding_dataset = loadInferenceEmbedding(args)
    # cot_embedding_dataset = loadCOTEmbedding(args)

    # Select_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Select_model = AutoModel.from_pretrained(args.model_name_or_path)

    # chain_of_thought_repair(args, Select_model, Select_tokenizer, cot_embedding_dataset, inference_embedding_dataset)


if __name__ =="__main__":
    main()
