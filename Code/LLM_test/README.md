---
## LLM_test 目录说明
本目录用于多LLM模型对比实验的全流程处理，包含推理、去重、得分计算、统计分析全链路工具脚本，以及各模型实验结果数据。

### 🛠️ 核心脚本功能说明
| 脚本名称 | 功能描述 | 使用方法 |
|---------|---------|---------|
| `run_llm_inference.py` | 批量调用LLM API进行推理的核心脚本，支持断点续传、多模型配置池 | `python run_llm_inference.py --use-model <模型名> --output <输出文件路径>`<br>可选参数：`--temperature <温度> --max-tokens <最大生成长度>` |
| `test_gpt_api.py` | GPT API连通性测试脚本 | `python test_gpt_api.py` 直接运行即可验证API配置是否正确 |
| `deduplicate_jsonl.py` | JSONL文件去重工具，默认按`generated_explanation`字段去重 | `python deduplicate_jsonl.py --input <输入文件> --output <输出文件>`<br>可选参数：`--deduplicate-field <去重字段名>` |
| `compute_reward_score.py` | 计算LLM输出的reward分项得分，调用patch_generation模块的reward函数 | 先修改脚本内的INPUT_PATH和OUTPUT_PATH，然后直接运行`python compute_reward_score.py` |
| `compute_groundtruth_reward.py` | 专门用于计算ground truth的reward得分，使用测试数据中的`generated_explanation`字段作为基准输出 | 直接运行`python compute_groundtruth_reward.py`，自动读取test_after_sft3.jsonl计算 |
| `stat_reward.py` | 统计reward得分并生成可视化图表，支持生成总reward和r_structure的直方图、箱线图 | `python stat_reward.py --input <带reward的jsonl文件> --prefix <模型名前缀>`<br>例：`python stat_reward.py --input gpt-5.4_output_with_reward.jsonl --prefix gpt-5.4` |
| `compare_r_structure.py` | 多模型r_structure得分对比工具，生成所有模型的r_structure分布对比图 | 直接运行`python compare_r_structure.py`，自动读取所有模型的得分文件生成对比图表 |

### 📁 文件命名规范
#### 数据文件类
| 文件名格式 | 含义说明 |
|-----------|---------|
| `test_after_sft3.jsonl` | 官方测试数据集，共286个样本，包含所有测试样本的prompt、ground truth等信息 |
| `{模型名}_output.jsonl` | 模型推理原始输出文件，未做任何处理 |
| `{模型名}_output_deduplicated.jsonl` | 去重后的模型输出文件 |
| `{模型名}_output_with_reward.jsonl` | 带reward全部分项得分的模型输出文件 |
| `groundtruth_with_reward.jsonl` | 基准ground truth的得分文件，使用测试集自带的generated_explanation计算 |

#### 结果文件类
| 文件名格式 | 含义说明 |
|-----------|---------|
| `{模型名}_reward_stat_result.txt` | 模型得分统计结果文本文件，包含各分项得分的均值、标准差、最值等 |
| `{模型名}_reward_histogram.png` | 总reward得分分布直方图 |
| `{模型名}_reward_boxplot.png` | reward所有分项得分对比箱线图 |
| `{模型名}_r_structure_histogram.png` | r_structure分项得分分布直方图 |
| `{模型名}_r_structure_boxplot.png` | r_structure分项得分单独箱线图 |
| `all_models_r_structure_histogram.png` | 所有模型r_structure得分直方图对比图 |
| `all_models_r_structure_boxplot.png` | 所有模型r_structure得分箱线图对比图 |

### 📋 典型使用流程（新增模型测试）
1. 在`run_llm_inference.py`的`MODEL_CONFIGS`字典中添加新模型的API配置
2. 运行推理：`python run_llm_inference.py --use-model <新模型名> --output <新模型名>_output.jsonl`
3. 去重处理：`python deduplicate_jsonl.py --input <新模型名>_output.jsonl --output <新模型名>_output_deduplicated.jsonl`
4. 修改`compute_reward_score.py`的输入输出路径为去重后的文件，运行计算得分
5. 生成统计图表：`python stat_reward.py --input <新模型名>_output_with_reward.jsonl --prefix <新模型名>`
6. 运行`compare_r_structure.py`更新多模型对比图表
---