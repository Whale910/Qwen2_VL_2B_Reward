# Qwen2-VL-2B Reward Model

本项目是基于 **Qwen2-VL-2B** 构建的多模态奖励模型（Reward Model），专门用于评估视觉-语言任务中的回答质量。

## 📁 项目结构

```
Qwen2_VL_2B_Reward/
├── models.py                 # 奖励模型网络定义
├── dataset.py                # Dataset 与 DataCollator
├── loss.py                   # 损失函数实现
├── trainer.py                # 主训练循环脚本
└── test_reward_model.py      # 模型推理与测试脚本
```

## 🛠️ 环境安装

### 1. 创建虚拟环境

```
conda create -n qwen2_reward python=3.10.19
conda activate qwen2_reward
```

### 2. 安装核心依赖

```
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
pip install transformers==4.57.6 accelerate==1.13.0 vllm==0.16.0
pip install peft==0.18.1 bitsandbytes==0.49.2
# 数据处理
pip install datasets==4.6.1 pandas==2.3.3 pyarrow==23.0.1 pillow==12.1.1 einops==0.8.2

# 实验监控 (推荐使用 SwanLab)
pip install swanlab==0.7.10 tqdm==4.67.3 loguru==0.7.3
```

## 📊 数据准备

本项目使用 [MMIF-23k](https://huggingface.co/datasets/ChrisDing1105/MMIF-23k)  作为主要训练集，并使用 [VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench) 作为评估数据集。

## 🚀 快速开始

### 1. 模型训练

使用 `trainer.py` 启动训练。

```
python trainer.py
```

### 2. 模型测试与评估

训练完成后，使用 `test_reward_model.py` 在VL_Benchmark上进行验证。

```
python test_reward_model.py \
    --checkpoint_path "./output/qwen2vl_reward_lora/checkpoint-1410" \
    --data_path "test-00000-of-00001.parquet" \
    --base_model_path Qwen/Qwen2-VL-2B-Instruct \
    --output_path ./reward_model_results.jsonl \
    --seed 42
```

## 🤗 模型权重 (Model Weights)

本项目训练好的奖励模型（包含 LoRA 适配器与 Score Head 权重）已上传至 Hugging Face：

* **模型地址**: [Whale0910/Qwen2_VL_2B_Reward](https://huggingface.co/Whale0910/Qwen2_VL_2B_Reward)
