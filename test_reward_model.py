import torch
import argparse
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from collections import defaultdict
import logging
import datetime

from transformers import Qwen2VLProcessor
from peft import PeftModel
from models import Qwen2VLRewardModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reward_model_test.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="测试Qwen2-VL奖励模型在VL_RewardBench上的性能")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='LoRA和score_head检查点路径')
    parser.add_argument('--base_model_path', type=str, default='Qwen/Qwen2-VL-2B-Instruct', help='基础模型路径')
    parser.add_argument('--data_path', type=str, required=True, help='VL_RewardBench测试数据路径(.parquet文件)')
    parser.add_argument('--output_path', type=str, default='reward_model_results.jsonl', help='结果输出路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_processor(args):
    """加载 Qwen2-VL 处理器、基座模型和 LoRA 权重"""
    logger.info(f"从 {args.base_model_path} 加载基础模型与处理器...")
    processor = Qwen2VLProcessor.from_pretrained(
        args.base_model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=320 * 28 * 28
    )

    base_model = Qwen2VLRewardModel(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    logger.info(f"正在挂载 Checkpoint: {args.checkpoint_path}")
    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint_path,
        is_trainable=False
    )
    model.eval()
    logger.info("模型加载成功！")

    return model, processor


def process_image(image_bytes):
    if isinstance(image_bytes, bytes):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    else:
        return image_bytes


def evaluate_pair(model, processor, image, query, response1, response2):
    """评估一对回答的质量"""
    processed_image = process_image(image)

    def build_message(resp):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": query + "\n" + resp},
                ],
            }
        ]

    text1 = processor.apply_chat_template(build_message(response1), tokenize=False, add_generation_prompt=False)
    text2 = processor.apply_chat_template(build_message(response2), tokenize=False, add_generation_prompt=False)

    inputs = processor(
        text=[text1, text2],
        images=[processed_image, processed_image],
        padding=True,
        return_tensors="pt"
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(**inputs)
        scores_tensor = outputs["scores"].squeeze()
        score1 = scores_tensor[0].item()
        score2 = scores_tensor[1].item()

    better_idx = 0 if score1 > score2 else 1
    scores = [float(score1), float(score2)]

    return better_idx, scores


def get_dataset_from_id(id_str):
    """根据ID确定数据集类型"""
    split_index = min([idx for idx in (id_str.find('_'), id_str.find('-')) if idx != -1] or [len(id_str)])
    id_prefix = id_str[:split_index]

    if id_prefix == "RLAIF":
        return "rlaif-v"
    elif id_prefix == "RLHF":
        return "rlhf-v"
    elif id_prefix in ["mathverse", "mmmu"]:
        return "reasoning_tasks"
    elif id_prefix == "wildvision":
        return "wildvision-battle"
    elif id_prefix == 'hallucination':
        return "povid"
    else:
        return "vlfeedback"


def test_model(model, processor, data_path, output_path):
    """执行 VL_RewardBench 评测流水线"""
    logger.info(f"加载测试数据: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"加载了 {len(df)} 条测试样本")

    results = []
    dataset_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="测试中"):
        item = row.to_dict()
        query = item["query"]
        responses = item["response"]
        human_ranking = item["human_ranking"]

        better_idx, scores = evaluate_pair(
            model,
            processor,
            item["image"]['bytes'],
            query,
            responses[0],
            responses[1]
        )

        correct = (better_idx == 0 and human_ranking[1] > human_ranking[0]) or \
                  (better_idx == 1 and human_ranking[0] > human_ranking[1])

        dataset_type = get_dataset_from_id(item["id"])
        dataset_results[dataset_type]["total"] += 1
        if correct:
            dataset_results[dataset_type]["correct"] += 1

        result = {
            "id": item["id"],
            "query": query,
            "response": responses,
            "human_ranking": human_ranking.tolist() if isinstance(human_ranking, np.ndarray) else human_ranking,
            "model_scores": scores,
            "model_choice": int(better_idx),
            "correct": bool(correct),
            "dataset": dataset_type
        }
        results.append(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json_result = json.dumps(result, ensure_ascii=False,
                                     default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
            f.write(json_result + '\n')

    total_correct = sum(res["correct"] for res in results)
    accuracy = total_correct / len(results)
    logger.info(f"总体准确率: {accuracy:.4f} ({total_correct}/{len(results)})")

    logger.info("各子数据集准确率:")
    group_mapping = {
        "vlfeedback": "general", "povid": "hallucination",
        "reasoning_tasks": "reasoning", "rlhf-v": "hallucination",
        "rlaif-v": "hallucination", "wildvision-battle": "general"
    }
    group_correct = defaultdict(int)
    group_total = defaultdict(int)

    for dataset, stats in dataset_results.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"  {dataset}: {acc:.4f} ({stats['correct']}/{stats['total']})")

        group = group_mapping.get(dataset, "other")
        group_correct[group] += stats["correct"]
        group_total[group] += stats["total"]

    logger.info("分组大类准确率:")
    task_list = ['reasoning', 'hallucination', 'general']
    for group in task_list:
        if group_total[group] > 0:
            acc = group_correct[group] / group_total[group]
            logger.info(f"  {group}: {acc:.4f} ({group_correct[group]}/{group_total[group]})")

    macro_avg = sum(group_correct[k] / group_total[k] for k in task_list if group_total[k] > 0) / len(
        [k for k in task_list if group_total[k] > 0])
    logger.info(f"宏平均准确率 (Macro-Avg): {macro_avg:.4f}")

    return accuracy


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(args.checkpoint_path.strip('/'))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_path.endswith('.jsonl'):
        output_path = args.output_path.replace('.jsonl', f'_{model_name}_{timestamp}.jsonl')
    else:
        output_path = f"{args.output_path}_{model_name}_{timestamp}.jsonl"

    model, processor = load_model_and_processor(args)
    test_model(model, processor, args.data_path, output_path)

    logger.info(f"评测日志和结果已保存到 {output_path}")

if __name__ == "__main__":
    main()