import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import Qwen2VLProcessor
from dataclasses import dataclass
from typing import Dict, List, Any
import os

class Qwen2VLPreferenceDataset(Dataset):
    """
    读取偏好数据集 (包含 image, question, chosen, rejected)
    """

    def __init__(self, data_path: str, processor: Qwen2VLProcessor, max_length: int = 4096, img_root: str = ""):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.img_root = img_root

        # JSON 格式， MMIF-23k 数据集
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"共 {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        img_rel_path = item['images'][0]
        img_path = os.path.join(self.img_root, img_rel_path)

        question = item['conversations'][0]['value']
        # 移除问题中的 <image> 占位符，因为 processor.apply_chat_template 会自动处理
        question = question.replace('<image>\n', '').replace('\n<image>', '').strip()

        chosen_text = item['chosen']['value']
        rejected_text = item['rejected']['value']

        image = Image.open(img_path).convert("RGB")

        # 构造 Qwen2-VL 的对话消息格式
        def create_msg(ans):
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]},
                {"role": "assistant", "content": ans}
            ]

        # 渲染为带有特殊 Token (如 <|vision_start|>) 的纯文本
        prompt_chosen = self.processor.apply_chat_template(create_msg(chosen_text), tokenize=False,
                                                           add_generation_prompt=False)
        prompt_rejected = self.processor.apply_chat_template(create_msg(rejected_text), tokenize=False,
                                                             add_generation_prompt=False)

        # 调用 processor
        inputs_chosen = self.processor(
            text=[prompt_chosen],
            images=[image],
            padding=False,
            return_tensors="pt"
        )

        inputs_rejected = self.processor(
            text=[prompt_rejected],
            images=[image],
            padding=False,
            return_tensors="pt"
        )

        # 剥离 batch 维度 (processor 默认会在最前面加一个 1 的维度)
        return {
            "input_ids_chosen": inputs_chosen["input_ids"][0],
            "attention_mask_chosen": inputs_chosen["attention_mask"][0],
            "input_ids_rejected": inputs_rejected["input_ids"][0],
            "attention_mask_rejected": inputs_rejected["attention_mask"][0],
            "pixel_values": inputs_chosen["pixel_values"],
            "image_grid_thw": inputs_chosen["image_grid_thw"]
        }


@dataclass
class RewardDataCollatorWithPadding:
    """
    对每个 Batch 处理，进行 Padding
    """
    processor: Qwen2VLProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_chosen = [f["input_ids_chosen"] for f in features]
        attention_mask_chosen = [f["attention_mask_chosen"] for f in features]
        input_ids_rejected = [f["input_ids_rejected"] for f in features]
        attention_mask_rejected = [f["attention_mask_rejected"] for f in features]

        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [f["image_grid_thw"] for f in features]

        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        # 对文本进行 Right Padding (右侧补齐)
        batch_input_ids_chosen = torch.nn.utils.rnn.pad_sequence(
            input_ids_chosen, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        batch_attention_mask_chosen = torch.nn.utils.rnn.pad_sequence(
            attention_mask_chosen, batch_first=True, padding_value=0
        )
        batch_input_ids_rejected = torch.nn.utils.rnn.pad_sequence(
            input_ids_rejected, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        batch_attention_mask_rejected = torch.nn.utils.rnn.pad_sequence(
            attention_mask_rejected, batch_first=True, padding_value=0
        )

        # 因为不同图片的切片数量不同，Qwen2 官方规定像素特征直接在第 0 维度进行拼接 (cat)
        batch_pixel_values = torch.cat(pixel_values, dim=0)
        batch_image_grid_thw = torch.cat(image_grid_thw, dim=0)

        return {
            "input_ids_chosen": batch_input_ids_chosen,
            "attention_mask_chosen": batch_attention_mask_chosen,
            "input_ids_rejected": batch_input_ids_rejected,
            "attention_mask_rejected": batch_attention_mask_rejected,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw
        }