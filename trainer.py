import torch
from transformers import Qwen2VLProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from models import Qwen2VLRewardModel
from dataset import Qwen2VLPreferenceDataset, RewardDataCollatorWithPadding
from loss import PairwiseRankingLoss
from transformers import BitsAndBytesConfig
from transformers.optimization import get_linear_schedule_with_warmup
from swanlab.integration.transformers import SwanLabCallback
import numpy as np
import random
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"随机种子已固定为: {seed}")


class RewardTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = PairwiseRankingLoss()
        self.ref_model = ref_model
        self._acc_total = 0
        self._acc_count = 0

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        重写此方法来自定义优化器和 Warmup
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999)
        )

        warmup_steps = self.args.get_warmup_steps(num_training_steps)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        def get_batch(suffix):
            return {
                "input_ids": inputs[f"input_ids_{suffix}"],
                "attention_mask": inputs[f"attention_mask_{suffix}"],
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"]
            }

        chosen_outputs = model(**get_batch("chosen"))
        rejected_outputs = model(**get_batch("rejected"))

        with torch.no_grad():
            ref_outputs = self.ref_model(**get_batch("chosen"))
            ref_chosen_logits = ref_outputs["logits"]

        loss, r_loss, kl_loss, lb_loss, z_loss, accuracy = self.loss_fn(
            chosen_rewards=chosen_outputs["scores"],
            rejected_rewards=rejected_outputs["scores"],
            chosen_logits=chosen_outputs["logits"],
            ref_chosen_logits=ref_chosen_logits, # 👈 传入真实的 Logits
            attention_mask=inputs["attention_mask_chosen"],
            router_logits=chosen_outputs.get("router_logits", None)
        )

        self._acc_total += accuracy.item()
        self._acc_count += 1

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "train/ranking_loss": r_loss.item(),
                 "train/kl_loss": kl_loss.item(),
                "train/moe_lb_loss": lb_loss.item(),
                "train/moe_z_loss": z_loss.item()
            })

        # print(f"KL Loss: {kl_loss.item():.6f}, Requires Grad: {kl_loss.requires_grad}")
        # print(f"lb Loss: {lb_loss.item():.6f}, Requires Grad: {lb_loss.requires_grad}")
        # print(f"rk Loss: {r_loss.item():.6f}, Requires Grad: {r_loss.requires_grad}")
        # print(f"z Loss: {z_loss.item():.6f}, Requires Grad: {z_loss.requires_grad}")

        return (loss, (loss, chosen_outputs, rejected_outputs)) if return_outputs else loss

    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=None,
    ):

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, None, None)

    def log(self, logs: dict, *args, **kwargs):
        """
        每逢 logging_steps，计算一次平均准确率发送给 SwanLab
        """
        if self._acc_count > 0:
            logs["train_accuracy"] = self._acc_total / self._acc_count
            self._acc_total = 0
            self._acc_count = 0

        super().log(logs, *args, **kwargs)


def main():
    seed_everything(42)
    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    data_path = "./v2/dpo/mmif_23k_4o_qwen2_5.json"

    processor = Qwen2VLProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=320 * 28 * 28,
                                                 use_fast=False)

    model = Qwen2VLRewardModel(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score_head"],  # score_head 全量训练并保存
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    model = get_peft_model(model, lora_config)

    try:
        text_embeddings = model.get_submodule("base_model.model.model.embed_tokens")
        text_embeddings.register_forward_hook(lambda module, input, output: output.requires_grad_(True))
    except AttributeError:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.register_forward_hook(lambda module, input, output: output.requires_grad_(True))

    ref_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("正在加载参考模型...")
    ref_model = Qwen2VLRewardModel(
        model_path,
        quantization_config=ref_bnb_config,
        device_map="auto"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    model.print_trainable_parameters()  # 查看有多少参数要训练

    train_dataset = Qwen2VLPreferenceDataset(data_path, processor, max_length=4096)
    data_collator = RewardDataCollatorWithPadding(processor)

    training_args = TrainingArguments(
        output_dir="./output/qwen2vl_reward_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_train_epochs=1,
        logging_steps=5,
        max_grad_norm=40.0,

        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
    )

    swanlab_callback = SwanLabCallback(
        project="Qwen2-VL-finetune",
        experiment_name="qwen2-vl-rm",
        config={
            "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct",
            "lora_rank": 64,
            "lora_alpha": 16,
        },
    )

    trainer = RewardTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
    )

    print("开始训练.")
    trainer.train()


if __name__ == "__main__":
    main()